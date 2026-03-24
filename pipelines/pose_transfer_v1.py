"""
pose_transfer_v1.py — Main pipeline: replace person in video with character JPEG.

Full pipeline:
  1. Extract frames + audio from input video
  2. YOLO auto-detects the person → SAM2 tracks them across all frames
  3. DWPose extracts skeleton from each frame
  4. ControlNet + IP-Adapter generates the character in each pose
  5. Compositor places the generated character onto the original background
  6. FFmpeg assembles the final video with original audio

Usage (SD1.5 — local):
  python pipelines/pose_transfer_v1.py \\
    --video input/video.mp4 \\
    --character input/character.jpg \\
    --output output/result.mp4 \\
    --backend sd15

Usage (SDXL — cloud GPU):
  python pipelines/pose_transfer_v1.py \\
    --video input/video.mp4 \\
    --character input/character.jpg \\
    --output output/result.mp4 \\
    --backend sdxl --device cuda

Usage (test mode — first 30 frames only):
  python pipelines/pose_transfer_v1.py \\
    --video input/video.mp4 --character input/character.jpg \\
    --output output/test.mp4 --max-frames 30
"""

import sys
import shutil
import time
import logging
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.video_io import (
    get_video_info,
    extract_frames,
    extract_audio,
    assemble_video,
)
from src.person_tracker import PersonTracker, track_frames_rembg
from src.pose_extractor import PoseExtractor
from src.character_generator import CharacterGenerator
from src.compositor_v2 import composite_all_frames


# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    log_dir = Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pose_transfer_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("pose_transfer")


logger = None


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    video_path: str,
    character_path: str,
    output_path: str,
    backend: str,
    device: str,
    max_frames: int,
    ip_scale: float,
    controlnet_scale: float,
    num_steps: int,
    sam2_model: str,
    use_rembg_fallback: bool,
    keep_temp: bool,
    prompt: str,
    fast: bool = False,
    only_generated: bool = False,
    output_fps: float = None,
):
    global logger

    logger.info("=" * 65)
    logger.info("🎬 VIDEO CHEF — POSE TRANSFER PIPELINE v1")
    logger.info("=" * 65)
    logger.info(f"  Video:       {video_path}")
    logger.info(f"  Character:   {character_path}")
    logger.info(f"  Output:      {output_path}")
    logger.info(f"  Backend:     {backend.upper()}")
    logger.info(f"  Device:      {device}")
    logger.info(f"  Max frames:  {max_frames or 'all'}")
    logger.info(f"  IP scale:    {ip_scale}")
    logger.info(f"  CN scale:    {controlnet_scale}")
    logger.info(f"  Steps:       {num_steps}")
    logger.info(f"  Fast Mode:   {fast}")
    logger.info(f"  Only Gen:    {only_generated}")
    logger.info("=" * 65)

    start = time.time()
    temp_dir = Path("temp_pose_transfer")
    temp_dir.mkdir(exist_ok=True)

    frames_dir     = temp_dir / "frames"
    poses_dir      = temp_dir / "poses"
    generated_dir  = temp_dir / "generated"
    composited_dir = temp_dir / "composited"

    # ── Step 1: Video info + extract frames ──────────────────────────────────
    logger.info("\n📐 STEP 1/6: Analyzing video + extracting frames...")
    info = get_video_info(video_path)
    fps = info["fps"]
    width = info["width"]
    height = info["height"]
    duration = info["duration"]

    logger.info(f"  {width}x{height} @ {fps:.2f} FPS  ({duration:.1f}s)")
    final_fps = output_fps if output_fps else fps
    if output_fps and output_fps != fps:
        logger.info(f"  Output FPS:  {final_fps} (overridden — output will be slower/longer)")

    frame_files = extract_frames(
        video_path=video_path,
        output_dir=str(frames_dir),
        max_frames=max_frames,
    )
    if not frame_files:
        logger.error("No frames extracted. Check your video file.")
        return

    logger.info(f"  → {len(frame_files)} frames")

    # ── Step 2: Extract audio ─────────────────────────────────────────────────
    logger.info("\n🔊 STEP 2/6: Extracting audio...")
    audio_path = str(temp_dir / "audio.aac")
    audio = extract_audio(video_path, audio_path)

    # ── Step 3: Person tracking (SAM2 or rembg fallback) ─────────────────────
    logger.info("\n🎯 STEP 3/6: Tracking person across frames...")

    if use_rembg_fallback:
        logger.info("  Using rembg fallback (--use-rembg flag set)")
        masks = track_frames_rembg(frame_files)
    else:
        try:
            tracker = PersonTracker(
                model_size=sam2_model,
                checkpoints_dir="checkpoints",
                device=device,
            )
            masks = tracker.track(
                frames_dir=str(frames_dir),
                frame_files=frame_files,
            )
        except (FileNotFoundError, ImportError) as e:
            logger.warning(f"  SAM2 unavailable ({e}) — falling back to rembg segmentation.")
            masks = track_frames_rembg(frame_files)

    logger.info(f"  → Masks for {len(masks)} frames")

    # ── Step 4: Pose extraction ───────────────────────────────────────────────
    logger.info("\n🦴 STEP 4/6: Extracting poses with DWPose...")
    extractor = PoseExtractor(
        detect_resolution=512,
        image_resolution=512,
        device=device,
    )
    pose_images = extractor.extract_batch(
        frame_files=frame_files,
        output_dir=str(poses_dir),
        save=True,
    )
    logger.info(f"  → {len(pose_images)} pose images")

    # ── Step 5: Character generation ─────────────────────────────────────────
    logger.info(f"\n🎨 STEP 5/6: Generating character frames ({backend.upper()})...")
    generator = CharacterGenerator(
        backend=backend,
        device=device,
        ip_scale=ip_scale,
        controlnet_scale=controlnet_scale,
        num_steps=num_steps,
        fast=fast,
    )

    from PIL import Image
    character_image = Image.open(character_path).convert("RGB")

    generated_paths = generator.generate_batch(
        pose_images=pose_images,
        character_image=character_image,
        output_size=(width, height),
        output_dir=str(generated_dir),
        prompt=prompt,
    )
    logger.info(f"  → {len(generated_paths)} generated frames")

    # ── Step 6: Compositing ───────────────────────────────────────────────────
    if only_generated:
        logger.info("\n⚡ STEP 6/6: Skipping composite — using raw AI frames directly.")
        final_paths = generated_paths
        final_dir = generated_dir
    else:
        logger.info("\n🖼️  STEP 6/6: Compositing character onto original background...")
        composited_paths = composite_all_frames(
            original_frame_files=frame_files,
            generated_frame_files=generated_paths,
            masks=masks,
            output_dir=str(composited_dir),
        )
        logger.info(f"  → {len(composited_paths)} composited frames")
        final_paths = composited_paths
        final_dir = composited_dir

    # ── Final: Assemble video ─────────────────────────────────────────────────
    logger.info("\n🎬 Assembling final video...")
    assemble_video(
        frames_dir=str(final_dir),
        output_path=output_path,
        fps=final_fps,
        audio_path=audio_path if audio else None,
    )

    # Cleanup
    if not keep_temp:
        logger.info("🧹 Cleaning up temp files...")
        shutil.rmtree(temp_dir, ignore_errors=True)

    elapsed = time.time() - start
    num_frames = len(final_paths)
    avg_per_frame = elapsed / num_frames if num_frames > 0 else 0

    logger.info("\n" + "=" * 65)
    logger.info("✅ DONE!")
    logger.info(f"   Total time:     {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"   Avg per frame:  {avg_per_frame:.2f}s")
    logger.info(f"   Output:         {output_path}")
    logger.info(f"   Frames:         {num_frames}")
    logger.info("=" * 65)

    print(f"\n✅ Done in {elapsed:.1f}s")
    print(f"📹 Output: {output_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

@click.command()
@click.option("--video",      required=True,  help="Input video path (mp4, mov).")
@click.option("--character",  required=True,  help="Character JPEG/PNG to replace the person.")
@click.option("--output",     default="output/result.mp4", show_default=True, help="Output video path.")
@click.option("--backend",    default="sd15", show_default=True, type=click.Choice(["sd15", "sdxl"]),
              help="Diffusion backend. sd15=faster/less VRAM, sdxl=higher quality.")
@click.option("--device",     default="auto", show_default=True, help="cuda / mps / cpu / auto (auto-detects best available).")
@click.option("--max-frames", default=None, type=int, help="Limit number of frames (for testing).")
@click.option("--ip-scale",   default=0.7, show_default=True, type=float,
              help="IP-Adapter scale (0–1). Higher = more like character JPEG.")
@click.option("--controlnet-scale", default=0.9, show_default=True, type=float,
              help="ControlNet pose scale (0–1). Higher = stricter pose adherence.")
@click.option("--steps",      default=25, show_default=True, type=int, help="Diffusion inference steps.")
@click.option("--sam2-model", default="large", show_default=True,
              type=click.Choice(["large", "base", "small", "tiny"]),
              help="SAM2 model size. Larger = better tracking, more VRAM.")
@click.option("--use-rembg",  is_flag=True,
              help="Use rembg instead of SAM2 for person segmentation (no SAM2 checkpoint needed).")
@click.option("--keep-temp",  is_flag=True, help="Keep temp files for debugging.")
@click.option("--log-level",  default="INFO", show_default=True, help="Logging level.")
@click.option("--prompt",     default="a person, full body, photorealistic, detailed, sharp",
              show_default=True, help="Positive prompt for generation.")
@click.option("--fast",           is_flag=True, help="Fast mode: lower resolution, 8 steps, 5 frames.")
@click.option("--only-generated", is_flag=True, help="Skip compositing; use raw AI frames (no original background).")
@click.option("--output-fps",     default=None, type=float, help="Override output FPS (lower = longer/slower video). Default: same as input.")
def main(video, character, output, backend, device, max_frames,
         ip_scale, controlnet_scale, steps, sam2_model,
         use_rembg, keep_temp, log_level, prompt, fast, only_generated, output_fps):
    """🍳 Video Chef — Pose-guided character replacement pipeline."""
    global logger
    logger = setup_logging(log_level)

    # Apply fast mode defaults if not specified
    if fast:
        if max_frames is None:
            max_frames = 5
        # We don't override steps if user manually set them
        # (check if default 25 is still there)
        if steps == 25:
            steps = 8
        use_rembg = True  # Fast mode uses rembg by default

    run_pipeline(
        video_path=video,
        character_path=character,
        output_path=output,
        backend=backend,
        device=device,
        max_frames=max_frames,
        ip_scale=ip_scale,
        controlnet_scale=controlnet_scale,
        num_steps=steps,
        sam2_model=sam2_model,
        use_rembg_fallback=use_rembg,
        keep_temp=keep_temp,
        prompt=prompt,
        fast=fast,
        only_generated=only_generated,
        output_fps=output_fps,
    )


if __name__ == "__main__":
    main()
