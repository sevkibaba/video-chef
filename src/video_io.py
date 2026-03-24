"""
video_io.py — Clean FFmpeg wrapper for all video I/O operations.
"""
import subprocess
import json
import shutil
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def get_video_info(video_path: str) -> dict:
    """Return dict with fps, width, height, duration, num_frames."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), {}
    )

    # Parse FPS (can be a fraction like "30000/1001")
    fps_raw = video_stream.get("r_frame_rate", "30/1")
    num, den = fps_raw.split("/")
    fps = float(num) / float(den)

    duration = float(data.get("format", {}).get("duration", 0))
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    num_frames = int(video_stream.get("nb_frames", int(duration * fps)))

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "duration": duration,
        "num_frames": num_frames,
    }


def extract_frames(
    video_path: str,
    output_dir: str,
    fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> list:
    """
    Extract frames as JPEGs into output_dir.
    Returns sorted list of Path objects.
    fps=None means extract every frame at original FPS.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if fps:
        cmd += ["-vf", f"fps={fps}"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-q:v", "2", str(output_dir / "frame_%06d.jpg")]

    logger.info(f"Extracting frames → {output_dir}")
    subprocess.run(cmd, capture_output=True, check=True)

    frames = sorted(output_dir.glob("frame_*.jpg"))
    logger.info(f"  Extracted {len(frames)} frames")
    return frames


def extract_audio(video_path: str, output_path: str) -> Optional[Path]:
    """Extract audio to AAC. Returns None if no audio stream."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-acodec", "copy", str(output_path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or not output_path.exists():
        logger.warning("No audio stream found or extraction failed.")
        return None
    logger.info(f"  Audio extracted → {output_path}")
    return output_path


def assemble_video(
    frames_dir: str,
    output_path: str,
    fps: float,
    audio_path: Optional[str] = None,
    crf: int = 18,
) -> Path:
    """
    Assemble JPEG frames into MP4 with optional audio.
    crf=18 is high quality (lower = bigger file).
    """
    frames_dir = Path(frames_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build input glob — frames must be named frame_000001.jpg etc.
    frame_pattern = str(frames_dir / "frame_%06d.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
    ]

    if audio_path and Path(audio_path).exists():
        cmd += ["-i", str(audio_path)]
        cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p"]
        cmd += ["-c:a", "aac", "-shortest"]
    else:
        cmd += ["-c:v", "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p"]

    cmd.append(str(output_path))

    logger.info(f"Assembling video → {output_path}")
    subprocess.run(cmd, capture_output=True, check=True)
    logger.info(f"  Done: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    return output_path


def resize_video(video_path: str, output_path: str, max_side: int = 720) -> Path:
    """Resize so the shorter side = max_side, keeping aspect ratio."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Scale: if width > height → scale height; else scale width
    vf = f"scale='if(gt(iw,ih),trunc(ih*{max_side}/ih)*2,trunc(iw*{max_side}/iw)*2)':-2"
    # Simpler: scale shortest side to max_side
    vf = f"scale=-2:{max_side}"

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "copy",
        str(output_path)
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
