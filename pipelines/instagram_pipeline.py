"""
Instagram Pipeline - Ana çalıştırma scripti.

Kullanım:
  python pipelines/instagram_pipeline.py \
    --video input/video.mp4 \
    --background input/background.png \
    --character input/character.png \
    --output output/result.mp4

Sadece arka plan değiştirme:
  python pipelines/instagram_pipeline.py \
    --video input/video.mp4 \
    --background input/background.png \
    --output output/result.mp4
"""

import sys
import shutil
import time
from pathlib import Path
import logging

import click
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.video_processor import resize_video, extract_frames, get_video_info
from src.segmenter import process_frames_batch
from src.compositor import process_all_frames
from src.video_assembler import frames_to_video, extract_audio


# ============== LOGGING SETUP ==============
def setup_logging(log_level: str = "INFO"):
    """Detaylı logging kurulumu"""
    log_dir = Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


logger = None


def load_config(config_path: str = "config.yaml") -> dict:
    if Path(config_path).exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def cleanup_temp(temp_dir: str):
    """Geçici dosyaları temizle."""
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)
        print(f"🧹 Temp dosyalar temizlendi: {temp_dir}")


@click.command()
@click.option("--video", required=True, help="Girdi video dosyası (.mp4, .mov)")
@click.option("--background", required=True, help="Yeni arka plan görseli (.png, .jpg)")
@click.option("--character", default=None, help="Yeni karakter görseli (opsiyonel)")
@click.option("--output", default="output/result.mp4", help="Çıktı video yolu")
@click.option("--config", default="config.yaml", help="Config dosyası")
@click.option("--keep-temp", is_flag=True, help="Geçici dosyaları sil")
@click.option("--test", is_flag=True, help="Test modu (ilk 30 frame)")
@click.option("--max-frames", default=None, type=int, help="Max frame sayısı (test için)")
@click.option("--log-level", default="INFO", help="Log seviyesi: DEBUG, INFO, WARNING, ERROR")
def run_pipeline(video, background, character, output, config, keep_temp, test, max_frames, log_level):
    global logger
    
    # Setup logging
    logger = setup_logging(log_level)
    
    logger.info("="*60)
    logger.info("🎬 VIDEO CHEF PIPELINE BAŞLADI")
    logger.info("="*60)
    """🍳 Video Chef - Video'nun arka planını ve karakterini değiştir."""

    start_time = time.time()
    cfg = load_config(config)

    # Log input parameters
    logger.info("📥 INPUT PARAMETERS:")
    logger.info(f"   Video:      {video}")
    logger.info(f"   Background: {background}")
    logger.info(f"   Character:  {character or 'None (skipping)'}")
    logger.info(f"   Output:     {output}")
    logger.info(f"   Config:     {config}")
    logger.info(f"   Test mode:  {test}")
    logger.info(f"   Max frames: {max_frames}")
    logger.info(f"   Log level:  {log_level}")
    logger.info("="*60)

    # Klasörleri oluştur
    temp_dir = cfg.get("paths", {}).get("temp_dir", "./temp")
    frames_dir = f"{temp_dir}/frames"
    masks_dir = f"{temp_dir}/masks"
    composited_dir = f"{temp_dir}/composited"
    resized_video = f"{temp_dir}/resized.mp4"

    # --- ADIM 1: Video küçült ---
    logger.info("📐 ADIM 1/5: Video resizing...")
    resolution = cfg.get("video", {}).get("target_resolution", 720)
    logger.info(f"   Target resolution: {resolution}p")
    resize_video(video, resized_video, resolution)

    # Video bilgisi
    info = get_video_info(resized_video)
    fps = info.get("fps", 30)
    logger.info(f"   Video info: {info.get('width')}x{info.get('height')} @ {fps:.1f} FPS")
    logger.info(f"   Duration: {info.get('duration', 0):.2f} seconds")

    # --- ADIM 2: Sesi çıkar ---
    logger.info("🔊 ADIM 2/5: Audio extraction...")
    audio_path = f"{temp_dir}/audio.aac"
    audio_result = extract_audio(video, audio_path)
    if audio_result:
        logger.info(f"   Audio saved: {audio_path}")
    else:
        logger.warning("   No audio found or extraction failed")

    # --- ADIM 3: Frame'leri çıkar ---
    logger.info("🎞️  ADIM 3/5: Frame extraction...")
    frames = extract_frames(resized_video, frames_dir, fps=int(fps))
    logger.info(f"   Total frames extracted: {len(frames)}")

    # Frame limit
    if test or max_frames:
        frames = frames[:max_frames or 30]
        logger.info(f"   Limited to {len(frames)} frames (test mode)")

    # --- ADIM 4: Segmentasyon ---
    logger.info(f"✂️  ADIM 4/5: Segmentation ({len(frames)} frames)...")
    model = cfg.get("segmentation", {}).get("model", "u2net_human_seg")
    device = cfg.get("segmentation", {}).get("device", "cuda")
    batch_size = cfg.get("segmentation", {}).get("batch_size", 2)
    
    logger.info(f"   Model: {model}")
    logger.info(f"   Device: {device}")
    logger.info(f"   Batch size: {batch_size}")

    masks = process_frames_batch(frames, masks_dir, model, device, batch_size)
    logger.info(f"   Masks generated: {len(masks)}")

    # --- ADIM 5: Compositing ---
    logger.info(f"🎨 ADIM 5/5: Compositing...")
    logger.info(f"   Background: {background}")
    logger.info(f"   Character: {character or 'None'}")
    
    final_frames = process_all_frames(
        frames, masks,
        background_path=background,
        character_path=character,
        output_dir=composited_dir,
        config=cfg
    )
    logger.info(f"   Composited frames: {len(final_frames)}")

    # --- Son: Video oluştur ---
    logger.info("🎬 Final video assembly...")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    frames_to_video(
        composited_dir,
        output,
        fps=fps,
        audio_path=audio_path if Path(audio_path).exists() else None,
        instagram_optimize=cfg.get("output", {}).get("instagram_optimize", True)
    )

    # Temizlik
    if not keep_temp:
        cleanup_temp(temp_dir)

    elapsed = time.time() - start_time
    
    logger.info("="*60)
    logger.info("✅ PIPELINE TAMAMLANDI!")
    logger.info(f"   Total time: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")
    logger.info(f"   Output: {output}")
    logger.info(f"   Frames processed: {len(frames)}")
    logger.info("="*60)
    
    print(f"\n✅ TAMAMLANDI! ({elapsed:.1f} saniye)")
    print(f"📱 Çıktı: {output}")
    print(f"🎉 Instagram'a hazır!")
    print(f"📝 Detaylı log: output/logs/")


if __name__ == "__main__":
    run_pipeline()
