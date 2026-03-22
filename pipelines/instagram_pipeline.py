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

import click
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.video_processor import resize_video, extract_frames, get_video_info
from src.segmenter import process_frames_batch
from src.compositor import process_all_frames
from src.video_assembler import frames_to_video, extract_audio


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
def run_pipeline(video, background, character, output, config, keep_temp, test):
    """🍳 Video Chef - Video'nun arka planını ve karakterini değiştir."""

    start_time = time.time()
    cfg = load_config(config)

    print("\n🎬 VIDEO CHEF BAŞLIYOR\n" + "="*40)
    print(f"📹 Video:      {video}")
    print(f"🖼️  Arka plan:  {background}")
    print(f"🎭 Karakter:   {character or 'Değiştirilmeyecek'}")
    print(f"💾 Çıktı:      {output}")
    print("="*40 + "\n")

    # Klasörleri oluştur
    temp_dir = cfg.get("paths", {}).get("temp_dir", "./temp")
    frames_dir = f"{temp_dir}/frames"
    masks_dir = f"{temp_dir}/masks"
    composited_dir = f"{temp_dir}/composited"
    resized_video = f"{temp_dir}/resized.mp4"

    # --- ADIM 1: Video küçült ---
    print("📐 ADIM 1/5: Video küçültülüyor...")
    resolution = cfg.get("video", {}).get("target_resolution", 720)
    resize_video(video, resized_video, resolution)

    # Video bilgisi
    info = get_video_info(resized_video)
    fps = info.get("fps", 30)
    print(f"   → {info.get('width')}x{info.get('height')} @ {fps:.1f} FPS")

    # --- ADIM 2: Sesi çıkar ---
    print("\n🔊 ADIM 2/5: Ses çıkarılıyor...")
    audio_path = f"{temp_dir}/audio.aac"
    extract_audio(video, audio_path)

    # --- ADIM 3: Frame'leri çıkar ---
    print("\n🎞️  ADIM 3/5: Frame'ler çıkarılıyor...")
    frames = extract_frames(resized_video, frames_dir, fps=int(fps))

    if test:
        frames = frames[:30]
        print(f"   → Test modu: Sadece ilk 30 frame işleniyor")

    # --- ADIM 4: Segmentasyon ---
    print(f"\n✂️  ADIM 4/5: Segmentasyon ({len(frames)} frame)...")
    model = cfg.get("segmentation", {}).get("model", "u2net_human_seg")
    device = cfg.get("segmentation", {}).get("device", "cuda")
    batch_size = cfg.get("segmentation", {}).get("batch_size", 2)

    masks = process_frames_batch(frames, masks_dir, model, device, batch_size)

    # --- ADIM 5: Compositing ---
    print(f"\n🎨 ADIM 5/5: Compositing...")
    final_frames = process_all_frames(
        frames, masks,
        background_path=background,
        character_path=character,
        output_dir=composited_dir,
        config=cfg
    )

    # --- Son: Video oluştur ---
    print(f"\n🎬 Final video oluşturuluyor...")
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
    print(f"\n✅ TAMAMLANDI! ({elapsed:.1f} saniye)")
    print(f"📱 Çıktı: {output}")
    print(f"🎉 Instagram'a hazır!\n")


if __name__ == "__main__":
    run_pipeline()
