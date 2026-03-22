"""
Video assembler: İşlenmiş frame'leri tekrar videoya birleştirir.
"""
import subprocess
import os
from pathlib import Path


def frames_to_video(
    frames_dir: str,
    output_path: str,
    fps: float = 30,
    audio_path: str = None,
    instagram_optimize: bool = True
) -> str:
    """
    Frame klasöründeki JPEG'leri video haline getirir.
    Opsiyonel: orijinal videodan sesi ekler.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame_pattern = os.path.join(frames_dir, "frame_%06d_final.jpg")

    # Temel video oluşturma komutu
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
    ]

    # Ses ekle (orijinal videodan)
    if audio_path and Path(audio_path).exists():
        cmd += ["-i", audio_path, "-c:a", "aac", "-b:a", "128k", "-shortest"]

    # Instagram optimizasyonu
    if instagram_optimize:
        cmd += [
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "slow",           # Daha iyi sıkıştırma
            "-profile:v", "high",
            "-level", "4.0",
            "-pix_fmt", "yuv420p",       # Instagram uyumluluğu
            "-movflags", "+faststart",   # Hızlı başlatma
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"  # Çift sayı boyutlar
        ]
    else:
        cmd += ["-c:v", "libx264", "-crf", "23"]

    cmd.append(output_path)

    print(f"🎬 Video oluşturuluyor: {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg hatası: {result.stderr}")

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"✅ Video hazır: {output_path} ({file_size:.1f} MB)")
    return output_path


def extract_audio(video_path: str, output_path: str) -> str:
    """
    Videodan ses trackini çıkarır.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "aac",
        "-b:a", "128k",
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"🔊 Ses çıkarıldı: {output_path}")
        return output_path
    return None


if __name__ == "__main__":
    print("Assembler test - bağımsız çalıştırmak için instagram_pipeline.py kullanın.")
