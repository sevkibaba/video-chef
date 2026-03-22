"""
Video processor: Video'yu küçültür ve frame'lere ayırır.
"""
import subprocess
import os
from pathlib import Path
import yaml


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def resize_video(input_path: str, output_path: str, resolution: int = 720) -> str:
    """
    Video'yu hedef çözünürlüğe küçültür.
    Orijinal aspect ratio korunur.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"scale=-2:{resolution}",   # En-boy oranını koru
        "-c:v", "libx264",
        "-crf", "23",
        "-preset", "fast",
        "-c:a", "aac",
        "-b:a", "128k",
        output_path
    ]

    print(f"📐 Video küçültülüyor: {input_path} → {resolution}p")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg hatası: {result.stderr}")

    print(f"✅ Video küçültüldü: {output_path}")
    return output_path


def extract_frames(video_path: str, output_dir: str, fps: int = 30) -> list:
    """
    Video'dan frame'leri PNG olarak çıkarır.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    frame_pattern = os.path.join(output_dir, "frame_%06d.png")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        frame_pattern
    ]

    print(f"🎞️  Frame'ler çıkarılıyor ({fps} FPS)...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg hatası: {result.stderr}")

    frames = sorted(Path(output_dir).glob("frame_*.png"))
    print(f"✅ {len(frames)} frame çıkarıldı → {output_dir}")
    return [str(f) for f in frames]


def get_video_info(video_path: str) -> dict:
    """
    Video hakkında bilgi döndürür (süre, fps, çözünürlük).
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        video_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    import json
    data = json.loads(result.stdout)
    video_stream = next(
        (s for s in data["streams"] if s["codec_type"] == "video"), None
    )

    if video_stream:
        fps_parts = video_stream.get("r_frame_rate", "30/1").split("/")
        fps = int(fps_parts[0]) / int(fps_parts[1])
        return {
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "fps": fps,
            "duration": float(video_stream.get("duration", 0))
        }
    return {}


if __name__ == "__main__":
    # Test
    info = get_video_info("input/video.mp4")
    print(f"Video bilgisi: {info}")
