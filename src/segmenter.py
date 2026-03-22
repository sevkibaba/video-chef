"""
Segmenter: rembg ile video frame'lerinden arkaplanı kaldırır,
insan/nesne maskesi oluşturur.
"""
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False
    print("⚠️  rembg kurulu değil: pip install rembg[gpu]")


def create_session(model: str = "u2net_human_seg", device: str = "cuda"):
    """
    rembg session oluşturur.
    model seçenekleri:
      - u2net_human_seg: İnsanlar için optimize, hafif (~176MB)
      - u2net: Genel objeler
      - isnet-general-use: Daha iyi kalite ama ağır
    """
    if not REMBG_AVAILABLE:
        raise RuntimeError("rembg kurulu değil!")

    print(f"🤖 Model yükleniyor: {model} ({device})")
    session = new_session(model)
    print(f"✅ Model hazır!")
    return session


def remove_background_frame(image_path: str, session, output_path: str) -> str:
    """
    Tek bir frame'den arkaplanı kaldırır.
    Returns: mask PNG (şeffaf arkaplan) yolu
    """
    img = Image.open(image_path)
    result = remove(img, session=session)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.save(output_path, "PNG")
    return output_path


def process_frames_batch(
    frame_paths: list,
    output_dir: str,
    model: str = "u2net_human_seg",
    device: str = "cuda",
    batch_size: int = 2
) -> list:
    """
    Tüm frame'leri işler, maskeleri kaydeder.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    session = create_session(model, device)

    mask_paths = []

    print(f"✂️  {len(frame_paths)} frame segmente ediliyor...")
    for frame_path in tqdm(frame_paths, desc="Segmentasyon"):
        frame_name = Path(frame_path).stem
        mask_path = str(Path(output_dir) / f"{frame_name}_mask.png")
        remove_background_frame(frame_path, session, mask_path)
        mask_paths.append(mask_path)

    print(f"✅ {len(mask_paths)} maske oluşturuldu → {output_dir}")
    return mask_paths


def extract_alpha_mask(masked_image_path: str) -> np.ndarray:
    """
    Şeffaf PNG'den alpha maskesini numpy array olarak döndürür.
    """
    img = Image.open(masked_image_path).convert("RGBA")
    r, g, b, alpha = img.split()
    return np.array(alpha)


if __name__ == "__main__":
    # Test - tek frame
    session = create_session()
    remove_background_frame(
        "input/test_frame.png",
        session,
        "temp/test_mask.png"
    )
    print("Test tamamlandı!")
