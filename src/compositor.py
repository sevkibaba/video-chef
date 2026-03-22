"""
Compositor: Segmente edilmiş karakter + yeni arka plan + yeni karakter
görsellerini birleştirerek final frame'i oluşturur.
"""
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np


def replace_background(
    original_frame_path: str,
    mask_path: str,
    new_background_path: str,
    output_path: str,
    edge_feather: int = 5
) -> str:
    """
    Original frame'deki arkaplanı yeni arka plan ile değiştirir.
    Karakter olduğu gibi kalır.
    """
    # Orijinal frame ve maskesi
    original = Image.open(original_frame_path).convert("RGBA")
    masked = Image.open(mask_path).convert("RGBA")

    # Arka planı frame boyutuna getir
    bg = Image.open(new_background_path).convert("RGBA")
    bg = bg.resize(original.size, Image.LANCZOS)

    # Kenar yumuşatma
    if edge_feather > 0:
        r, g, b, alpha = masked.split()
        alpha = alpha.filter(ImageFilter.GaussianBlur(edge_feather))
        masked = Image.merge("RGBA", (r, g, b, alpha))

    # Arka planın üstüne karakteri yapıştır
    result = bg.copy()
    result.paste(masked, (0, 0), masked)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.convert("RGB").save(output_path, "JPEG", quality=92)

    return output_path


def replace_character(
    composited_frame_path: str,
    character_image_path: str,
    mask_path: str,
    output_path: str,
    scale: float = 1.0,
    edge_feather: int = 5
) -> str:
    """
    Videodaki karakterin üstüne yeni karakter görselini yapıştırır.
    Maskeyi kullanarak karakterin pozisyonu ve boyutunu tespit eder.
    """
    frame = Image.open(composited_frame_path).convert("RGBA")
    mask_img = Image.open(mask_path).convert("RGBA")
    character = Image.open(character_image_path).convert("RGBA")

    # Maskeden bounding box bul (karakterin nerede olduğunu bul)
    alpha = np.array(mask_img.split()[-1])
    rows = np.any(alpha > 10, axis=1)
    cols = np.any(alpha > 10, axis=0)

    if not rows.any() or not cols.any():
        # Frame'de kimse yok, olduğu gibi kaydet
        frame.convert("RGB").save(output_path, "JPEG", quality=92)
        return output_path

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Karakter boyutunu bounding box'a göre ayarla
    char_width = int((cmax - cmin) * scale)
    char_height = int((rmax - rmin) * scale)
    character_resized = character.resize((char_width, char_height), Image.LANCZOS)

    # Kenar yumuşatma
    if edge_feather > 0:
        r, g, b, a = character_resized.split()
        a = a.filter(ImageFilter.GaussianBlur(edge_feather))
        character_resized = Image.merge("RGBA", (r, g, b, a))

    # Yapıştır
    paste_x = cmin + (cmax - cmin - char_width) // 2
    paste_y = rmin + (rmax - rmin - char_height) // 2
    frame.paste(character_resized, (paste_x, paste_y), character_resized)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frame.convert("RGB").save(output_path, "JPEG", quality=92)

    return output_path


def process_all_frames(
    frame_paths: list,
    mask_paths: list,
    background_path: str,
    character_path: str,
    output_dir: str,
    config: dict = None
) -> list:
    """
    Tüm frame'leri işler: arka plan + karakter değişimi.
    """
    from tqdm import tqdm

    edge_feather = config.get("compositing", {}).get("edge_feather", 5) if config else 5
    replace_char = character_path and Path(character_path).exists()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_paths = []

    print(f"🎨 Compositing başlıyor ({len(frame_paths)} frame)...")
    for i, (frame_path, mask_path) in enumerate(tqdm(
        zip(frame_paths, mask_paths), total=len(frame_paths), desc="Compositing"
    )):
        frame_name = Path(frame_path).stem
        output_path = str(Path(output_dir) / f"{frame_name}_final.jpg")

        # 1. Arka planı değiştir
        bg_replaced = replace_background(
            frame_path, mask_path, background_path,
            output_path, edge_feather
        )

        # 2. Karakter swap (opsiyonel)
        if replace_char:
            replace_character(
                bg_replaced, character_path, mask_path,
                output_path, edge_feather=edge_feather
            )

        output_paths.append(output_path)

    print(f"✅ {len(output_paths)} frame composite edildi")
    return output_paths


if __name__ == "__main__":
    print("Compositor test - bağımsız çalıştırmak için instagram_pipeline.py kullanın.")
