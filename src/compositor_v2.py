"""
compositor_v2.py — Composite the generated character onto the original video background.

Strategy:
  - The generated frame contains the character in the right pose but with a
    diffusion-model background (which we don't want — we keep the original BG).
  - We use the SAM2 mask (of the ORIGINAL person) to define WHERE to paste.
  - We separately remove the background of the GENERATED frame (rembg) to get
    the generated character cutout.
  - We paste the generated character cutout onto the original background frame.

This gives us: original background + AI-generated character in correct pose.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Union
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Background removal from generated frames
# ──────────────────────────────────────────────────────────────────────────────

def _get_rembg_session(model: str = "u2net_human_seg"):
    try:
        from rembg import new_session
        return new_session(model)
    except ImportError:
        raise ImportError("pip install rembg")


def remove_bg_from_generated(
    generated_image: Image.Image,
    session=None,
    edge_feather: int = 8,
) -> Image.Image:
    """
    Remove background from a generated frame → returns RGBA image (character only).

    Args:
        generated_image: PIL Image (RGB) from CharacterGenerator.
        session:         rembg session (reuse across calls for speed).
        edge_feather:    GaussianBlur radius for edge softening.

    Returns:
        RGBA PIL Image with background removed.
    """
    try:
        from rembg import remove

        rgba = remove(generated_image.convert("RGB"), session=session)

        # Soften edges
        if edge_feather > 0:
            r, g, b, a = rgba.split()
            a = a.filter(ImageFilter.GaussianBlur(edge_feather))
            rgba = Image.merge("RGBA", (r, g, b, a))

        return rgba

    except ImportError:
        raise ImportError("pip install rembg")


# ──────────────────────────────────────────────────────────────────────────────
# Smart placement: scale/position generated character to match original mask
# ──────────────────────────────────────────────────────────────────────────────

def get_mask_bbox(mask: np.ndarray) -> Optional[tuple]:
    """
    Get bounding box of the non-zero region in a bool mask.
    Returns (x_min, y_min, x_max, y_max) or None if mask is empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Return bottom-right coordinates as exclusive (standard for PIL crops)
    return (int(x_min), int(y_min), int(x_max + 1), int(y_max + 1))


def composite_frame(
    original_frame: Image.Image,
    generated_char: Image.Image,   # RGBA, background removed
    person_mask: np.ndarray,       # bool mask from SAM2 (H, W)
    edge_feather: int = 8,
    scale_padding: float = 0.05,   # Slight padding around bounding box
) -> Image.Image:
    """
    Place generated_char onto original_frame's background.

    The character is scaled to fit the original person's bounding box.

    Args:
        original_frame: Original video frame (RGB PIL Image).
        generated_char: Generated character, background removed (RGBA PIL Image).
        person_mask:    SAM2 mask for the original person (bool H×W numpy).
        edge_feather:   Blur radius for blending edges.
        scale_padding:  Extra space added around person bounding box.

    Returns:
        Composited RGB PIL Image.
    """
    W, H = original_frame.size
    result = original_frame.copy().convert("RGBA")

    # Get person bounding box from SAM2 mask
    bbox = get_mask_bbox(person_mask)
    if bbox is None:
        logger.debug("Empty mask — frame returned unchanged.")
        return original_frame.copy()

    x_min, y_min, x_max, y_max = bbox

    # Add padding around original person box
    pad_x = int((x_max - x_min) * scale_padding)
    pad_y = int((y_max - y_min) * scale_padding)
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(W, x_max + pad_x)
    y_max = min(H, y_max + pad_y)

    target_w = x_max - x_min
    target_h = y_max - y_min
    # 1. Determine local character bounds in the generated frame
    # We use a threshold to ignore rembg noise/artifacts near the edges.
    gen_alpha = np.array(generated_char.getchannel("A"))
    mask_to_crop = (gen_alpha > 50) # Ignore semi-transparent noise
    
    # Force generic 3px border to False to ignore any junk at the frame edges
    if mask_to_crop.shape[0] > 10 and mask_to_crop.shape[1] > 10:
        mask_to_crop[0:3, :] = False
        mask_to_crop[-3:, :] = False
        mask_to_crop[:, 0:3] = False
        mask_to_crop[:, -3:] = False
        
    gen_bbox = get_mask_bbox(mask_to_crop)
    
    if gen_bbox is None:
        # If no character found, return original background
        return original_frame.copy()
        
    gx_min, gy_min, gx_max, gy_max = gen_bbox
    
    # Check if detected bbox is basically the whole frame (implies rembg failed)
    if (gx_max - gx_min) > 0.9 * W and (gy_max - gy_min) > 0.9 * H:
        logger.debug("BBox covers whole frame - using direct overlay.")
        result.paste(generated_char, (0, 0), generated_char)
        return result.convert("RGB")

    # 2. Crop the character and resize to fit target person's bounding box
    char_crop = generated_char.crop((gx_min, gy_min, gx_max, gy_max))
    char_w, char_h = char_crop.size
    char_ratio = char_w / char_h
    target_ratio = target_w / target_h

    if char_ratio > target_ratio:
        # Character is wider — constrain by width
        new_w = target_w
        new_h = int(new_w / char_ratio)
    else:
        # Character is taller — constrain by height
        new_h = target_h
        new_w = int(new_h * char_ratio)

    char_resized = char_crop.resize((new_w, new_h), Image.LANCZOS)

    # 3. Center and paste
    paste_x = x_min + (target_w - new_w) // 2
    paste_y = y_min + (target_h - new_h) // 2
    result.paste(char_resized, (paste_x, paste_y), char_resized)

    return result.convert("RGB")


# ──────────────────────────────────────────────────────────────────────────────
# Batch compositing
# ──────────────────────────────────────────────────────────────────────────────

def composite_all_frames(
    original_frame_files: list,
    generated_frame_files: list,
    masks: dict,                    # {frame_idx: np.ndarray bool (H,W)}
    output_dir: str,
    edge_feather: int = 8,
    rembg_model: str = "u2net_human_seg",
) -> list:
    """
    Composite all frames. Saves results as frame_XXXXXX.jpg.

    Args:
        original_frame_files:  Sorted list of original frame Paths.
        generated_frame_files: Sorted list of generated frame Paths (from CharacterGenerator).
        masks:                 Dict of SAM2 masks per frame index.
        output_dir:            Where to save composited frames.
        edge_feather:          Edge softening radius.
        rembg_model:           rembg model for generated frame BG removal.

    Returns:
        Sorted list of output frame Paths.
    """
    from tqdm import tqdm

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load rembg session once
    logger.info("Loading rembg for generated frame background removal...")
    rembg_session = _get_rembg_session(rembg_model)

    output_paths = []
    total = min(len(original_frame_files), len(generated_frame_files))

    logger.info(f"Compositing {total} frames...")

    for i in tqdm(range(total), desc="Compositing"):
        out_path = output_dir / f"frame_{i:06d}.jpg"

        try:
            original = Image.open(str(original_frame_files[i])).convert("RGB")
            generated = Image.open(str(generated_frame_files[i])).convert("RGB")

            # Remove BG from generated frame
            generated_rgba = remove_bg_from_generated(generated, rembg_session, edge_feather)

            # Get mask for this frame (fallback to empty mask)
            mask = masks.get(i, np.zeros(
                (original.size[1], original.size[0]), dtype=bool
            ))

            # Composite
            composited = composite_frame(
                original_frame=original,
                generated_char=generated_rgba,
                person_mask=mask,
                edge_feather=edge_feather,
            )

            composited.save(str(out_path), "JPEG", quality=95)
            output_paths.append(out_path)

        except Exception as e:
            logger.error(f"  Compositing frame {i} failed: {e}")
            # Fallback: save original frame unchanged
            Image.open(str(original_frame_files[i])).save(str(out_path), "JPEG", quality=95)
            output_paths.append(out_path)

    logger.info(f"Done: {len(output_paths)} frames composited → {output_dir}")
    return sorted(output_paths)
