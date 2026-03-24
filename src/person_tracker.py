"""
person_tracker.py — Track a person across all video frames using SAM2.

Flow:
  1. YOLO detects the most prominent person in frame 0 → gives click point
  2. SAM2 is initialized with that click point
  3. SAM2 propagates the mask through every frame

Returns: dict {frame_idx: np.ndarray (H, W) bool mask}
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional
import cv2

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# YOLO: auto-detect person center in a frame
# ──────────────────────────────────────────────────────────────────────────────

def detect_person_center(frame_path: str, conf_threshold: float = 0.4) -> tuple:
    """
    Run YOLOv8 on a frame and return (x, y) center of the most prominent person.
    Falls back to image center if no person is found.

    Args:
        frame_path: Path to the first frame image.
        conf_threshold: Minimum confidence for YOLO detection.

    Returns:
        (cx, cy) pixel coordinates.
    """
    try:
        from ultralytics import YOLO
        import torch

        logger.info("Loading YOLOv8n for person detection...")
        model = YOLO("yolov8n.pt")  # auto-downloaded on first run

        frame = cv2.imread(str(frame_path))
        h, w = frame.shape[:2]

        results = model(frame, classes=[0], conf=conf_threshold, verbose=False)  # class 0 = person

        if not results or len(results[0].boxes) == 0:
            logger.warning("No person detected by YOLO — using image center as fallback.")
            return (w // 2, h // 2)

        boxes = results[0].boxes.xyxy.cpu().numpy()  # [N, 4] x1 y1 x2 y2
        confs = results[0].boxes.conf.cpu().numpy()

        # Score by: confidence * box area (pick largest+most-confident = main subject)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        scores = confs * areas
        best = int(np.argmax(scores))

        x1, y1, x2, y2 = boxes[best]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        logger.info(f"  YOLO detected person at ({cx}, {cy}), conf={confs[best]:.2f}")
        return (cx, cy)

    except ImportError:
        logger.warning("ultralytics not installed — using image center as SAM2 prompt.")
        frame = cv2.imread(str(frame_path))
        h, w = frame.shape[:2]
        return (w // 2, h // 2)


# ──────────────────────────────────────────────────────────────────────────────
# SAM2: track person mask through video
# ──────────────────────────────────────────────────────────────────────────────

class PersonTracker:
    """
    Uses SAM2 video predictor to generate per-frame binary masks.

    Usage:
        tracker = PersonTracker(checkpoint="checkpoints/sam2_hiera_large.pt")
        masks = tracker.track(frames_dir="temp/frames", click_point=None)
        # masks[0] → np.ndarray bool (H, W)
    """

    SAM2_CONFIGS = {
        "large":  ("sam2_hiera_l.yaml",  "sam2_hiera_large.pt"),
        "base":   ("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
        "small":  ("sam2_hiera_s.yaml",  "sam2_hiera_small.pt"),
        "tiny":   ("sam2_hiera_t.yaml",  "sam2_hiera_tiny.pt"),
    }

    def __init__(
        self,
        model_size: str = "large",
        checkpoints_dir: str = "checkpoints",
        device: str = "cuda",
    ):
        self.device = device
        self.model_size = model_size
        self.checkpoints_dir = Path(checkpoints_dir)
        self.predictor = None

    def _load(self):
        if self.predictor is not None:
            return

        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor

            cfg_name, ckpt_name = self.SAM2_CONFIGS[self.model_size]
            ckpt_path = self.checkpoints_dir / ckpt_name

            if not ckpt_path.exists():
                logger.warning(
                    f"SAM2 checkpoint not found at {ckpt_path}. "
                    f"Download with: bash scripts/download_models.sh"
                )
                raise FileNotFoundError(f"Missing: {ckpt_path}")

            logger.info(f"Loading SAM2 ({self.model_size}) from {ckpt_path}...")
            self.predictor = build_sam2_video_predictor(
                cfg_name, str(ckpt_path), device=self.device
            )
            logger.info("  SAM2 loaded.")

        except ImportError:
            raise ImportError(
                "SAM2 not installed. Run:\n"
                "  pip install git+https://github.com/facebookresearch/sam2.git"
            )

    def track(
        self,
        frames_dir: str,
        click_point: Optional[tuple] = None,
        frame_files: Optional[list] = None,
    ) -> dict:
        """
        Track person through all frames.

        Args:
            frames_dir: Directory containing frame_000001.jpg etc.
            click_point: (x, y) pixel in frame 0. If None, auto-detected via YOLO.
            frame_files: Optional sorted list of frame Paths (if already known).

        Returns:
            dict {frame_idx (int): np.ndarray bool mask (H, W)}
        """
        import torch

        self._load()

        frames_dir = Path(frames_dir)
        if frame_files is None:
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))

        if len(frame_files) == 0:
            raise ValueError(f"No frames found in {frames_dir}")

        # Auto-detect click point if not provided
        if click_point is None:
            logger.info("Auto-detecting person position with YOLO...")
            click_point = detect_person_center(str(frame_files[0]))

        logger.info(f"SAM2 tracking with click point {click_point} across {len(frame_files)} frames...")

        masks = {}

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            # SAM2 needs a directory of JPEG frames
            inference_state = self.predictor.init_state(video_path=str(frames_dir))

            # Prompt: single point on the person in frame 0
            points = np.array([list(click_point)], dtype=np.float32)
            labels = np.array([1], dtype=np.int32)  # 1 = foreground

            _, obj_ids, mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )

            # Propagate forward through video
            for frame_idx, obj_ids, mask_logits in self.predictor.propagate_in_video(inference_state):
                # mask_logits shape: [N_objects, 1, H, W]
                mask = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(bool)
                masks[frame_idx] = mask

                if frame_idx % 30 == 0:
                    logger.info(f"  Tracked frame {frame_idx}/{len(frame_files)}")

        logger.info(f"Done. Tracked {len(masks)} frames.")
        return masks


# ──────────────────────────────────────────────────────────────────────────────
# Fallback: rembg-based per-frame segmentation (if SAM2 not available)
# ──────────────────────────────────────────────────────────────────────────────

def segment_frame_rembg(frame_path: str, session=None) -> np.ndarray:
    """
    Fallback: use rembg to get a person mask from a single frame.
    Returns bool mask (H, W).
    """
    try:
        from rembg import remove, new_session
        from PIL import Image

        if session is None:
            session = new_session("u2net_human_seg")

        img = Image.open(frame_path)
        result = remove(img, session=session)
        alpha = np.array(result)[:, :, 3]
        return alpha > 10

    except ImportError:
        raise ImportError("rembg not installed: pip install rembg")


def track_frames_rembg(frame_files: list, model: str = "u2net_human_seg") -> dict:
    """
    Fallback tracker using rembg (no SAM2 needed).
    Slower and less accurate but works locally without SAM2 weights.
    """
    try:
        from rembg import new_session
    except ImportError:
        raise ImportError("pip install rembg")

    logger.info(f"Using rembg fallback tracker ({model})...")
    session = None
    try:
        from rembg import new_session
        session = new_session(model)
    except Exception:
        pass

    masks = {}
    for i, fp in enumerate(frame_files):
        masks[i] = segment_frame_rembg(str(fp), session)
        if i % 30 == 0:
            logger.info(f"  rembg: {i}/{len(frame_files)}")

    return masks
