"""
pose_extractor.py — Extract skeleton/pose images from video frames using DWPose.

DWPose is the best open-source 2D pose estimator and is natively supported
by the controlnet_aux library (used by diffusers ControlNet workflows).

Output: PIL Image of pose skeleton — exactly what ControlNet OpenPose expects.
"""

import logging
from pathlib import Path
from typing import Union
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class PoseExtractor:
    """
    Wraps DWPose (via controlnet_aux) to extract OpenPose-format skeleton images.

    Args:
        detect_resolution: Internal resolution for pose detection (512 or 768).
        image_resolution:  Output pose image resolution.
        device:            "cuda" or "cpu".

    Usage:
        extractor = PoseExtractor(device="cuda")
        pose_img = extractor.extract(Image.open("frame.jpg"))
        pose_img.save("pose.png")
    """

    def __init__(
        self,
        detect_resolution: int = 512,
        image_resolution: int = 512,
        device: str = "cuda",
        include_hand: bool = True,
        include_face: bool = False,
    ):
        self.detect_resolution = detect_resolution
        self.image_resolution = image_resolution
        self.device = device
        self.include_hand = include_hand
        self.include_face = include_face
        self._detector = None
        self._mode = None  # "dwpose" or "openpose"

    def _load(self):
        if self._detector is not None:
            return
        try:
            from controlnet_aux import DWposeDetector
            logger.info("Loading DWPose detector (downloading ONNX weights if needed)...")
            # from_pretrained downloads lightweight ONNX weights from HuggingFace.
            # Does NOT require mmcv / mmpose / mmdet.
            self._detector = DWposeDetector.from_pretrained("lllyasviel/Annotators")
            self._mode = "dwpose"
            logger.info("  DWPose loaded.")
        except Exception as e:
            logger.warning(f"DWPose failed ({e}) — falling back to OpenposeDetector.")
            try:
                from controlnet_aux import OpenposeDetector
                self._detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
                self._mode = "openpose"
                logger.info("  OpenposeDetector loaded (fallback).")
            except Exception as e2:
                raise ImportError(
                    f"Neither DWPose nor OpenPose could be loaded: {e2}\n"
                    "Run: pip install controlnet-aux"
                )

    def extract(self, image: Union[Image.Image, str, Path]) -> Image.Image:
        """
        Extract pose from a single image.

        Args:
            image: PIL Image, or path to an image file.

        Returns:
            PIL Image with rendered skeleton (RGB, same size as input).
        """
        self._load()

        if not isinstance(image, Image.Image):
            image = Image.open(str(image)).convert("RGB")

        original_size = image.size  # (W, H)

        if self._mode == "dwpose":
            pose_image = self._detector(
                image,
                detect_resolution=self.detect_resolution,
                image_resolution=self.image_resolution,
                include_hand=self.include_hand,
                include_face=self.include_face,
            )
        else:
            # OpenposeDetector fallback
            pose_image = self._detector(
                image,
                detect_resolution=self.detect_resolution,
                image_resolution=self.image_resolution,
            )

        # Resize back to original dimensions so it matches the video frame
        if pose_image.size != original_size:
            pose_image = pose_image.resize(original_size, Image.LANCZOS)

        return pose_image

    def extract_batch(
        self,
        frame_files: list,
        output_dir: str,
        save: bool = True,
    ) -> list:
        """
        Extract poses from a list of frame paths.

        Args:
            frame_files: List of Path or str to frame images.
            output_dir:  Where to save pose images.
            save:        If True, saves pose PNGs to output_dir.

        Returns:
            List of PIL Images (pose images).
        """
        self._load()
        output_dir = Path(output_dir)
        if save:
            output_dir.mkdir(parents=True, exist_ok=True)

        pose_images = []
        total = len(frame_files)

        for i, fp in enumerate(frame_files):
            try:
                pose = self.extract(fp)

                if save:
                    stem = Path(fp).stem
                    pose.save(output_dir / f"{stem}_pose.png")

                pose_images.append(pose)

                if i % 30 == 0:
                    logger.info(f"  Pose extraction: {i}/{total}")

            except Exception as e:
                logger.warning(f"  Pose extraction failed for frame {i}: {e}")
                # Use blank black frame as fallback
                img = Image.open(str(fp)).convert("RGB")
                pose_images.append(Image.new("RGB", img.size, (0, 0, 0)))

        logger.info(f"  Done: {len(pose_images)} pose images extracted.")
        return pose_images


# ──────────────────────────────────────────────────────────────────────────────
# Fallback: MediaPipe pose (no controlnet_aux needed)
# ──────────────────────────────────────────────────────────────────────────────

def extract_pose_mediapipe(image: Image.Image) -> Image.Image:
    """
    Fallback pose extractor using MediaPipe.
    Lower quality than DWPose but very lightweight and no extra model downloads.
    """
    try:
        import mediapipe as mp
        import cv2
        import numpy as np

        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils

        frame_np = np.array(image.convert("RGB"))
        h, w = frame_np.shape[:2]

        # Create black canvas for skeleton drawing
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5,
        ) as pose:
            results = pose.process(cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR))
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    canvas,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(128, 128, 255), thickness=2),
                )

        return Image.fromarray(canvas)

    except ImportError:
        raise ImportError("mediapipe not installed: pip install mediapipe")
