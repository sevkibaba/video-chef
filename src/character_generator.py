"""
character_generator.py — Generate character frames using ControlNet (pose) + IP-Adapter (appearance).

Two backends supported:
  - SD15:  runwayml/stable-diffusion-v1-5 + control_v11p_sd15_openpose + ip-adapter_sd15
           Recommended for: ~8GB VRAM (RTX 3080, A10), local machines
  - SDXL:  stabilityai/stable-diffusion-xl-base-1.0 + controlnet-openpose-sdxl + ip-adapter_sdxl
           Recommended for: ~16GB VRAM (A100, H100), cloud GPUs

Device selection (pass --device auto to detect automatically):
  - cuda:  NVIDIA GPU            → fastest
  - mps:   Apple Silicon (M1/M2/M3) → medium speed, native Mac only
  - cpu:   Fallback              → slow, ~5-15 min/frame

The IP-Adapter provides character appearance from your JPEG.
ControlNet forces the generated character into the exact pose from the video.
"""

import logging
import os
import gc
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import torch
import numpy as np

# Prevent PyTorch from throwing artificial OOM errors on Mac unified memory
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

logger = logging.getLogger("pose_transfer")


# ──────────────────────────────────────────────────────────────────────────────
# Device auto-detection
# ──────────────────────────────────────────────────────────────────────────────

def auto_device() -> str:
    """
    Pick the best available device:
      1. CUDA  (NVIDIA GPU)
      2. MPS   (Apple Silicon — native Mac only, not available inside Docker)
      3. CPU   (fallback)
    """
    if torch.cuda.is_available():
        logger.info("Device: CUDA (NVIDIA GPU)")
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Device: MPS (Apple Silicon GPU)")
        return "mps"
    logger.info("Device: CPU (no GPU found — inference will be slow)")
    return "cpu"




# ──────────────────────────────────────────────────────────────────────────────
# Model configs
# ──────────────────────────────────────────────────────────────────────────────

SD15_CONFIG = {
    "base_model":   "runwayml/stable-diffusion-v1-5",
    "controlnet":   "lllyasviel/control_v11p_sd15_openpose",
    "ip_adapter_repo": "h94/IP-Adapter",
    "ip_adapter_subfolder": "models",
    "ip_adapter_weight": "ip-adapter_sd15.bin",
    "default_size": (512, 768),   # (W, H) — portrait for full-body
}

SDXL_CONFIG = {
    "base_model":   "stabilityai/stable-diffusion-xl-base-1.0",
    "controlnet":   "thibaud/controlnet-openpose-sdxl-1.0",
    "ip_adapter_repo": "h94/IP-Adapter",
    "ip_adapter_subfolder": "sdxl_models",
    "ip_adapter_weight": "ip-adapter_sdxl.bin",
    "default_size": (1024, 1024),
}


# ──────────────────────────────────────────────────────────────────────────────
# CharacterGenerator
# ──────────────────────────────────────────────────────────────────────────────

class CharacterGenerator:
    """
    Generates frames where the target character (from a JPEG) is placed
    into the original video's pose.

    Args:
        backend:        "sd15" or "sdxl"
        device:         "cuda" or "cpu"
        ip_scale:       How strongly to apply the character appearance (0.0–1.0).
                        0.6–0.8 works well. Higher = more like the JPEG character.
        controlnet_scale: How strongly to apply pose (0.0–1.0). 0.8–1.0 recommended.
        num_steps:      Diffusion steps. 20–30 is a good balance.
        guidance_scale: CFG scale. 5–8 typical.
        seed:           Fixed seed for temporal consistency across frames.
        fast:           Fast mode for testing (lower resolution).
    """

    def __init__(
        self,
        backend: str = "sd15",
        device: str = "auto",
        ip_scale: float = 0.7,
        controlnet_scale: float = 0.9,
        num_steps: int = 25,
        guidance_scale: float = 7.0,
        seed: int = 42,
        fast: bool = False,
    ):
        self.backend = backend.lower()
        self.device = auto_device() if device == "auto" else device
        self.ip_scale = ip_scale
        self.controlnet_scale = controlnet_scale
        self.num_steps = num_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.fast = fast
        self.cfg = SD15_CONFIG if self.backend == "sd15" else SDXL_CONFIG
        self._pipe = None

    def load(self):
        """Load all models. Call once before processing."""
        if self._pipe is not None:
            return

        logger.info(f"Loading {self.backend.upper()} pipeline...")

        if self.backend == "sd15":
            self._load_sd15()
        elif self.backend == "sdxl":
            self._load_sdxl()
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Use 'sd15' or 'sdxl'.")

        logger.info(f"Pipeline ready on: {self.device}")

        logger.info("Models loaded. Ready to generate.")

    def _load_sd15(self):
        from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        logger.info(f"  Loading ControlNet: {self.cfg['controlnet']}")
        controlnet = ControlNetModel.from_pretrained(
            self.cfg["controlnet"],
            torch_dtype=dtype,
        )

        logger.info(f"  Loading base model: {self.cfg['base_model']}")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.cfg["base_model"],
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )

        # Memory optimizations
        # cpu_offload only works on CUDA; on MPS/CPU load normally
        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)

        # Load IP-Adapter
        logger.info(f"  Loading IP-Adapter: {self.cfg['ip_adapter_weight']}")
        pipe.load_ip_adapter(
            self.cfg["ip_adapter_repo"],
            subfolder=self.cfg["ip_adapter_subfolder"],
            weight_name=self.cfg["ip_adapter_weight"],
        )
        pipe.set_ip_adapter_scale(self.ip_scale)

        self._pipe = pipe

    def _load_sdxl(self):
        from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel

        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        controlnet = ControlNetModel.from_pretrained(
            self.cfg["controlnet"],
            torch_dtype=dtype,
        )

        logger.info(f"  Loading SDXL base: {self.cfg['base_model']}")
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.cfg["base_model"],
            controlnet=controlnet,
            torch_dtype=dtype,
        )

        if self.device == "cuda":
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(self.device)

        logger.info(f"  Loading IP-Adapter (SDXL): {self.cfg['ip_adapter_weight']}")
        pipe.load_ip_adapter(
            self.cfg["ip_adapter_repo"],
            subfolder=self.cfg["ip_adapter_subfolder"],
            weight_name=self.cfg["ip_adapter_weight"],
        )
        pipe.set_ip_adapter_scale(self.ip_scale)

        self._pipe = pipe

    def generate(
        self,
        pose_image: Image.Image,
        character_image: Image.Image,
        output_size: tuple,
        prompt: str = "a person, full body, photorealistic, detailed",
        negative_prompt: str = (
            "deformed, ugly, bad anatomy, blurry, extra limbs, "
            "watermark, text, artifacts, low quality"
        ),
    ) -> Image.Image:
        """
        Generate a single frame showing the character in the given pose.

        Args:
            pose_image:      OpenPose skeleton image (from PoseExtractor).
            character_image: The character JPEG reference image.
            output_size:     (W, H) to resize final output to.
            prompt:          Positive text prompt.
            negative_prompt: Negative text prompt.

        Returns:
            Generated PIL Image (RGB), resized to output_size.
        """
        self.load()

        # 1. Determine generation size based on video aspect ratio
        orig_w, orig_h = output_size
        aspect_ratio = orig_w / orig_h
        
        # Use height from config as base, adjust width for aspect ratio
        gen_h = self.cfg["default_size"][1]
        if self.fast:
            gen_h = 480  # Much faster for testing
            
        gen_w = int(gen_h * aspect_ratio)
        
        # Ensure multiples of 8 for diffusion models
        gen_w = (gen_w // 8) * 8
        gen_h = (gen_h // 8) * 8
        
        if self.fast:
            logger.info(f"  [FAST] Using lower resolution: {gen_w}x{gen_h}")

        # 2. Resize pose to generation size
        pose_resized = pose_image.resize((gen_w, gen_h), Image.LANCZOS)

        # MPS generators must be created on CPU
        gen_device = "cpu" if self.device == "mps" else self.device
        generator = torch.Generator(device=gen_device).manual_seed(self.seed)

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=pose_resized,                  # ControlNet condition
            ip_adapter_image=character_image,    # IP-Adapter appearance
            num_inference_steps=self.num_steps,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=self.controlnet_scale,
            # Adjust IP-scale properly; higher means more like the original image
            cross_attention_kwargs={"scale": self.ip_scale},
            generator=generator,
            width=gen_w,
            height=gen_h,
        ).images[0]

        # Resize to match original video frame size
        if result.size != output_size:
            result = result.resize(output_size, Image.LANCZOS)

        return result

    def generate_batch(
        self,
        pose_images: list,
        character_image: Union[Image.Image, str, Path],
        output_size: tuple,
        output_dir: str,
        prompt: str = "(masterpiece, best quality:1.2), high quality, photorealistic, identical character to reference, detailed",
        negative_prompt: str = (
            "deformed, ugly, bad anatomy, blurry, extra limbs, "
            "watermark, text, artifacts, low quality, different person, malformed face, bad proportions"
        ),
    ) -> list:
        """
        Generate frames for all pose images. Saves to output_dir.

        Args:
            pose_images:    List of PIL pose images (from PoseExtractor).
            character_image: Character JPEG as PIL Image or path.
            output_size:    (W, H) of the original video.
            output_dir:     Where to save generated frames.
            prompt / negative_prompt: See generate().

        Returns:
            List of Path objects pointing to generated frame files.
        """
        self.load()

        if not isinstance(character_image, Image.Image):
            character_image = Image.open(str(character_image)).convert("RGB")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_paths = []
        total = len(pose_images)

        for i, pose_img in enumerate(pose_images):
            out_path = output_dir / f"frame_{i:06d}.jpg"

            try:
                gen_frame = self.generate(
                    pose_image=pose_img,
                    character_image=character_image,
                    output_size=output_size,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                )
                gen_frame.save(str(out_path), "JPEG", quality=95)
                generated_paths.append(out_path)

                if i % 10 == 0:
                    logger.info(f"  Generated frame {i}/{total}")

            except Exception as e:
                logger.error(f"  Frame {i} generation failed: {e}")
                # Fallback: black frame
                Image.new("RGB", output_size, (0, 0, 0)).save(str(out_path))
                generated_paths.append(out_path)

            # --- Critical fix for Mac (Apple Silicon) Out-Of-Memory ---
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"  Done: {len(generated_paths)} frames generated → {output_dir}")
        return generated_paths
