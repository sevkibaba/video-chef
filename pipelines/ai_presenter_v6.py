"""
Video Chef Pipeline V6 - AI Presenter Style

Creates a new video from character image + background.
Similar framing to original video, preserves audio.

Approach:
1. Extract audio from original video
2. Get original video dimensions/fps
3. Create new frames: character centered + new background
4. Use similar duration as original video
5. Add original audio

This creates a "talking head" style video with your character!
"""

import sys
import shutil
import time
from pathlib import Path
import logging

import click
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.video_processor import resize_video, extract_frames, get_video_info, extract_audio
from src.video_assembler import frames_to_video


# ============== LOGGING ==============
def setup_logging():
    log_dir = Path("output/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"pipeline_v6_{time.strftime('%Y%m%d_%H%M%S')}.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = None


def create_ai_presenter_frame(
    character_path: str,
    background_path: str,
    output_size: tuple,
    position: str = "center",
    scale: float = 0.7,
    breathing: bool = True
) -> Image.Image:
    """
    Create a presenter-style frame with character + background.
    
    Args:
        character_path: Path to character image (JPEG/PNG)
        background_path: Path to background image
        output_size: (width, height) - match original video
        position: center, left, right, top, bottom
        scale: character size as percentage of frame
        breathing: add subtle animation effect
    """
    # Load images
    character = Image.open(character_path).convert("RGBA")
    background = Image.open(background_path).convert("RGBA")
    
    w, h = output_size
    
    # Resize background to match output
    background = background.resize((w, h), Image.LANCZOS)
    
    # Calculate character size
    char_w, char_h = character.size
    char_ratio = char_w / char_h
    
    # Target character size based on scale
    target_h = int(h * scale)
    target_w = int(target_h * char_ratio)
    
    # Resize character
    character = character.resize((target_w, target_h), Image.LANCZOS)
    
    # Apply edge feathering
    r, g, b, a = character.split()
    a = a.filter(ImageFilter.GaussianBlur(10))
    a = a.filter(ImageFilter.GaussianBlur(5))
    character = Image.merge('RGBA', (r, g, b, a))
    
    # Calculate position
    if position == "center":
        pos_x = (w - target_w) // 2
        pos_y = (h - target_h) // 2
    elif position == "left":
        pos_x = int(w * 0.1)
        pos_y = (h - target_h) // 2
    elif position == "right":
        pos_x = int(w * 0.6)
        pos_y = (h - target_h) // 2
    elif position == "top":
        pos_x = (w - target_w) // 2
        pos_y = int(h * 0.1)
    elif position == "bottom":
        pos_x = (w - target_w) // 2
        pos_y = int(h * 0.7)
    else:
        pos_x = (w - target_w) // 2
        pos_y = (h - target_h) // 2
    
    # Ensure within bounds
    pos_x = max(0, min(pos_x, w - target_w))
    pos_y = max(0, min(pos_y, h - target_h))
    
    # Composite
    output = background.copy().convert('RGBA')
    output.paste(character, (pos_x, pos_y), character)
    
    return output.convert('RGB')


def create_video_with_ai_presenter(
    original_video_path: str,
    character_path: str,
    background_path: str,
    output_path: str,
    num_frames: int = None,
    position: str = "center",
    scale: float = 0.7,
    breathing: bool = True,
    keep_temp: bool = False
) -> bool:
    """Main function to create AI presenter video."""
    global logger
    
    temp_dir = Path("temp_v6")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Step 1: Get original video info
        logger.info("Step 1: Analyzing original video...")
        info = get_video_info(original_video_path)
        fps = int(info.get('fps', 30))
        width = info.get('width', 1280)
        height = info.get('height', 720)
        duration = info.get('duration', 10)
        
        logger.info(f"Original: {width}x{height} @ {fps} FPS, {duration:.1f}s")
        
        # Calculate number of frames
        if num_frames is None:
            num_frames = int(duration * fps)
        
        # Step 2: Extract audio (keep original!)
        logger.info("Step 2: Extracting original audio...")
        audio_path = str(temp_dir / "audio.aac")
        extract_audio(original_video_path, audio_path)
        
        # Step 3: Create new frames
        logger.info(f"Step 3: Creating {num_frames} new frames...")
        
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        for i in range(num_frames):
            # Add slight breathing animation
            if breathing and i % 10 == 0:
                scale_mod = scale * 1.02  # Slight zoom
            else:
                scale_mod = scale
            
            frame = create_ai_presenter_frame(
                character_path,
                background_path,
                (width, height),
                position=position,
                scale=scale_mod
            )
            
            frame.save(frames_dir / f"frame_{i:06d}.jpg", "JPEG", quality=92)
            
            if i % 30 == 0:
                logger.info(f"  Created {i}/{num_frames} frames")
        
        # Step 4: Create video with audio
        logger.info("Step 4: Creating final video...")
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        frames_to_video(
            str(frames_dir),
            output_path,
            fps=fps,
            audio_path=audio_path if Path(audio_path).exists() else None,
            instagram_optimize=False
        )
        
        logger.info(f"Done! Output: {output_path}")
        
        # Cleanup temp if requested
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


@click.command()
@click.option("--video", required=True, help="Original video (ses için)")
@click.option("--character", required=True, help="Karakter görseli (JPEG/PNG)")
@click.option("--background", required=True, help="Arka plan görseli")
@click.option("--output", default="output/ai_presenter.mp4", help="Çıktı video")
@click.option("--position", default="center", help="Karakter pozisyonu: center, left, right, top, bottom")
@click.option("--scale", default=0.7, type=float, help="Karakter boyutu (0.1-1.0)")
@click.option("--fps", default=None, type=int, help="FPS (varsayılan: orijinal)")
@click.option("--duration", default=None, type=int, help="Süre saniye (varsayılan: orijinal)")
@click.option("--no-breathing", is_flag=True, help="Breathing animasyonu kapat")
@click.option("--keep-temp", is_flag=True, help="Geçici dosyaları silme")
def run_v6(video, character, background, output, position, scale, fps, duration, no_breathing, keep_temp):
    global logger
    logger = setup_logging()
    
    logger.info("="*60)
    logger.info("VIDEO CHEF V6 - AI PRESENTER")
    logger.info("="*60)
    
    start_time = time.time()
    
    # Calculate number of frames
    info = get_video_info(video)
    original_duration = info.get('duration', 10)
    original_fps = fps or int(info.get('fps', 30))
    
    target_duration = duration or original_duration
    target_fps = fps or original_fps
    
    num_frames = int(target_duration * target_fps)
    
    logger.info(f"Creating {num_frames} frames at {target_fps} FPS for {target_duration}s video")
    
    result = create_video_with_ai_presenter(
        original_video_path=video,
        character_path=character,
        background_path=background,
        output_path=output,
        num_frames=num_frames,
        position=position,
        scale=scale,
        breathing=not no_breathing,
        keep_temp=keep_temp
    )
    
    elapsed = time.time() - start_time
    
    if result:
        logger.info("="*60)
        logger.info(f"SUCCESS! Created in {elapsed:.1f}s")
        logger.info(f"Output: {output}")
        logger.info("="*60)
    else:
        logger.error("FAILED!")


if __name__ == "__main__":
    run_v6()