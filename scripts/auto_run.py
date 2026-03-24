import os
import subprocess
from pathlib import Path

def run_bulk():
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Find all subdirectories in input/
    subdirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"No subdirectories found in {input_dir}")
        return

    for subdir in subdirs:
        print(f"\n📂 Checking {subdir.name}...")
        
        # Find video file
        video_files = list(subdir.glob("*.mp4")) + list(subdir.glob("*.mov"))
        # Find character image
        char_files = list(subdir.glob("*.jpeg")) + list(subdir.glob("*.jpg")) + list(subdir.glob("*.png"))
        
        if not video_files:
            print(f"  ❌ No video file (.mp4, .mov) found in {subdir}")
            continue
        if not char_files:
            print(f"  ❌ No character image (.jpeg, .jpg, .png) found in {subdir}")
            continue
            
        video_path = video_files[0]
        char_path = char_files[0]
        output_path = output_dir / f"{subdir.name}_result.mp4"
        
        print(f"  ✅ Found Video: {video_path.name}")
        print(f"  ✅ Found Character: {char_path.name}")
        print(f"  🚀 Launching pipeline for {subdir.name}...")
        
        cmd = [
            "python3", "pipelines/pose_transfer_v1.py",
            "--video", str(video_path),
            "--character", str(char_path),
            "--output", str(output_path),
            "--max-frames", "30",  # Test run by default
            "--use-rembg"           # Use rembg as default for Mac testing
        ]
        
        print(f"  Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            print(f"  ✨ Finished {subdir.name}. Result: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Error running pipeline for {subdir.name}: {e}")

if __name__ == "__main__":
    run_bulk()
