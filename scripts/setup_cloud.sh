#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# setup_cloud.sh — One-shot setup for cloud GPU machines (RunPod, vast.ai, etc.)
# Tested on: Ubuntu 22.04, CUDA 12.1, A100/H100
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "=============================="
echo " Video Chef — Cloud Setup"
echo "=============================="

# 1. System packages
echo "[1/5] Installing system packages..."
apt-get update -q
apt-get install -y -q ffmpeg git wget curl

# 2. Python dependencies
echo "[2/5] Installing Python packages..."
pip install --upgrade pip

# PyTorch (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core packages
pip install \
  diffusers>=0.27.0 \
  transformers>=4.36.0 \
  accelerate>=0.25.0 \
  safetensors>=0.4.0 \
  controlnet-aux>=0.0.9 \
  ultralytics>=8.0.0 \
  "rembg[gpu]>=2.0.50" \
  onnxruntime-gpu>=1.16.0 \
  Pillow>=10.0.0 \
  numpy>=1.24.0 \
  opencv-python>=4.8.0 \
  click>=8.1.0 \
  tqdm>=4.65.0 \
  pyyaml>=6.0

# 3. SAM2
echo "[3/5] Installing SAM2..."
pip install git+https://github.com/facebookresearch/sam2.git

# 4. Download SAM2 checkpoints
echo "[4/5] Downloading SAM2 checkpoints..."
mkdir -p checkpoints
cd checkpoints

# SAM2 Large (~900MB) — best quality
wget -q --show-progress \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

# SAM2 Small (~185MB) — faster, still good
wget -q --show-progress \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt

cd ..

# 5. Create input/output directories
echo "[5/5] Creating directory structure..."
mkdir -p input output output/logs temp_pose_transfer

echo ""
echo "=============================="
echo " ✅ Setup complete!"
echo ""
echo " Run the pipeline:"
echo ""
echo "   python pipelines/pose_transfer_v1.py \\"
echo "     --video input/video.mp4 \\"
echo "     --character input/character.jpg \\"
echo "     --output output/result.mp4 \\"
echo "     --backend sdxl \\"
echo "     --device cuda"
echo ""
echo " Test with 30 frames first:"
echo "   --max-frames 30"
echo "=============================="
