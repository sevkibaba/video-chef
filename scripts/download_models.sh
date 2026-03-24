#!/bin/bash
# download_models.sh — Download SAM2 checkpoints for local use
set -e

mkdir -p checkpoints
cd checkpoints

echo "Downloading SAM2 Large (best quality, ~900MB)..."
wget -q --show-progress \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt

echo "Downloading SAM2 Small (faster, ~185MB)..."
wget -q --show-progress \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt

echo "Done. Checkpoints saved to: checkpoints/"
