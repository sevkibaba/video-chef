# Docker Build & Run Guide

## Quick Start (CPU)

```bash
cd video-chef

# Build
docker-compose build

# Run
docker-compose run video-chef \
  --video input/video.mp4 \
  --background input/background.png \
  --output output/result.mp4
```

## Build Variants

### CPU Build
```bash
docker build -t video-chef:cpu --build-arg RUNTIME=cpu .
```

### GPU Build (NVIDIA CUDA)
```bash
docker build -t video-chef:gpu --build-arg RUNTIME=gpu .
```

## Run Examples

### Docker (CPU)
```bash
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output:rw \
  video-chef:cpu \
  --video input/video.mp4 \
  --background input/background.png \
  --output output/result.mp4
```

### Docker (GPU)
```bash
docker run --rm \
  --gpus all \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output:rw \
  video-chef:gpu \
  --video input/video.mp4 \
  --background input/background.png \
  --output output/result.mp4
```

### Docker Compose (CPU)
```bash
# docker-compose.yml'de RUNTIME=cpu ayarlanmışsa:
docker-compose run video-chef \
  --video input/video.mp4 \
  --background input/background.png \
  --character input/character.png \
  --output output/result.mp4
```

### Docker Compose (GPU)
```bash
# GPU servisini kullan:
docker-compose run video-chef-gpu \
  --video input/video.mp4 \
  --background input/background.png \
  --character input/character.png \
  --output output/result.mp4
```

## GPU Setup

### Windows (Docker Desktop)

1. **Requirements:**
   - Windows 11 Pro/Enterprise
   - WSL2 with Ubuntu
   - NVIDIA Docker support enabled

2. **Install NVIDIA Docker Runtime:**
   ```bash
   # WSL2 terminal'de
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-docker2
   sudo systemctl restart docker
   ```

3. **Verify GPU:**
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.2-base nvidia-smi
   ```

### Linux

```bash
# NVIDIA Docker Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.2-base nvidia-smi
```

## Performance Comparison

| Config | Duration | Speed | VRAM |
|--------|----------|-------|------|
| CPU (4 core) | ~30 min | 1x | - |
| GPU (RTX 3060) | ~2 min | 15x | 6GB |
| GPU (RTX 4090) | ~30 sec | 60x | 24GB |

*15 saniyelik video, 720p @ 30fps*

## Troubleshooting

### GPU not detected
```bash
# Check NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:12.2-base nvidia-smi

# Check Docker config
docker info | grep nvidia
```

### Permission denied (volumes)
```bash
chmod 777 input output temp
```

### Out of memory
```bash
# Dockerfile'da batch_size'ı düşür (config.yaml)
# veya --memory flag'ı kullan
docker run --memory=4g video-chef:gpu ...
```

### WSL2 Issues
```bash
# WSL2'de çalıştır
wsl -d Ubuntu
cd /mnt/c/Users/simit/Documents/repos/video-chef
docker-compose build
docker-compose run video-chef-gpu --help
```

## File Structure

```
video-chef/
├── input/           # Video + görseller (read-only volume)
├── output/          # Sonuç videosu (write volume)
├── temp/            # Geçici dosyalar (write volume)
├── Dockerfile       # CPU/GPU build logic
├── docker-compose.yml  # Orkestrasyonu
└── DOCKER.md        # Bu dosya
```

## Cleanup

```bash
# Stop container
docker-compose down

# Remove image
docker rmi video-chef:latest

# Remove all temp files
rm -rf temp/* output/*
```
