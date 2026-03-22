# Docker Build & Run Guide

## Build

```bash
cd video-chef
docker build -t video-chef:latest .
```

## Run (Docker)

```bash
# Basit arka plan değişimi
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output:rw \
  video-chef:latest \
  --video input/video.mp4 \
  --background input/background.png \
  --output output/result.mp4

# Karakter + arka plan değişimi
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output:rw \
  video-chef:latest \
  --video input/video.mp4 \
  --background input/background.png \
  --character input/character.png \
  --output output/result.mp4
```

## Run (Docker Compose)

```bash
# Build
docker-compose build

# Run
docker-compose run video-chef \
  --video input/video.mp4 \
  --background input/background.png \
  --character input/character.png \
  --output output/result.mp4
```

## GPU Support

NVIDIA GPU kullanmak için:

1. NVIDIA Container Toolkit kur: https://github.com/NVIDIA/nvidia-docker
2. `docker-compose.yml`'de GPU bölümünün commentini aç
3. Base image değiştir: `python:3.11` → `nvidia/cuda:12.2-devel-ubuntu22.04`

```dockerfile
FROM nvidia/cuda:12.2-devel-ubuntu22.04
...
```

## Troubleshooting

### GPU detected değilse:
```bash
docker run --gpus all nvidia/cuda:12.2-base nvidia-smi
```

### Volume permissions:
```bash
chmod 777 input output temp
```

### Log:
```bash
docker logs -f <container_id>
```

## WSL2 kullanıyorsan:
```bash
# WSL2 terminal'de:
cd /mnt/c/Users/simit/Documents/repos/video-chef
docker-compose build
docker-compose run video-chef --help
```
