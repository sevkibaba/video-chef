# 🎬 Video Chef v2 — Pose-Guided Character Replacement

> Replace any person in a video with your character image — keeping the original
> poses, movement, and background — using open-source AI models running 100% locally
> or on a cloud GPU.

---

## 🏗️ How It Works

```
Input video + Character JPEG
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  STEP 1 │ Extract Frames + Audio (FFmpeg)               │
├─────────────────────────────────────────────────────────┤
│  STEP 2 │ Person Tracking (YOLO → SAM2)                 │
│          YOLO detects person center in frame 0           │
│          SAM2 tracks them across ALL frames              │
│          → Per-frame binary masks                        │
├─────────────────────────────────────────────────────────┤
│  STEP 3 │ Pose Extraction (DWPose)                       │
│          Skeleton image per frame                        │
│          (OpenPose format for ControlNet)                │
├─────────────────────────────────────────────────────────┤
│  STEP 4 │ Character Generation (ControlNet + IP-Adapter) │
│          IP-Adapter: use your JPEG as appearance ref     │
│          ControlNet: force exact same pose               │
│          → New frame: your character, original pose      │
├─────────────────────────────────────────────────────────┤
│  STEP 5 │ Compositing                                    │
│          Remove BG from generated frame (rembg)          │
│          Scale to original person bounding box           │
│          Paste onto original background                  │
├─────────────────────────────────────────────────────────┤
│  STEP 6 │ Video Assembly (FFmpeg + original audio)       │
└─────────────────────────────────────────────────────────┘
        │
        ▼
Output video.mp4
```

---

## 📦 Project Structure

```
video-chef/
├── pipelines/
│   └── pose_transfer_v1.py     ← Main pipeline (run this)
│
├── src/
│   ├── video_io.py             ← FFmpeg wrapper
│   ├── person_tracker.py       ← SAM2 + YOLO auto-detect
│   ├── pose_extractor.py       ← DWPose skeleton extraction
│   ├── character_generator.py  ← ControlNet + IP-Adapter
│   └── compositor_v2.py        ← Composite char onto BG
│
├── scripts/
│   ├── setup_cloud.sh          ← One-shot cloud GPU setup
│   └── download_models.sh      ← Download SAM2 checkpoints
│
├── input/                      ← Put your files here
│   ├── video.mp4
│   └── character.jpg
│
├── output/                     ← Results go here
├── checkpoints/                ← SAM2 model weights
└── requirements.txt
```

---

## 🚀 Setup

### 🍎 Running on Mac

> **Three ways to run on Mac — choose based on your goal:**

| Mode | What it is | Speed | Good for |
|---|---|---|---|
| **Docker (CPU)** | Pipeline in a container, CPU only | 🐢 Very slow | Verifying the pipeline works |
| **Native Python + MPS** | Python directly on Mac, Metal GPU | 🐇 Medium | Apple Silicon M1/M2/M3 |
| **Cloud GPU** | SSH into RunPod/vast.ai | 🚀 Fast | Real results |

---

#### Option A — Docker on Mac (CPU, for testing only)

> ⚠️ Docker on Mac has **no GPU access** (no CUDA, no Metal). Everything runs on CPU.
> Expect ~5–15 minutes per frame. Use `--max-frames 5` to validate the pipeline works, then switch to cloud or native Python for real processing.

**Prerequisites:** [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop/) — give it at least 8GB RAM in Settings → Resources.

```bash
# 1. Build the Mac image (CPU-only, ~first build takes 5-10 min)
docker compose -f docker-compose.mac.yml build

# 2. Put your files in input/
cp /path/to/your/video.mp4      input/video.mp4
cp /path/to/your/character.jpg  input/character.jpg

# 3. Run a quick 5-frame test (validates the whole pipeline)
docker compose -f docker-compose.mac.yml run --rm video-chef-mac \
  --video input/video.mp4 \
  --character input/character.jpg \
  --output output/test.mp4 \
  --max-frames 5 \
  --steps 10 \
  --use-rembg \
  --device cpu

# Result will be at: output/test.mp4
```

> **Note:** The first run downloads ~5–8GB of models (SD1.5, ControlNet, IP-Adapter).
> These are cached in `./model_cache/` — subsequent runs start instantly.

---

#### Option B — Native Python on Mac (Apple Silicon, recommended)

This runs **outside Docker**, using MPS (Metal) which is ~3–5× faster than CPU on M1/M2/M3.

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install CPU/MPS PyTorch (no CUDA needed)
pip install torch torchvision

# 3. Install dependencies
pip install -r requirements.mac.txt

# 4. Install FFmpeg
brew install ffmpeg

# 5. Run (device auto-detects MPS on Apple Silicon)
python pipelines/pose_transfer_v1.py \
  --video input/video.mp4 \
  --character input/character.jpg \
  --output output/result.mp4 \
  --backend sd15 \
  --max-frames 30 \
  --use-rembg
```

> MPS is **automatically detected** — no `--device mps` flag needed (it's the default on Apple Silicon).

---

#### Mac Performance Guide

| Mac | SD1.5, 25 steps | Est. per frame |
|---|---|---|
| M1 / M2 (8GB RAM) | CPU only | ~8–12 min |
| M1 Pro / M2 Pro (16GB RAM) | MPS | ~3–5 min |
| M1 Max / M2 Max / M3 Max (32GB RAM) | MPS | ~1–2 min |
| M4 Max (48GB RAM) | MPS | ~30–60 sec |

**For a 30s video at 30fps = 900 frames:**
- M2 Pro native (MPS): ~45–75 hours 😅 — use cloud GPU for real videos
- Use Mac to **test 5–30 frames**, then push to cloud for full processing

---

### Local Machine (Linux + NVIDIA GPU)

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 2. Install PyTorch (adjust CUDA version if needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install all other dependencies
pip install -r requirements.txt

# 4. Install SAM2
pip install git+https://github.com/facebookresearch/sam2.git

# 5. Download SAM2 checkpoints
bash scripts/download_models.sh

# 6. Install FFmpeg (if not already installed)
# Mac:   brew install ffmpeg
# Linux: sudo apt install ffmpeg
```

### Cloud GPU (RunPod / vast.ai / Lambda Labs)

```bash
# Clone the repo
git clone https://github.com/sevkibaba/video-chef.git
cd video-chef

# Run setup script (installs everything automatically)
bash scripts/setup_cloud.sh
```

> Recommended cloud instance: **A100 40GB** or **H100 80GB**
> A100 80GB can process ~1min of video in ~20 minutes.

---

## 🎬 Usage

### Basic (SD1.5 — works on 8GB VRAM)

```bash
python pipelines/pose_transfer_v1.py \
  --video input/video.mp4 \
  --character input/character.jpg \
  --output output/result.mp4 \
  --backend sd15 \
  --device cuda
```

### High Quality (SDXL — recommended on cloud A100)

```bash
python pipelines/pose_transfer_v1.py \
  --video input/video.mp4 \
  --character input/character.jpg \
  --output output/result.mp4 \
  --backend sdxl \
  --device cuda
```

### Test Run (first 30 frames only)

```bash
python pipelines/pose_transfer_v1.py \
  --video input/video.mp4 \
  --character input/character.jpg \
  --output output/test.mp4 \
  --max-frames 30 \
  --keep-temp
```

### No SAM2 (fallback mode with rembg only)

```bash
python pipelines/pose_transfer_v1.py \
  --video input/video.mp4 \
  --character input/character.jpg \
  --output output/result.mp4 \
  --use-rembg
```

---

## ⚙️ All Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Input video path |
| `--character` | required | Character JPEG/PNG |
| `--output` | `output/result.mp4` | Output video path |
| `--backend` | `sd15` | `sd15` (faster) or `sdxl` (better) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--max-frames` | all | Limit frames (for testing) |
| `--ip-scale` | `0.7` | Character appearance strength (0–1) |
| `--controlnet-scale` | `0.9` | Pose adherence strength (0–1) |
| `--steps` | `25` | Diffusion inference steps |
| `--sam2-model` | `large` | SAM2 size: `large`, `base`, `small`, `tiny` |
| `--use-rembg` | off | Use rembg instead of SAM2 |
| `--keep-temp` | off | Keep intermediate files for debugging |
| `--prompt` | see below | Positive text prompt for generation |

**Default prompt:**
```
a person, full body, photorealistic, detailed, sharp
```

---

## 🔧 Tuning Tips

### Character looks too different from the JPEG?
Increase `--ip-scale` (e.g., `0.85`)

### Character pose doesn't match the video?
Increase `--controlnet-scale` (e.g., `0.95`)

### Compositing looks unnatural (hard edges)?
This is controlled in `compositor_v2.py` → `edge_feather` parameter (default: 8px).

### Temporal flickering between frames?
The pipeline uses a fixed seed (`--seed` not exposed in CLI yet but set in `CharacterGenerator`).
For smoother results, try `AnimateDiff` as a future upgrade.

### Character too big/small in the frame?
This is driven by the SAM2 mask bounding box. If the mask is wrong, try:
- Using `--sam2-model large` for better tracking
- Or provide a manual `click_point` in `person_tracker.py`

---

## 📊 Performance Estimates

| Machine | Backend | 30s video @ 30fps | Est. Time |
|---------|---------|-------------------|-----------|
| RTX 3080 (10GB) | SD1.5 | 900 frames | ~45 min |
| RTX 4090 (24GB) | SDXL | 900 frames | ~30 min |
| A100 40GB | SDXL | 900 frames | ~15 min |
| A100 80GB | SDXL | 900 frames | ~10 min |

> Tip: Use `--max-frames 30` first to validate quality before running the full video.

---

## 🤖 Models Used

| Model | Purpose | Size | License |
|-------|---------|------|---------|
| YOLOv8n | Auto-detect person in frame 0 | ~6MB | AGPL-3.0 |
| SAM2 Large | Track person across video | ~900MB | Apache 2.0 |
| DWPose | Skeleton extraction | ~400MB | Apache 2.0 |
| ControlNet OpenPose (SD1.5) | Pose-conditioned generation | ~1.4GB | OpenRAIL-M |
| IP-Adapter (SD1.5) | Appearance from JPEG | ~300MB | Apache 2.0 |
| SD1.5 | Base diffusion model | ~4GB | OpenRAIL-M |
| rembg (u2net_human_seg) | BG removal from generated frames | ~176MB | MIT |

> All models download automatically on first run (via HuggingFace Hub / ultralytics).
> Only SAM2 requires a manual checkpoint download (see setup above).

---

## 🗺️ Roadmap

- [x] SAM2 person tracking
- [x] DWPose skeleton extraction
- [x] ControlNet + IP-Adapter generation
- [x] Smart compositing with bounding box alignment
- [ ] Temporal consistency (AnimateDiff / frame interpolation)
- [ ] 3D model support (Blender render → IP-Adapter reference)
- [ ] Lighting adaptation (match character lighting to scene)
- [ ] Batch processing (multiple videos)
- [ ] GPU memory optimization (xformers, attention slicing)

---

*Video Chef v2 — Cook your videos with AI 🍳*
