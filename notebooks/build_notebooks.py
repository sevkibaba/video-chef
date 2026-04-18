#!/usr/bin/env python3
"""Generate both Colab notebooks with proper JSON formatting."""
import json, os

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Shared helper: the model-check + download cell source lines ──
def make_download_cell(title, repo_id, ckpt_dir, is_fstring=False):
    """Build the source lines for a robust model-check + download cell.
    
    If is_fstring=True, REPO_ID and CKPT_DIR are emitted as f-strings
    so that variables like {MODEL} are resolved at notebook runtime.
    """
    if is_fstring:
        repo_line = f'REPO_ID  = f"{repo_id}"\n'
        ckpt_line = f'CKPT_DIR = f"{ckpt_dir}"\n'
    else:
        repo_line = f'REPO_ID  = "{repo_id}"\n'
        ckpt_line = f'CKPT_DIR = "{ckpt_dir}"\n'
    return [
        f"# @title {title}\n",
        "import os, shutil, glob\n",
        "from huggingface_hub import snapshot_download\n",
        "\n",
        repo_line,
        ckpt_line,
        "\n",
        "os.makedirs(CKPT_DIR, exist_ok=True)\n",
        "\n",
        "# --- Storage check ---\n",
        '_, _, free = shutil.disk_usage("/")\n',
        'print(f"Local disk: {free // (2**30)} GB free")\n',
        "\n",
        "# --- Check if model already exists on Drive ---\n",
        "existing_files = os.listdir(CKPT_DIR) if os.path.isdir(CKPT_DIR) else []\n",
        'print(f"Files already in {CKPT_DIR}: {len(existing_files)}")\n',
        "for f in sorted(existing_files):\n",
        '    full = os.path.join(CKPT_DIR, f)\n',
        '    if os.path.isdir(full):\n',
        '        sub_count = len(os.listdir(full))\n',
        '        print(f"  [DIR]  {f}/  ({sub_count} files)")\n',
        '    else:\n',
        '        sz = os.path.getsize(full) / (1024**2)\n',
        '        print(f"  [FILE] {f}  ({sz:.1f} MB)")\n',
        "\n",
        "# --- Detect model weights: look for .safetensors or .bin in any subfolder ---\n",
        "weight_files = (\n",
        '    glob.glob(os.path.join(CKPT_DIR, "**", "*.safetensors"), recursive=True)\n',
        '    + glob.glob(os.path.join(CKPT_DIR, "**", "*.bin"), recursive=True)\n',
        ")\n",
        'print(f"Weight files found: {len(weight_files)}")\n',
        "\n",
        "if len(weight_files) >= 1:\n",
        '    print(f"\\u2705 Model weights found on Drive at {CKPT_DIR}. Skipping download.")\n',
        "else:\n",
        '    print(f"\\u274c No model weights found on Drive. Downloading {REPO_ID} ...")\n',
        "    snapshot_download(\n",
        "        repo_id=REPO_ID,\n",
        "        local_dir=CKPT_DIR,\n",
        "    )\n",
        '    print("Download complete.")\n',
        "\n",
        'print(f"\\nModel path for inference: {CKPT_DIR}")\n',
    ]


# ─────────────────────────────────────────────────────────
# Notebook 1: Text-to-Video
# ─────────────────────────────────────────────────────────
nb1 = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Video Chef -- Text-to-Video (Wan 2.2)\n",
                "\n",
                "Generate a short video from a text prompt using the open-source **Wan 2.2** family (Apache 2.0).\n",
                "\n",
                "**Default model: `TI2V-5B`** -- runs on a single **L4 (24 GB)**.\n",
                "For higher quality, switch to `T2V-A14B` (requires A100 40 GB).\n",
                "\n",
                "| Model | Task | Min VRAM | Colab tier | Speed (5 s @ 720p) |\n",
                "|---|---|---|---|---|\n",
                "| `ti2v-5B` | T2V + I2V | ~24 GB | L4 | ~9 min |\n",
                "| `t2v-A14B` | T2V (MoE 27B/14B active) | ~40 GB | A100 40 GB | ~15-25 min |\n",
                "\n",
                "> Runtime > Change runtime type > **L4** (or A100 for A14B).\n",
            ],
        },
        # ── 0. Check GPU ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ["# @title 0. Check GPU\n", "!nvidia-smi"],
        },
        # ── 1. Mount Drive ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# @title 1. Mount Google Drive (shared model cache)\n",
                "import os\n",
                "from google.colab import drive\n",
                "\n",
                "if not os.path.exists('/content/drive'):\n",
                "    drive.mount('/content/drive')\n",
                "\n",
                "!mkdir -p /content/drive/MyDrive/Wan2.2/outputs\n",
                "print('Drive mounted. Shared weights root: /content/drive/MyDrive/Wan2.2/')\n",
            ],
        },
        # ── 2. Clone + deps ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# @title 2. Clone Wan 2.2 repo + install dependencies\n",
                "%cd /content\n",
                "![ -d Wan2.2 ] || git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git\n",
                "%cd /content/Wan2.2\n",
                "!pip install -q -e .\n",
                "!pip install -q -r requirements.txt\n",
                "!pip install -q ftfy regex decord 'huggingface_hub[cli]'\n",
                "print('Setup done.')\n",
            ],
        },
        # ── 3. Download weights (shared on Drive) ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# @title 3a. Pick model\n",
                "# @markdown Pick the model. TI2V-5B = L4-friendly. T2V-A14B = A100 only.\n",
                'MODEL = "TI2V-5B"  # @param ["TI2V-5B", "T2V-A14B"]\n',
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": make_download_cell(
                title="3b. Download model weights (shared on Drive across notebooks)",
                repo_id='Wan-AI/Wan2.2-{MODEL}',
                ckpt_dir='/content/drive/MyDrive/Wan2.2/Wan2.2-{MODEL}',
                is_fstring=True,
            ),
        },
        # ── 4. Prompt & settings ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# @title 4. Prompt & settings\n",
                "# @markdown ### Prompt and generation settings\n",
                'PROMPT = "A cinematic shot of a fox running through a snowy forest at golden hour, 4k, shallow depth of field"  # @param {type:"string"}\n',
                'SIZE   = "1280*704"  # @param ["1280*704", "704*1280", "832*480", "480*832"]\n',
                "SEED   = 42  # @param {type:\"integer\"}\n",
                "# @markdown ### Duration (frame count must be 4k+1, rendered at 16fps)\n",
                'FRAME_NUM = 81  # @param [33, 49, 65, 81, 97, 113, 129] {type:"raw"}\n',
                "# @markdown Leave IMAGE empty for pure text-to-video. Provide a path for image-to-video (TI2V-5B only).\n",
                'IMAGE  = ""  # @param {type:"string"}\n',
                'print(f"Duration: ~{FRAME_NUM/16:.1f}s ({FRAME_NUM} frames @ 16fps)")\n',
                'print(f"MODEL={MODEL}  SIZE={SIZE}  SEED={SEED}  FRAMES={FRAME_NUM}")\n',
                'print(f"PROMPT={PROMPT}")\n',
            ],
        },
        # ── 5. Inference (flash_attn disabled) ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# @title 5. Run inference\n",
                "import time\n",
                "%cd /content/Wan2.2\n",
                "\n",
                'if MODEL == "TI2V-5B":\n',
                '    task  = "ti2v-5B"\n',
                '    flags = "--offload_model True --convert_model_dtype --t5_cpu"\n',
                "else:\n",
                '    task  = "t2v-A14B"\n',
                '    flags = "--offload_model True --convert_model_dtype"\n',
                "\n",
                "img_arg = f'--image \"{IMAGE}\"' if IMAGE else \"\"\n",
                "cmd = (\n",
                "    f'python generate.py --task {task} --size {SIZE} '\n",
                "    f'--ckpt_dir \"{CKPT_DIR}\" --base_seed {SEED} --frame_num {FRAME_NUM} {flags} {img_arg} '\n",
                "    f'--use_flash_attn false '\n",
                "    f'--prompt \"{PROMPT}\"'\n",
                ")\n",
                'print("Running:\\n", cmd, "\\n")\n',
                "t0 = time.time()\n",
                "!{cmd}\n",
                'print(f"\\nElapsed: {(time.time()-t0)/60:.1f} min")\n',
            ],
        },
        # ── 6. Show result ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# @title 6. Show result + save to Drive\n",
                "import glob, shutil, os, time\n",
                "from IPython.display import HTML, display\n",
                "from base64 import b64encode\n",
                "\n",
                'vids = sorted(glob.glob("/content/Wan2.2/*.mp4"), key=os.path.getmtime, reverse=True)\n',
                'assert vids, "No mp4 produced - check the inference cell output above."\n',
                "latest = vids[0]\n",
                "\n",
                "# Copy to Drive for safekeeping\n",
                'ts = time.strftime("%Y%m%d_%H%M%S")\n',
                'drive_out = f"/content/drive/MyDrive/Wan2.2/outputs/t2v_{MODEL}_{ts}.mp4"\n',
                "shutil.copy(latest, drive_out)\n",
                'print(f"Saved: {drive_out}")\n',
                "\n",
                "data_url = \"data:video/mp4;base64,\" + b64encode(open(latest, 'rb').read()).decode()\n",
                "display(HTML(f'<video width=720 controls src=\"{data_url}\"></video>'))\n",
            ],
        },
    ],
}

# ─────────────────────────────────────────────────────────
# Notebook 2: Character Replacement
# ─────────────────────────────────────────────────────────
nb2 = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "L4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Video Chef -- Character Replacement (Wan 2.2 Animate-14B)\n",
                "\n",
                "Replace the person in an input video with a character from a single reference image,\n",
                "preserving motion, facial expressions, and scene lighting (Relighting LoRA).\n",
                "\n",
                "**Model: `Wan2.2-Animate-14B`** (Apache 2.0)\n",
                "\n",
                "| Colab tier | GPU | Feasibility |\n",
                "|---|---|---|\n",
                "| Pay-as-you-go / Pro | **L4 24 GB** | Works with `--offload_model True --convert_model_dtype` (slow) |\n",
                "| Pay-as-you-go / Pro+ | **A100 40 GB** | Recommended - 2-3x faster |\n",
                "| Free | T4 16 GB | Too little VRAM, skip |\n",
                "\n",
                "> Runtime > Change runtime type > **L4** or **A100**.\n",
                ">\n",
                "> Pipeline: preprocess (pose + face + mask + bg) > generate > save.\n",
            ],
        },
        # ── 0. Check GPU ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ["# @title 0. Check GPU\n", "!nvidia-smi"],
        },
        # ── 1. Mount Drive ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# @title 1. Mount Google Drive\n",
                "import os\n",
                "from google.colab import drive\n",
                "\n",
                "if not os.path.exists('/content/drive'):\n",
                "    drive.mount('/content/drive')\n",
                "\n",
                "!mkdir -p /content/drive/MyDrive/Wan-Inputs /content/drive/MyDrive/Wan2.2/outputs\n",
                "print('Drive mounted.')\n",
                "print('Weights: /content/drive/MyDrive/Wan2.2/Wan2.2-Animate-14B')\n",
                "print('Inputs:  /content/drive/MyDrive/Wan-Inputs/')\n",
            ],
        },
        # ── 2. Clone + deps ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# @title 2. Clone Wan 2.2 + install animate deps\n",
                "%cd /content\n",
                "![ -d Wan2.2 ] || git clone --depth 1 https://github.com/Wan-Video/Wan2.2.git\n",
                "%cd /content/Wan2.2\n",
                "!pip install -q -e .\n",
                "!pip install -q -r requirements.txt\n",
                "!pip install -q -r requirements_animate.txt\n",
                "!pip install -q ftfy regex decord 'huggingface_hub[cli]'\n",
                "print('Setup done. Animation dependencies installed.')\n",
            ],
        },
        # ── 3. Download weights (shared on Drive) ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": make_download_cell(
                title="3. Download Animate-14B weights (shared on Drive across notebooks)",
                repo_id="Wan-AI/Wan2.2-Animate-14B",
                ckpt_dir="/content/drive/MyDrive/Wan2.2/Wan2.2-Animate-14B",
            ),
        },
        # ── 4. Select inputs ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# @title 4. Select inputs\n",
                "# @markdown Paths to your input video and reference character image.\n",
                "# @markdown Default: pulls first matching files from /content/drive/MyDrive/Wan-Inputs/\n",
                "import os, glob, shutil\n",
                "\n",
                'VIDEO_PATH = ""  # @param {type:"string"}\n',
                'IMAGE_PATH = ""  # @param {type:"string"}\n',
                "\n",
                'INPUTS_DIR = "/content/drive/MyDrive/Wan-Inputs"\n',
                "if not VIDEO_PATH:\n",
                '    mp4s = glob.glob(f"{INPUTS_DIR}/*.mp4")\n',
                '    VIDEO_PATH = mp4s[0] if mp4s else ""\n',
                "if not IMAGE_PATH:\n",
                '    imgs = glob.glob(f"{INPUTS_DIR}/*.jpg") + glob.glob(f"{INPUTS_DIR}/*.jpeg") + glob.glob(f"{INPUTS_DIR}/*.png")\n',
                '    IMAGE_PATH = imgs[0] if imgs else ""\n',
                "\n",
                'assert VIDEO_PATH and os.path.exists(VIDEO_PATH), f"Video not found: {VIDEO_PATH}"\n',
                'assert IMAGE_PATH and os.path.exists(IMAGE_PATH), f"Image not found: {IMAGE_PATH}"\n',
                "\n",
                'WORK_DIR = "/content/Wan2.2/examples/wan_animate/replace"\n',
                'PROC_DIR = f"{WORK_DIR}/process_results"\n',
                "os.makedirs(WORK_DIR, exist_ok=True)\n",
                'shutil.copy(VIDEO_PATH, f"{WORK_DIR}/video.mp4")\n',
                'shutil.copy(IMAGE_PATH, f"{WORK_DIR}/image{os.path.splitext(IMAGE_PATH)[1]}")\n',
                'print(f"VIDEO: {VIDEO_PATH}")\n',
                'print(f"IMAGE: {IMAGE_PATH}")\n',
                'print(f"WORK:  {WORK_DIR}")\n',
                "\n",
                "# @markdown ### Duration (frame count must be 4k+1, rendered at 16fps)\n",
                'FRAME_NUM = 81  # @param [33, 49, 65, 81, 97, 113, 129] {type:"raw"}\n',
                'print(f"Duration: ~{FRAME_NUM/16:.1f}s ({FRAME_NUM} frames @ 16fps)")\n',
            ],
        },
        # ── 5. Preprocess ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# @title 5. Preprocess (pose + face + mask + background)\n",
                "# @markdown Area of the generated video (W H). Use 832x480 for faster tests.\n",
                "RES_W = 832  # @param {type:\"integer\"}\n",
                "RES_H = 480  # @param {type:\"integer\"}\n",
                "\n",
                "%cd /content/Wan2.2\n",
                "img_ext = os.path.splitext(IMAGE_PATH)[1]\n",
                "import time; t0 = time.time()\n",
                "!python ./wan/modules/animate/preprocess/preprocess_data.py \\\n",
                '    --ckpt_path "{CKPT_DIR}/process_checkpoint" \\\n',
                '    --video_path "{WORK_DIR}/video.mp4" \\\n',
                '    --refer_path "{WORK_DIR}/image{img_ext}" \\\n',
                '    --save_path "{PROC_DIR}" \\\n',
                "    --resolution_area {RES_W} {RES_H} \\\n",
                "    --iterations 3 --k 7 --w_len 1 --h_len 1 \\\n",
                "    --replace_flag\n",
                'print(f"Preprocess elapsed: {(time.time()-t0)/60:.1f} min")\n',
            ],
        },
        # ── 6. Inference (flash_attn disabled) ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# @title 6. Run Wan-Animate (replacement mode, flash_attn disabled)\n",
                "import time\n",
                "%cd /content/Wan2.2\n",
                "t0 = time.time()\n",
                "!python generate.py --task animate-14B \\\n",
                '    --ckpt_dir "{CKPT_DIR}" \\\n',
                '    --src_root_path "{PROC_DIR}/" \\\n',
                "    --refert_num 1 \\\n",
                "    --replace_flag \\\n",
                "    --use_relighting_lora \\\n",
                "    --offload_model True \\\n",
                "    --convert_model_dtype \\\n",
                "    --use_flash_attn false \\\n",
                "    --t5_cpu \\\n",
                "    --frame_num {FRAME_NUM}\n",
                'print(f"Inference elapsed: {(time.time()-t0)/60:.1f} min")\n',
            ],
        },
        # ── 7. Show result ──
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# @title 7. Show result + save to Drive\n",
                "import glob, shutil, os, time\n",
                "from IPython.display import HTML, display\n",
                "from base64 import b64encode\n",
                "\n",
                'vids = sorted(glob.glob("/content/Wan2.2/*.mp4"), key=os.path.getmtime, reverse=True)\n',
                'assert vids, "No mp4 produced - check inference output."\n',
                "latest = vids[0]\n",
                "\n",
                'ts = time.strftime("%Y%m%d_%H%M%S")\n',
                'drive_out = f"/content/drive/MyDrive/Wan2.2/outputs/replace_{ts}.mp4"\n',
                "shutil.copy(latest, drive_out)\n",
                'print(f"Saved: {drive_out}")\n',
                "\n",
                "data_url = \"data:video/mp4;base64,\" + b64encode(open(latest, 'rb').read()).decode()\n",
                "display(HTML(f'<video width=720 controls src=\"{data_url}\"></video>'))\n",
            ],
        },
    ],
}

# ─────────────────────────────────────────────────────────
# Write both notebooks
# ─────────────────────────────────────────────────────────
for fname, nb in [("01_text_to_video.ipynb", nb1), ("02_character_replace.ipynb", nb2)]:
    path = os.path.join(NOTEBOOK_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=True)
    print(f"Wrote {path}")
