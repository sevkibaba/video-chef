# 🎬 Video Chef

Colab-first toolkit for AI video workflows using the open-source **Wan 2.2** model family
(Apache 2.0).

Two notebooks, no local setup required:

| Notebook | What it does | Model | Default GPU |
|---|---|---|---|
| [`notebooks/01_text_to_video.ipynb`](notebooks/01_text_to_video.ipynb) | Generate a 2–8 second video from a text prompt (optionally from an image) | `Wan2.2-TI2V-5B` (default) / `T2V-A14B` | L4 24 GB |
| [`notebooks/02_character_replace.ipynb`](notebooks/02_character_replace.ipynb) | Replace the person in an input video with your character image, preserving motion + lighting | `Wan2.2-Animate-14B` | L4 24 GB / A100 40 GB |

---

## 🚀 Quick start (Colab)

1. Open a notebook on Colab:
   - T2V: https://colab.research.google.com/github/sevkibaba/video-chef/blob/main/notebooks/01_text_to_video.ipynb
   - Character replace: https://colab.research.google.com/github/sevkibaba/video-chef/blob/main/notebooks/02_character_replace.ipynb
2. **Runtime → Change runtime type → GPU → L4** (or A100 if you have Pro+).
3. (Character replace only) put your inputs in `MyDrive/Wan-Inputs/` — see [Drive layout](#-google-drive-layout--inputs) below.
4. Run the cells top to bottom. First run downloads model weights and caches them to **Google Drive** under `MyDrive/Wan2.2/` — subsequent runs start in seconds.
5. Outputs are saved to `MyDrive/Wan2.2/outputs/`.

To run from **VS Code** instead of the Colab UI, see [`VSCODE_SETUP.md`](VSCODE_SETUP.md).

---

## 📂 Google Drive layout & inputs

The notebooks use the following structure on your Google Drive. If you have already
downloaded `Wan2.2-Animate-14B` into `MyDrive/Wan2.2/`, it will be reused — no re-download.

```
MyDrive/
├── Wan2.2/                                   ← weights root (auto-created / reused)
│   ├── Wan2.2-Animate-14B/                   ← ~30 GB, used by notebook 02
│   │   └── process_checkpoint/               ← required sub-folder for preprocessing
│   ├── Wan2.2-TI2V-5B/                       ← ~10 GB, used by notebook 01 (default)
│   ├── Wan2.2-T2V-A14B/                      ← ~28 GB, used by notebook 01 (optional, A100)
│   └── outputs/                              ← generated .mp4 files go here
│       ├── t2v_TI2V-5B_20260418_153012.mp4
│       └── replace_20260418_161245.mp4
│
└── Wan-Inputs/                               ← ONLY for notebook 02 (character replace)
    ├── video.mp4                             ← the source video (the person to replace)
    └── character.jpg                         ← the new character to put into the video
```

### Input files for notebook 02 (character replacement)

Place **exactly two files** in `MyDrive/Wan-Inputs/`:

| File | Required name | Format | Notes |
|---|---|---|---|
| Source video | any `*.mp4` (e.g. `video.mp4`) | MP4, H.264 | 5–15 seconds recommended. Must clearly show the person you want to replace. |
| Character image | any `*.jpg`, `*.jpeg`, or `*.png` (e.g. `character.jpg`) | JPEG/PNG | Full-body or upper-body. Clean background preferred. Single person. |

The notebook auto-picks the **first** `.mp4` and the **first** image it finds in that folder.
If you keep multiple videos/images in there, either (a) pick specific paths by filling in
`VIDEO_PATH` and `IMAGE_PATH` in cell 4, or (b) make sure the intended files sort first
alphabetically.

### Notebook 01 (text-to-video) takes no input files

It only needs a text prompt (edited in cell 4 of the notebook). No Drive inputs required.

### Duration control

Both notebooks include a **`FRAME_NUM`** setting to control output video length.
The frame count must follow `4k+1` (Wan 2.2 renders at 16 fps):

| `FRAME_NUM` | Duration |
|---|---|
| 33 | ~2 sec |
| 49 | ~3 sec |
| 65 | ~4 sec |
| **81** | **~5 sec (default)** |
| 97 | ~6 sec |
| 113 | ~7 sec |
| 129 | ~8 sec |

### Re-using an existing download

If `MyDrive/Wan2.2/Wan2.2-Animate-14B/` already exists with the full weights (your case),
cell 3 of notebook 02 will detect it and **skip the download** — `huggingface_hub.snapshot_download`
verifies existing files and only fetches what's missing. Same for `TI2V-5B` / `T2V-A14B` in notebook 01.

---

## 💻 GPU / tier compatibility

Pay-as-you-go Colab gives you T4, **L4**, and (rarely) A100 40GB. H100 and A100 80GB are
Colab Enterprise only.

| Task | T4 16GB | L4 24GB | A100 40GB |
|---|---|---|---|
| T2V — `TI2V-5B` (720p @ 24fps) | ❌ | ✅ default, ~9 min / 5s | ✅ fastest |
| T2V — `T2V-A14B` MoE | ❌ | ⚠️ OOM even with offload | ✅ with `--offload_model True` |
| Character replace — `Animate-14B` | ❌ | ⚠️ slow (~25–40 min / clip) | ✅ recommended |

Both notebooks set the correct `--offload_model True --convert_model_dtype --t5_cpu` flags
automatically based on the selected GPU.

---

## 🤖 Why Wan 2.2?

After evaluating current (April 2026) open-source options, Wan 2.2 is the best fit because:

- **T2V quality** — `T2V-A14B` (MoE, 27B total / 14B active) is competitive with closed Veo3/PixVerse V5 and runs locally.
- **Efficient variant** — `TI2V-5B` with 16×16×4 VAE compression is one of the fastest open 720p@24fps models and fits in 24 GB.
- **Unified character replacement** — `Animate-14B` handles both "animate a still image" and "replace character in video" with the same weights, plus a Relighting LoRA for scene-consistent lighting.
- **Apache 2.0** license, actively maintained by Alibaba Tongyi Lab.
- Well-integrated with Diffusers and ComfyUI if you later want to switch runtime.

### Other open-source models considered (and why not defaulted)

| Model | Strength | Why not default |
|---|---|---|
| HunyuanVideo 13B (Tencent) | Sharp textures, strong T2V | Needs ≥24 GB FP8, ≥40 GB FP16 — comparable to Wan T2V-A14B but without the character-replacement counterpart |
| LTX-Video / LTX-2 (Lightricks) | Fastest open model | Lower visual quality; good for drafting not final |
| CogVideoX-5B (Zhipu) | Well-supported in Diffusers | Lower resolution (720×480), weaker motion |
| Mochi 1 (Genmo) | High T2V quality | 24 GB minimum, 480p only in practice |
| Allegro (rhymes-ai) | 9 GB with CPU offload | Quality behind Wan 2.2 |
| Alice 14B (Mirage) | 4-step inference, 7× faster | New; less tooling around it |
| LongCat-Video (Meituan) | Long-form video | Same VRAM class as Wan, newer / less validated |
| AnimateAnyone / MimicMotion / Champ / EchoMimicV2 | Pose-driven character animation | All superseded by Wan-Animate per their own authors |
| SwapAnyone (PKU) | Purpose-built person swap | Narrower scope than Wan-Animate; good secondary option |
| Wan 2.5 / 2.6, Sora 2, Veo 3, Kling | Higher quality | **Closed source / API only** — not runnable on Colab |

---

## 📁 Repo structure

```
video-chef/
├── notebooks/
│   ├── 01_text_to_video.ipynb       # Wan 2.2 TI2V-5B / T2V-A14B
│   ├── 02_character_replace.ipynb   # Wan 2.2 Animate-14B (replace mode)
│   ├── CHARACTER_REPLACE_GUIDE.md   # How to prepare inputs for character replacement
│   └── build_notebooks.py          # Script to regenerate notebooks with valid JSON
├── README.md
├── VSCODE_SETUP.md                  # Running notebooks from VS Code
└── .gitignore
```

No local Python package, no Docker, no src/ code — every step runs inside the notebook
on a Colab GPU against the upstream `github.com/Wan-Video/Wan2.2` repo.

---

## 🔗 Upstream

- Wan 2.2 code: https://github.com/Wan-Video/Wan2.2
- Weights: https://huggingface.co/Wan-AI
- License: Apache 2.0

