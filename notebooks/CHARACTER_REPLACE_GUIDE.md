# 🎭 Video Chef — Character Replacement Prompt Guide

How to get great results from **Wan 2.2 Animate-14B** character replacement.

---

## How Character Replacement Works

Unlike text-to-video, character replacement takes **two inputs**:

1. **Source Video** — The original video with a person whose body/motion you want to keep.
2. **Reference Image** — A single photo of the character you want to swap in.

The model extracts **pose, face landmarks, and body mask** from the source video, then re-renders the scene with the reference character's appearance while preserving the original motion.

> **There is no text prompt in character replacement mode.** The model is driven entirely by the video motion + reference image.

---

## Input Requirements

### Source Video

| Rule | Details |
|---|---|
| **Single person** | The video should contain **one clearly visible person**. Multi-person scenes confuse the pose extractor. |
| **Clear body visibility** | Full or upper body is best. Avoid extreme close-ups (face only) or very far away shots. |
| **Stable camera** | Moderate camera movement is fine, but avoid shaky handheld footage. |
| **Duration** | 2–8 seconds works best. Use `FRAME_NUM` to control output length. |
| **Format** | `.mp4` (H.264). Place in `/content/drive/MyDrive/Wan-Inputs/`. |
| **Resolution** | Any resolution works — the preprocessor will resize. 720p–1080p input is ideal. |

### Reference Image

| Rule | Details |
|---|---|
| **Single person** | One clearly visible character in the image. |
| **Face visible** | The model needs to see the face to transfer facial features. |
| **Neutral pose** | A front-facing or 3/4 pose works best. Avoid extreme angles. |
| **Clean background** | Plain or simple backgrounds help the model isolate the character. |
| **High quality** | Sharp, well-lit photos produce better results. Avoid blurry or low-res images. |
| **Format** | `.jpg`, `.jpeg`, or `.png`. Place in `/content/drive/MyDrive/Wan-Inputs/`. |

---

## Settings That Matter

### Resolution (`RES_W` × `RES_H`)

Set in **Cell 5 (Preprocess)**. This controls the output video resolution.

| Setting | Use Case |
|---|---|
| `832 × 480` | Fast test runs (~2× faster than 720p) |
| `1280 × 720` | Production quality (recommended for final output) |

> ⚠️ Higher resolution = more VRAM + longer processing time.

### Frame Count (`FRAME_NUM`)

Set in **Cell 4 (Select inputs)**. Controls output duration.

| Frames | Duration @ 16fps |
|---|---|
| 33 | ~2 sec |
| 49 | ~3 sec |
| 65 | ~4 sec |
| **81** | **~5 sec (default)** |
| 97 | ~6 sec |
| 113 | ~7 sec |
| 129 | ~8 sec |

> 💡 If your source video is 3 seconds, set `FRAME_NUM = 49`. Don't exceed the source video length.

### Relighting LoRA (`--use_relighting_lora`)

This is **enabled by default** in the notebook. It adjusts the lighting on the swapped character to match the scene. Keep it on for best results.

---

## Tips for Best Results

### ✅ Do

- **Match body type**: If your source video has a tall person, use a reference image of a similar build.
- **Match clothing style**: The model blends appearance, so similar clothing styles produce more coherent results.
- **Use well-lit inputs**: Both the video and reference image should have good, even lighting.
- **Start with short clips**: Test with `FRAME_NUM = 33` (2 sec) first to check quality before committing to longer renders.
- **Use 832×480 for testing**: Switch to 1280×720 only for the final render.

### ❌ Don't

- **Don't use multi-person videos**: The pose extractor picks one person and results are unpredictable with crowds.
- **Don't use extreme motion**: Very fast movements (jumping, running) can cause artifacts.
- **Don't use occluded subjects**: If the person is behind objects or partially hidden, the mask extraction fails.
- **Don't exceed source duration**: Setting `FRAME_NUM` higher than the source video's actual frame count produces garbage.
- **Don't use cartoon/anime reference images**: The model is trained on real human data. Stylized characters produce poor results.

---

## Example Workflows

### Basic Character Swap

1. Place `dance_video.mp4` and `my_character.jpg` in `/content/drive/MyDrive/Wan-Inputs/`
2. Run cells 0–3 (setup, drive, deps, model download)
3. Cell 4: Leave `VIDEO_PATH` and `IMAGE_PATH` empty (auto-detected)
4. Cell 4: Set `FRAME_NUM = 49` for a quick 3-sec test
5. Cell 5: Set `RES_W = 832`, `RES_H = 480` for a fast test
6. Run cells 5–7

### Production Quality Render

1. Same as above, but in Cell 5 set `RES_W = 1280`, `RES_H = 720`
2. In Cell 4 set `FRAME_NUM = 81` (or match your source video length)
3. Expect ~15–25 min on A100, ~30–45 min on L4

### Multiple Characters

To swap in different characters using the same source video:
1. Run cells 0–5 once (preprocessing only needs to happen once per video)
2. Change `IMAGE_PATH` in Cell 4 to your new character image
3. **Skip Cell 5** (preprocessing is already done)
4. Re-run Cell 6 (inference) and Cell 7 (save)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Face looks wrong | Use a clearer, front-facing reference image |
| Body proportions are off | Match source video body type to reference image |
| Flickering/artifacts | Reduce resolution or use a more stable source video |
| OOM (Out of Memory) | Use `832×480` resolution, or switch to A100 GPU |
| Output is black/empty | Check that preprocessing (Cell 5) completed without errors |
| Wrong person detected | Ensure only one person is visible in the source video |
