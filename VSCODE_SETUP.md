# 🖥️ Running the notebooks from VS Code

You don't need to leave VS Code — there are three ways to run the notebooks, each connecting to
a different remote GPU. Pick the one that matches how you buy compute.

> The notebooks themselves (`notebooks/01_text_to_video.ipynb`, `notebooks/02_character_replace.ipynb`)
> are standard Jupyter `.ipynb` files and work with any of the options below.

---

## Prerequisites (once)

1. **Install VS Code**: https://code.visualstudio.com/
2. **Install these extensions** (all from Microsoft):
   - `Python` (`ms-python.python`)
   - `Jupyter` (`ms-toolsai.jupyter`)
   - `Remote - SSH` (`ms-vscode-remote.remote-ssh`) — only for Option C
3. **Clone this repo:**
   ```bash
   git clone https://github.com/sevkibaba/video-chef.git
   cd video-chef
   code .
   ```

---

## Option A — Local VS Code connected to a Google Colab runtime (recommended for pay-as-you-go)

This gives you VS Code's editor + a real Colab GPU.

1. Open Colab in a browser: https://colab.research.google.com
2. File → Upload notebook → pick `notebooks/01_text_to_video.ipynb` (or open it directly via the GitHub link in `README.md`).
3. **Runtime → Change runtime type → GPU → L4** (or A100 if available), Save.
4. **Runtime → Connect** to start the VM.
5. **Tools → Settings → Advanced → "Enable connections via the Jupyter kernel"** → Save.
6. **Tools → Command Palette → "Connect to local runtime"**, copy the Jupyter URL that Colab prints (looks like `http://localhost:8888/?token=...`). Colab also provides a **Colab-hosted** connection URL; use that one.
7. In VS Code: open the notebook file → click the kernel picker (top right) → **Select Another Kernel → Existing Jupyter Server** → paste the URL from step 6.

Now every cell you run in VS Code executes on Colab's L4/A100. Your Drive mount and weights cache from the Colab UI carry over.

> **Note:** Colab restricts the kernel URL to the browser tab that opened it. The official supported path is "Colab UI in browser + editor in VS Code" via the **"Open with Colab"** GitHub integration — see Option B.

---

## Option B — VS Code + GitHub → one-click Colab

Fastest workflow:

1. Push your fork to GitHub.
2. In VS Code's terminal, open the GitHub URL of a notebook and prepend `colab.research.google.com/github/`:
   ```
   https://colab.research.google.com/github/<your-user>/video-chef/blob/main/notebooks/01_text_to_video.ipynb
   ```
3. The notebook opens on Colab with a live GPU. Edit in VS Code, commit, push, reload Colab.

This is what the "Quick start" links in `README.md` do.

---

## Option C — VS Code Remote SSH into a rented GPU box (cheapest for heavy use)

When pay-as-you-go Colab compute units get expensive, rent a GPU directly. Prices as of April 2026:

| Provider | A100 40GB | A100 80GB | H100 80GB |
|---|---|---|---|
| **RunPod** | ~$0.79/h | ~$1.19/h | ~$2.49/h |
| **Vast.ai** | ~$0.50/h | ~$0.80/h | ~$1.80/h |
| **Lambda** | ~$1.29/h | ~$1.79/h | ~$2.99/h |

### Steps

1. **Create a pod** on RunPod or Vast.ai with an **A100 40GB** (or larger) and the **"PyTorch 2.4 CUDA 12.1"** template. Make sure to allocate at least **100 GB disk**.
2. Copy the SSH command they give you, e.g. `ssh root@ssh.runpod.io -p 12345 -i ~/.ssh/id_ed25519`.
3. In VS Code: **Command Palette → Remote-SSH: Add New SSH Host** → paste the command.
4. **Remote-SSH: Connect to Host** → pick the new host. A new VS Code window opens inside the pod.
5. In the remote VS Code terminal:
   ```bash
   apt-get update && apt-get install -y git ffmpeg
   git clone https://github.com/sevkibaba/video-chef.git
   cd video-chef
   ```
6. Open `notebooks/01_text_to_video.ipynb` in the remote VS Code. It will prompt you to pick/install a Python + Jupyter kernel — accept.
7. **Skip the `drive.mount(...)` cell** (no Google Drive in a rented pod). Adjust weight/input/output paths to local dirs — e.g. change `/content/drive/MyDrive/Wan2.2/...` to `/workspace/Wan2.2/...` and `/content/drive/MyDrive/Wan-Inputs` to `/workspace/Wan-Inputs`.
8. Run the cells. Stop the pod when done to avoid charges.

> **Tip:** mount a persistent volume for `weights/` so you don't re-download 30 GB every session.

---

## Option D — Google Colab Enterprise (for A100 80GB / H100)

If you need H100s or A100 80GB inside the Colab UI itself, that requires **Colab Enterprise** via Google Cloud (billed through GCP). Pricing is ~$3.5/h (A100 80GB) to ~$11/h (H100). The notebooks work unchanged — just pick a larger machine type in Vertex AI. Not needed for the defaults in this repo.

---

## Which option should I pick?

- **Trying it out / occasional runs** → Option B (Colab link in README).
- **Want VS Code editor UX specifically, pay-as-you-go Colab credits** → Option A.
- **Running many videos, cost-sensitive** → Option C (RunPod or Vast.ai).
- **Enterprise / need H100 in Colab UI** → Option D.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `flash_attn` build fails during install | Ignore — Wan 2.2 falls back to SDPA attention. The `|| true` in the install cell handles this. |
| `CUDA out of memory` on L4 during Animate-14B | Lower `RES_W`/`RES_H` to `832 480` in the preprocess cell; ensure `--offload_model True --convert_model_dtype --t5_cpu` flags are set. |
| First download very slow | Expected — Wan2.2-Animate-14B is ~30 GB. Drive caching means it only happens once. |
| Kernel disconnects on Colab | Free tier times out after ~90 min idle. Upgrade to pay-as-you-go or use Option C. |
| Notebook can't find input video | Put your `.mp4` and character `.jpg` in `MyDrive/Wan-Inputs/` before running cell 4. |

