# Live Demo Guide: Real User Flow on Google Colab GPU

This walks through demoing the real pipeline — open the frontend, upload a
lecture video, get back a real condensed summary video — with the backend
running on a Colab GPU runtime (fast Whisper/ResNet/ViT inference) and the
frontend running locally on your laptop, connected to Colab over a public
tunnel URL.

## Part 1 — Colab: environment setup

New Colab notebook, GPU runtime (`Runtime > Change runtime type > GPU`).

```python
# Cell 1 - Mount Drive (for checkpoints/features too large for GitHub)
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# Cell 2 - Clone repo and checkout your branch
!git clone https://github.com/<your-org>/lecture-video-summarizer.git
%cd lecture-video-summarizer
!git checkout <your-branch>
```

```bash
# Cell 3 - Recreate data/processed (gitignored, doesn't exist in a fresh clone)
!mkdir -p data/processed
!ln -sfn /content/drive/MyDrive/INTEGRA_Data/raw data/raw
!ln -sfn /content/drive/MyDrive/INTEGRA_Data/features data/processed/features
```

```bash
# Cell 4 - Install dependencies
!pip install -r requirements.txt

# Fix the recurring torch/torchvision/torchaudio CUDA-mismatch bug
# (requirements.txt's CUDA-13 nvidia-* packages can force a plain-PyPI torch
# wheel while Colab's pre-installed torchvision/torchaudio stay on a
# different CUDA build - verify with the print at the end of this cell,
# and only re-run the force-reinstall line if it reports a mismatch):
!pip install --force-reinstall --no-deps torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# easyocr is pinned in requirements.txt but has been seen missing from a
# plain pip install in this environment before - confirm it's importable:
!python -c "import easyocr" || pip install easyocr==1.7.2

!python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Part 2 — Checkpoints (gitignored, must come from Drive)

```bash
# Cell 5 - Copy/symlink the three trained checkpoints from Drive
!cp /content/drive/MyDrive/INTEGRA_Data/checkpoints/best_module1_model.pt .
!mkdir -p src/module2_summarization/bert_model_v2
!cp /content/drive/MyDrive/INTEGRA_Data/checkpoints/model.safetensors src/module2_summarization/bert_model_v2/
!mkdir -p models/module3/vit_slide_classifier
!cp /content/drive/MyDrive/INTEGRA_Data/checkpoints/vit_model.safetensors models/module3/vit_slide_classifier/model.safetensors
```

## Part 3 — `.env` in Colab

```bash
# Cell 6 - Write a .env for this Colab session
%%writefile .env
SUPABASE_URL=<your real Supabase URL>
SUPABASE_KEY=<your real Supabase key>
SUPABASE_JOBS_TABLE=jobs
GEMINI_API_KEY=<a freshly regenerated key from https://aistudio.google.com/apikey>
GEMINI_MODEL=gemini-flash-lite-latest
GEMINI_FALLBACK_MODEL=gemini-flash-latest
```

Your current `GEMINI_API_KEY` was confirmed 401ing this session (`ACCESS_TOKEN_TYPE_UNSUPPORTED`) — regenerate it before the demo. If you don't get to it in time, Module 3 still runs via its existing OCR fallback, just with lower-quality "Critical" frame analysis (no Gemini semantic enrichment). Ollama/GPT-4o keys aren't needed — Pipeline A (what's now wired into Module 4) doesn't use them.

## Part 4 — Start the backend

```bash
# Cell 7 - Start FastAPI in the background, confirm it's up
!nohup uvicorn src.backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
import time; time.sleep(5)
!curl -s http://localhost:8000/
```

You should see `{"message":"INTEGRA backend is running"}`. If not, check `backend.log`.

## Part 5 — Expose it publicly

```bash
# Cell 8 - pyngrok (requires a free account/authtoken - sign up at ngrok.com
# ahead of time in case this is a last-minute blocker)
!pip install pyngrok -q
from pyngrok import ngrok
ngrok.set_auth_token("<your ngrok authtoken>")
public_url = ngrok.connect(8000, "http")
print("Public backend URL:", public_url)
```

**No-signup fallback** if ngrok access is an issue on demo day:
```bash
!npm install -g localtunnel
!nohup lt --port 8000 > lt.log 2>&1 &
import time; time.sleep(5)
!cat lt.log   # prints the public https://*.loca.lt URL
```

Verify from a separate device/network (not just from within Colab) that the printed URL responds:
```bash
!curl -s <public_url>/
```

## Part 6 — Point the local frontend at Colab

On your laptop, in `frontend/`:

```bash
echo "VITE_API_URL=<public_url from Part 5>" > .env
npm run dev
```

Open `http://localhost:3000`, and run through the real flow: drop a video in, click "Run Pipeline". The frontend now makes real `POST /api/upload` and polls real `GET /api/jobs/{id}/status` calls against the Colab-hosted backend (confirmed working the same way against a local backend this session — same code path, just a different `API_BASE`).

**Test with `scripts_scratch_test_clip.mp4` first**, not a full 60-minute lecture, before doing this live in front of the panel. Even on Colab GPU, a full lecture still takes real time end-to-end (Whisper transcription + ResNet/ViT inference + video rendering), and you want to catch any Colab-specific path issue (missing checkpoint, wrong Drive path, tunnel drop) on a fast, low-stakes run first.

## Troubleshooting quick-reference

| Symptom | Cause | Fix |
|---|---|---|
| `torch`/`torchvision` CUDA version mismatch error | requirements.txt's CUDA-13 packages forced a fresh torch wheel | Re-run the `--force-reinstall --no-deps torch==... torchvision==... torchaudio==...` line in Cell 4 |
| `ModuleNotFoundError: easyocr` | Not always installed by a plain `pip install -r requirements.txt` | `pip install easyocr==1.7.2` |
| Module 2 stuck "running" for a very long time | Whisper `large-v3` (2.88GB) download interrupted/corrupted, re-downloading on a slow connection | Check Colab's own network speed (usually fast) — this was a *local laptop* network issue in earlier testing, should not recur on Colab's connection, but if it does, delete `~/.cache/whisper/large-v3.pt` and retry, or temporarily switch to a smaller Whisper model for the demo |
| Job status `failed`, error mentions Module 3 | Missing ViT/BERT checkpoint, or `easyocr` missing | Re-check Part 2's checkpoint copy step and Part 1's `easyocr` check |
| Frontend shows nothing after upload / CORS error in console | Tunnel URL wrong or backend not actually reachable | Re-run the `curl <public_url>/` check from Part 5 before touching the frontend |
