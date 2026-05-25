# INTEGRA — Complete Environment Setup Guide
### Automated Lecture Video Summarization | Setup

> **Project Plan Alignment:** 
> *"Install Python 3.10, PyTorch, CUDA drivers, OpenCV on all 4 machines.
> Verify GPU is detected (`torch.cuda.is_available()`). Use Google Colab if any machine lacks a GPU."*

---

## Who Does What

| Member | Module | Must Complete Before |
|--------|--------|----------------------|
| Rashmi | Module 1 — ResNet-50 + BiLSTM | Starting frame extraction script |
| Ravindu | Module 2 — Whisper + BERT | Running Whisper on sample lectures |
| Fazly | Module 3 — ViT + TrOCR | Running slide extraction script |
| Lathisana | Module 4 — MoviePy + FFmpeg + GPT-4o | Test clip cutting & TTS test |

**All four members follow Sections 1–5 identically. Then each follows their module-specific Section 7 additions.**

---

## Section 1: Check GPU First

Before installing anything, check whether your machine has an NVIDIA GPU.

### Windows
```powershell
nvidia-smi
```

**If `nvidia-smi` shows a GPU table → your machine has a compatible GPU. Continue to Section 2.**

**If `nvidia-smi` is not found → go to Section 6 (Google Colab Fallback) immediately.**

---

## Section 2: Install Python 3.10

> [!IMPORTANT]
> The project requires exactly **Python 3.10.x**. Python 3.11+ has incompatibilities with `openai-whisper`. Do not skip this version check.

### Option A — Direct Python Installer (Simplest)

1. Download **Python 3.10.14** from: https://www.python.org/downloads/release/python-31014/
   - Windows: choose "Windows installer (64-bit)"
2. Run the installer. **Check "Add python.exe to PATH"** before clicking Install.
3. Verify:
```powershell
python --version
# Expected output: Python 3.10.14
```

### Option B — Miniconda (Recommended)

1. Download Miniconda from: https://docs.conda.io/en/latest/miniconda.html
2. Install, then:
```bash
conda create -n integra python=3.10 -y
conda activate integra
python --version   # Python 3.10.x
```

> Always activate the environment before working: `conda activate integra`

---

## Section 3: Install PyTorch with CUDA

> [!IMPORTANT]
> **Do NOT run `pip install torch` alone** — that installs the CPU-only build. Use the CUDA-specific wheel URL below.

### Step 3a: Check your CUDA version

```powershell
nvidia-smi
# Look for "CUDA Version: XX.X" in the top-right corner
```

### Step 3b: Install PyTorch matching your CUDA version

**CUDA 12.1 (RTX 30xx, 40xx series — most modern machines):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 11.8 (GTX 10xx, 16xx, RTX 20xx — older machines):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Not sure?** Use CUDA 11.8 — it works on all NVIDIA GPUs from 2016 onwards.

### Step 3c: Verify GPU detection (Critical Project Plan Check)

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.version.cuda)"
```

**Expected output (GPU machine):**
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3060
PyTorch: 2.x.x+cu121 | CUDA: 12.1
```

If CUDA is `False` after reinstalling → go to Section 6 (Colab).

---

## Section 4: Install OpenCV

```bash
pip install opencv-python>=4.8.0
```

**Verify:**
```bash
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

---

## Section 5: Install All Project Dependencies + FFmpeg

### 5a — Python packages

From the repository root (where `requirements.txt` is):

```bash
pip install -r requirements.txt
```

This takes 5–15 minutes. It installs Transformers, Whisper, MoviePy, Pillow, scikit-learn, rouge-score, jiwer, yt-dlp, and everything else.

### 5b — FFmpeg (system binary — NOT installed by pip)

**Windows Option A — Winget (Recommended):**
```powershell
winget install --id=Gyan.FFmpeg -e
```

**Windows Option B — Chocolatey:**
```powershell
choco install ffmpeg
```

**Windows Option C — Manual:**
1. Download from https://www.gyan.dev/ffmpeg/builds/ → "ffmpeg-release-essentials.zip"
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to System PATH (System Properties → Environment Variables → Path → New)
4. Restart terminal

**Verify:**
```bash
ffmpeg -version
```

---

## Section 5c: Run the Full Verification Script

```bash
python scripts/verify_env.py
```

This runs all checks in one pass with clear PASS/FAIL output. **Every team member must run this and confirm no FAIL lines before the Week 1 group call.**

---

## Section 6: Google Colab Fallback (No GPU Machine)

> [!WARNING]
> If your machine has no NVIDIA GPU, use Google Colab immediately — do not try to install CUDA on non-NVIDIA hardware.

### Setting Up Colab

1. Go to https://colab.research.google.com
2. Click **Runtime → Change runtime type → T4 GPU** → Save
3. In a new notebook, run:

```python
# Cell 1 — Clone repo and switch to your branch
!git clone https://github.com/rashmiJayawardhana/lecture-video-summarizer.git
%cd lecture-video-summarizer
!git checkout module-X   # Replace X with your module number (1, 2, 3, or 4)

# Cell 2 — Verify GPU
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# Cell 3 — Install dependencies
!pip install -r requirements.txt --quiet

# Cell 4 — Verify FFmpeg (pre-installed on Colab)
!ffmpeg -version
```

### Saving Work Between Sessions

Colab sessions reset after disconnection. Always save to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Save checkpoints and outputs
import shutil
shutil.copy('outputs/module_output.json', '/content/drive/MyDrive/INTEGRA/outputs/')
```

Push code back to GitHub at the end of every session:
```bash
!git add . && git commit -m "Week 1 progress" && git push
```

---

## Section 7: Module-Specific Setup Tests

### Module 1 — Rashmi (ResNet-50 + BiLSTM)

```bash
python -c "
import torch
import torchvision.models as models
model = models.resnet50(weights='IMAGENET1K_V1')
model.eval()
print('ResNet-50 loaded:', sum(p.numel() for p in model.parameters()), 'parameters')
# Expected: ResNet-50 loaded: 25557032 parameters
"
```

### Module 2 — Ravindu (Whisper + BERT)

```bash
# Load Whisper tiny (fast test, no GPU needed)
python -c "
import whisper
model = whisper.load_model('tiny')
print('Whisper tiny loaded OK')
"

# Pre-download large-v3 weights (~3GB, do this once)
python -c "import whisper; whisper.load_model('large-v3'); print('Whisper large-v3 cached')"
```

> NOTE: Whisper large-v3 download takes 5–10 minutes. It is cached in `~/.cache/whisper/`.

### Module 3 — Fazly (ViT + TrOCR + OpenCV)

```bash
# Test ViT-base loading
python -c "
from transformers import ViTForImageClassification
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
print('ViT-base loaded OK')
"

# Test TrOCR loading
python -c "
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
print('TrOCR loaded OK')
"
```

### Module 4 — Lathisana (MoviePy + edge-tts + GPT-4o)

```bash
# Test MoviePy
python -c "from moviepy.editor import VideoFileClip; print('MoviePy OK')"

# Test edge-tts AriaNeural voice
python -m edge_tts --voice en-US-AriaNeural --text "INTEGRA environment test." --write-media test_tts.mp3
# Play test_tts.mp3 to verify, then delete it

# Set up OpenAI API key in .env:
copy .env.example .env
# Edit .env: OPENAI_API_KEY=sk-your-key-here

# Test API key
python -c "
from openai import OpenAI
client = OpenAI()
models = client.models.list()
print('OpenAI API key valid')
"
```

---

## Section 8: Environment Variables

```bash
copy .env.example .env
```

Key fields in `.env`:

```env
# GPU
CUDA_VISIBLE_DEVICES=0        # Use first GPU; set to "" for CPU-only mode

# Module 4 (Lathisana only initially)
OPENAI_API_KEY=sk-your-key-here

# Paths
DATA_DIR=./data
OUTPUT_DIR=./outputs
MODELS_DIR=./models
```

---

## Section 9: Troubleshooting

| Symptom | Fix |
|---------|-----|
| `torch.cuda.is_available()` → `False` | Reinstall with `--index-url` (Section 3b). Check NVIDIA driver with `nvidia-smi` |
| `ImportError: No module named 'whisper'` | Run `pip install openai-whisper` (NOT `pip install whisper`) |
| FFmpeg not found by MoviePy | Run `winget install --id=Gyan.FFmpeg -e` and restart terminal |
| CUDA out of memory during training | Reduce `batch_size` in the relevant `configs/moduleX_config.yaml` to 4 or 2 |
| Whisper large-v3 OOM | Use `whisper.load_model("medium")` during development |
| Python version wrong | Recreate env: `conda create -n integra python=3.10 -y` |

---

## Week 1 Sign-Off Checklist

Post this completed checklist in the group chat or GitHub Discussions:

```
Member name:
Module:

[ ] Python 3.10.x installed and verified
[ ] PyTorch installed with CUDA (or Colab T4 confirmed)
[ ] torch.cuda.is_available() returns True (or Colab T4 confirmed)
[ ] OpenCV installed and cv2.__version__ printed
[ ] All requirements.txt packages installed (no errors)
[ ] FFmpeg installed and ffmpeg -version works
[ ] scripts/verify_env.py runs with NO FAIL lines
[ ] .env file created from .env.example
[ ] Module-specific test (Section 7) passes
[ ] Correct GitHub branch checked out (module-1/2/3/4)
```

---

