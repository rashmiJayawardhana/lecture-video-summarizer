# 🗺️ INTEGRA Codebase Guide — For Beginners

> Open this file in VS Code preview (Ctrl+Shift+V) for best reading experience.

---

## How the System Works (Big Picture)

```
┌──────────────────────────────────────────────────────────────┐
│                 INPUT: 60-minute Lecture Video                │
└──────────────────────┬───────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
   ┌─────────────┐ ┌──────────┐ ┌──────────────┐
   │  MODULE 1   │ │ MODULE 2 │ │  MODULE 3    │
   │  Rashmi     │ │ Ravindu  │ │  Fazly       │
   │             │ │          │ │              │
   │  "Is this   │ │ "Is what │ │ "Is this     │
   │   frame     │ │  they're │ │  slide       │
   │   visually  │ │  saying  │ │  showing     │
   │   important │ │  import- │ │  critical,   │
   │   ?"        │ │  ant?"   │ │  important,  │
   │             │ │          │ │  or skip?"   │
   │ score_V     │ │ score_T  │ │ label+text   │
   └──────┬──────┘ └────┬─────┘ └──────┬───────┘
          │              │              │
          │    JSON      │    JSON      │    JSON
          │    files     │    files     │    files
          ▼              ▼              ▼
   ┌───────────────────────────────────────────┐
   │              MODULE 4 — Lathisana         │
   │                                           │
   │  Step 1: FUSE scores                      │
   │    S = 0.33×V + 0.33×T + 0.34×L          │
   │                                           │
   │  Step 2: SELECT top segments              │
   │    (keep S ≥ 0.3, fill 10 minutes)        │
   │                                           │
   │  Step 3: BUILD video                      │
   │    Pipeline A → highlight.mp4             │
   │    Pipeline B → slideshow.mp4             │
   └───────────────────┬───────────────────────┘
                       │
                       ▼
   ┌───────────────────────────────────────────┐
   │        OUTPUT: 10-minute Summary Video    │
   └───────────────────────────────────────────┘
```

---

## Your Files — What to Read First

### If you are Rashmi (Module 1):
1. Read `src/module1_importance/model.py` — has BEGINNER comments
2. Read `configs/module1_config.yaml` — your settings
3. Read `src/module1_importance/feature_extractor.py` — how frames become numbers
4. **Write next**: `train.py` and `inference.py`

### If you are Ravindu (Module 2):
1. Read `src/module2_summarization/transcriber.py` — how Whisper works
2. Read `src/module2_summarization/model.py` — how BERT classifies sentences
3. Read `configs/module2_config.yaml` — your settings
4. **Write next**: `train.py` and `inference.py`

### If you are Fazly (Module 3):
1. Read `src/module3_visual/slide_extractor.py` — how slides are detected
2. Read `src/module3_visual/model.py` — how ViT classifies slides
3. Read `src/module3_visual/ocr.py` — how TrOCR reads text from slides
4. **Write next**: `train.py` and `inference.py`

### If you are Lathisana (Module 4):
1. Read `src/module4_synthesis/fusion.py` — has BEGINNER comments + demo
2. Read `src/module4_synthesis/pipeline_a.py` — highlight video
3. Read `src/module4_synthesis/pipeline_b.py` — AI slideshow
4. **Run first**: `python src/module4_synthesis/fusion.py` to see the demo

### Everyone should read:
- `src/utils/json_schema.py` — the JSON format contract between modules
- `README.md` — project overview
- This file! (`docs/CODEBASE_GUIDE.md`)

---

## Key Concepts in 1 Sentence Each

| Concept | One-Sentence Explanation |
|---------|------------------------|
| **Pre-trained model** | A model someone else already trained on millions of examples; we download it and adjust it for our task |
| **Fine-tuning** | Taking a pre-trained model and training it a little more on our specific data |
| **ResNet-50** | A 50-layer image recognition model that converts any image into 2048 numbers |
| **BiLSTM** | A memory network that reads a sequence forward AND backward to understand patterns over time |
| **BERT** | A text understanding model that reads sentences and understands their meaning |
| **ViT** | Vision Transformer — like BERT but for images instead of text |
| **TrOCR** | A model that reads text from images (like reading a slide's content) |
| **Whisper** | OpenAI's speech-to-text model — listens to audio and types out every word |
| **GPT-4o** | OpenAI's language model — we use it to write narration scripts for slides |
| **edge-tts** | Microsoft's text-to-speech — converts narration scripts into spoken audio |
| **MoviePy** | Python library for cutting and stitching video clips |
| **FFmpeg** | The engine that encodes the final MP4 video file |
| **JSON schema** | The agreed format for data files passed between modules |
| **Fusion** | Combining scores from 3 modules into 1 final score per segment |
| **Config file** | YAML file with settings — change numbers here, NOT in code |

---

## Day 1 Commands (Copy-Paste Ready)

```bash
# 1. Clone
git clone https://github.com/rashmiJayawardhana/lecture-video-summarizer.git
cd lecture-video-summarizer

# 2. Create your branch (pick YOUR module number)
git checkout -b module-1

# 3. Create environment
python -m venv .venv
.venv\Scripts\activate

# 4. Install PyTorch (with GPU support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 5. Install everything else
pip install -r requirements.txt

# 6. Check GPU works
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# 7. Run the JSON schema test
python src/utils/json_schema.py

# 8. Run the fusion demo (Module 4)
python src/module4_synthesis/fusion.py

# 9. Run Module 1 model test
python src/module1_importance/model.py
```

---

## How Modules Talk to Each Other

```
Module 1 saves:  outputs/module_outputs/module1_output.json
Module 2 saves:  outputs/module_outputs/module2_output.json
Module 3 saves:  outputs/module_outputs/module3_output.json
                         │
                         ▼
Module 4 reads ALL THREE files and produces:
                 outputs/summaries/highlight.mp4
                 outputs/summaries/slideshow.mp4
```

**Rule**: Modules NEVER import each other's code. They only share JSON files.

---

## Git Workflow (Simple Version)

```bash
# 1. Make sure you're on YOUR branch
git checkout module-1

# 2. Write code, test it

# 3. Save your changes
git add .
git commit -m "[Module1] Add training script"

# 4. Push to GitHub
git push origin module-1

# 5. Every Friday: create a Pull Request on GitHub to merge into main
```
