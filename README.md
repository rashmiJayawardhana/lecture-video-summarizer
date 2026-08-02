# INTEGRA - Automated Lecture Video Summarization with Condensed Video Output

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Transform 60-minute IT theory lectures into coherent condensed videos using deep learning - with zero manual steps.

## 🎯 Project Overview

This research project develops a four-module deep learning pipeline that automatically produces a semantically meaningful condensed video from a full-length IT theory lecture.

- **Input**: an uploaded lecture video (MP4)
- **Output**: a single, fully-narrated condensed MP4 summary video. 
Module 4 aligns the visual, speech, and slide analysis by timestamp, generates a structured per-slide summary with a locally fine-tuned BART-base model, verifies that summary against the source data with a cloud-hosted model, and renders the verified content into the final video with Pillow + MoviePy + edge-tts + FFmpeg. No raw lecture footage is used in the output; every slide is included, narrated end to end.

## 🏗️ Architecture

The system consists of 4 modules with a weighted score fusion feeding a generation-and-verification synthesis pipeline:

```
Input Video
    │
    ├──→ Module 1: Keyframe Detection & Importance Scoring ──→ score_V
    │         (frozen ResNet-50 + 2-layer BiLSTM + classifier head)
    │
    ├──→ Module 2: Speech-to-Text & Content Summarization ──→ importance_ratio_T
    │         (Whisper large-v3 + fine-tuned BERT-base + 4 novel features)
    │
    ├──→ Module 3: Visual Content Understanding & Slide Extraction ──→ label (→ score_L)
    │         (fine-tuned ViT-base + EasyOCR, Gemini for Critical frames)
    │
    └──→ Module 4: Video Synthesis & Intelligent Editing
              │
              ├──→ Source Alignment: joins slide OCR data, importance scores, and audio
              │         transcript by timestamp window (align_source.py)
              │
              ├──→ Content Generation: locally fine-tuned BART-base (domain-adapted on
              │         QMSum) generates each slide's summary, on CPU
              │
              ├──→ Content Verification: a cloud-hosted model (Ollama Cloud) checks the
              │         generated content against the aligned source and corrects false
              │         attribution/omissions before rendering
              │
              └──→ Video Rendering: Pillow (slide images) + edge-tts (narration) +
                        MoviePy/FFmpeg (assembly) → Condensed Video (narrated MP4)
```

### Module 1: Keyframe Detection & Importance Scoring
- **Owner**: Jayawardhana G.G.R.M. (214093E)
- **Model**: frozen ResNet-50 feature extractor + 2-layer BiLSTM
- **Task**: Score 10-second video segments by instructional importance
- **Output**: `{ segment_id, timestamp_start, timestamp_end, score_V }`
- **Status**: Trained and evaluated across 5 runs on 61 manually-annotated lecture videos (23,402 segments). Best run: F1 = 0.870 at the ≥5 threshold. Integrated into the real backend and verified with live inference.

### Module 2: Speech-to-Text & Content Summarization
- **Owner**: Jayaweera B.R.D. (214095L)
- **Model**: OpenAI Whisper (large-v3) + fine-tuned BERT-base, enhanced with 4 novel features: IT-keyword boost (+0.10), definition-pattern boost (+0.15), repetition scoring (+0.05), and confidence-weighted output
- **Task**: Transcribe speech and classify sentences as important/not important
- **Output**: `{ sentence, timestamp_start, timestamp_end, is_important, confidence, keyword_boost, definition_match, repetition_boost, importance_ratio_T }`
- **Status**: BERT F1 = 0.83 (exceeds the 0.70 target; 80% precision at the standard threshold; 4 of 6 thresholds pass F1 > 0.70) on 5,050 manually-labelled sentences across 20 IT lecture videos, an 80/20 train/test split. Integrated into the real backend and verified with live inference (requires `ffmpeg` on PATH for Whisper's audio decoding).

### Module 3: Visual Content Understanding & Slide Extraction
- **Owner**: Ahamed M.F.F. (214008C)
- **Model**: fine-tuned ViT-base + EasyOCR (adopted in place of the originally-proposed TrOCR for faster CPU inference and simpler setup), with Gemini used only for deeper semantic analysis of Critical frames
- **Task**: Classify slide frames (Critical/Important/Skip) and extract text
- **Output**: `{ frame_time, label, original_label, label_score, confidence, threshold_applied, ocr_text, analysis_source }`
- **Status**: 74.87% test accuracy on 5,009 annotated frames from 40 lecture videos. Integrated into the real backend and verified with live inference.

### Module 4: Video Synthesis & Intelligent Editing
- **Owner**: Lathisana T. (214116F)
- **Pipeline**: Source Alignment (`align_source.py` joins Module 1/2/3 outputs by timestamp window) → Content Generation (a locally fine-tuned `facebook/bart-base` checkpoint, domain-adapted on QMSum, generates each slide's title, key concepts, code examples, and voiceover script; inference runs on CPU) → Content Verification (Ollama Cloud checks the generated content against the aligned source data and corrects false attribution or omissions, used strictly for verification, never generation) → Video Rendering (Pillow renders slide images, edge-tts synthesizes narration with the en-US-AriaNeural voice, MoviePy/FFmpeg assemble and encode the final MP4).
- **Status**: Stage 1 BART fine-tuning done; content verification done; full end-to-end run executes on real lecture data with zero schema-validation errors. Formal human evaluation of output quality is planned.
- **Output**: a single narrated `summarized_video.mp4` (every slide included, no raw footage) + `module4_final_output.json`

## 📋 Shared JSON Schema

Defined in `src/utils/json_schema.py`, which also provides validators for each module's output.

```json
// Module 1 Output
{
  "segment_id": "seg_001",
  "timestamp_start": 12.0,
  "timestamp_end": 22.0,
  "score_V": 0.82
}

// Module 2 Output
{
  "sentence": "A binary search tree is a data structure...",
  "timestamp_start": 12.0,
  "timestamp_end": 15.5,
  "is_important": true,
  "importance_ratio_T": 0.75
}

// Module 3 Output
{
  "frame_time": 14.5,
  "label": "Critical",
  "ocr_text": "Binary Search Tree: O(log n) average case"
}
```

**Fusion formula**: `S = w1·V + w2·T + w3·L`, default weights `w1=0.33, w2=0.33, w3=0.34` (`src/module4_synthesis/fusion.py`), used to align and prioritise segments before Module 4's BART-based content generation.

## 📁 Repository Structure

```
lecture-video-summarizer/
├── README.md
├── LICENSE
├── requirements.txt
├── .env.example                       # Environment variables template
│
├── frontend/                          # Web interface (Vanilla JS + Vite, NOT React)
│   ├── index.html
│   ├── js/main.js                     # Real fetch()-based upload/status-poll/result display
│   ├── package.json
│   └── vite.config.js                 # Dev proxy: /api -> http://localhost:8000
│
├── src/
│   ├── module1_importance/            # Keyframe Detection
│   │   ├── model.py                   # VideoImportanceScorer (BiLSTM)
│   │   ├── feature_extractor.py       # ResNet-50 feature extraction
│   │   ├── train.py                   # python -m src.module1_importance.train
│   │   └── inference.py
│   │
│   ├── module2_summarization/         # Content Summarization
│   │   ├── transcribe_video.py        # Whisper transcription
│   │   ├── classify_with_features.py  # BERT + 4 novel features
│   │   └── bert_model_v2/             # Fine-tuned BERT checkpoint (gitignored)
│   │
│   ├── module3_visual/                # Visual Understanding
│   │   ├── slide_extractor.py         # OpenCV slide extraction
│   │   ├── inference.py               # ViT inference
│   │   ├── ocr.py                     # EasyOCR text extraction
│   │   └── train.py                   # --data_dir/--output_dir/--model_name/--epochs
│   │
│   ├── module4_synthesis/             # Video Synthesis
│   │   ├── fusion.py                  # ScoreFusion (S = w1.V + w2.T + w3.L)
│   │   ├── align_source.py            # Timestamp window join across Module 1/2/3 outputs
│   │   ├── pipeline_b.py / slideshow_video.py  # BART content generation + Pillow/MoviePy/edge-tts rendering
│   │   ├── ollama_cloud.py            # Ollama Cloud source-grounded content verification
│   │   ├── schema_repair.py           # Verified-JSON schema repair before rendering
│   │   └── pipeline_a.py              # Real-footage highlight-reel generator (earlier design)
│   │
│   ├── backend/                       # Real FastAPI backend
│   │   ├── main.py                    # FastAPI app entrypoint
│   │   ├── api/                       # routes_upload.py, routes_jobs.py, routes_results.py
│   │   ├── services/                  # pipeline_orchestrator.py, storage_service.py, job_service.py
│   │   ├── workers/                   # module_runners.py - calls each module's real code
│   │   └── core/                      # config.py, paths.py, supabase_client.py
│   │
│   ├── evaluation/metrics.py          # ROUGE-L, Precision/Recall, WER, F1
│   └── utils/json_schema.py           # Inter-module JSON schema & validators
│
├── data/                              # gitignored: raw videos, processed features, annotations
├── models/module3/vit_slide_classifier/  # ViT checkpoint (gitignored)
├── best_module1_model.pt              # Module 1 checkpoint (gitignored, project root)
│
├── storage/jobs/{job_id}/             # Per-upload runtime data (gitignored)
│   ├── input/                         # Uploaded video
│   ├── temp/                          # Intermediate per-module files
│   └── outputs/                       # module{1,2,3}_output.json, module4_final_output.json,
│                                       #   summarized_video.mp4
│
├── scripts/                           # Annotation tools, evaluation scripts, Module 4 export
├── notebooks/                         # Colab training/setup notebooks
├── docs/                              # SETUP_GUIDE.md, COLAB_DEMO_GUIDE.md, CODEBASE_GUIDE.md
└── tests/
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13, PyTorch 2.11 (CUDA build recommended)
- **FFmpeg installed and on PATH** (required by both Module 2's Whisper audio decoding and Module 4's video rendering; verify with `ffmpeg -version`)
- Node.js 20+ (frontend)
- A Supabase project (free tier) for job status tracking
- A Gemini API key (free tier) for Module 3's Critical-frame analysis (Module 3 still runs without it via its OCR fallback, at lower quality)
- An Ollama Cloud account/API access for Module 4's content verification step

### Backend Installation

```bash
git clone https://github.com/rashmiJayawardhana/lecture-video-summarizer.git
cd lecture-video-summarizer

python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt

copy .env.example .env
# Edit .env: SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY

python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Place the trained checkpoints (not in git, distributed separately; see the team's shared storage):
- `best_module1_model.pt` at the project root
- `src/module2_summarization/bert_model_v2/model.safetensors`
- `models/module3/vit_slide_classifier/model.safetensors`
- Module 4's fine-tuned BART-base checkpoint (domain-adapted on QMSum; see `notebooks/module4_stage1_qmsum_finetune.ipynb`)

### Run the backend

```bash
python -m uvicorn src.backend.main:app --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev   # http://localhost:3000, proxies /api to http://localhost:8000
```

Upload a video through the UI, or directly via the API:

```bash
curl -X POST http://localhost:8000/api/upload -F "file=@lecture.mp4"
# {"job_id": "job_xxxx", ...}
curl http://localhost:8000/api/jobs/job_xxxx/status
curl http://localhost:8000/api/jobs/job_xxxx/download-video -o summarized_video.mp4
```

For running the backend on Google Colab GPU (for faster Whisper/ResNet/ViT inference during a demo) and connecting either a local or Vercel-deployed frontend to it, see [docs/COLAB_DEMO_GUIDE.md](docs/COLAB_DEMO_GUIDE.md).

## 📊 Dataset

| Module | Data Type | Total | Notes |
|--------|-----------|-------|-------|
| Module 1 | Segment scores (0-10), 4-criterion rubric | 61 videos / 23,402 segments | Split: video ≤45 train, 46-50 val, >50 test |
| Module 2 | Sentence importance labels | 5,050 sentences / 20 videos | 80/20 train-test split |
| Module 3 | Slide labels (Critical/Important/Skip) | 5,009 frames / 40 videos | Split: 3,506 train / 751 val / 752 test |

## 🎓 Training

```bash
# Module 1: Importance Scoring (loads data/processed/features + module1_annotations.json)
python -m src.module1_importance.train

# Module 3: Visual Understanding (ViT-base)
python src/module3_visual/train.py --data_dir data/datasets/module3 --output_dir models/module3/vit_slide_classifier --epochs 3
```

Module 2's BERT fine-tuning was run interactively via Google Colab notebook rather than a checked-in `train.py` script; see `notebooks/`.

Module 4's BART-base content-generation model is domain-adapted on the QMSum meeting-summarization dataset via `notebooks/module4_stage1_qmsum_finetune.ipynb`, then targeted toward the aligned lecture data to produce the structured per-slide summary schema Module 4 renders from.

## 📈 Evaluation Results

| Module | Metric | Result |
|--------|--------|--------|
| Module 1 | Segment F1 at ≥5 threshold | 0.870 |
| Module 2 | BERT F1 | 0.83 |
| Module 3 | ViT test accuracy | 74.87% |
| Module 4 | End-to-end run on real lecture data | Zero schema-validation errors; formal human evaluation planned |

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- Project Repository: [GitHub](https://github.com/rashmiJayawardhana/lecture-video-summarizer)
- Issues: [GitHub Issues](https://github.com/rashmiJayawardhana/lecture-video-summarizer/issues)
