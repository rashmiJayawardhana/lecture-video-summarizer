# INTEGRA — Automated Lecture Video Summarization with Condensed Video Output

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Transform 60-minute IT theory lectures into coherent 10-minute condensed videos using deep learning — with zero manual steps.

## 🎯 Project Overview

This research project develops a four-module deep learning pipeline that automatically produces a duration-controlled, semantically meaningful condensed video from a full-length IT theory lecture. The system generates two output formats:

- **Pipeline A** — Real lecture highlight reel with original audio, transitions, and subtitles
- **Pipeline B** — Fully synthetic narrated slideshow using AI-rendered slides and TTS voiceover

**Input**: 60-minute IT theory lecture video  
**Output**: 10-minute condensed MP4 video

## 🏗️ Architecture

The system consists of 4 modules with a weighted score fusion:

```
Input Video (60 min)
    │
    ├──→ Module 1: Keyframe Detection & Importance Scoring ──→ score_V
    │         (ResNet-50 + BiLSTM)
    │
    ├──→ Module 2: Speech-to-Text & Content Summarization ──→ importance_ratio_T
    │         (Whisper large-v3 + Fine-tuned BERT-base)
    │
    ├──→ Module 3: Visual Content Understanding & Slide Extraction ──→ label (→ score_L)
    │         (ViT-base + TrOCR)
    │
    └──→ Module 4: Video Synthesis & Integration
              │
              ├──→ Score Fusion: S = w1·V + w2·T + w3·L
              │
              ├──→ Pipeline A: highlight.mp4 (MoviePy + FFmpeg)
              └──→ Pipeline B: slideshow.mp4 (GPT-4o + edge-tts + Pillow)
```

### Module 1: Keyframe Detection & Importance Scoring
- **Owner**: Jayawardhana G.G.R.M. (214093E)
- **Model**: ResNet-50 + BiLSTM
- **Task**: Score 10-second video segments by instructional importance
- **Output**: `{ segment_id, timestamp_start, timestamp_end, score_V }`
- **References**: Rahman et al. 2020, Zhang et al. 2016, Lin et al. 2022

### Module 2: Speech-to-Text & Content Summarization
- **Owner**: Jayaweera B.R.D. (214095L)
- **Model**: OpenAI Whisper (large-v3) + Fine-tuned BERT-base
- **Task**: Transcribe speech and classify sentences as important/not important
- **Output**: `{ sentence, timestamp_start, timestamp_end, is_important, importance_ratio_T }`
- **References**: Radford et al. 2023, Gonzalez et al. 2023

### Module 3: Visual Content Understanding & Slide Extraction
- **Owner**: Ahamed M.F.F. (214008C)
- **Model**: ViT-base + TrOCR
- **Task**: Classify slide frames (Critical/Important/Skip) and extract text
- **Output**: `{ frame_time, label, ocr_text }`
- **References**: Biswas et al. 2025, Li et al. 2023

### Module 4: Video Synthesis & Integration
- **Owner**: Lathisana T. (214116F)
- **Technologies**: MoviePy, FFmpeg, GPT-4o, edge-tts (AriaNeural), Pillow
- **Task**: Fuse scores and produce two output videos (Pipeline A & B)
- **Output**: `highlight.mp4` and `slideshow.mp4`
- **References**: Gonzalez et al. 2023, Ahmed et al. 2025

## 📋 Shared JSON Schema

**Agreed Week 1 — Frozen after Week 5 integration meeting.**

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

**Fusion Formula**: `S = w1·V + w2·T + w3·L`  
Starting weights: `w1=0.33, w2=0.33, w3=0.34` (tuned via grid search)

## 📁 Repository Structure

```
lecture-video-summarizer/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── setup.py                           # Package installation
├── .gitignore                         # Git ignore rules
├── .env.example                       # Environment variables template
│
├── configs/                           # Configuration files
│   ├── pipeline_config.yaml           # Pipeline & fusion settings
│   ├── module1_config.yaml            # Module 1: ResNet-50 + BiLSTM
│   ├── module2_config.yaml            # Module 2: Whisper + BERT
│   ├── module3_config.yaml            # Module 3: ViT + TrOCR
│   └── module4_config.yaml            # Module 4: Pipeline A & B
│
├── frontend/                          # Web interface
│   ├── index.html                     # Main entry point
│   ├── css/                           # Stylesheets
│   ├── js/                            # JavaScript logic
│   ├── package.json                   # Frontend dependencies
│   └── vite.config.js                 # Vite configuration
│
├── src/                               # Source code
│   ├── module1_importance/            # Keyframe Detection 
│   │   ├── model.py                   # VideoImportanceScorer (ResNet-50 + BiLSTM)
│   │   ├── feature_extractor.py       # ResNet-50 feature extraction
│   │   ├── train.py                   # Training script
│   │   └── inference.py               # Inference script
│   │
│   ├── module2_summarization/         # Content Summarization 
│   │   ├── model.py                   # LectureSentenceClassifier (BERT-base)
│   │   ├── transcriber.py             # Whisper transcription
│   │   ├── train.py                   # Fine-tuning script
│   │   └── inference.py               # Inference script
│   │
│   ├── module3_visual/                # Visual Understanding 
│   │   ├── model.py                   # SlideImportanceClassifier (ViT-base)
│   │   ├── slide_extractor.py         # OpenCV slide detection
│   │   ├── ocr.py                     # TrOCR text extraction
│   │   ├── train.py                   # Fine-tuning script
│   │   └── inference.py               # Inference script
│   │
│   ├── module4_synthesis/             # Video Synthesis 
│   │   ├── fusion.py                  # Score fusion (S = w1·V + w2·T + w3·L)
│   │   ├── pipeline_a.py             # Highlight video generator
│   │   └── pipeline_b.py             # Synthetic slideshow generator
│   │
│   ├── pipeline/                      # Integration pipeline
│   │   ├── summarizer.py             # Main LectureVideoSummarizer
│   │   └── config.py                 # Pipeline configuration
│   │
│   ├── evaluation/                    # Evaluation metrics
│   │   ├── metrics.py                # ROUGE-L, Precision/Recall, WER, F1
│   │   └── user_study.py            # User study tools
│   │
│   ├── data/                          # Data processing utilities
│   │   ├── video_loader.py           # Video loading
│   │   └── preprocessing.py         # Data preprocessing
│   │
│   └── utils/                         # Shared utilities
│       ├── json_schema.py            # Inter-module JSON schema & validators
│       ├── logger.py                 # Logging configuration
│       └── config.py                 # Global configuration
│
├── data/                              # Data directory (gitignored)
│   ├── raw/                           # Original lecture videos
│   ├── processed/                     # Processed data (transcripts, slides)
│   ├── annotations/                   # Manual annotations
│   │   ├── module1/                   # Segment importance scores (0-10)
│   │   ├── module2/                   # Sentence importance labels
│   │   └── module3/                   # Slide labels (Critical/Important/Skip)
│   └── datasets/                      # Training datasets
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/                            # Trained models (gitignored)
│   ├── module1/                       # ResNet-50 + BiLSTM checkpoints
│   ├── module2/                       # BERT-base fine-tuned
│   └── module3/                       # ViT-base fine-tuned
│
├── scripts/                           # Utility scripts
│   ├── download_videos.py            # Download lecture videos
│   └── run_pipeline.py               # Run full pipeline
│
├── notebooks/                         # Jupyter notebooks for prototyping
├── tools/                             # Annotation tools
├── tests/                             # Unit tests
├── outputs/                           # Generated outputs (gitignored)
│   ├── summaries/                     # Output videos
│   ├── logs/                          # Training & pipeline logs
│   └── results/                       # Evaluation results
│
├── docs/                              # Documentation
│   ├── architecture.md               # System architecture
│   └── installation.md               # Setup instructions
│
└── research/                          # Research materials
    └── meeting_notes/                 # Team meeting notes
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (RTX 3060 12GB or better recommended)
- 32GB RAM
- FFmpeg installed and in PATH
- OpenAI API key (for Module 4 Pipeline B)

### Installation

```bash
# Clone the repository
git clone https://github.com/rashmiJayawardhana/lecture-video-summarizer.git
cd lecture-video-summarizer

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
copy .env.example .env
# Edit .env — add your OPENAI_API_KEY

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### Basic Usage

```python
from src.pipeline.summarizer import LectureVideoSummarizer

# Initialize summarizer
summarizer = LectureVideoSummarizer()

# Summarize a lecture video
summary_video, metadata = summarizer.summarize(
    video_path='data/raw/lecture_001.mp4',
    target_duration=600  # 10 minutes
)

print(f"Summary created: {metadata['compression_ratio']:.1f}x compression")
```

## 📊 Dataset

### Data Collection
- **Source**: MIT OpenCourseWare, YouTube, Coursera, AVLectures, LVVO dataset
- **Size**: 50 IT theory lecture videos (45–90 minutes each)
- **Content**: Algorithms, networking, databases, software engineering

### Annotation Requirements

| Module | Data Type | Total | Train | Val | Test |
|--------|-----------|-------|-------|-----|------|
| Module 1 | Segment scores (0-10) | 50 videos | 35 | 5 | 10 |
| Module 2 | Sentence labels | 50 transcripts | 35 | 5 | 10 |
| Module 3 | Slide labels (Critical/Important/Skip) | 800-1000 slides | 600 | 100 | 100-300 |

## 🎓 Training

### Train Individual Modules

```bash
# Module 1: Importance Scoring (ResNet-50 + BiLSTM)
python src/module1_importance/train.py --config configs/module1_config.yaml

# Module 2: Content Summarization (BERT-base)
python src/module2_summarization/train.py --config configs/module2_config.yaml

# Module 3: Visual Understanding (ViT-base)
python src/module3_visual/train.py --config configs/module3_config.yaml
```

## 📈 Evaluation Targets

| Metric | Target | Measured With |
|--------|--------|---------------|
| ROUGE-L Score | > 0.40 | `rouge-score` library |
| Segment Precision | > 0.75 | Custom (10 test videos) |
| Segment Recall | > 0.75 | Custom (10 test videos) |
| ViT Classification Accuracy | > 75% | 100-300 test slides |
| BERT F1 Score | > 0.70 | 10 test transcripts |
| ASR Word Error Rate | < 15% | `jiwer` library |
| Knowledge Retention (User Study) | > 70% | 20 IT undergraduates |
| Video Duration | ≤ 10 min | Enforced programmatically |

## 📚 References

1. Rahman et al. (2020) — Visual Summarization of Lecture Video Segments
2. Zhang et al. (2016) — Video Summarization with LSTM
3. Lin et al. (2022) — Lecture Video Keyframe Detection
4. Radford et al. (2023) — Robust Speech Recognition (Whisper)
5. Gonzalez et al. (2023) — Automatically Generated Summaries of Video Lectures
6. Biswas et al. (2025) — Visual Content Detection with Transfer Learning
7. Li et al. (2023) — TrOCR: Transformer-based OCR
8. Kaur & Ragha (2024) — Optimized Deep Learning Lecture Summarization
9. Ahmed et al. (2025) — Multimodal Video Summarization
10. Benedetto et al. (2024) — Abstractive Video Lecture Summarization

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MIT OpenCourseWare for lecture videos
- OpenAI Whisper for speech recognition
- Hugging Face for transformer models (BERT, ViT, TrOCR)
- MoviePy for video editing capabilities
- OpenAI GPT-4o for narration generation
- Microsoft edge-tts for neural speech synthesis

## Contact

- Project Repository: [GitHub](https://github.com/rashmiJayawardhana/lecture-video-summarizer)
- Issues: [GitHub Issues](https://github.com/rashmiJayawardhana/lecture-video-summarizer/issues)



