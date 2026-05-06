# INTEGRA — System Architecture

This document explains the organization and architecture of the lecture-video-summarizer repository.

## High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Input: 60-min IT Lecture Video                     │
└───────────────┬─────────────────────┬─────────────────┬───────────────┘
                │                     │                 │
                ▼                     ▼                 ▼
┌───────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  Module 1         │  │  Module 2            │  │  Module 3            │
│  Keyframe Det.    │  │  Speech-to-Text &    │  │  Visual Content &    │
│  & Importance     │  │  Content Summary     │  │  Slide Extraction    │
│                   │  │                      │  │                      │
│  ResNet-50        │  │  Whisper (large-v3)  │  │  ViT-base            │
│  + BiLSTM         │  │  + BERT-base         │  │  + TrOCR             │
│                   │  │                      │  │                      │
│  Output: score_V  │  │  Output:             │  │  Output: label,      │
│  per 10s segment  │  │  importance_ratio_T  │  │  ocr_text per frame  │
└───────┬───────────┘  └──────────┬───────────┘  └──────────┬───────────┘
        │                        │                          │
        └────────────┬───────────┴──────────────┬───────────┘
                     │                          │
                     ▼                          │
        ┌────────────────────────┐              │
        │  Score Fusion          │              │
        │  S = w1·V + w2·T      │◄─────────────┘
        │      + w3·L           │
        │  Threshold: S >= 0.3  │
        └───────┬────────┬──────┘
                │        │
                ▼        ▼
    ┌───────────────┐  ┌───────────────────┐
    │  Pipeline A   │  │  Pipeline B       │
    │  Highlight    │  │  Synthetic        │
    │  Video        │  │  Slideshow        │
    │               │  │                   │
    │  MoviePy +    │  │  GPT-4o +         │
    │  FFmpeg       │  │  edge-tts +       │
    │               │  │  Pillow           │
    └───────┬───────┘  └─────────┬─────────┘
            │                    │
            ▼                    ▼
    ┌───────────────┐  ┌───────────────────┐
    │ highlight.mp4 │  │ slideshow.mp4     │
    │ (≤ 10 min)    │  │ (≤ 10 min)        │
    └───────────────┘  └───────────────────┘
```

## Module Details

### Module 1: Keyframe Detection & Importance Scoring
- **Owner**: Rashmi (214093E)
- **Architecture**: ResNet-50 (frozen early layers) → BiLSTM (2 layers, 512 hidden) → Linear classifier
- **Input**: 30-frame sequences extracted at 1fps from 10-second segments
- **Output**: Importance score (0–1) per segment → `score_V`
- **Training**: 50 annotated IT lecture videos (35 train / 5 val / 10 test)
- **Key Files**:
  - `src/module1_importance/model.py` — `VideoImportanceScorer` class
  - `src/module1_importance/feature_extractor.py` — ResNet-50 feature extraction
  - `configs/module1_config.yaml` — Hyperparameters

### Module 2: Speech-to-Text & Content Summarization
- **Owner**: Ravindu (214095L)
- **Architecture**: Whisper large-v3 (ASR) → BERT-base fine-tuned (binary classifier)
- **Classification Criteria**: Concept introduction, term definition, key explanation
- **Output**: Per sentence → `{ sentence, is_important, importance_ratio_T }`
- **Training**: 50 labeled lecture transcripts (35 train / 5 val / 10 test)
- **Key Files**:
  - `src/module2_summarization/model.py` — `LectureSentenceClassifier` class
  - `src/module2_summarization/transcriber.py` — Whisper integration
  - `configs/module2_config.yaml` — Hyperparameters

### Module 3: Visual Content Understanding & Slide Extraction
- **Owner**: Fazly (214008C)
- **Architecture**: ViT-base-patch16-224 (3-class classifier) + TrOCR (text extraction)
- **Labels**: Critical (diagrams, formulas) / Important (definitions, examples) / Skip (title, blank)
- **Output**: Per frame → `{ frame_time, label, ocr_text }`
- **Training**: 800–1000 annotated slide images (600 train / 100 val / 100–300 test)
- **Fallback**: ResNet-50 + linear head if ViT accuracy < 70%
- **Key Files**:
  - `src/module3_visual/model.py` — `SlideImportanceClassifier` + fallback
  - `src/module3_visual/slide_extractor.py` — OpenCV SSIM-based extraction
  - `src/module3_visual/ocr.py` — TrOCR integration (NOT Tesseract)
  - `configs/module3_config.yaml` — Hyperparameters

### Module 4: Video Synthesis & Integration
- **Owner**: Lathisana (214116F)
- **Type**: Engineering module — **NO trainable DL model**
- **Fusion**: `S = w1·V + w2·T + w3·L` with `min_threshold = 0.3`
- **Pipeline A**: highlight.mp4 — Real clips + transitions + subtitles + chapter banners
- **Pipeline B**: slideshow.mp4 — AI slides + GPT-4o narration + AriaNeural TTS
- **Key Files**:
  - `src/module4_synthesis/fusion.py` — Weighted score fusion & segment selection
  - `src/module4_synthesis/pipeline_a.py` — Highlight video generator
  - `src/module4_synthesis/pipeline_b.py` — Synthetic slideshow generator
  - `configs/module4_config.yaml` — Pipeline A & B settings

## Shared JSON Schema

**Agreed Week 1. Frozen after Week 5.**

Defined and validated in `src/utils/json_schema.py`.

```json
// Module 1 → Module 4
{ "segment_id": "seg_001", "timestamp_start": 12.0, "timestamp_end": 22.0, "score_V": 0.82 }

// Module 2 → Module 4
{ "sentence": "...", "timestamp_start": 12.0, "timestamp_end": 15.5, "is_important": true, "importance_ratio_T": 0.75 }

// Module 3 → Module 4
{ "frame_time": 14.5, "label": "Critical", "ocr_text": "..." }
```

## Directory Structure

```
lecture-video-summarizer/
├── src/                          # Source code
│   ├── module1_importance/       # Rashmi — ResNet-50 + BiLSTM
│   ├── module2_summarization/    # Ravindu — Whisper + BERT
│   ├── module3_visual/           # Fazly — ViT + TrOCR
│   ├── module4_synthesis/        # Lathisana — Fusion + Pipelines A/B
│   ├── pipeline/                 # End-to-end orchestration
│   ├── evaluation/               # Metrics (ROUGE-L, F1, WER, etc.)
│   ├── data/                     # Data loading utilities
│   └── utils/                    # JSON schema, logging, config
│
├── configs/                      # YAML configuration files
├── data/                         # Raw videos, annotations, datasets
├── models/                       # Trained checkpoints (Modules 1-3 only)
├── scripts/                      # Download, training, pipeline scripts
├── notebooks/                    # Jupyter experiments
├── tools/                        # Annotation interfaces
├── tests/                        # Unit tests
├── outputs/                      # Generated videos, logs, results
├── docs/                         # Documentation
└── research/                     # Literature, meeting notes
```

## Git Workflow

### Branches
- `main` — Stable, tested code
- `module-1` — Rashmi's keyframe detection work
- `module-2` — Ravindu's transcription & classification work
- `module-3` — Fazly's slide extraction & classification work
- `module-4` — Lathisana's video synthesis work

### Commit Messages
```
[Module{N}] Brief description

Detailed explanation if needed.
```

Example:
```
[Module1] Add BiLSTM temporal modeling

Implemented bidirectional LSTM on top of ResNet-50 features.
Trained with 30-frame sequences on 35 annotated IT lecture videos.
```

### Code Review Process
1. Create feature branch from your module branch
2. Make changes and test locally
3. Push and create a Pull Request
4. At least one team member reviews before merging
5. Weekly integration meeting merges module branches into `main`

## Technology Stack

| Module | Technologies | Literature Reference |
|--------|-------------|---------------------|
| Module 1 | ResNet-50, BiLSTM, OpenCV, PyTorch | Rahman et al. 2020, Zhang et al. 2016 |
| Module 2 | Whisper (large-v3), BERT-base, Hugging Face Trainer, jiwer | Radford et al. 2023, Gonzalez et al. 2023 |
| Module 3 | ViT-base, TrOCR, OpenCV, Hugging Face | Biswas et al. 2025, Li et al. 2023 |
| Module 4 | MoviePy, FFmpeg, Pillow, GPT-4o, edge-tts, scikit-learn | Gonzalez et al. 2023, Ahmed et al. 2025 |
| Evaluation | rouge-score, jiwer, scikit-learn | Benedetto et al. 2024, Kaur & Ragha 2024 |

## Evaluation Targets

| Metric | Target | Module |
|--------|--------|--------|
| ROUGE-L Score | > 0.40 | All |
| Segment Precision | > 0.75 | Module 1 + 4 |
| Segment Recall | > 0.75 | Module 1 + 4 |
| ViT Classification Accuracy | > 75% | Module 3 |
| BERT F1 Score | > 0.70 | Module 2 |
| ASR Word Error Rate | < 15% | Module 2 |
| Knowledge Retention | > 70% | User Study |
| Video Duration | ≤ 600s | Module 4 |

## Quick Reference

### Run Training
```bash
python src/module1_importance/train.py --config configs/module1_config.yaml
python src/module2_summarization/train.py --config configs/module2_config.yaml
python src/module3_visual/train.py --config configs/module3_config.yaml
```

### Run Full Pipeline
```bash
python scripts/run_pipeline.py --input data/raw/lecture_001.mp4 --output outputs/summaries/
```

### Run Tests
```bash
pytest tests/ -v
```

### Validate JSON Schema
```bash
python src/utils/json_schema.py
```
