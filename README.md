# Automated Lecture Video Summarization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> Transform 60-minute lecture videos into concise 10-minute summaries using deep learning

## 🎯 Project Overview

This research project develops an end-to-end system that automatically summarizes lecture videos by:
- Detecting important visual moments and scene changes
- Extracting and summarizing speech content
- Identifying critical slides and diagrams
- Synthesizing a coherent condensed video with smooth transitions

**Input**: 60-minute lecture video  
**Output**: 10-minute summarized video containing key segments

## 🏗️ Architecture

The system consists of 4 trainable modules:

### Module 1: Keyframe Detection & Importance Scoring
- **Owner**: Member 1
- **Model**: ResNet50 + LSTM
- **Task**: Score video segments by importance using visual and temporal features
- **Output**: Timestamp ranges with importance scores

### Module 2: Speech-to-Text + Content Summarization
- **Owner**: Member 2
- **Model**: Whisper + Fine-tuned BERT
- **Task**: Transcribe speech and identify key sentences
- **Output**: Text summary aligned with timestamps

### Module 3: Visual Content Understanding & Slide Extraction
- **Owner**: Member 3
- **Model**: Vision Transformer (ViT)
- **Task**: Extract and classify slides by importance
- **Output**: Key visual frames with metadata

### Module 4: Video Synthesis & Intelligent Editing
- **Owner**: Member 4
- **Model**: 3D CNN for transition quality
- **Task**: Combine segments into coherent video
- **Output**: Final summarized video with transitions

## 📁 Repository Structure

```
lecture-video-summarizer/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package installation
├── .gitignore                        # Git ignore rules
├── .env.example                      # Environment variables template
│
├── docs/                             # Documentation
│   ├── architecture.md               # System architecture
│   ├── installation.md               # Setup instructions
│   ├── usage.md                      # Usage guide
│   ├── api.md                        # API documentation
│   └── papers/                       # Research papers (PDFs)
│
├── data/                             # Data directory (gitignored)
│   ├── raw/                          # Raw lecture videos
│   ├── processed/                    # Processed data
│   ├── annotations/                  # Manual annotations
│   │   ├── module1/                  # Segment importance scores
│   │   ├── module2/                  # Sentence labels
│   │   ├── module3/                  # Slide importance labels
│   │   └── module4/                  # Transition quality ratings
│   └── datasets/                     # Training datasets
│       ├── train/
│       ├── val/
│       └── test/
│
├── models/                           # Trained models (gitignored)
│   ├── module1/                      # Importance scoring models
│   ├── module2/                      # Summarization models
│   ├── module3/                      # Slide classification models
│   └── module4/                      # Transition quality models
│
├── src/                              # Source code
│   ├── __init__.py
│   │
│   ├── module1_importance/           # Module 1: Keyframe Detection
│   │   ├── __init__.py
│   │   ├── model.py                  # VideoImportanceScorer
│   │   ├── train.py                  # Training script
│   │   ├── inference.py              # Inference script
│   │   ├── feature_extractor.py      # ResNet feature extraction
│   │   └── utils.py                  # Helper functions
│   │
│   ├── module2_summarization/        # Module 2: Content Summarization
│   │   ├── __init__.py
│   │   ├── model.py                  # LectureSentenceClassifier
│   │   ├── train.py                  # Training script
│   │   ├── inference.py              # Inference script
│   │   ├── transcriber.py            # Whisper integration
│   │   └── utils.py                  # Helper functions
│   │
│   ├── module3_visual/               # Module 3: Visual Understanding
│   │   ├── __init__.py
│   │   ├── model.py                  # SlideImportanceClassifier
│   │   ├── train.py                  # Training script
│   │   ├── inference.py              # Inference script
│   │   ├── slide_extractor.py        # Slide detection
│   │   ├── ocr.py                    # OCR processing
│   │   └── utils.py                  # Helper functions
│   │
│   ├── module4_synthesis/            # Module 4: Video Synthesis
│   │   ├── __init__.py
│   │   ├── model.py                  # TransitionQualityPredictor
│   │   ├── train.py                  # Training script
│   │   ├── inference.py              # Inference script
│   │   ├── video_editor.py           # MoviePy integration
│   │   └── utils.py                  # Helper functions
│   │
│   ├── pipeline/                     # Integration pipeline
│   │   ├── __init__.py
│   │   ├── summarizer.py             # Main LectureVideoSummarizer
│   │   ├── segment_selector.py       # Segment selection logic
│   │   └── config.py                 # Pipeline configuration
│   │
│   ├── data/                         # Data processing
│   │   ├── __init__.py
│   │   ├── video_loader.py           # Video loading utilities
│   │   ├── preprocessing.py          # Data preprocessing
│   │   └── dataset.py                # PyTorch datasets
│   │
│   ├── evaluation/                   # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── metrics.py                # Quantitative metrics
│   │   ├── user_study.py             # User study tools
│   │   └── visualize.py              # Visualization tools
│   │
│   └── utils/                        # Shared utilities
│       ├── __init__.py
│       ├── logger.py                 # Logging configuration
│       ├── config.py                 # Global configuration
│       └── helpers.py                # Common helper functions
│
├── scripts/                          # Utility scripts
│   ├── download_videos.py            # Download lecture videos
│   ├── setup_environment.sh          # Environment setup
│   ├── train_all_modules.sh          # Train all modules
│   └── run_pipeline.py               # Run full pipeline
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb     # Data analysis
│   ├── 02_module1_prototype.ipynb    # Module 1 experiments
│   ├── 03_module2_prototype.ipynb    # Module 2 experiments
│   ├── 04_module3_prototype.ipynb    # Module 3 experiments
│   ├── 05_module4_prototype.ipynb    # Module 4 experiments
│   └── 06_integration_demo.ipynb     # Full pipeline demo
│
├── tools/                            # Annotation tools
│   ├── annotate_segments.py          # Segment importance annotation
│   ├── annotate_sentences.py         # Sentence importance annotation
│   ├── annotate_slides.py            # Slide importance annotation
│   └── annotate_transitions.py       # Transition quality annotation
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_module1.py
│   ├── test_module2.py
│   ├── test_module3.py
│   ├── test_module4.py
│   └── test_pipeline.py
│
├── configs/                          # Configuration files
│   ├── module1_config.yaml           # Module 1 settings
│   ├── module2_config.yaml           # Module 2 settings
│   ├── module3_config.yaml           # Module 3 settings
│   ├── module4_config.yaml           # Module 4 settings
│   └── pipeline_config.yaml          # Pipeline settings
│
├── outputs/                          # Generated outputs (gitignored)
│   ├── summaries/                    # Generated summary videos
│   ├── logs/                         # Training logs
│   └── results/                      # Evaluation results
│
└── research/                         # Research materials
    ├── literature_review.md          # Paper summaries
    ├── experiment_log.md             # Experiment tracking
    └── meeting_notes/                # Team meeting notes
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (RTX 3060 or better recommended)
- 32GB RAM
- 100GB+ free disk space

### Installation

```bash
# Clone the repository
git clone https://github.com/rashmiJayawardhana/lecture-video-summarizer.git
cd lecture-video-summarizer

# Create virtual environment
conda create -n lecture-summarizer python=3.10
conda activate lecture-summarizer

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (required for video processing)
# Ubuntu/Debian:
sudo apt-get install ffmpeg
# macOS:
brew install ffmpeg
# Windows: Download from https://ffmpeg.org/download.html

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
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
- **Source**: MIT OpenCourseWare, Khan Academy, Coursera
- **Size**: 50 lecture videos (30-90 minutes each)
- **Subjects**: Computer Science, Mathematics, Physics

### Annotation Requirements

| Module | Data Type | Quantity | Time Estimate |
|--------|-----------|----------|---------------|
| Module 1 | Segment scores | 50 videos | 60-80 hours |
| Module 2 | Sentence labels | 40 transcripts | 50-70 hours |
| Module 3 | Slide labels | 800-1000 slides | 40-50 hours |
| Module 4 | Transition ratings | 500 pairs | 30-40 hours |

## 🎓 Training

### Train Individual Modules

```bash
# Module 1: Importance Scoring
python src/module1_importance/train.py --config configs/module1_config.yaml

# Module 2: Content Summarization
python src/module2_summarization/train.py --config configs/module2_config.yaml

# Module 3: Visual Understanding
python src/module3_visual/train.py --config configs/module3_config.yaml

# Module 4: Video Synthesis
python src/module4_synthesis/train.py --config configs/module4_config.yaml
```

### Train All Modules

```bash
bash scripts/train_all_modules.sh
```

## 📈 Evaluation Metrics

### Quantitative Metrics
- **Compression Ratio**: 6:1 (60 min → 10 min)
- **Content Coverage**: ROUGE-L > 0.40
- **Segment Precision/Recall**: > 0.75
- **Transition Smoothness**: > 7/10
- **Processing Time**: < 45 min per 1-hour lecture

### User Study Metrics
- **Comprehension**: Quiz score > 70% of full-video watchers
- **Satisfaction**: > 8/10 rating
- **Usefulness**: > 80% would use for study

## 🗓️ Project Timeline

- **Month 1**: Setup + Data Collection + Literature Review
- **Month 2**: Data Annotation + Baseline Implementation
- **Month 3**: Model Development & Training
- **Month 4**: Integration + Pipeline Development
- **Month 5**: Evaluation + User Studies
- **Month 6**: Refinement + Documentation + Presentation

## 👥 Team

| Member | Module | Responsibility |
|--------|--------|----------------|
| Member 1 | Module 1 | Keyframe Detection & Importance Scoring |
| Member 2 | Module 2 | Speech-to-Text + Content Summarization |
| Member 3 | Module 3 | Visual Content Understanding |
| Member 4 | Module 4 | Video Synthesis & Editing |

## 📚 References

1. Rahman et al. (2020) - Visual Summarization of Lecture Video Segments
2. IEEE T4E (2018) - Automated Summarization of Lecture Videos
3. ACL BEA (2023) - Automatically Generated Summaries of Video Lectures
4. [Full bibliography in docs/papers/](docs/papers/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MIT OpenCourseWare for lecture videos
- OpenAI Whisper for speech recognition
- Hugging Face for transformer models
- MoviePy for video editing capabilities

## Contact

For questions or collaboration:
- Project Repository: [GitHub](https://github.com/rashmiJayawardhana/lecture-video-summarizer)
- Issues: [GitHub Issues](https://github.com/rashmiJayawardhana/lecture-video-summarizer/issues)

---

**Status**: In Development | **Version**: 0.1.0 | **Last Updated**: February 2026
