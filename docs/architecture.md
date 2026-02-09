# Project Structure Documentation

This document explains the organization of the lecture-video-summarizer repository.

## Directory Overview

```
lecture-video-summarizer/
├── src/                    # Source code (main implementation)
├── data/                   # Data storage (gitignored)
├── models/                 # Trained models (gitignored)
├── configs/                # Configuration files
├── docs/                   # Documentation
├── scripts/                # Utility scripts
├── tools/                  # Annotation tools
├── notebooks/              # Jupyter notebooks
├── tests/                  # Unit tests
├── outputs/                # Generated outputs (gitignored)
└── research/               # Research materials
```

## Source Code (`src/`)

### Module Structure

Each module follows the same structure:

```
src/module{N}_{name}/
├── __init__.py            # Module initialization
├── model.py               # Neural network model
├── train.py               # Training script
├── inference.py           # Inference script
└── utils.py               # Helper functions
```

### Module 1: `module1_importance/`
- **Purpose**: Keyframe detection and importance scoring
- **Model**: ResNet50 + LSTM
- **Owner**: Member 1
- **Key Files**:
  - `model.py`: VideoImportanceScorer class
  - `feature_extractor.py`: ResNet feature extraction
  - `train.py`: Training loop
  - `inference.py`: Score video segments

### Module 2: `module2_summarization/`
- **Purpose**: Speech-to-text and content summarization
- **Model**: Whisper + BERT
- **Owner**: Member 2
- **Key Files**:
  - `model.py`: LectureSentenceClassifier
  - `transcriber.py`: Whisper integration
  - `train.py`: Fine-tune BERT
  - `inference.py`: Generate summaries

### Module 3: `module3_visual/`
- **Purpose**: Visual content understanding and slide extraction
- **Model**: Vision Transformer (ViT)
- **Owner**: Member 3
- **Key Files**:
  - `model.py`: SlideImportanceClassifier
  - `slide_extractor.py`: Slide detection
  - `ocr.py`: Tesseract OCR integration
  - `train.py`: Train ViT classifier

### Module 4: `module4_synthesis/`
- **Purpose**: Video synthesis and intelligent editing
- **Model**: 3D CNN for transitions
- **Owner**: Member 4
- **Key Files**:
  - `model.py`: TransitionQualityPredictor
  - `video_editor.py`: MoviePy integration
  - `train.py`: Train transition model
  - `inference.py`: Create final video

### Pipeline (`pipeline/`)
- **Purpose**: Integrate all modules
- **Key Files**:
  - `summarizer.py`: Main LectureVideoSummarizer class
  - `segment_selector.py`: Segment selection algorithms
  - `config.py`: Pipeline configuration

### Shared Utilities

#### `data/`
- `video_loader.py`: Load and preprocess videos
- `preprocessing.py`: Data augmentation
- `dataset.py`: PyTorch Dataset classes

#### `evaluation/`
- `metrics.py`: ROUGE, precision, recall, etc.
- `user_study.py`: User study tools
- `visualize.py`: Result visualization

#### `utils/`
- `logger.py`: Logging configuration
- `config.py`: Configuration management
- `helpers.py`: Common utilities

## Data Directory (`data/`)

```
data/
├── raw/                   # Original lecture videos
├── processed/             # Preprocessed data
├── annotations/           # Manual annotations
│   ├── module1/          # Segment importance scores
│   ├── module2/          # Sentence labels
│   ├── module3/          # Slide importance labels
│   └── module4/          # Transition quality ratings
└── datasets/             # Training datasets
    ├── train/
    ├── val/
    └── test/
```

## Models Directory (`models/`)

```
models/
├── module1/              # Importance scoring models
│   ├── best_model.pth
│   └── checkpoints/
├── module2/              # Summarization models
├── module3/              # Slide classification models
└── module4/              # Transition quality models
```

## Configuration Files (`configs/`)

- `module1_config.yaml`: Module 1 hyperparameters
- `module2_config.yaml`: Module 2 hyperparameters
- `module3_config.yaml`: Module 3 hyperparameters
- `module4_config.yaml`: Module 4 hyperparameters
- `pipeline_config.yaml`: End-to-end pipeline settings

## Scripts (`scripts/`)

Utility scripts for common tasks:
- `download_videos.py`: Download lecture videos
- `setup_environment.sh`: Environment setup
- `train_all_modules.sh`: Train all modules sequentially
- `run_pipeline.py`: Run full summarization pipeline

## Tools (`tools/`)

Annotation interfaces:
- `annotate_segments.py`: Gradio UI for segment annotation
- `annotate_sentences.py`: Sentence importance labeling
- `annotate_slides.py`: Slide importance labeling
- `annotate_transitions.py`: Transition quality rating

## Notebooks (`notebooks/`)

Jupyter notebooks for exploration and prototyping:
1. `01_data_exploration.ipynb`: Explore video datasets
2. `02_module1_prototype.ipynb`: Module 1 experiments
3. `03_module2_prototype.ipynb`: Module 2 experiments
4. `04_module3_prototype.ipynb`: Module 3 experiments
5. `05_module4_prototype.ipynb`: Module 4 experiments
6. `06_integration_demo.ipynb`: Full pipeline demo

## Tests (`tests/`)

Unit tests for each module:
- `test_module1.py`: Test importance scoring
- `test_module2.py`: Test summarization
- `test_module3.py`: Test slide extraction
- `test_module4.py`: Test video synthesis
- `test_pipeline.py`: Test end-to-end pipeline

Run tests:
```bash
pytest tests/
```

## Outputs (`outputs/`)

```
outputs/
├── summaries/            # Generated summary videos
├── logs/                 # Training logs
│   ├── module1/
│   ├── module2/
│   ├── module3/
│   └── module4/
└── results/              # Evaluation results
```

## Research Materials (`research/`)

- `literature_review.md`: Summary of research papers
- `experiment_log.md`: Track experiments and results
- `meeting_notes/`: Team meeting notes

## File Naming Conventions

### Python Files
- Classes: `PascalCase` (e.g., `VideoImportanceScorer`)
- Functions: `snake_case` (e.g., `extract_features`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_SEQUENCE_LENGTH`)

### Data Files
- Videos: `lecture_XXX.mp4` (e.g., `lecture_001.mp4`)
- Annotations: `lecture_XXX_annotations.json`
- Models: `module{N}_epoch{E}.pth` (e.g., `module1_epoch50.pth`)

### Configuration Files
- YAML format: `{module}_config.yaml`
- Environment: `.env` (not committed)

## Git Workflow

### Branches
- `main`: Stable code
- `develop`: Integration branch
- `feature/module{N}-{feature}`: Feature branches
- `fix/{issue}`: Bug fixes

### Commit Messages
```
[Module{N}] Brief description

Detailed explanation if needed
```

Example:
```
[Module1] Add ResNet feature extraction

Implemented feature extraction using pre-trained ResNet50.
Added caching to speed up repeated extractions.
```

## Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/module1-training
   ```

2. **Make changes and test**
   ```bash
   pytest tests/test_module1.py
   ```

3. **Commit changes**
   ```bash
   git add .
   git commit -m "[Module1] Implement training loop"
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/module1-training
   ```

5. **Code review and merge**

## Best Practices

### Code Organization
- Keep modules independent
- Use configuration files for hyperparameters
- Write docstrings for all functions/classes
- Add type hints where possible

### Data Management
- Never commit large files (videos, models)
- Use `.gitignore` properly
- Document data sources
- Version datasets

### Collaboration
- Regular team meetings
- Code reviews for all PRs
- Update documentation
- Track experiments in `experiment_log.md`

## Quick Reference

### Run Training
```bash
python src/module1_importance/train.py --config configs/module1_config.yaml
```

### Run Inference
```bash
python src/pipeline/summarizer.py --input data/raw/lecture_001.mp4 --output outputs/summaries/
```

### Run Tests
```bash
pytest tests/ -v
```

### View Logs
```bash
tensorboard --logdir outputs/logs/
```
