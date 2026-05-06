# Data Directory

This directory contains all data for the INTEGRA lecture video summarization project.

## Structure

```
data/
├── raw/                   # Original lecture videos (gitignored)
├── processed/             # Preprocessed data (gitignored)
│   ├── transcripts/       # Whisper-generated transcripts (JSON)
│   └── slides/            # Extracted slide frames (PNG)
├── annotations/           # Manual annotations (gitignored)
│   ├── module1/           # Segment importance scores (0-10 per 10-sec segment)
│   ├── module2/           # Sentence importance labels
│   └── module3/           # Slide labels (Critical / Important / Skip)
└── datasets/              # Training datasets (gitignored)
    ├── train/
    ├── val/
    └── test/
```

## Data Sources

### Raw Videos
- MIT OpenCourseWare
- YouTube
- Coursera
- AVLectures dataset
- LVVO dataset

### Content Focus
- **IT theory lectures only**: algorithms, networking, databases, software engineering
- **Duration**: 45–90 minutes per lecture
- **Total**: 50 annotated IT lecture videos

### Download Videos
```bash
python scripts/download_videos.py --num-videos 50
```

## Annotation Guidelines

### Module 1 — Segment Importance (Rashmi)
- Rate each 10-second segment on a scale of **0–10** for instructional importance
- Scoring criteria: concept introduction, formula, worked example, visual emphasis

### Module 2 — Sentence Importance (Ravindu)
- Label each sentence as **Important** or **Not Important**
- Classification criteria:
  1. **Concept introduction** — "A binary search tree is..."
  2. **Term definition** — "This is called dynamic programming..."
  3. **Key explanation** — "The reason this works is because..."

### Module 3 — Slide Classification (Fazly)
- Label each slide frame as:
  - **Critical** — Key diagrams, important formulas, summary slides
  - **Important** — Definitions, step explanations, worked examples
  - **Skip** — Title slides, duplicates, blank frames
- Each slide takes ~5–8 seconds to annotate

## Dataset Splits

| Module | Total | Train | Val | Test |
|--------|-------|-------|-----|------|
| Module 1 | 50 videos | 35 | 5 | 10 |
| Module 2 | 50 transcripts | 35 | 5 | 10 |
| Module 3 | 800-1000 slides | 600 | 100 | 100-300 |

## Important Notes

- **All data files are gitignored** to avoid repository bloat
- Videos should be in MP4 format, max 720p
- Store annotations as JSON files
- Use the shared JSON schema defined in `src/utils/json_schema.py`
