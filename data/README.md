# Data Directory

This directory contains all data for the lecture video summarization project.

## Structure

```
data/
├── raw/                   # Original lecture videos (gitignored)
├── processed/             # Preprocessed data (gitignored)
├── annotations/           # Manual annotations (gitignored)
│   ├── module1/          # Segment importance scores
│   ├── module2/          # Sentence labels
│   ├── module3/          # Slide importance labels
│   └── module4/          # Transition quality ratings
└── datasets/             # Training datasets (gitignored)
    ├── train/
    ├── val/
    └── test/
```

## Data Sources

### Raw Videos
- MIT OpenCourseWare
- Khan Academy
- Coursera
- YouTube EDU

### Download Videos
```bash
python scripts/download_videos.py --num-videos 5
```

## Annotation Guidelines

See individual module annotation tools in `tools/` directory.

## Important Notes

- **All data files are gitignored** to avoid repository bloat
- Videos should be in MP4 format
- Target resolution: 720p
- Store annotations as JSON files
