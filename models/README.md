# Models Directory

This directory stores trained models for all modules.

## Structure

```
models/
├── module1/              # Importance scoring models
│   ├── best_model.pth
│   ├── checkpoints/
│   └── config.json
├── module2/              # Summarization models
├── module3/              # Slide classification models
└── module4/              # Transition quality models
```

## Model Files

All model files are gitignored due to size. Download pre-trained models:

```bash
# Download from team shared drive or train yourself
python src/module1_importance/train.py --config configs/module1_config.yaml
```

## Naming Convention

- Best model: `best_model.pth`
- Checkpoints: `checkpoint_epoch{N}.pth`
- Configuration: `config.json`

## Model Sizes

| Module | Model Size | VRAM Required |
|--------|-----------|---------------|
| Module 1 | ~100MB | 6GB |
| Module 2 | ~500MB | 4GB |
| Module 3 | ~350MB | 6GB |
| Module 4 | ~200MB | 8GB |
