# Models Directory

This directory stores trained model checkpoints for Modules 1, 2, and 3.

> **Note**: Module 4 is an engineering module (MoviePy + FFmpeg + GPT-4o + edge-tts).
> It does **not** have a trainable model — it consumes fusion scores from Modules 1–3.

## Structure

```
models/
├── module1/              # ResNet-50 + BiLSTM checkpoints (Rashmi)
│   ├── best_model.pth
│   ├── checkpoints/
│   └── config.json
├── module2/              # Fine-tuned BERT-base checkpoints (Ravindu)
│   ├── best_model/       # Hugging Face model directory
│   └── checkpoints/
└── module3/              # Fine-tuned ViT-base checkpoints (Fazly)
    ├── best_model/       # Hugging Face model directory
    └── checkpoints/
```

## Model Files

All model files are gitignored due to size. Train models using:

```bash
# Module 1
python src/module1_importance/train.py --config configs/module1_config.yaml

# Module 2
python src/module2_summarization/train.py --config configs/module2_config.yaml

# Module 3
python src/module3_visual/train.py --config configs/module3_config.yaml
```

## Naming Convention

- Best model: `best_model.pth` (PyTorch) or `best_model/` (Hugging Face)
- Checkpoints: `checkpoint_epoch{N}.pth`
- Configuration: `config.json`

## Model Sizes (Approximate)

| Module | Architecture | Model Size | VRAM Required |
|--------|-------------|-----------|---------------|
| Module 1 | ResNet-50 + BiLSTM | ~100MB | 6GB |
| Module 2 | BERT-base fine-tuned | ~440MB | 4GB |
| Module 3 | ViT-base fine-tuned | ~330MB | 6GB |

## Pre-trained Model Downloads

The following pre-trained models are downloaded automatically from Hugging Face:
- `torchvision.models.resnet50` (ImageNet weights)
- `bert-base-uncased`
- `google/vit-base-patch16-224`
- `microsoft/trocr-base-printed`
