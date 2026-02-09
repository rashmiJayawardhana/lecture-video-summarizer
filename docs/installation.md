# Installation Guide

## Prerequisites

Before starting, ensure you have:
- **Python 3.10 or higher**
- **CUDA-capable GPU** (RTX 3060 12GB or better recommended)
- **32GB RAM** minimum
- **100GB+ free disk space**
- **Git** installed

## Step 1: Clone the Repository

```bash
git clone https://github.com/rashmiJayawardhana/lecture-video-summarizer.git
cd lecture-video-summarizer
```

## Step 2: Create Virtual Environment

### Using Conda (Recommended)

```bash
# Create environment
conda create -n lecture-summarizer python=3.10
conda activate lecture-summarizer
```

### Using venv

```bash
# Create environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

## Step 3: Install PyTorch with CUDA Support

```bash
# For CUDA 11.8 (check your CUDA version first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}')"
```

## Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 5: Install FFmpeg

### Windows
1. Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to PATH

Or using Chocolatey:
```bash
choco install ffmpeg
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### macOS
```bash
brew install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```

## Step 6: Install Tesseract OCR (for Module 3)

### Windows
1. Download installer from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install to `C:\Program Files\Tesseract-OCR`
3. Add to PATH

### Linux (Ubuntu/Debian)
```bash
sudo apt-get install tesseract-ocr
```

### macOS
```bash
brew install tesseract
```

Verify installation:
```bash
tesseract --version
```

## Step 7: Set Up Environment Variables

```bash
# Copy template
cp .env.example .env

# Edit .env file with your settings
# Set CUDA_VISIBLE_DEVICES, paths, API keys, etc.
```

## Step 8: Install Package in Development Mode

```bash
pip install -e .
```

## Step 9: Download Pre-trained Models (Optional)

```bash
# This will download Whisper, BERT, ViT models to cache
python scripts/download_pretrained_models.py
```

## Step 10: Verify Installation

```bash
# Run tests
pytest tests/

# Or run a quick check
python -c "from src.pipeline.summarizer import LectureVideoSummarizer; print('Installation successful!')"
```

## Troubleshooting

### CUDA Not Available
- Check NVIDIA driver: `nvidia-smi`
- Reinstall PyTorch with correct CUDA version
- Verify CUDA toolkit installation

### FFmpeg Not Found
- Ensure FFmpeg is in PATH
- Restart terminal after installation

### Out of Memory Errors
- Reduce batch size in config files
- Use smaller models (e.g., Whisper "tiny" instead of "base")
- Process videos at lower resolution

### Import Errors
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

## Next Steps

After installation:
1. Download sample lecture videos: `python scripts/download_videos.py`
2. Follow the [Usage Guide](usage.md)
3. Start with [Data Exploration Notebook](../notebooks/01_data_exploration.ipynb)

## GPU Requirements by Module

| Module | Min VRAM | Recommended VRAM |
|--------|----------|------------------|
| Module 1 | 6GB | 12GB |
| Module 2 | 4GB | 8GB |
| Module 3 | 6GB | 12GB |
| Module 4 | 8GB | 16GB |
| Full Pipeline | 12GB | 24GB |

## Cloud GPU Options

If you don't have a local GPU:

### Google Colab
- Free tier: Limited GPU hours
- Pro: $10/month
- Pro+: $50/month

### AWS EC2
- p3.2xlarge: ~$3/hour (V100 16GB)
- g4dn.xlarge: ~$0.50/hour (T4 16GB)

### Paperspace
- Free tier available
- GPU instances from $0.45/hour

## Support

For issues:
- Check [GitHub Issues](https://github.com/your-team/lecture-video-summarizer/issues)
- Contact team members
- Review documentation in `docs/`
