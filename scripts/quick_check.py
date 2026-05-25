# ============================================================
# INTEGRA — Quick GPU & Environment Check
# Run this in PowerShell from the project root:
#   .\.venv\Scripts\python.exe scripts\quick_check.py
# Or if using conda:
#   python scripts\quick_check.py
# ============================================================

import sys, importlib, subprocess

OK   = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

print("\n" + "="*55)
print("  INTEGRA Environment Quick Check")
print("="*55)

# ── 1. Python version ────────────────────────────────────
v = sys.version_info
py_ok = (v.major == 3 and v.minor == 10)
print(f"\n{'Python 3.10':<35} {OK if py_ok else FAIL}  ({v.major}.{v.minor}.{v.micro})")
if not py_ok:
    print("  !! Must be Python 3.10.x  (openai-whisper incompatible with 3.11+)")

# ── 2. PyTorch + CUDA ───────────────────────────────────
try:
    import torch
    cuda = torch.cuda.is_available()
    print(f"\n{'PyTorch':<35} {OK}  ({torch.__version__})")
    print(f"{'torch.cuda.is_available()':<35} {OK if cuda else FAIL}  ({'GPU detected' if cuda else 'NO GPU — use Google Colab'})")
    if cuda:
        print(f"{'GPU name':<35} {OK}  {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"{'VRAM':<35} {'OK' if vram>=4 else WARN}  {vram:.1f} GB")
        print(f"{'CUDA build version':<35}     {torch.version.cuda}")
    else:
        print()
        print("  ACTION REQUIRED: No GPU detected.")
        print("  -> Option 1: Re-install PyTorch with CUDA:")
        print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("  -> Option 2: Use Google Colab (see docs/SETUP_GUIDE.md Section 6)")
except ImportError:
    print(f"\n{'PyTorch':<35} {FAIL}  NOT INSTALLED")
    print("  -> Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

# ── 3. Core packages ────────────────────────────────────
print("\n--- Core Packages ---")
pkgs = {
    "cv2":            "opencv-python",
    "transformers":   "transformers",
    "whisper":        "openai-whisper",
    "moviepy":        "moviepy",
    "PIL":            "Pillow",
    "edge_tts":       "edge-tts",
    "openai":         "openai",
    "rouge_score":    "rouge-score",
    "jiwer":          "jiwer",
    "yt_dlp":         "yt-dlp",
    "numpy":          "numpy",
    "sklearn":        "scikit-learn",
}
for mod, pkg in pkgs.items():
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "ok")
        print(f"  {pkg:<30} {OK}  ({ver})")
    except ImportError:
        print(f"  {pkg:<30} {FAIL}  -> pip install {pkg}")

# ── 4. FFmpeg ───────────────────────────────────────────
print("\n--- System Binaries ---")
r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
if r.returncode == 0:
    line = r.stdout.splitlines()[0]
    print(f"  {'FFmpeg':<30} {OK}  ({line[:40]})")
else:
    print(f"  {'FFmpeg':<30} {FAIL}  -> winget install --id=Gyan.FFmpeg -e")

# ── 5. Summary ──────────────────────────────────────────
print("\n" + "="*55)
print("  Run full checks: python scripts/verify_env.py")
print("  Setup guide:     docs/SETUP_GUIDE.md")
print("="*55 + "\n")
