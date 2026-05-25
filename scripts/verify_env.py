"""
INTEGRA — Environment Verification Script
==========================================
Run this after completing the setup in docs/SETUP_GUIDE.md.
Every team member must run this and confirm all checks pass before
starting annotation work.

Usage:
    python scripts/verify_env.py
"""

import sys
import subprocess
import importlib
import platform

# ─── ANSI colours ─────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

PASS = f"{GREEN}[PASS]{RESET}"
FAIL = f"{RED}[FAIL]{RESET}"
WARN = f"{YELLOW}[WARN]{RESET}"
INFO = f"{CYAN}[INFO]{RESET}"


def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*60}{RESET}")


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = PASS if ok else FAIL
    line = f"  {status}  {label}"
    if detail:
        line += f"  →  {detail}"
    print(line)
    return ok


def try_import(pkg: str, min_version: str = "") -> tuple[bool, str]:
    """Import a package and return (success, version_string)."""
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "unknown")
        return True, ver
    except ImportError:
        return False, "NOT INSTALLED"


# ══════════════════════════════════════════════════════════════
# 1. Python version
# ══════════════════════════════════════════════════════════════
section("1. Python Runtime")
py_ver = sys.version_info
py_ok = py_ver.major == 3 and py_ver.minor == 10
check(
    f"Python 3.10  (found {py_ver.major}.{py_ver.minor}.{py_ver.micro})",
    py_ok,
    "Must be exactly 3.10.x for compatibility" if not py_ok else "",
)
print(f"  {INFO}  Platform: {platform.platform()}")

# ══════════════════════════════════════════════════════════════
# 2. PyTorch + CUDA  ← the most critical check
# ══════════════════════════════════════════════════════════════
section("2. PyTorch & CUDA  [CRITICAL]")
torch_ok, torch_ver = try_import("torch")
check("torch installed", torch_ok, torch_ver)

if torch_ok:
    import torch

    cuda_available = torch.cuda.is_available()
    check(
        "torch.cuda.is_available()",
        cuda_available,
        "GPU detected ✓" if cuda_available else "No GPU — use Google Colab (see SETUP_GUIDE.md §6)",
    )

    if cuda_available:
        gpu_name  = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        vram_gb   = torch.cuda.get_device_properties(0).total_memory / 1e9
        check("GPU name",  True, gpu_name)
        check("GPU count", True, str(gpu_count))
        check(
            "VRAM ≥ 4 GB (minimum for BERT fine-tuning)",
            vram_gb >= 4,
            f"{vram_gb:.1f} GB",
        )
        check(
            "VRAM ≥ 6 GB (recommended for ViT / ResNet-50)",
            vram_gb >= 6,
            f"{vram_gb:.1f} GB",
        )
        print(f"  {INFO}  CUDA version PyTorch built with: {torch.version.cuda}")
    else:
        print(f"\n  {WARN}  GPU not detected.  Options:")
        print(f"         → Use Google Colab (free T4 GPU)")
        print(f"         → Run: python scripts/verify_env.py --colab-check")

    # Quick tensor computation on GPU/CPU
    try:
        device = "cuda" if cuda_available else "cpu"
        t = torch.randn(3, 3).to(device)
        _ = t @ t
        check(f"Tensor computation on {device}", True, "matrix multiply OK")
    except Exception as e:
        check("Tensor computation", False, str(e))

# ══════════════════════════════════════════════════════════════
# 3. Hugging Face ecosystem
# ══════════════════════════════════════════════════════════════
section("3. Hugging Face Transformers")
for pkg in ["transformers", "tokenizers", "datasets", "accelerate"]:
    ok, ver = try_import(pkg)
    check(pkg, ok, ver)

# ══════════════════════════════════════════════════════════════
# 4. Computer Vision
# ══════════════════════════════════════════════════════════════
section("4. Computer Vision (OpenCV + Pillow)")
cv2_ok, cv2_ver = try_import("cv2")
check("opencv-python", cv2_ok, cv2_ver)

if cv2_ok:
    import cv2
    # Quick test: create a blank frame
    import numpy as np
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    check("cv2 colour conversion", gray.shape == (224, 224), f"output shape {gray.shape}")

pil_ok, pil_ver = try_import("PIL")
check("Pillow", pil_ok, pil_ver)

sk_ok, sk_ver = try_import("skimage")
check("scikit-image (SSIM)", sk_ok, sk_ver)

# ══════════════════════════════════════════════════════════════
# 5. ASR — OpenAI Whisper
# ══════════════════════════════════════════════════════════════
section("5. OpenAI Whisper (Module 2)")
whisper_ok, whisper_ver = try_import("whisper")
check("openai-whisper", whisper_ok, whisper_ver)

if whisper_ok:
    import whisper
    try:
        # Load the tiny model as a quick smoke test (no GPU needed)
        model = whisper.load_model("tiny", device="cpu")
        check("Whisper model load (tiny / CPU)", True, "model loaded OK")
        del model
    except Exception as e:
        check("Whisper model load", False, str(e))

# ══════════════════════════════════════════════════════════════
# 6. Video processing
# ══════════════════════════════════════════════════════════════
section("6. Video Processing (MoviePy + FFmpeg)")
mp_ok, mp_ver = try_import("moviepy")
check("moviepy", mp_ok, mp_ver)

ffmpeg_result = subprocess.run(
    ["ffmpeg", "-version"], capture_output=True, text=True
)
ffmpeg_ok = ffmpeg_result.returncode == 0
ffmpeg_line = ffmpeg_result.stdout.splitlines()[0] if ffmpeg_ok else "not found"
check("FFmpeg (system binary)", ffmpeg_ok, ffmpeg_line[:60])

# ══════════════════════════════════════════════════════════════
# 7. Text-to-Speech (Module 4 Pipeline B)
# ══════════════════════════════════════════════════════════════
section("7. Text-to-Speech — edge-tts (Module 4)")
tts_ok, tts_ver = try_import("edge_tts")
check("edge-tts", tts_ok, tts_ver)

# ══════════════════════════════════════════════════════════════
# 8. OpenAI API (Module 4 Pipeline B)
# ══════════════════════════════════════════════════════════════
section("8. OpenAI API (Module 4 Pipeline B)")
openai_ok, openai_ver = try_import("openai")
check("openai SDK", openai_ok, openai_ver)

import os
from pathlib import Path

# Check .env for OPENAI_API_KEY
env_path = Path(__file__).parent.parent / ".env"
api_key_set = False
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            if line.startswith("OPENAI_API_KEY=") and len(line.strip()) > 16:
                api_key_set = True
                break

env_api = os.environ.get("OPENAI_API_KEY", "")
api_key_set = api_key_set or (len(env_api) > 10)
check(
    "OPENAI_API_KEY set",
    api_key_set,
    "found in .env or environment" if api_key_set else "missing — add to .env (Module 4 only)",
)

# ══════════════════════════════════════════════════════════════
# 9. Evaluation libraries
# ══════════════════════════════════════════════════════════════
section("9. Evaluation Libraries")
for pkg in ["rouge_score", "jiwer", "nltk", "sklearn"]:
    ok, ver = try_import(pkg)
    check(pkg, ok, ver)

# ══════════════════════════════════════════════════════════════
# 10. Data utilities
# ══════════════════════════════════════════════════════════════
section("10. Data & Utility Libraries")
for pkg in ["numpy", "pandas", "scipy", "yaml", "dotenv", "tqdm", "yt_dlp"]:
    ok, ver = try_import(pkg)
    check(pkg, ok, ver)

# ══════════════════════════════════════════════════════════════
# 11. JSON schema smoke-test
# ══════════════════════════════════════════════════════════════
section("11. Inter-Module JSON Schema Smoke Test")
import json

SAMPLE_M1 = {"segment_id": "seg_001", "timestamp_start": 0.0, "timestamp_end": 10.0, "score_V": 0.82}
SAMPLE_M2 = {"sentence": "A stack is a LIFO data structure.", "timestamp_start": 0.0, "timestamp_end": 4.2, "is_important": True, "importance_ratio_T": 0.75}
SAMPLE_M3 = {"frame_time": 5.5, "label": "Critical", "ocr_text": "Stack: push(), pop(), peek()"}

for name, sample in [("Module 1", SAMPLE_M1), ("Module 2", SAMPLE_M2), ("Module 3", SAMPLE_M3)]:
    try:
        serialized   = json.dumps(sample)
        deserialized = json.loads(serialized)
        check(f"{name} schema round-trip", deserialized == sample)
    except Exception as e:
        check(f"{name} schema round-trip", False, str(e))

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
section("SUMMARY")
if torch_ok and (True if not torch_ok else torch.cuda.is_available()) and cv2_ok and whisper_ok and mp_ok and ffmpeg_ok:
    print(f"\n  {GREEN}{BOLD}✓ Environment is ready for INTEGRA development.{RESET}")
    print(f"  {INFO}  Next step: python scripts/download_videos.py\n")
else:
    print(f"\n  {YELLOW}{BOLD}⚠  Some checks failed. Fix them before starting model training.{RESET}")
    print(f"  {INFO}  See docs/SETUP_GUIDE.md for remediation steps.\n")
