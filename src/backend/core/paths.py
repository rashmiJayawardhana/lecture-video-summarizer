from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

STORAGE_DIR = PROJECT_ROOT / "storage"
JOBS_DIR = STORAGE_DIR / "jobs"

JOBS_DIR.mkdir(parents=True, exist_ok=True)