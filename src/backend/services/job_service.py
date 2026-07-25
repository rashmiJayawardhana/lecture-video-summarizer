import json
from datetime import datetime
from pathlib import Path

from src.backend.core.paths import JOBS_DIR


def get_job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def get_status_path(job_id: str) -> Path:
    return get_job_dir(job_id) / "status.json"


def write_status(job_id: str, status_data: dict) -> None:
    status_data["updated_at"] = datetime.now().isoformat()

    status_path = get_status_path(job_id)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    with status_path.open("w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=4)


def read_status(job_id: str) -> dict:
    status_path = get_status_path(job_id)

    if not status_path.exists():
        return {
            "job_id": job_id,
            "status": "not_found"
        }

    with status_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def init_job_status(job_id: str, filename: str) -> None:
    write_status(job_id, {
        "job_id": job_id,
        "filename": filename,
        "status": "uploaded",
        "module1": "waiting",
        "module2": "waiting",
        "module3": "waiting",
        "module4": "waiting",
        "final_video": None,
        "final_json": None,
        "error": None,
        "created_at": datetime.now().isoformat(),
    })


def update_job_status(job_id: str, **updates) -> None:
    status = read_status(job_id)
    status.update(updates)
    write_status(job_id, status)