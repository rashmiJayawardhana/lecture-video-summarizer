import shutil
import uuid
from pathlib import Path
from fastapi import UploadFile

from src.backend.core.paths import JOBS_DIR


def create_job_folders() -> dict:
    job_id = f"job_{uuid.uuid4().hex[:12]}"

    job_dir = JOBS_DIR / job_id
    input_dir = job_dir / "input"
    output_dir = job_dir / "outputs"
    temp_dir = job_dir / "temp"
    logs_dir = job_dir / "logs"

    for folder in [input_dir, output_dir, temp_dir, logs_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    return {
        "job_id": job_id,
        "job_dir": job_dir,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "temp_dir": temp_dir,
        "logs_dir": logs_dir,
    }


def save_uploaded_video(file: UploadFile, input_dir: Path) -> Path:
    video_path = input_dir / file.filename

    with video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return video_path