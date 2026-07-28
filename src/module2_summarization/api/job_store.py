"""In-memory job store for the standalone Module 2 API.

Resets on server restart -- fine for a research-module demo, no DB required.
"""

import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def create_job(filename: str) -> str:
    job_id = f"m2job_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "step": "queued",
            "filename": filename,
            "result": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }

    return job_id


def update_job(job_id: str, **updates: Any) -> None:
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()

    with _lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(updates)


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None
