from datetime import datetime, timezone

from src.backend.core.config import SUPABASE_JOBS_TABLE
from src.backend.core.supabase_client import supabase


def init_job_status(job_id: str, filename: str) -> None:
    now = datetime.now(timezone.utc).isoformat()

    supabase.table(SUPABASE_JOBS_TABLE).insert({
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
        "created_at": now,
        "updated_at": now,
    }).execute()


def read_status(job_id: str) -> dict:
    response = (
        supabase.table(SUPABASE_JOBS_TABLE)
        .select("*")
        .eq("job_id", job_id)
        .limit(1)
        .execute()
    )

    if not response.data:
        return {
            "job_id": job_id,
            "status": "not_found"
        }

    return response.data[0]


def update_job_status(job_id: str, **updates) -> None:
    updates["updated_at"] = datetime.now(timezone.utc).isoformat()

    supabase.table(SUPABASE_JOBS_TABLE).update(updates).eq("job_id", job_id).execute()
