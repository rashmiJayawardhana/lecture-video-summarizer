from fastapi import APIRouter

from src.backend.services.job_service import read_status

router = APIRouter()


@router.get("/jobs/{job_id}/status")
def get_job_status(job_id: str):
    return read_status(job_id)