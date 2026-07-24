from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.backend.services.job_service import read_status

router = APIRouter()


@router.get("/jobs/{job_id}/result")
def get_job_result(job_id: str):
    status = read_status(job_id)

    if status.get("status") != "completed":
        return {
            "job_id": job_id,
            "status": status.get("status"),
            "message": "Result is not ready yet.",
            "details": status
        }

    return {
        "job_id": job_id,
        "status": "completed",
        "final_video": status.get("final_video"),
        "final_json": status.get("final_json")
    }


@router.get("/jobs/{job_id}/download-json")
def download_final_json(job_id: str):
    status = read_status(job_id)

    final_json = status.get("final_json")

    if not final_json or not Path(final_json).exists():
        raise HTTPException(status_code=404, detail="Final JSON not found")

    return FileResponse(
        final_json,
        media_type="application/json",
        filename="module4_final_output.json"
    )


@router.get("/jobs/{job_id}/download-video")
def download_final_video(job_id: str):
    status = read_status(job_id)

    final_video = status.get("final_video")

    if not final_video or not Path(final_video).exists():
        raise HTTPException(status_code=404, detail="Final video not found")

    return FileResponse(
        final_video,
        media_type="video/mp4",
        filename="summarized_video.mp4"
    )