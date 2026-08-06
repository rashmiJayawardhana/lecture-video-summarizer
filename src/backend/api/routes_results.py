import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from src.backend.core.paths import JOBS_DIR
from src.backend.services.job_service import read_status

router = APIRouter()

_MODULE_OUTPUT_FILENAMES = {
    "module1": "module1_output.json",
    "module2": "module2_output.json",
    "module3": "module3_output.json",
    "module4": "module4_final_output.json",
}


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


@router.get("/jobs/{job_id}/module/{module_name}")
def get_module_output(job_id: str, module_name: str):
    """
    Raw output JSON for a single module (module1/module2/module3/module4),
    available as soon as that specific module finishes - independent of
    whether the overall job has reached status=completed. Used by the
    frontend's Results tabs to render real data instead of placeholders.
    """
    filename = _MODULE_OUTPUT_FILENAMES.get(module_name)
    if not filename:
        raise HTTPException(status_code=400, detail=f"Unknown module: {module_name}")

    output_path = JOBS_DIR / job_id / "outputs" / filename
    if not output_path.exists():
        raise HTTPException(status_code=404, detail=f"{module_name} output not available yet")

    with output_path.open("r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/jobs/{job_id}/frame")
def get_frame_image(job_id: str, path: str):
    """
    Serves a single extracted slide frame (e.g. for Module 3's real
    thumbnails), given the frame_path already returned in that job's own
    module3_output.json. Restricted to files inside this job's own storage
    directory, regardless of what `path` contains, so it can't be used to
    read arbitrary files elsewhere on disk.
    """
    job_dir = (JOBS_DIR / job_id).resolve()
    requested = Path(path).resolve()

    try:
        requested.relative_to(job_dir)
    except ValueError:
        raise HTTPException(status_code=404, detail="Frame not found")

    if not requested.exists():
        raise HTTPException(status_code=404, detail="Frame not found")

    return FileResponse(requested, media_type="image/jpeg")