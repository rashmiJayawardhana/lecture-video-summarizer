from fastapi import APIRouter, UploadFile, File, BackgroundTasks

from src.backend.services.storage_service import (
    create_job_folders,
    save_uploaded_video,
)
from src.backend.services.job_service import init_job_status
from src.backend.services.pipeline_orchestrator import process_pipeline

router = APIRouter()


@router.post("/upload")
def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    folders = create_job_folders()

    video_path = save_uploaded_video(file, folders["input_dir"])

    init_job_status(
        job_id=folders["job_id"],
        filename=file.filename
    )

    background_tasks.add_task(
        process_pipeline,
        folders["job_id"],
        str(video_path),
        str(folders["output_dir"])
    )

    return {
        "message": "Video uploaded successfully. Processing started.",
        "job_id": folders["job_id"],
        "filename": file.filename,
        "video_path": str(video_path)
    }