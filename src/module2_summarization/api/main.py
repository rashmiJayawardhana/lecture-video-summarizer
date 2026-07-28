"""
Standalone Module 2 API -- transcribe / classify / enrich / process.

Run with:
    uvicorn src.module2_summarization.api.main:app --reload
"""

import shutil
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

from src.module2_summarization.api import job_store, pipeline
from src.module2_summarization.api.schemas import (
    ClassifyRequest,
    EnrichRequest,
    JobResult,
    JobStatus,
)

app = FastAPI(title="Module 2 -- Lecture Summarization API", version="1.0.0")

UPLOADS_DIR = Path(__file__).resolve().parents[1] / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(file: UploadFile) -> Path:
    dest = UPLOADS_DIR / f"{uuid.uuid4().hex[:12]}_{file.filename}"
    with dest.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return dest


@app.get("/")
def root():
    return {"message": "Module 2 (Lecture Summarization) API is running"}


@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    video_path = _save_upload(file)
    try:
        sentences = pipeline.run_transcribe(str(video_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"filename": file.filename, "sentences": sentences}


@app.post("/classify")
def classify(request: ClassifyRequest):
    sentences = [s.model_dump() for s in request.sentences]
    try:
        segments = pipeline.run_classify(sentences)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"segments": segments}


@app.post("/enrich")
def enrich(request: EnrichRequest):
    segments = [s.model_dump() for s in request.segments]
    try:
        blocks = pipeline.run_enrich(segments)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"blocks": blocks}


@app.post("/process")
def process(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    video_path = _save_upload(file)
    job_id = job_store.create_job(filename=file.filename)

    background_tasks.add_task(pipeline.run_full_pipeline, job_id, str(video_path))

    return {
        "message": "Video uploaded successfully. Processing started.",
        "job_id": job_id,
        "filename": file.filename,
    }


@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        step=job["step"],
        filename=job["filename"],
        error=job["error"],
    )


@app.get("/result/{job_id}", response_model=JobResult)
def get_result(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] == "processing":
        raise HTTPException(status_code=409, detail=f"Job {job_id} is still processing (step: {job['step']})")

    return JobResult(
        job_id=job["job_id"],
        status=job["status"],
        result=job["result"],
        error=job["error"],
    )
