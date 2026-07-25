from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.backend.api.routes_upload import router as upload_router
from src.backend.api.routes_jobs import router as jobs_router
from src.backend.api.routes_results import router as results_router

app = FastAPI(
    title="INTEGRA Lecture Video Summarizer Backend",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later change this to frontend URL only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/api")
app.include_router(jobs_router, prefix="/api")
app.include_router(results_router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": "INTEGRA backend is running"
    }