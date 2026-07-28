"""Step functions used by both the individual endpoints and /process's background task."""

from src.module2_summarization import classify_with_features, gemini_enrich
from src.module2_summarization.api import job_store, model_cache
from src.module2_summarization.transcribe_video import transcribe_to_sentences


def run_transcribe(video_path: str) -> list:
    transcriber = model_cache.get_transcriber()
    return transcribe_to_sentences(transcriber, video_path)


def run_classify(sentences: list) -> list:
    model_cache.get_bert()
    return classify_with_features.classify_sentences(sentences)


def run_enrich(segments: list) -> list:
    return gemini_enrich.enrich_segments(segments)


def run_full_pipeline(job_id: str, video_path: str) -> None:
    """Sequential transcribe -> classify -> enrich, updating job_store after each step."""
    try:
        job_store.update_job(job_id, step="transcribing")
        sentences = run_transcribe(video_path)

        job_store.update_job(job_id, step="classifying")
        segments = run_classify(sentences)

        job_store.update_job(job_id, step="enriching")
        enriched = run_enrich(segments)

        job_store.update_job(
            job_id,
            status="done",
            step="done",
            result=enriched,
        )
    except Exception as e:
        job_store.update_job(
            job_id,
            status="failed",
            error=str(e),
        )
