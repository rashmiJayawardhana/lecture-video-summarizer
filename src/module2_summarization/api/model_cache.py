"""Lazy singleton loaders for the heavy Module 2 models.

Loaded on first use (not at server startup) so /transcribe keeps working even
before a trained BERT checkpoint exists, and vice versa -- one missing model
doesn't take down the whole API. Guarded by a lock so concurrent requests
don't trigger a double-load.
"""

import os
import threading

from src.module2_summarization import classify_with_features
from src.module2_summarization.transcriber import LectureTranscriber

_transcriber: LectureTranscriber | None = None
_transcriber_lock = threading.Lock()

_bert_loaded = False
_bert_lock = threading.Lock()

DEFAULT_BERT_MODEL_PATH = classify_with_features.DEFAULT_MODEL_PATH


def get_transcriber() -> LectureTranscriber:
    global _transcriber

    if _transcriber is None:
        with _transcriber_lock:
            if _transcriber is None:
                model_size = os.getenv("MODULE2_WHISPER_MODEL", "large-v3")
                _transcriber = LectureTranscriber(model_size=model_size)

    return _transcriber


def get_bert(model_path: str = DEFAULT_BERT_MODEL_PATH) -> None:
    """Ensures classify_with_features.tokenizer/model are loaded for model_path."""
    global _bert_loaded

    if not _bert_loaded:
        with _bert_lock:
            if not _bert_loaded:
                classify_with_features.load_bert(model_path)
                _bert_loaded = True
