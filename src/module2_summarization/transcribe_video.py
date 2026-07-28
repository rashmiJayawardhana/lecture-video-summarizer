"""
Single-video CLI wrapper around LectureTranscriber, for backend integration.

Usage:
    python transcribe_video.py --video path/to/video.mp4 --output_json path/to/out.json [--model large-v3]
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.module2_summarization.transcriber import LectureTranscriber


def transcribe_to_sentences(transcriber: LectureTranscriber, video_path: str) -> list:
    """Transcribe video_path with an already-loaded transcriber and return sentence-level records."""
    result = transcriber.transcribe(video_path)
    return transcriber.extract_sentences(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Transcribe a lecture video into sentence-level JSON.")
    parser.add_argument("--video", required=True, help="Path to the lecture video file.")
    parser.add_argument("--output_json", required=True, help="Path to write the sentence-level JSON output.")
    parser.add_argument(
        "--model",
        default=os.getenv("MODULE2_WHISPER_MODEL", "large-v3"),
        help="Whisper model size (default: MODULE2_WHISPER_MODEL env var, or large-v3).",
    )
    args = parser.parse_args()

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    transcriber = LectureTranscriber(model_size=args.model)
    sentences = transcribe_to_sentences(transcriber, args.video)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(sentences, f, indent=2, ensure_ascii=False)

    print(f"Transcribed {len(sentences)} sentences -> {output_path}")


if __name__ == "__main__":
    main()
