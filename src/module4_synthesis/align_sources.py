"""
Aligns the two real Module 4 input sources for lecture_021 by timestamp:
  - lecture_021_module3_final_output.json (slide/OCR semantic analysis, 59 slides)
  - LecVideo_001_enriched.json            (audio transcript semantic analysis, 17 blocks)

Coverage is sparse (only 17 audio blocks for 59 slides, non-overlapping time
ranges at the edges) — most slides have no audio block in their window. Per
the slide-only fallback decision: audio is attached only when it genuinely
falls inside a slide's window, and is never required. Slides with no match
still get a full output entry, grounded only in their own visual/OCR data.

This is the shared source-of-truth builder: Step A (the fine-tuned content
model) and Step B (the Ollama Cloud check-and-fill step) both read its
output rather than the two raw files directly.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

# The last slide has no "next slide" timestamp to bound its window, so cap
# how far past its own frame_time it can still claim audio as its own.
MAX_TRAILING_WINDOW_SECONDS = 180.0


def _load(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _audio_entry(block: Dict[str, Any]) -> Dict[str, Any]:
    sa = block.get("semantic_analysis", {}) or {}
    return {
        "block_id": block.get("block_id", ""),
        "timestamp_start": block.get("timestamp_start"),
        "timestamp_end": block.get("timestamp_end"),
        "topic": sa.get("topic", ""),
        "summary": sa.get("summary", ""),
        "key_points": sa.get("key_points", []) or [],
    }


def align(slides_path: str, audio_path: str) -> Dict[str, Any]:
    slides = sorted(_load(slides_path), key=lambda s: s.get("frame_time", 0.0))
    audio_blocks = sorted(_load(audio_path), key=lambda b: b.get("timestamp_start", 0.0))

    aligned = []
    used_audio_ids = set()

    for i, slide in enumerate(slides):
        start = slide.get("frame_time", 0.0)
        if i + 1 < len(slides):
            end = slides[i + 1].get("frame_time", start)
        else:
            end = start + MAX_TRAILING_WINDOW_SECONDS

        matched = [b for b in audio_blocks if start <= b.get("timestamp_start", -1) < end]
        used_audio_ids.update(b.get("block_id") for b in matched)

        sa = slide.get("semantic_analysis", {}) or {}
        aligned.append({
            "slide_number": i + 1,
            "frame_time": start,
            "visual": {
                "visual_topic": sa.get("visual_topic", ""),
                "key_points": sa.get("key_points", []) or [],
                "slide_summary": slide.get("slide_summary", ""),
                "ocr_text": slide.get("ocr_text", ""),
                "module4_usage": sa.get("module4_usage", ""),
            },
            "audio_blocks": [_audio_entry(b) for b in matched],
        })

    orphaned = [b for b in audio_blocks if b.get("block_id") not in used_audio_ids]

    return {
        "slides": aligned,
        "stats": {
            "total_slides": len(slides),
            "total_audio_blocks": len(audio_blocks),
            "slides_with_audio": sum(1 for s in aligned if s["audio_blocks"]),
            "slides_without_audio": sum(1 for s in aligned if not s["audio_blocks"]),
            "orphaned_audio_blocks": [b.get("block_id") for b in orphaned],
        },
    }


# Candidate input filenames, checked in order. Lets a fresh rerun just drop
# lecture_slide.json / lecture_audio.json into input/ without renaming
# anything or touching the original lecture_021 / LecVideo_001 sample files.
SLIDE_FILENAME_CANDIDATES = ["lecture_slide.json", "lecture_021_module3_final_output.json"]
AUDIO_FILENAME_CANDIDATES = ["lecture_audio.json", "LecVideo_001_enriched.json"]


def _resolve_input(input_dir: Path, candidates: List[str]) -> Path:
    for name in candidates:
        path = input_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"None of {candidates} found in {input_dir}"
    )


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    input_dir = repo_root / "input"

    slides_path = _resolve_input(input_dir, SLIDE_FILENAME_CANDIDATES)
    audio_path = _resolve_input(input_dir, AUDIO_FILENAME_CANDIDATES)
    print(f"Using slides file: {slides_path.name}")
    print(f"Using audio file : {audio_path.name}")

    result = align(
        slides_path=str(slides_path),
        audio_path=str(audio_path),
    )

    out_path = repo_root / "input" / "aligned_sources.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    stats = result["stats"]
    print(f"Wrote {out_path}")
    print(f"  {stats['total_slides']} slides, {stats['total_audio_blocks']} audio blocks")
    print(f"  {stats['slides_with_audio']} slides have matching audio, "
          f"{stats['slides_without_audio']} rely on slide-only fallback")
    if stats["orphaned_audio_blocks"]:
        print(f"  Orphaned audio blocks (no slide window matched): "
              f"{stats['orphaned_audio_blocks']}")
