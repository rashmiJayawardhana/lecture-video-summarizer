import argparse
import json
import os
import time

from google import genai

# ── Settings ──────────────────────────────────────────────
DEFAULT_INPUT_FILE  = "src/module2_summarization/final_output/LecVideo_001_module2_final.json"
DEFAULT_OUTPUT_FILE = "src/module2_summarization/final_output/LecVideo_001_enriched.json"
DEFAULT_MIN_RATIO    = 0.7
DEFAULT_MODEL_NAME   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_GAP_THRESHOLD = 30.0
DEFAULT_SLEEP_S       = 4.5
# ──────────────────────────────────────────────────────────


def group_segments(segs: list, gap_threshold: float = DEFAULT_GAP_THRESHOLD) -> list:
    if not segs:
        return []
    blocks = [[segs[0]]]
    for seg in segs[1:]:
        last_end = blocks[-1][-1]["timestamp_end"]
        if seg["timestamp_start"] - last_end <= gap_threshold:
            blocks[-1].append(seg)
        else:
            blocks.append([seg])
    return blocks


def analyze_block(client, model_name: str, block: list) -> dict:
    all_sentences = []
    for seg in block:
        for s in seg["sentences"]:
            if s["is_important"]:
                all_sentences.append(s["sentence"])

    text = " ".join(all_sentences)

    prompt = f"""Analyze this lecture excerpt from an IT/database course. Return ONLY a valid JSON object with these fields:
- topic: short topic name (max 8 words)
- summary: one clean sentence describing the concept
- key_points: array of 2-4 short bullet points

Lecture text:
{text}

Return ONLY the JSON object, no markdown, no code blocks, no explanation."""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        result_text = response.text.strip()

        # Clean markdown if present
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]

        return json.loads(result_text.strip())
    except Exception as e:
        return {
            "topic": "Unknown",
            "summary": "Error processing block",
            "key_points": [],
            "error": str(e)
        }


def enrich_segments(
    segments: list,
    api_key: str | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    min_ratio: float = DEFAULT_MIN_RATIO,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    sleep_s: float = DEFAULT_SLEEP_S,
) -> list:
    """
    Group high-importance segments into topic blocks and enrich each with a
    Gemini-generated topic/summary/key_points. Called lazily by both the CLI
    and the API so importing this module never requires GEMINI_API_KEY to be set.
    """
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not found. Add it to your .env file.")

    client = genai.Client(api_key=api_key)

    important_segments = [s for s in segments if s["importance_ratio_T"] >= min_ratio]
    blocks = group_segments(important_segments, gap_threshold)

    enriched = []
    for i, block in enumerate(blocks):
        block_start = block[0]["timestamp_start"]
        block_end   = block[-1]["timestamp_end"]

        analysis = analyze_block(client, model_name, block)

        enriched.append({
            "block_id"        : f"block_{i+1:03d}",
            "timestamp_start" : block_start,
            "timestamp_end"   : block_end,
            "segments_count"  : len(block),
            "segment_ids"     : [s["segment_id"] for s in block],
            "semantic_analysis": analysis
        })

        if i < len(blocks) - 1:
            time.sleep(sleep_s)

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich high-importance segments with Gemini topic/summary blocks.")
    parser.add_argument("--input_json", default=DEFAULT_INPUT_FILE, help="Classified segments JSON (classify_with_features.py output).")
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_FILE, help="Where to write the enriched blocks JSON.")
    parser.add_argument("--min_ratio", type=float, default=DEFAULT_MIN_RATIO, help="Minimum importance_ratio_T for a segment to be enriched.")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Gemini model name.")
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        segments = json.load(f)

    print(f"Total segments: {len(segments)}")

    enriched = enrich_segments(
        segments,
        model_name=args.model,
        min_ratio=args.min_ratio,
    )

    print(f"Grouped into {len(enriched)} topic blocks")
    for block in enriched:
        print(f"   [{block['block_id']}] {block['semantic_analysis'].get('topic', 'N/A')}")

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    print("=" * 50)
    print("✅ GEMINI ENRICHMENT COMPLETE!")
    print(f"Total blocks: {len(enriched)}")
    print(f"Saved to: {args.output_json}")


if __name__ == "__main__":
    main()
