import argparse
import json
import os
import time
from collections import Counter

import gemini_visual_analyzer
from gemini_visual_analyzer import analyze_visual_with_gemini
from visual_type_detector import detect_visual_type


def clean_text(text):
    if not text:
        return ""
    return " ".join(str(text).split())


def build_text_only_analysis(ocr_text, label):
    text = clean_text(ocr_text)

    return {
        "semantic_visual_type": "text",
        "visual_topic": text[:80] if text else "Text slide",
        "visual_explanation": (
            f"This {label} frame contains OCR-based lecture content. "
            f"Extracted text: {text}"
        ),
        "key_points": [text[:180]] if text else [],
        "module4_usage": "Use this frame as supporting visual-text evidence together with audio and topic scores.",
        "confidence": 0.6 if text else 0.3
    }


def should_send_to_gemini(label):
    """
    Final Module 3 rule:

    Critical  -> Gemini API semantic analysis
    Important -> OCR only
    Skip      -> ignored from final output
    """
    label = str(label).strip().lower()
    return label == "critical"


def is_gemini_failure(semantic):
    if not isinstance(semantic, dict):
        return True

    topic = str(semantic.get("visual_topic", "")).lower()
    explanation = str(semantic.get("visual_explanation", "")).lower()
    confidence = semantic.get("confidence", 0)

    if "gemini analysis failed" in topic:
        return True

    if "gemini analysis failed" in explanation:
        return True

    if confidence == 0:
        return True

    return False


def process_file(input_json, output_json, limit=None, sleep_seconds=2.0):
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    # sleep_seconds is the intended per-key spacing. Calls round-robin across
    # however many Gemini keys are configured (GEMINI_API_KEY_FALLBACK), so
    # each individual key is still only hit once every sleep_seconds even
    # though the loop itself moves proportionally faster with more keys.
    num_keys = gemini_visual_analyzer.num_configured_keys()
    effective_sleep = sleep_seconds / num_keys
    if num_keys > 1:
        print(f"{num_keys} Gemini keys configured; using {effective_sleep:.1f}s "
              f"between calls (round-robined) instead of {sleep_seconds}s.")

    with open(input_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    if limit:
        records = records[:limit]

    output_records = []

    for index, item in enumerate(records, start=1):
        frame_path = item.get("frame_path", "")
        ocr_text = item.get("ocr_text", "")
        label = item.get("label", "Unknown")
        label_lower = str(label).strip().lower()

        print(f"\n[{index}/{len(records)}] Processing:")
        print("Frame:", frame_path)
        print("Label:", label)

        # Skip frames should not go to final Module 3 JSON
        if label_lower == "skip":
            print("Skip frame ignored. Not added to final JSON.")
            continue

        if label_lower not in {"critical", "important"}:
            print("Unknown label ignored:", label)
            continue

        if not frame_path or not os.path.exists(frame_path):
            item["semantic_analysis"] = {
                "semantic_visual_type": "unknown",
                "visual_topic": "Missing frame image",
                "visual_explanation": "The original frame image could not be found.",
                "key_points": [],
                "module4_usage": "Do not use this frame for visual semantic analysis.",
                "confidence": 0.0
            }
            item["gemini_analysis_status"] = "frame_missing"
            item["analysis_source"] = "frame_missing"
            output_records.append(item)
            continue

        # Detect visual type if it is not already available
        if "visual_type" not in item:
            result = detect_visual_type(frame_path, ocr_text)
            item["visual_type"] = result["visual_type"]
            item["detected_visual_types"] = result["detected_visual_types"]
            item["visual_debug"] = result["debug"]

        visual_type = item.get("visual_type", "unknown")
        print("Visual type:", visual_type)

        send_gemini = should_send_to_gemini(label)
        print("Send to Gemini:", send_gemini)

        try:
            if send_gemini:
                print("Critical frame detected. Sending to Gemini for semantic visual analysis...")

                semantic = analyze_visual_with_gemini(
                    image_path=frame_path,
                    visual_type=visual_type,
                    ocr_text=ocr_text,
                    label=label
                )

                if is_gemini_failure(semantic):
                    print("Gemini failed. Using OCR fallback for this Critical frame.")
                    item["gemini_analysis_status"] = "failed"
                    item["analysis_source"] = "gemini_failed_ocr_fallback"
                    item["gemini_failure_details"] = semantic
                    semantic = build_text_only_analysis(ocr_text, label)
                else:
                    item["gemini_analysis_status"] = "success"
                    item["analysis_source"] = "gemini"

                time.sleep(effective_sleep)

            else:
                print("Important frame detected. Gemini not used. OCR-only analysis applied.")
                semantic = build_text_only_analysis(ocr_text, label)
                item["gemini_analysis_status"] = "not_used_for_important"
                item["analysis_source"] = "ocr"

        except Exception as e:
            print("Gemini error:", str(e))

            item["gemini_analysis_status"] = "failed"
            item["analysis_source"] = "gemini_failed_ocr_fallback"
            item["gemini_failure_details"] = {
                "error": str(e)
            }

            semantic = build_text_only_analysis(ocr_text, label)

        item["semantic_analysis"] = semantic

        item["visual_elements"] = [
            {
                "type": semantic.get("semantic_visual_type", visual_type),
                "topic": semantic.get("visual_topic", ""),
                "content": semantic.get("visual_explanation", ""),
                "key_points": semantic.get("key_points", [])
            }
        ]

        item["slide_summary"] = semantic.get(
            "visual_explanation",
            item.get("slide_summary", "")
        )

        output_records.append(item)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output_records, f, indent=4, ensure_ascii=False)

    print("\nSaved:", output_json)
    print("Total output records:", len(output_records))
    print("Labels:", Counter(item.get("label", "unknown") for item in output_records))
    print("Visual types:", Counter(item.get("visual_type", "unknown") for item in output_records))
    print("Gemini status:", Counter(item.get("gemini_analysis_status", "unknown") for item in output_records))
    print("Analysis source:", Counter(item.get("analysis_source", "unknown") for item in output_records))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        default="outputs/module3/enhanced_module3_output_v2.json"
    )

    parser.add_argument(
        "--output_json",
        default="outputs/module3/enhanced_module3_output_gemini.json"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0
    )

    args = parser.parse_args()

    process_file(
        input_json=args.input_json,
        output_json=args.output_json,
        limit=args.limit,
        sleep_seconds=args.sleep
    )


if __name__ == "__main__":
    main()
