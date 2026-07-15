import argparse
import json
import os
from collections import Counter

from visual_type_detector import detect_visual_type
from text_extractor import extract_text_content
from table_extractor import extract_table_content
from graph_extractor import extract_graph_content
from diagram_extractor import extract_diagram_content


def clean_text(text):
    if not text:
        return ""
    return " ".join(str(text).split())


def build_slide_summary(label, visual_type, visual_elements, ocr_text):
    text = clean_text(ocr_text)

    if text:
        short_text = text[:180]
    else:
        short_text = "No clear OCR text was extracted."

    element_types = [e.get("type", "unknown") for e in visual_elements]

    if visual_type == "mixed_slide":
        return (
            f"This {label} frame contains mixed visual content "
            f"({', '.join(element_types)}). Main extracted text: {short_text}"
        )

    if visual_type == "table_slide":
        return (
            f"This {label} frame contains table-based visual information. "
            f"Main extracted text: {short_text}"
        )

    if visual_type == "graph_slide":
        return (
            f"This {label} frame contains a graph or chart. "
            f"Main extracted text: {short_text}"
        )

    if visual_type == "diagram_slide":
        return (
            f"This {label} frame contains diagram-based visual information. "
            f"Main extracted text: {short_text}"
        )

    if visual_type == "text_slide":
        return (
            f"This {label} frame contains text-based lecture content. "
            f"Main extracted text: {short_text}"
        )

    return (
        f"This {label} frame has uncertain visual structure. "
        f"Main extracted text: {short_text}"
    )


def process_record(item):
    frame_path = item.get("frame_path", "")
    ocr_text = item.get("ocr_text", "")
    label = item.get("label", "Unknown")

    result = detect_visual_type(frame_path, ocr_text)

    visual_type = result["visual_type"]
    detected_visual_types = result["detected_visual_types"]
    debug = result["debug"]

    visual_elements = []

    # If mixed, run all detected extractors.
    # If single type, run only that extractor.
    types_to_run = detected_visual_types.copy()

    if visual_type == "text_slide" and "text" not in types_to_run:
        types_to_run.append("text")

    if visual_type == "table_slide" and "table" not in types_to_run:
        types_to_run.append("table")

    if visual_type == "graph_slide" and "graph" not in types_to_run:
        types_to_run.append("graph")

    if visual_type == "diagram_slide" and "diagram" not in types_to_run:
        types_to_run.append("diagram")

    if not types_to_run:
        types_to_run = ["text"] if ocr_text else ["unknown"]

    for t in types_to_run:
        if t == "text":
            visual_elements.append(extract_text_content(ocr_text))

        elif t == "table":
            visual_elements.append(extract_table_content(frame_path, ocr_text, debug))

        elif t == "graph":
            visual_elements.append(extract_graph_content(frame_path, ocr_text, debug))

        elif t == "diagram":
            visual_elements.append(extract_diagram_content(frame_path, ocr_text, debug))

        else:
            visual_elements.append({
                "type": "unknown",
                "content": clean_text(ocr_text)
            })

    item["visual_type"] = visual_type
    item["detected_visual_types"] = detected_visual_types
    item["visual_elements"] = visual_elements
    item["slide_summary"] = build_slide_summary(
        label=label,
        visual_type=visual_type,
        visual_elements=visual_elements,
        ocr_text=ocr_text
    )
    item["extraction_status"] = "success" if visual_type != "unknown" else "low_confidence"
    item["visual_debug"] = debug

    return item


def process_file(input_json, output_json):
    with open(input_json, "r", encoding="utf-8") as f:
        records = json.load(f)

    enhanced = []

    for item in records:
        enhanced.append(process_record(item))

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(enhanced, f, indent=4, ensure_ascii=False)

    print("Enhanced output saved:", output_json)
    print("Total records:", len(enhanced))
    print("Original label distribution:", Counter(item.get("label", "Unknown") for item in enhanced))
    print("Visual type distribution:", Counter(item.get("visual_type", "unknown") for item in enhanced))
    print("Extraction status:", Counter(item.get("extraction_status", "unknown") for item in enhanced))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_json",
        default="outputs/module3/module3_output.json"
    )

    parser.add_argument(
        "--output_json",
        default="outputs/module3/enhanced_module3_output.json"
    )

    args = parser.parse_args()

    process_file(args.input_json, args.output_json)


if __name__ == "__main__":
    main()
