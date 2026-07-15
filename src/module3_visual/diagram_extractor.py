import re


def clean_text(text):
    if not text:
        return ""
    return " ".join(str(text).split())


def split_possible_labels(text):
    text = clean_text(text)

    if not text:
        return []

    # Simple label extraction from OCR text
    parts = re.split(r"[|,;:\n\r\t]+", text)
    labels = []

    for p in parts:
        p = clean_text(p)
        if len(p) >= 2:
            labels.append(p)

    if len(labels) <= 1:
        words = text.split()
        labels = words[:20]

    return labels[:20]


def extract_diagram_content(image_path, ocr_text, debug=None):
    text = clean_text(ocr_text)
    debug = debug or {}

    rectangles = debug.get("rectangles", 0)
    hough_lines = debug.get("hough_lines", 0)

    labels = split_possible_labels(text)

    if labels:
        label_text = ", ".join(labels)
    else:
        label_text = "No clear diagram labels were extracted."

    content = (
        "The frame appears to contain a diagram, flowchart, architecture, or visual process. "
        f"Detected visual structure includes approximately {rectangles} rectangular regions "
        f"and {hough_lines} line segments. "
        f"Detected labels/text: {label_text}"
    )

    if len(labels) >= 3:
        content += (
            ". This may represent a relationship or process flow between the detected labels."
        )

    return {
        "type": "diagram",
        "content": content,
        "detected_labels": labels,
        "structure": {
            "rectangles": rectangles,
            "hough_lines": hough_lines
        }
    }
