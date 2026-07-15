def clean_text(text):
    if not text:
        return ""
    return " ".join(str(text).split())


def extract_table_content(image_path, ocr_text, debug=None):
    text = clean_text(ocr_text)
    debug = debug or {}

    h_lines = debug.get("horizontal_lines", 0)
    v_lines = debug.get("vertical_lines", 0)
    intersections = debug.get("intersections", 0)

    content = (
        "The frame appears to contain a table or grid-based comparison. "
        f"Detected structure includes approximately {h_lines} horizontal lines, "
        f"{v_lines} vertical lines, and {intersections} line intersections. "
    )

    if text:
        content += f"Extracted table-related text: {text}"
    else:
        content += "No clear table text was extracted."

    return {
        "type": "table",
        "content": content,
        "table_text": text,
        "structure": {
            "horizontal_lines": h_lines,
            "vertical_lines": v_lines,
            "intersections": intersections
        }
    }
