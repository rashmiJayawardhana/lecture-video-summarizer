import cv2
import numpy as np
import re


def clean_text(text):
    if not text:
        return ""
    return " ".join(str(text).split())


def detect_chart_type(image_path, debug=None):
    debug = debug or {}

    rectangles = debug.get("rectangles", 0)
    hough_lines = debug.get("hough_lines", 0)
    number_count = debug.get("number_count", 0)

    # Simple first-version chart type detection
    if rectangles >= 5 and number_count >= 2:
        return "bar_chart"

    if hough_lines >= 8 and number_count >= 2:
        return "line_chart"

    # Try circle/pie detection
    image = cv2.imread(image_path)
    if image is not None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=80,
            param1=80,
            param2=40,
            minRadius=30,
            maxRadius=300
        )

        if circles is not None:
            return "pie_chart_or_circular_chart"

    return "unknown_chart"


def extract_possible_labels(ocr_text):
    text = clean_text(ocr_text)

    labels = {
        "title_or_text": text,
        "numbers": re.findall(r"\d+(\.\d+)?%?", text)
    }

    return labels


def extract_graph_content(image_path, ocr_text, debug=None):
    text = clean_text(ocr_text)
    chart_type = detect_chart_type(image_path, debug)
    labels = extract_possible_labels(text)

    if chart_type == "line_chart":
        summary = (
            "The frame appears to contain a line graph. "
            "It may show a trend or change over a sequence such as time, epochs, or ordered categories."
        )

    elif chart_type == "bar_chart":
        summary = (
            "The frame appears to contain a bar chart. "
            "It may compare values across different categories."
        )

    elif chart_type == "pie_chart_or_circular_chart":
        summary = (
            "The frame appears to contain a pie chart or circular chart. "
            "It may show percentage or proportion-based information."
        )

    else:
        summary = (
            "The frame appears to contain a graph or chart, but the exact chart type is uncertain."
        )

    if text:
        summary += f" Extracted chart-related text: {text}"
    else:
        summary += " No clear chart text was extracted."

    return {
        "type": "graph",
        "chart_type": chart_type,
        "content": summary,
        "extracted_labels": labels
    }
