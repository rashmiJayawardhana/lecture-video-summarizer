import cv2
import numpy as np
import re


def clean_text(text):
    if not text:
        return ""
    return " ".join(str(text).split())


def count_words(text):
    return len(clean_text(text).split())


def count_numbers(text):
    return len(re.findall(r"\d+(\.\d+)?%?", clean_text(text)))


def count_components(mask, min_area=100):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sum(1 for c in contours if cv2.contourArea(c) >= min_area)


def detect_line_structure(gray):
    """
    Detect stronger structural lines only.
    Longer kernels reduce false table detection from normal text, bullets, and slide decoration.
    """
    bw = cv2.adaptiveThreshold(
        ~gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15,
        -2
    )

    h, w = gray.shape

    # Stronger/longer kernels than previous version
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(60, w // 8), 1)
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(60, h // 8))
    )

    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vertical_kernel)

    horizontal_count = count_components(horizontal, min_area=250)
    vertical_count = count_components(vertical, min_area=250)

    intersections = cv2.bitwise_and(horizontal, vertical)
    intersection_count = count_components(intersections, min_area=5)

    horizontal_pixels = int(np.count_nonzero(horizontal))
    vertical_pixels = int(np.count_nonzero(vertical))

    total_pixels = h * w

    horizontal_ratio = horizontal_pixels / total_pixels
    vertical_ratio = vertical_pixels / total_pixels

    return {
        "horizontal_count": horizontal_count,
        "vertical_count": vertical_count,
        "intersection_count": intersection_count,
        "horizontal_pixels": horizontal_pixels,
        "vertical_pixels": vertical_pixels,
        "horizontal_ratio": horizontal_ratio,
        "vertical_ratio": vertical_ratio
    }


def detect_rectangles(gray):
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rect_count = 0

    for c in contours:
        area = cv2.contourArea(c)

        if area < 900:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 4:
            rect_count += 1

    return rect_count


def detect_hough_lines(gray):
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=60,
        maxLineGap=10
    )

    if lines is None:
        return 0

    return len(lines)


def detect_visual_type(image_path, ocr_text=""):
    image = cv2.imread(image_path)

    if image is None:
        return {
            "visual_type": "unknown",
            "detected_visual_types": [],
            "debug": {
                "error": "image_not_found",
                "image_path": image_path
            }
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    line_info = detect_line_structure(gray)

    horizontal_lines = line_info["horizontal_count"]
    vertical_lines = line_info["vertical_count"]
    intersections = line_info["intersection_count"]
    horizontal_ratio = line_info["horizontal_ratio"]
    vertical_ratio = line_info["vertical_ratio"]

    rectangles = detect_rectangles(gray)
    hough_lines = detect_hough_lines(gray)

    text = clean_text(ocr_text)
    word_count = count_words(text)
    number_count = count_numbers(text)

    lower_text = text.lower()

    graph_keywords = [
        "graph", "chart", "axis", "accuracy", "loss", "epoch",
        "precision", "recall", "trend", "rate", "percentage", "%"
    ]

    table_keywords = [
        "table", "row", "column", "accuracy", "precision", "recall",
        "comparison", "model", "value", "confidentiality", "integrity",
        "availability"
    ]

    diagram_keywords = [
        "input", "output", "process", "architecture", "flow", "system",
        "module", "layer", "network", "classification", "pipeline",
        "deep learning", "neural network", "ai"
    ]

    graph_keyword_count = sum(1 for k in graph_keywords if k in lower_text)
    table_keyword_count = sum(1 for k in table_keywords if k in lower_text)
    diagram_keyword_count = sum(1 for k in diagram_keywords if k in lower_text)

    table_score = 0
    graph_score = 0
    diagram_score = 0
    text_score = 0

    # -----------------------------
    # STRICT TABLE DETECTION
    # -----------------------------
    # A real table usually has:
    # - many horizontal + vertical structural lines
    # - many intersections
    # - or clear table/comparison words
    #
    # This avoids classifying normal slide layouts as tables.
    if horizontal_lines >= 4 and vertical_lines >= 4 and intersections >= 20:
        table_score += 3

    if horizontal_lines >= 6 and vertical_lines >= 3 and intersections >= 12 and table_keyword_count >= 1:
        table_score += 2

    if table_keyword_count >= 2 and intersections >= 8:
        table_score += 2

    # If OCR has no useful text, do not classify weak grids as table unless structure is very strong.
    if word_count == 0 and intersections < 25:
        table_score = max(0, table_score - 2)

    # Very tiny line ratio means lines are probably not strong table grid
    if horizontal_ratio < 0.001 and vertical_ratio < 0.001:
        table_score = max(0, table_score - 1)

    # -----------------------------
    # GRAPH DETECTION
    # -----------------------------
    if horizontal_lines >= 1 and vertical_lines >= 1 and hough_lines >= 8:
        graph_score += 2

    if number_count >= 3:
        graph_score += 1

    if graph_keyword_count >= 2:
        graph_score += 2
    elif graph_keyword_count >= 1:
        graph_score += 1

    # Avoid graph if table is very strong
    if table_score >= 3:
        graph_score = max(0, graph_score - 2)

    # -----------------------------
    # DIAGRAM DETECTION
    # -----------------------------
    if rectangles >= 2:
        diagram_score += 2

    if hough_lines >= 12 and word_count >= 1:
        diagram_score += 1

    if diagram_keyword_count >= 2:
        diagram_score += 2
    elif diagram_keyword_count >= 1:
        diagram_score += 1

    # AI / Deep Learning / Neural Network slides are often diagrams or concept slides,
    # not tables, unless table structure is very strong.
    if diagram_keyword_count >= 1 and table_score < 4:
        diagram_score += 1

    # -----------------------------
    # TEXT DETECTION
    # -----------------------------
    if word_count >= 8:
        text_score += 2
    elif word_count >= 3:
        text_score += 1

    detected = []

    if table_score >= 4:
        detected.append("table")

    if graph_score >= 3:
        detected.append("graph")

    if diagram_score >= 3:
        detected.append("diagram")

    if text_score >= 2:
        detected.append("text")

    # If no strong visual structure, default to text if OCR exists
    if len(detected) == 0:
        if word_count > 0:
            detected.append("text")
            visual_type = "text_slide"
        else:
            visual_type = "unknown"

    elif len(detected) >= 2:
        visual_type = "mixed_slide"

    elif detected[0] == "table":
        visual_type = "table_slide"

    elif detected[0] == "graph":
        visual_type = "graph_slide"

    elif detected[0] == "diagram":
        visual_type = "diagram_slide"

    elif detected[0] == "text":
        visual_type = "text_slide"

    else:
        visual_type = "unknown"

    debug = {
        "horizontal_lines": horizontal_lines,
        "vertical_lines": vertical_lines,
        "intersections": intersections,
        "horizontal_ratio": horizontal_ratio,
        "vertical_ratio": vertical_ratio,
        "rectangles": rectangles,
        "hough_lines": hough_lines,
        "word_count": word_count,
        "number_count": number_count,
        "table_score": table_score,
        "graph_score": graph_score,
        "diagram_score": diagram_score,
        "text_score": text_score,
        "graph_keyword_count": graph_keyword_count,
        "table_keyword_count": table_keyword_count,
        "diagram_keyword_count": diagram_keyword_count
    }

    return {
        "visual_type": visual_type,
        "detected_visual_types": detected,
        "debug": debug
    }
