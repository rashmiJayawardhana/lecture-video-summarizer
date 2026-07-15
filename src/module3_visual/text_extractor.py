def clean_text(text):
    if not text:
        return ""
    return " ".join(str(text).split())


def extract_text_content(ocr_text):
    text = clean_text(ocr_text)

    return {
        "type": "text",
        "content": text if text else "No clear text was extracted from this frame."
    }
