import os
import json
import argparse
import cv2
import easyocr
from tqdm import tqdm


def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resize 2x to improve OCR on small slide text
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    cv2.imwrite(output_path, gray)

    return output_path


def clean_ocr_results(results, confidence_threshold=0.60):
    clean_text = []

    for bbox, text, confidence in results:
        text = text.strip()

        # Keep only confident OCR results
        if confidence < confidence_threshold:
            continue

        # Remove very short noisy outputs like "0", "7", "S"
        if len(text) < 3:
            continue

        # Remove text that is only symbols/punctuation
        if not any(char.isalpha() for char in text):
            continue

        clean_text.append(text)

    return " ".join(clean_text)


def run_ocr_on_predictions(input_json, output_json, limit=None, confidence_threshold=0.60):
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input JSON not found: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if limit is not None:
        data_to_process = data[:limit]
    else:
        data_to_process = data

    print(f"Total records loaded: {len(data)}")
    print(f"Records to process: {len(data_to_process)}")
    print(f"OCR confidence threshold: {confidence_threshold}")

    reader = easyocr.Reader(["en"], gpu=False)

    debug_dir = "outputs/module3/ocr_debug"
    os.makedirs(debug_dir, exist_ok=True)

    updated_data = []

    for index, item in enumerate(tqdm(data_to_process)):
        label = item.get("label", "Skip")
        frame_path = item.get("frame_path", "")

        # Apply OCR only to Critical and Important frames
        if label in ["Critical", "Important"] and os.path.exists(frame_path):
            try:
                preprocessed_path = os.path.join(debug_dir, f"ocr_temp_{index}.jpg")

                preprocess_image(frame_path, preprocessed_path)

                results = reader.readtext(preprocessed_path, detail=1)

                ocr_text = clean_ocr_results(
                    results,
                    confidence_threshold=confidence_threshold
                )

            except Exception as e:
                print(f"OCR failed for {frame_path}: {e}")
                ocr_text = ""
        else:
            ocr_text = ""

        item["ocr_text"] = ocr_text
        updated_data.append(item)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(updated_data, f, indent=4)

    print(f"OCR output saved to: {output_json}")
    print(f"Total records written: {len(updated_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply OCR to Module 3 prediction JSON.")

    parser.add_argument("--input_json", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.60)

    args = parser.parse_args()

    run_ocr_on_predictions(
        input_json=args.input_json,
        output_json=args.output_json,
        limit=args.limit,
        confidence_threshold=args.threshold
    )