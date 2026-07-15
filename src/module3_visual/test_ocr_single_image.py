import cv2
import easyocr
import argparse
import os


def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Resize image 2x for better OCR
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Save preprocessed image
    cv2.imwrite(output_path, gray)

    return output_path


def run_ocr(image_path, confidence_threshold=0.50):
    reader = easyocr.Reader(["en"], gpu=False)

    results = reader.readtext(image_path, detail=1)

    clean_text = []

    print("\nDetected OCR Text with Confidence:\n")

    for bbox, text, confidence in results:
        if confidence >= confidence_threshold:
            clean_text.append(text)
            print(f"{text}  --> confidence: {confidence:.4f}")

    print("\nClean OCR Text:\n")
    print(" ".join(clean_text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--threshold", type=float, default=0.50)

    args = parser.parse_args()

    output_dir = "outputs/module3/ocr_debug"
    os.makedirs(output_dir, exist_ok=True)

    preprocessed_path = os.path.join(output_dir, "preprocessed_test_image.jpg")

    preprocess_image(args.image, preprocessed_path)

    print(f"Preprocessed image saved to: {preprocessed_path}")

    run_ocr(preprocessed_path, args.threshold)