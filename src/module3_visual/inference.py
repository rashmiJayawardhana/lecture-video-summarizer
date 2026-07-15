import os
import json
import argparse
import re
import torch

from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, AutoImageProcessor


LABEL_SCORE = {
    "Critical": 1.0,
    "Important": 0.5,
    "Skip": 0.0
}


def extract_lecture_id(filename):
    match = re.search(r"(lecture_\d+)", filename)
    return match.group(1) if match else "unknown"


def extract_time_from_filename(filename):
    """
    Example:
    lecture_001_frame_00005_00m25s.jpg -> 25.0
    """
    match = re.search(r"_(\d+)m(\d+)s", filename)

    if not match:
        return 0.0

    minutes = int(match.group(1))
    seconds = int(match.group(2))

    return float(minutes * 60 + seconds)


def get_transform(model_dir):
    processor = AutoImageProcessor.from_pretrained(model_dir)

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_mean,
            std=processor.image_std
        )
    ])


def normalize_label(label):
    label = str(label).lower()

    if label == "critical":
        return "Critical"
    if label == "important":
        return "Important"
    if label == "skip":
        return "Skip"

    return "Skip"


def predict_frames(model_dir, frames_dir, output_json):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model folder not found: {model_dir}")

    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Frames folder not found: {frames_dir}")

    model = ViTForImageClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    transform = get_transform(model_dir)

    image_files = sorted([
        f for f in os.listdir(frames_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"Frames directory: {frames_dir}")
    print(f"Total image frames found: {len(image_files)}")

    if len(image_files) == 0:
        raise ValueError("No image frames found in the frames directory.")

    results = []

    with torch.no_grad():
        for filename in image_files:
            image_path = os.path.join(frames_dir, filename)

            image = Image.open(image_path).convert("RGB")
            pixel_values = transform(image).unsqueeze(0).to(device)

            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits

            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_id = torch.max(probabilities, dim=1)

            predicted_id = predicted_id.item()
            confidence = confidence.item()

            raw_label = model.config.id2label.get(
                predicted_id,
                model.config.id2label.get(str(predicted_id), "Skip")
            )

            label = normalize_label(raw_label)

    

            frame_time = extract_time_from_filename(filename)
            lecture_id = extract_lecture_id(filename)

            results.append({
                "lecture_id": lecture_id,
                "frame_time": frame_time,
                "label": label,
                "label_score": LABEL_SCORE[label],
                "confidence": round(confidence, 4),
                "ocr_text": "",
                "frame_path": image_path
            })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"Prediction JSON saved to: {output_json}")
    print(f"Total records written: {len(results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Module 3 ViT inference on extracted lecture frames.")

    parser.add_argument(
        "--model_dir",
        default="models/module3/vit_slide_classifier",
        help="Path to trained ViT model"
    )

    parser.add_argument(
        "--frames_dir",
        required=True,
        help="Path to extracted lecture frames"
    )

    parser.add_argument(
        "--output_json",
        required=True,
        help="Path to save Module 3 output JSON"
    )

    args = parser.parse_args()

    predict_frames(
        model_dir=args.model_dir,
        frames_dir=args.frames_dir,
        output_json=args.output_json
    )