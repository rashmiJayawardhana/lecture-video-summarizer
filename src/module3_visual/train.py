import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

from transformers import ViTForImageClassification, AutoImageProcessor
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


def get_transforms(model_name):
    processor = AutoImageProcessor.from_pretrained(model_name)

    image_mean = processor.image_mean
    image_std = processor.image_std

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_mean, std=image_std)
    ])


def evaluate(model, dataloader, device, class_names):
    model.eval()

    all_preds = []
    all_labels = []

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total if total > 0 else 0

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0
    )

    return accuracy, report, all_labels, all_preds


def train(args):
    model_name = args.model_name
    data_dir = args.data_dir
    output_dir = args.output_dir

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("outputs/results/module3", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    transform = get_transforms(model_name)

    train_dataset = ImageFolder(train_dir, transform=transform)
    val_dataset = ImageFolder(val_dir, transform=transform)
    test_dataset = ImageFolder(test_dir, transform=transform)

    class_names = train_dataset.classes

    print("Class names:", class_names)
    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    id2label = {
        0: "Critical",
        1: "Important",
        2: "Skip"
    }

    label2id = {
        "Critical": 0,
        "Important": 1,
        "Skip": 2
    }

    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )

    # For first prototype, freeze the ViT backbone.
    # This makes training faster on CPU.
    for param in model.vit.parameters():
        param.requires_grad = False

    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate
    )

    loss_fn = nn.CrossEntropyLoss()

    best_val_accuracy = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        for images, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(pixel_values=images)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        val_accuracy, val_report, _, _ = evaluate(
            model,
            val_loader,
            device,
            class_names
        )

        print(f"Training Loss: {avg_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("\nValidation Report:")
        print(val_report)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            model.save_pretrained(output_dir)
            processor = AutoImageProcessor.from_pretrained(model_name)
            processor.save_pretrained(output_dir)

            print(f"Best model saved to: {output_dir}")

    print("\nTraining completed.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")

    print("\nEvaluating on test set...")

    test_accuracy, test_report, y_true, y_pred = evaluate(
        model,
        test_loader,
        device,
        class_names
    )

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nTest Classification Report:")
    print(test_report)

    report_path = "outputs/results/module3/classification_report.txt"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n\n")
        f.write(test_report)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    disp.plot()
    plt.title("Module 3 ViT Classification Confusion Matrix")
    plt.savefig("outputs/results/module3/confusion_matrix.png")
    plt.close()

    print(f"Classification report saved to: {report_path}")
    print("Confusion matrix saved to: outputs/results/module3/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT-base for Module 3 slide classification.")

    parser.add_argument(
        "--data_dir",
        default="data/datasets/module3",
        help="Path to Module 3 dataset folder"
    )

    parser.add_argument(
        "--output_dir",
        default="models/module3/vit_slide_classifier",
        help="Path to save trained model"
    )

    parser.add_argument(
        "--model_name",
        default="google/vit-base-patch16-224-in21k",
        help="Pretrained ViT model name"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )

    args = parser.parse_args()

    train(args)