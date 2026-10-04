#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model():
    model = models.efficientnet_b0(weights=None)
    features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(features, 2),
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument(
        "--model",
        default="models/luxury_authenticator.pth",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = get_device()
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
    std = checkpoint.get("std", [0.229, 0.224, 0.225])
    image_size = checkpoint.get("image_size", 224)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_path = Path(args.data) / "test"

    if not test_path.exists():
        raise FileNotFoundError(f"Test directory not found: {test_path}")

    test_ds = datasets.ImageFolder(
        test_path,
        transform=transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = build_model()
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    labels = []
    predictions = []
    probabilities = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            labels.extend(targets.numpy().tolist())
            predictions.extend(preds.cpu().numpy().tolist())
            probabilities.extend(probs[:, 1].cpu().numpy().tolist())

    y_true = np.array(labels)
    y_pred = np.array(predictions)
    y_prob = np.array(probabilities)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = float("nan")

    print()
    print("Luxury Authenticator - Evaluation")
    print()
    print(f"Device: {device}")
    print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
    print(f"Best epoch: {checkpoint.get('best_epoch', 'N/A')}")
    print(f"Classes: {test_ds.class_to_idx}")
    print(f"Test images: {len(test_ds)}")
    print()

    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print()
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["fake", "real"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()