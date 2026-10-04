#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model():
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    features = model.classifier[1].in_features

    model.classifier[1] = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(features, 2),
    )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--out",
        default="models/luxury_authenticator.pth",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = get_device()

    print("Luxury Authenticator - Training")
    print()
    print(f"Device: {device}")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.75, 1.0),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    root = Path(args.data)
    train_path = root / "train"
    val_path = root / "val"

    if not train_path.exists():
        raise FileNotFoundError(f"Training directory not found: {train_path}")

    if not val_path.exists():
        raise FileNotFoundError(f"Validation directory not found: {val_path}")

    train_ds = datasets.ImageFolder(
        train_path,
        transform=train_transform,
    )

    val_ds = datasets.ImageFolder(
        val_path,
        transform=eval_transform,
    )

    print(f"Classes: {train_ds.class_to_idx}")
    print(f"Train images: {len(train_ds)}")
    print(f"Val images: {len(val_ds)}")

    workers = 0 if device.type == "mps" else 2

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model().to(device)

    counts = np.bincount(
        train_ds.targets,
        minlength=2,
    ).astype(np.float32)

    class_weights = counts.sum() / np.maximum(counts, 1) / 2.0

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_acc = -1.0
    best_epoch = 0
    no_improvement = 0

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda"):
                    logits = model(images)
                    loss = criterion(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (
                logits.argmax(dim=1) == targets
            ).sum().item()
            train_total += images.size(0)

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)

                logits = model(images)
                loss = criterion(logits, targets)

                val_loss += loss.item() * images.size(0)
                val_correct += (
                    logits.argmax(dim=1) == targets
                ).sum().item()
                val_total += images.size(0)

        train_loss /= train_total
        val_loss /= val_total
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        scheduler.step(val_acc)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.3f} "
            f"lr={lr:.2e}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            no_improvement = 0

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "class_to_idx": train_ds.class_to_idx,
                    "image_size": 224,
                    "mean": mean,
                    "std": std,
                    "best_val_acc": best_acc,
                    "best_epoch": best_epoch,
                },
                output_path,
            )
        else:
            no_improvement += 1

            if no_improvement >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print()
    print(f"Best validation accuracy: {best_acc:.3f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Checkpoint: {output_path}")


if __name__ == "__main__":
    main()