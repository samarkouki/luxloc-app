#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms


class Predictor:
    def __init__(
        self,
        model_path="models/luxury_authenticator.pth",
    ):
        self.device = self.get_device()

        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        self.model = models.efficientnet_b0(weights=None)

        features = self.model.classifier[1].in_features

        self.model.classifier[1] = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(features, 2),
        )

        self.model.load_state_dict(
            checkpoint["state_dict"]
        )

        self.model.to(self.device)
        self.model.eval()

        image_size = checkpoint.get("image_size", 224)
        mean = checkpoint.get(
            "mean",
            [0.485, 0.456, 0.406],
        )
        std = checkpoint.get(
            "std",
            [0.229, 0.224, 0.225],
        )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.class_to_idx = checkpoint.get(
            "class_to_idx",
            {"fake": 0, "real": 1},
        )

        self.idx_to_class = {
            index: name
            for name, index in self.class_to_idx.items()
        }

    @staticmethod
    def get_device():
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def predict(self, image):
        image = image.convert("RGB")
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)

        with torch.no_grad():
            probabilities = torch.softmax(
                self.model(image),
                dim=1,
            )[0]

        fake_probability = probabilities[
            self.class_to_idx["fake"]
        ].item()

        real_probability = probabilities[
            self.class_to_idx["real"]
        ].item()

        if real_probability > fake_probability:
            prediction = "real"
        else:
            prediction = "fake"

        confidence = max(
            fake_probability,
            real_probability,
        )

        if confidence >= 0.80:
            verdict = (
                "AUTHENTIC LIKELY"
                if prediction == "real"
                else "FAKE LIKELY"
            )
        else:
            verdict = "UNCERTAIN"

        return {
            "prediction": prediction,
            "verdict": verdict,
            "fake_probability": fake_probability,
            "real_probability": real_probability,
            "confidence": confidence,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument(
        "--model",
        default="models/luxury_authenticator.pth",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    predictor = Predictor(args.model)

    image = Image.open(image_path)
    result = predictor.predict(image)

    print()
    print("Luxury Authenticator")
    print()
    print(f"Device: {predictor.device}")
    print(f"Prediction: {result['prediction'].upper()}")
    print(f"Verdict: {result['verdict']}")
    print(
        f"Fake probability: "
        f"{result['fake_probability'] * 100:.2f}%"
    )
    print(
        f"Real probability: "
        f"{result['real_probability'] * 100:.2f}%"
    )
    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )
    print()
    print(
        "Result is advisory. Final authentication "
        "must be performed by an expert."
    )


if __name__ == "__main__":
    main()