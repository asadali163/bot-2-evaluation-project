"""Shape detection: which of the eight BOT-2 items a drawing shows (developed in notebooks/shape_detector/).

The drawing is cleaned with the shared preprocessing, cropped to the ink and passed through a frozen EfficientNet-B0
(ImageNet weights); a logistic regression on these features decides the shape. No flips or rotations are used anywhere,
because the Diamond is a rotated Square and the orientation separates other items too.
"""
import json
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from torchvision import models

from src import preprocessing as pp

# Folder in Data/Shapes -> shape key (the same keys as the folders in models/) and display name.
FOLDER_TO_SHAPE = {
    "Circle": "circle",
    "Square": "square",
    "Triangle": "triangle",
    "Diagonal": "diamond",
    "Star": "star",
    "Wave": "wave",
    "Overlapped circle": "overlapped_circle",
    "Overlapped pencils": "overlapped_pencils",
}
SHAPE_NAMES = {
    "circle": "Circle",
    "square": "Square",
    "triangle": "Triangle",
    "diamond": "Diamond",
    "star": "Star",
    "wave": "Wavy line",
    "overlapped_circle": "Overlapping circles",
    "overlapped_pencils": "Overlapping pencils",
}
INPUT_SIZE = 224
MARGIN = 0.08  # margin around the drawing, as a fraction of its larger side
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def detection_input(image):
    """Cleaned ink mask cropped to the drawing, padded to a square and resized (uint8, 1 = ink)."""
    mask, _ = pp.extract_ink(image)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((INPUT_SIZE, INPUT_SIZE), np.uint8)
    margin = int(MARGIN * max(np.ptp(xs), np.ptp(ys)))
    top, left = max(ys.min() - margin, 0), max(xs.min() - margin, 0)
    crop = mask[top:ys.max() + margin + 1, left:xs.max() + margin + 1]
    small = cv2.resize(pp.pad_to_square(crop).astype(np.float32), (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    return (small >= 0.25).astype(np.uint8)


BACKBONE_FILE = Path(__file__).resolve().parents[1] / "models" / "shape_detector" / "efficientnet_b0_imagenet.pth"


def load_backbone():
    """Frozen EfficientNet-B0 without its classification layer (1280 features per image).

    The ImageNet weights (torchvision IMAGENET1K_V1) are stored in models/shape_detector/, so no download is needed;
    if that file is missing, torchvision downloads them.
    """
    if BACKBONE_FILE.exists():
        network = models.efficientnet_b0(weights=None)
        network.load_state_dict(torch.load(BACKBONE_FILE, map_location="cpu"))
    else:
        network = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    network.classifier = torch.nn.Identity()
    return network.eval()


@torch.no_grad()
def embed(backbone, inputs, batch_size=32):
    """Feature vectors for a stack of detection inputs (black ink on white, ImageNet normalisation)."""
    features = []
    for start in range(0, len(inputs), batch_size):
        images = 1.0 - inputs[start:start + batch_size].astype(np.float32)
        images = np.repeat(images[..., None], 3, axis=-1)
        images = (images - IMAGENET_MEAN) / IMAGENET_STD
        features.append(backbone(torch.from_numpy(images.transpose(0, 3, 1, 2).copy())).numpy())
    return np.concatenate(features)


class ShapeDetector:
    """Loads the trained detector from models/shape_detector/ and predicts the shape of one image."""

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        self.classifier = joblib.load(model_dir / "classifier.joblib")
        self.config = json.loads((model_dir / "config.json").read_text())
        self.backbone = load_backbone()

    def predict(self, image):
        """Return (shape key, {shape key: probability}) for a BGR image."""
        features = embed(self.backbone, detection_input(image)[None])
        probabilities = self.classifier.predict_proba(features)[0]
        ranked = dict(sorted(zip(self.classifier.classes_, probabilities.round(4)), key=lambda kv: -kv[1]))
        return next(iter(ranked)), ranked
