"""Overlapping-pencils scorer (notebooks overlapped_pencils/02 and 05).

Final method: the fine-tuned EfficientNet-B0 of notebook 05 (QWK 0.55, accuracy 63.7% in cross-validation). Each
network gives the probability that each of the six facets earns its point; the total is the most probable total
under the manual's basic-shape rule, exactly as in notebook 05. For a new drawing the ten networks trained in the
cross-validation (5 folds x 2 runs) are averaged. The measurements of notebook 02 are shown as extra information.
"""
import itertools
import json
import math

import cv2
import numpy as np
import torch
from torch import nn
from torchvision import models

from src import overlapped_pencils_measurements as pm
from src import preprocessing as pp
from src.scoring.base import MODELS_DIR, Facet, ScoreResult
from src.shape_scorers.overlapped_pencils_scorer.prep import crop_canvas, preprocess

FACETS = ["basic_shape", "closure", "edges", "orientation", "overlap", "overall_size"]
PENCIL_MEASUREMENTS = ["left_angle", "right_angle", "min_elongation", "edges_ratio", "points_correctly",
                       "overlap_fraction"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def network():
    """EfficientNet-B0 with one output per facet (notebook 05); the weights come from the saved models."""
    model = models.efficientnet_b0(weights=None)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, len(FACETS)))
    return model.eval()


def stroke_image(path, canvas):
    """The cleaned stroke image of notebook 05 (black ink on white, cropped onto a square canvas)."""
    return crop_canvas(preprocess(str(path)), canvas)[0]


@torch.no_grad()
def facet_probabilities(model, image, size, views, canvas):
    """Probability that each facet earns its point, averaged over the test-time views (notebook 05)."""
    images = []
    for angle, zoom in views:
        matrix = cv2.getRotationMatrix2D((canvas / 2, canvas / 2), angle, zoom)
        view = cv2.warpAffine(image, matrix, (canvas, canvas), borderValue=255)
        images.append(cv2.resize(view, (size, size), interpolation=cv2.INTER_AREA))
    batch = torch.from_numpy(np.stack(images).astype(np.float32) / 255)[:, None].repeat(1, 3, 1, 1)
    return torch.sigmoid(model((batch - MEAN) / STD)).numpy().mean(axis=0)


def total_distribution(p):
    """Probability of every total 0..6 from the facet probabilities (basic shape 0 => total 0), as in notebook 05."""
    distribution = np.array([1.0])
    for q in p[1:]:
        distribution = np.convolve(distribution, [1 - q, q])
    total = np.zeros(len(FACETS) + 1)
    total[1:] += p[0] * distribution
    total[0] += 1 - p[0]
    return total


def most_likely_facets(p, total):
    """The most probable facet decisions whose total (with the basic-shape rule) equals the predicted total."""
    best, best_probability = None, -1.0
    for decisions in itertools.product((0, 1), repeat=len(FACETS)):
        gated = decisions if decisions[0] else (0,) * len(FACETS)
        if sum(gated) != total:
            continue
        probability = np.prod([q if d else 1 - q for q, d in zip(p, decisions)])
        if probability > best_probability:
            best, best_probability = gated, probability
    return dict(zip(FACETS, best))


class OverlappedPencilsScorer:
    shape = "overlapped_pencils"

    def __init__(self, model_dir=MODELS_DIR / "overlapped_pencils"):
        self.rules = json.loads((model_dir / "rule_config.json").read_text())
        self.config = json.loads((model_dir / "finetuned_config.json").read_text())
        self.models = []  # (run, input size, network)
        for run, settings in self.config["runs"].items():
            for fold in range(self.config["folds"]):
                model = network()
                model.load_state_dict(torch.load(model_dir / "finetuned" / f"run{run}_fold{fold}.pt", map_location="cpu"))
                self.models.append((run, settings["size"], model))

    def probabilities(self, path, folds=None):
        """Facet probabilities: mean over the runs of the mean over their fold networks (all folds by default)."""
        image = stroke_image(path, self.config["canvas"])
        per_run = {}
        for i, (run, size, model) in enumerate(self.models):
            fold = i % self.config["folds"]
            if folds is None or fold in folds:
                per_run.setdefault(run, []).append(
                    facet_probabilities(model, image, size, self.config["views"], self.config["canvas"]))
        return np.mean([np.mean(v, axis=0) for v in per_run.values()], axis=0)

    def score(self, path):
        p = self.probabilities(path)
        total = int(total_distribution(p).argmax())
        facets = most_likely_facets(p, total)

        image = pp.load_image(path)
        mask = pp.normalise_size(pp.extract_ink(image)[0])
        m, _ = pm.measure_pencils(mask, image)
        m = {**{k: math.nan for k in PENCIL_MEASUREMENTS}, **m}
        split = not math.isnan(m["left_angle"])
        r = self.rules
        measured = {
            "basic_shape": (f"split into two pencils; the thinner one is {m['min_elongation']:.1f} × longer than wide"
                            if split else "the drawing could not be split into two pencils"),
            "closure": f"thickest line {m['thick_ratio']:.1f} × the usual width, longest tail {m['longest_tail']:.0f} px",
            "edges": f"larger pencil {m['edges_ratio']:.2f} × the smaller (rubric: at most 1.5)" if split else "not measurable",
            "orientation": (("left pencil points down and right pencil up" if m["points_correctly"]
                             else "the pencils do not point as in the stimulus") if split else "not measurable"),
            "overlap": (f"overlap {m['overlap_fraction']:.0%} of the smaller pencil (typical "
                        f"{r['overlap']['min_overlap_fraction']:.0%}–{r['overlap']['max_overlap_fraction']:.0%})"
                        if split else "not measurable"),
            "overall_size": f"largest extent {m['size']:.0%} of the box",
        }
        details = {k: f"network: probability that the facet earns its point {p[i]:.2f}; measured: {measured[k]}"
                   for i, k in enumerate(FACETS)}
        return ScoreResult(
            shape=self.shape, total=total, max_score=len(FACETS),
            facets=[Facet(k, bool(facets[k]), details[k], "fine-tuned CNN") for k in FACETS],
            method="Fine-tuned EfficientNet-B0 (10 networks averaged); measurements shown for information",
            mask=mask)
