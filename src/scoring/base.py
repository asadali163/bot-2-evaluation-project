"""Common parts of the per-shape scorers used by the dashboard.

Every scorer loads its final settings and models from models/<shape>/ and scores one image file. The measurement code
is the same code the notebooks used, so a drawing gets the same measurements as in the notebooks.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src import preprocessing as pp

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
FACET_NAMES = {
    "basic_shape": "Basic shape",
    "closure": "Closure",
    "edges": "Edges",
    "orientation": "Orientation",
    "overlap": "Overlap",
    "overall_size": "Overall size",
}


@dataclass
class Facet:
    key: str
    passed: bool
    detail: str  # what was measured and the limit, in words
    decided_by: str = "rule"  # "rule" or "learned model"

    @property
    def name(self):
        return FACET_NAMES[self.key]


@dataclass
class ScoreResult:
    shape: str
    total: int
    max_score: int
    facets: list
    method: str
    mask: np.ndarray  # the cleaned 512 x 512 ink mask that was measured
    notes: list = field(default_factory=list)


def load_and_clean(path, extract=pp.extract_ink):
    """Original image and its cleaned 512 x 512 ink mask (extract = the shape's own ink extraction)."""
    image = pp.load_image(path)
    return image, pp.normalise_size(extract(image)[0])


def apply_gate(facets):
    """Manual rule: if basic shape is 0, every facet (and the total) is 0."""
    if not facets["basic_shape"]:
        return {key: 0 for key in facets}
    return facets


def probability_fails(model, features, columns, limit):
    """Probability that a facet failed (from a saved model) and the decision at the saved limit."""
    import pandas as pd

    row = pd.DataFrame([features]).reindex(columns=columns).astype(float)
    probability = float(model.predict_proba(row)[0, 1])
    return probability, probability >= limit
