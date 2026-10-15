"""Circle scorer (notebooks/circle/03, 04, 07 and 08).

Final method: gradient boosting on my 23 features plus the 93 `circle_scorer` features predicts the total score
(QWK 0.67, accuracy 91.6% in cross-validation). The rules of notebook 03, calibrated on all drawings, explain the
facets.
"""
import json
import math

import joblib
import pandas as pd

from src import circle_measurements as cm
from src.scoring.base import MODELS_DIR, Facet, ScoreResult, apply_gate, load_and_clean
from src.shape_scorers.circle_scorer import circlefeat

FACETS = ["basic_shape", "closure", "edges", "overall_size"]
MAX_EDGES_RATIO = 3  # manual: longest diameter less than 3 x the shortest


def apply_rules(m, params):
    """Facet scores of notebook 03 for one drawing's measurements (dict or row)."""
    has_corner = m["sharpest_turn"] > params["corner_angle"]  # False when not measurable (open drawing)
    facets = {
        "basic_shape": int(not has_corner),
        "closure": int(m["gap"] <= params["pixels_per_inch"] / 8 and m["overlap"] <= params["pixels_per_inch"] / 4),
        "edges": int(m["axis_ratio"] < MAX_EDGES_RATIO),
        "overall_size": int(m["size_fraction"] >= params["min_size_fraction"]),
    }
    return apply_gate(facets)


class CircleScorer:
    shape = "circle"

    def __init__(self, model_dir=MODELS_DIR / "circle"):
        self.rules = json.loads((model_dir / "rule_config.json").read_text())
        self.config = json.loads((model_dir / "final_config.json").read_text())
        self.model = joblib.load(model_dir / self.config["model_file"])

    def score(self, path):
        image, mask = load_and_clean(path)
        ours = cm.all_features(image, mask)
        theirs = circlefeat.extract(str(path))
        row = {**{k: ours[k] for k in cm.ML_FEATURES}, **{f"ext_{k}": theirs.get(k) for k in self.config["their_features"]}}
        X = pd.DataFrame([row])[self.config["features"]].astype(float)  # missing values: the model's median imputer
        total = int(self.model.predict(X)[0])

        params = self.rules
        facets = apply_rules(ours, params)
        corner = "no closed outline to measure" if math.isnan(ours["sharpest_turn"]) else \
            f"sharpest turn {ours['sharpest_turn']:.0f}° (corner above {params['corner_angle']:.0f}°)"
        details = {
            "basic_shape": corner,
            "closure": f"gap {ours['gap']:.0f} px (limit {params['pixels_per_inch'] / 8:.0f}), "
                       f"overlap {ours['overlap']:.0f} px (limit {params['pixels_per_inch'] / 4:.0f})",
            "edges": f"longest / shortest diameter {ours['axis_ratio']:.2f} (limit {MAX_EDGES_RATIO})",
            "overall_size": f"diameter {ours['size_fraction']:.0%} of the box (at least {params['min_size_fraction']:.0%})",
        }
        notes = []
        if total != sum(facets.values()):
            notes.append(f"The learned model gives {total}; the facet rules alone would give {sum(facets.values())}.")
        return ScoreResult(
            shape=self.shape, total=total, max_score=len(FACETS),
            facets=[Facet(k, bool(facets[k]), details[k]) for k in FACETS],
            method="Total from gradient boosting on 116 features; facets explained by the calibrated rules",
            mask=mask, notes=notes)
