"""Circle scorer (notebooks/circle/03, 04, 07 and 08).

Final method: gradient boosting on my 23 features plus the 93 `circle_scorer` features predicts the total score
(QWK 0.67, accuracy 91.6% in cross-validation). The facets come from the rules of notebook 03, calibrated on all
drawings; where they disagree with the learned total, they are adjusted so that they always add up to it.
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


def fail_ratios(m, params):
    """How far each measurement is from its rule limit: above 1 = the rule fails the facet, below 1 = it passes."""
    turn = m["sharpest_turn"]
    return {
        "basic_shape": 0.0 if math.isnan(turn) else turn / params["corner_angle"],
        "closure": max(m["gap"] / (params["pixels_per_inch"] / 8), m["overlap"] / (params["pixels_per_inch"] / 4)),
        "edges": m["axis_ratio"] / MAX_EDGES_RATIO,
        "overall_size": params["min_size_fraction"] / max(m["size_fraction"], 1e-6),
    }


# Facets other than basic shape, in the order the expert fails them most often in the circle data
# (closure 115, overall size 6, edges 5 failures): the learned total model mainly reflects closure.
FAIL_ORDER = ["closure", "overall_size", "edges"]


def facets_for_total(total, ratios, rule_facets):
    """Facet decisions that add up to the learned total under the manual's basic-shape rule.

    Total 0 means the basic shape failed (every facet is 0) and the maximum means every facet passed. Otherwise the
    basic shape passed, and the rule decisions of the other facets (before the basic-shape rule) are kept as far as
    possible: if the learned total is higher, the failed facets closest to their limits are passed; if it is lower,
    passed facets are failed in the order the expert fails them most often.
    """
    if total == 0:
        return {f: 0 for f in FACETS}
    if total == len(FACETS):
        return {f: 1 for f in FACETS}
    facets = {"basic_shape": 1, **{f: rule_facets[f] for f in FACETS[1:]}}
    while sum(facets.values()) < total:
        facets[min((f for f in FACETS[1:] if not facets[f]), key=lambda f: ratios[f])] = 1
    while sum(facets.values()) > total:
        facets[next(f for f in FAIL_ORDER if facets[f])] = 0
    return facets


def apply_rules(m, params, gate=True):
    """Facet scores of notebook 03 for one drawing's measurements (dict or row); gate=False skips the basic-shape rule."""
    has_corner = m["sharpest_turn"] > params["corner_angle"]  # False when not measurable (open drawing)
    facets = {
        "basic_shape": int(not has_corner),
        "closure": int(m["gap"] <= params["pixels_per_inch"] / 8 and m["overlap"] <= params["pixels_per_inch"] / 4),
        "edges": int(m["axis_ratio"] < MAX_EDGES_RATIO),
        "overall_size": int(m["size_fraction"] >= params["min_size_fraction"]),
    }
    return apply_gate(facets) if gate else facets


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
        rule_facets = apply_rules(ours, params)
        facets = facets_for_total(total, fail_ratios(ours, params), apply_rules(ours, params, gate=False))
        corner = "no closed outline to measure" if math.isnan(ours["sharpest_turn"]) else \
            f"sharpest turn {ours['sharpest_turn']:.0f}° (corner above {params['corner_angle']:.0f}°)"
        details = {
            "basic_shape": corner,
            "closure": f"gap {ours['gap']:.0f} px (limit {params['pixels_per_inch'] / 8:.0f}), "
                       f"overlap {ours['overlap']:.0f} px (limit {params['pixels_per_inch'] / 4:.0f})",
            "edges": f"longest / shortest diameter {ours['axis_ratio']:.2f} (limit {MAX_EDGES_RATIO})",
            "overall_size": f"diameter {ours['size_fraction']:.0%} of the box (at least {params['min_size_fraction']:.0%})",
        }
        changed = [k for k in FACETS if facets[k] != rule_facets[k]]
        notes = []
        if changed:
            notes.append(f"The learned model gives {total}, the rules alone would give {sum(rule_facets.values())}; "
                         f"the facets marked 'learned model' follow the learned total.")
        for k in changed:
            rule = "passes" if rule_facets[k] else "fails"
            if k != "basic_shape" and facets["basic_shape"] == 0:
                details[k] += f"; 0 because the basic shape failed (rule alone: {rule})"
            else:
                details[k] += f"; rule alone: {rule}"
        return ScoreResult(
            shape=self.shape, total=total, max_score=len(FACETS),
            facets=[Facet(k, bool(facets[k]), details[k], "learned model" if k in changed else "rule") for k in FACETS],
            method="Total from gradient boosting on 116 features; facets from the calibrated rules, adjusted to agree "
                   "with the total", mask=mask, notes=notes)
