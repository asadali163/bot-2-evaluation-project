"""Overlapping-circles scorer (notebooks overlapped_circle/02, 03 and 05).

Final method: the rules of notebook 02, the closure and overlap models of notebook 03, and the basic-shape check of
`overlapped_circle_scorer` in place of my basic-shape rule (QWK 0.54, accuracy 77.0% in cross-validation).
As in notebook 05, the rule facets are those of notebook 02 (all 0 when my basic-shape rule fails).
"""
import json
import math

import joblib

from src import overlapped_circle_measurements as om
from src import preprocessing as pp
from src.overlapped_circle_preprocessing import extract_ink_overlapped_circle
from src.scoring.base import MODELS_DIR, Facet, ScoreResult, apply_gate, probability_fails
from src.shape_scorers.overlapped_circle_scorer import features as circle_features
from src.shape_scorers.overlapped_circle_scorer import scorer as circle_scorer

FACETS = ["basic_shape", "closure", "edges", "orientation", "overlap", "overall_size"]
CIRCLE_MEASUREMENTS = ["fit_error", "on_circles", "elongation", "radius_ratio", "area_ratio", "tilt", "overlap_depth",
                       "centre_distance", "gap", "doubled_length", "off_circles_length", "loose_ends", "overlap"]


class OverlappedCircleScorer:
    shape = "overlapped_circle"

    def __init__(self, model_dir=MODELS_DIR / "overlapped_circle"):
        self.rules = json.loads((model_dir / "rule_config.json").read_text())
        steps = json.loads((model_dir / "ml_config.json").read_text())["steps"]
        self.ml = {"closure": steps["+ ML closure"], "overlap": steps["+ ML overlap"]}
        self.models = {facet: joblib.load(model_dir / cfg["file"]) for facet, cfg in self.ml.items()}

    def score(self, path):
        image = pp.load_image(path)
        mask = pp.normalise_size(extract_ink_overlapped_circle(image)[0])
        m, circles = om.measure_circles(mask, image)
        m = {**{k: math.nan for k in CIRCLE_MEASUREMENTS}, **m}  # two circles not fitted: measurements missing
        r = self.rules
        params = {"min_on_circles": r["basic_shape"]["min_on_circles"], "max_gap": r["closure"]["max_gap_px"],
                  "max_overlap": r["closure"]["max_overlap_px"], "max_radius_ratio": r["edges"]["max_radius_ratio"],
                  "max_tilt": r["orientation"]["max_tilt_deg"], "min_depth": r["overlap"]["min_overlap_depth"],
                  "max_depth": r["overlap"]["max_overlap_depth"], "min_size": r["overall_size"]["min_extent_fraction_of_box"]}
        rule = om.score_drawing(m, params)
        facets = {"basic_shape": rule["basic_shape"], "closure": rule["closure"], "edges": rule["edges"],
                  "orientation": rule["orientation"], "overlap": rule["overlap_facet"], "overall_size": rule["overall_size"]}

        probabilities = {}
        for facet, cfg in self.ml.items():
            probabilities[facet], failed = probability_fails(self.models[facet], m, cfg["features"],
                                                             cfg["fail_if_probability_at_least"])
            facets[facet] = int(not failed)
        theirs = circle_features.extract(str(path))
        facets["basic_shape"] = int(circle_scorer.shape_ok(theirs))
        facets = apply_gate(facets)

        fitted = not math.isnan(m["fit_error"])
        details = {
            "basic_shape": f"two-ellipse check: {theirs.get('frac_inlier', math.nan):.0%} of the ink on two ellipses "
                           f"(at least 87%), no pronounced corner" if theirs.get("fit_ok") else "two circles could not be fitted",
            "closure": f"learned model: probability of a closure error {probabilities['closure']:.2f} (fails from "
                       f"{self.ml['closure']['fail_if_probability_at_least']:.2f}); doubled line {m['doubled_length']:.0f} px",
            "edges": f"larger circle {m['radius_ratio']:.2f} × the smaller radius (limit {params['max_radius_ratio']:.2f}), "
                     f"most stretched circle {m['elongation']:.2f} (limit 3)" if fitted else "not measurable",
            "orientation": f"centres {m['tilt']:.0f}° from vertical (limit {params['max_tilt']:.1f}°)" if fitted else "not measurable",
            "overlap": f"learned model: probability of an overlap error {probabilities['overlap']:.2f} (fails from "
                       f"{self.ml['overlap']['fail_if_probability_at_least']:.2f}); overlap depth {m['overlap_depth']:.2f} "
                       f"(typical {params['min_depth']:.2f}–{params['max_depth']:.2f})",
            "overall_size": f"largest extent {m['size']:.0%} of the box (at least {params['min_size']:.0%})",
        }
        decided = {"basic_shape": "basic-shape check", "closure": "learned model", "overlap": "learned model"}
        return ScoreResult(
            shape=self.shape, total=sum(facets.values()), max_score=len(FACETS),
            facets=[Facet(k, bool(facets[k]), details[k], decided.get(k, "rule")) for k in FACETS],
            method="Rules (two fitted circles) + learned closure and overlap models + two-ellipse basic-shape check",
            mask=mask)
