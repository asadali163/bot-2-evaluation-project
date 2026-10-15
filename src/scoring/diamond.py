"""Diamond scorer (notebooks diamond/02 and 03).

Final method: the rules of notebook 02, with the basic-shape check of `diamond_scorer` added, and the random-forest
closure model of notebook 03 (QWK 0.67, accuracy 88.8% in cross-validation).
"""
import json

import joblib

from src import diamond_measurements as dm
from src import preprocessing as pp
from src.diamond_preprocessing import extract_ink_diamond
from src.scoring.base import MODELS_DIR, Facet, ScoreResult, apply_gate, probability_fails
from src.shape_scorers.diamond_scorer import scorer as diamond_scorer

MIN_QUAD_IOU, MAX_PENTAGON_GAIN = 0.80, 0.06  # the basic-shape check of diamond_scorer


class DiamondScorer:
    shape = "diamond"

    def __init__(self, model_dir=MODELS_DIR / "diamond"):
        self.rules = json.loads((model_dir / "rule_config.json").read_text())
        self.ml = json.loads((model_dir / "ml_config.json").read_text())["steps"]["+ ML closure"]
        self.closure_model = joblib.load(model_dir / self.ml["file"])

    def score(self, path):
        image = pp.load_image(path)
        mask = pp.normalise_size(extract_ink_diamond(image)[0])
        m, corners = dm.measure_diamond(mask, image)
        r = self.rules
        params = {"max_gap": r["closure"]["max_gap_px"], "max_overlap": r["closure"]["max_overlap_px"],
                  "max_doubled": r["closure"]["max_doubled_px"], "max_side_ratio": r["edges"]["max_side_ratio"],
                  "max_rotation": r["orientation"]["max_rotation_deg"],
                  "min_size": r["overall_size"]["min_longest_diagonal_fraction_of_box"]}
        facets = dm.score_diamond(m, params)
        rule_basic = facets["basic_shape"]

        theirs = diamond_scorer._feat(str(path))
        quad_iou, pent_iou = theirs.get("quad_iou", float("nan")), theirs.get("pent_iou", float("nan"))
        their_basic = int(quad_iou >= MIN_QUAD_IOU and pent_iou - quad_iou <= MAX_PENTAGON_GAIN)

        features = {**m, **{f"ext_{k}": v for k, v in theirs.items()}}
        probability, failed = probability_fails(self.closure_model, features, self.ml["features"],
                                                self.ml["fail_if_probability_at_least"])
        facets = {k: facets[k] for k in dm.FACETS}
        facets["basic_shape"] = int(rule_basic and their_basic)
        facets["closure"] = int(not failed)
        facets = apply_gate(facets)

        details = {
            "basic_shape": f"four-sided fit: {m['on_outline']:.0%} of the line on the outline (at least 58%), corners "
                           f"{m['smallest_angle']:.0f}°–{m['largest_angle']:.0f}° (30°–150°); four-sided overlap "
                           f"{quad_iou:.2f} (at least {MIN_QUAD_IOU}), gain with five sides {pent_iou - quad_iou:.2f} "
                           f"(at most {MAX_PENTAGON_GAIN})",
            "closure": f"learned model: probability of a closure error {probability:.2f} "
                       f"(fails from {self.ml['fail_if_probability_at_least']:.2f}); overlap {m['overlap']:.0f} px, "
                       f"gap {m['gap']:.0f} px",
            "edges": f"longest side {m['side_ratio']:.2f} × the shortest (limit {params['max_side_ratio']:.2f})",
            "orientation": f"diagonals rotated {m['rotation']:.1f}° (limit {params['max_rotation']:.1f}°)",
            "overall_size": f"longest diagonal {m['size']:.0%} of the box (at least {params['min_size']:.0%})",
        }
        return ScoreResult(
            shape=self.shape, total=sum(facets.values()), max_score=len(dm.FACETS),
            facets=[Facet(k, bool(facets[k]), details[k], "learned model" if k == "closure" else "rule")
                    for k in dm.FACETS],
            method="Rules (four-sided fit) + basic-shape check + learned closure model", mask=mask)
