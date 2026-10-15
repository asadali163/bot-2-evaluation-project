"""Star scorer (notebook star/02).

Final method: the rules of notebook 02, calibrated on all drawings (QWK 0.67, accuracy 63.7% in cross-validation).
"""
import json
import math

from src import preprocessing as pp
from src import star_measurements as sm
from src.scoring.base import MODELS_DIR, Facet, ScoreResult
from src.star_preprocessing import extract_ink_star

STAR_MEASUREMENTS = ["length_ratio", "width_ratio", "edges_ratio", "rotation", "on_outline", "gap", "overlap"]


class StarScorer:
    shape = "star"

    def __init__(self, model_dir=MODELS_DIR / "star"):
        self.rules = json.loads((model_dir / "rule_config.json").read_text())

    def score(self, path):
        image = pp.load_image(path)
        mask = pp.normalise_size(extract_ink_star(image)[0])
        m, points = sm.measure_star(mask, image)
        m = {**{k: math.nan for k in STAR_MEASUREMENTS}, **m}  # five tips not found: measurements missing
        r = self.rules
        no_limit = lambda v: math.inf if v is None else v
        params = {"min_on_outline": r["basic_shape"]["min_on_outline"],
                  "max_gap": no_limit(r["closure"]["max_gap_px"]), "max_overlap": no_limit(r["closure"]["max_overlap_px"]),
                  "max_edges_ratio": r["edges"]["max_edges_ratio"], "max_rotation": r["orientation"]["max_rotation_deg"],
                  "min_size": r["overall_size"]["min_extent_fraction_of_box"]}
        facets = sm.score_star(m, params)
        five_tips = not math.isnan(m["length_ratio"])
        details = {
            "basic_shape": (f"five points found, {m['on_outline']:.0%} of the line on the star outline "
                            f"(at least {params['min_on_outline']:.0%})") if five_tips else "five separate points not found",
            "closure": f"gap {m['gap']:.0f} px (limit {params['max_gap']:.0f}), overlap {m['overlap']:.0f} px "
                       f"(limit {params['max_overlap']:.0f})",
            "edges": f"largest point {m['edges_ratio']:.2f} × the smallest (limit {params['max_edges_ratio']:.2f})",
            "orientation": f"rotated {m['rotation']:.1f}° from one point straight up (limit {params['max_rotation']:.1f}°)",
            "overall_size": f"largest extent {m['size']:.0%} of the box (at least {params['min_size']:.0%})",
        }
        return ScoreResult(
            shape=self.shape, total=facets["total"], max_score=len(sm.FACETS),
            facets=[Facet(k, bool(facets[k]), details[k]) for k in sm.FACETS],
            method="Rules (five tips and five inner corners)", mask=mask)
