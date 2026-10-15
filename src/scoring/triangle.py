"""Triangle scorer (notebook triangle/02).

Final method: the rules of notebook 02, calibrated on all drawings (QWK 0.70, accuracy 92.6% in cross-validation).
"""
import json

from src import preprocessing as pp
from src import triangle_measurements as tm
from src.scoring.base import MODELS_DIR, Facet, ScoreResult
from src.triangle_preprocessing import extract_ink_triangle


class TriangleScorer:
    shape = "triangle"

    def __init__(self, model_dir=MODELS_DIR / "triangle"):
        self.rules = json.loads((model_dir / "rule_config.json").read_text())

    def score(self, path):
        image = pp.load_image(path)
        mask = pp.normalise_size(extract_ink_triangle(image)[0])
        m, corners = tm.measure_triangle(mask, image)
        r = self.rules
        params = {"pixels_per_inch": r["closure"]["pixels_per_inch"], "max_doubled": r["closure"]["max_doubled_px"],
                  "max_side_ratio": r["edges"]["max_side_ratio"], "max_rotation": r["orientation"]["max_rotation_deg"],
                  "min_size": r["overall_size"]["min_longest_side_fraction_of_box"]}
        facets = tm.score_triangle(m, params)
        details = {
            "basic_shape": f"triangle fit: smallest corner {m['smallest_angle']:.0f}° (at least 25°), "
                           f"{m['on_outline']:.0%} of the line on the outline (at least 75%)",
            "closure": f"overlap {m['overlap']:.0f} px (limit {params['pixels_per_inch'] / 4:.0f}), gap {m['gap']:.0f} px "
                       f"(limit {params['pixels_per_inch'] / 8:.0f}), doubled line {m['doubled_length']:.0f} px "
                       f"(limit {params['max_doubled']:.0f})",
            "edges": f"longest side {m['side_ratio']:.2f} × the shortest (limit {params['max_side_ratio']:.2f})",
            "orientation": f"rotated {m['rotation']:.1f}° (limit {params['max_rotation']:.1f}°), "
                           f"apex {'above' if m['apex_above'] > 0 else 'below'} the base",
            "overall_size": f"longest side {m['size']:.0%} of the box (at least {params['min_size']:.0%})",
        }
        return ScoreResult(
            shape=self.shape, total=facets["total"], max_score=len(tm.FACETS),
            facets=[Facet(k, bool(facets[k]), details[k]) for k in tm.FACETS],
            method="Rules (triangle fit)", mask=mask)
