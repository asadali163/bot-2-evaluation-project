"""Square scorer (notebooks/square/02 and 03).

Final method: the rules of notebook 02 for basic shape, edges, orientation and size, and the random-forest closure
model of notebook 03 (QWK 0.82, accuracy 96.4% in cross-validation).
"""
import json

import cv2
import joblib

from src import preprocessing as pp
from src import square_measurements as sq
from src.scoring.base import MODELS_DIR, Facet, ScoreResult, apply_gate, probability_fails
from src.square_preprocessing import extract_ink_square


class SquareScorer:
    shape = "square"

    def __init__(self, model_dir=MODELS_DIR / "square"):
        self.rules = json.loads((model_dir / "rule_config.json").read_text())
        self.ml = json.loads((model_dir / "ml_config.json").read_text())["closure_model"]
        self.closure_model = joblib.load(model_dir / self.ml["file"])

    def score(self, path):
        image = pp.load_image(path)
        full_mask = extract_ink_square(image)[0]
        mask = pp.normalise_size(full_mask)
        m, corners = sq.measure_square(mask, image)
        r = self.rules
        params = {"pixels_per_inch": r["closure"]["pixels_per_inch"], "max_tilt": r["orientation"]["max_tilt_deg"],
                  "min_side_fraction": r["overall_size"]["min_side_fraction_of_box"]}
        facets = sq.score_square(m, params)

        features = {**m, "pieces": cv2.connectedComponents(full_mask)[0] - 1, "ink_fraction": full_mask.mean()}
        probability, failed = probability_fails(self.closure_model, features, self.ml["features"],
                                                self.ml["fail_if_probability_at_least"])
        facets["closure"] = int(not failed)
        facets = apply_gate({k: facets[k] for k in sq.FACETS})

        details = {
            "basic_shape": f"four-sided fit: {m['on_outline']:.0%} of the line on the outline (at least 75%), "
                           f"corners within {m['corner_deviation']:.0f}° of square (limit 45°)",
            "closure": f"learned model: probability of a closure error {probability:.2f} "
                       f"(fails from {self.ml['fail_if_probability_at_least']:.2f}); overlap {m['overlap']:.0f} px, "
                       f"gap {m['gap']:.0f} px",
            "edges": f"longest side {m['side_ratio']:.2f} × the shortest (limit 1.5)",
            "orientation": f"sides tilted {m['tilt']:.1f}° (limit {params['max_tilt']:.1f}°)",
            "overall_size": f"side {m['side_fraction']:.0%} of the box (at least {params['min_side_fraction']:.0%})",
        }
        return ScoreResult(
            shape=self.shape, total=sum(facets.values()), max_score=len(sq.FACETS),
            facets=[Facet(k, bool(facets[k]), details[k], "learned model" if k == "closure" else "rule")
                    for k in sq.FACETS],
            method="Rules (four-sided fit) + learned closure model", mask=mask)
