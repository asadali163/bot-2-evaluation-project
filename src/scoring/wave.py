"""Wavy-line scorer (notebook wave/02).

Final method: the rules of notebook 02, calibrated on all drawings (QWK 0.68, accuracy 87.1% in cross-validation).
"""
import json

from src import preprocessing as pp
from src import wave_measurements as wm
from src.scoring.base import MODELS_DIR, Facet, ScoreResult, load_and_clean


class WaveScorer:
    shape = "wave"

    def __init__(self, model_dir=MODELS_DIR / "wave"):
        self.rules = json.loads((model_dir / "rule_config.json").read_text())

    def score(self, path):
        _, mask = load_and_clean(path, pp.extract_ink)
        m, line, split = wm.measure_wave(mask)
        r = self.rules
        params = {"min_arc_turn": r["basic_shape"]["arc_counts_if_turning_at_least_deg"],
                  "max_width_ratio": r["edges"]["max_width_ratio"],
                  "max_opening_angle": r["orientation"]["max_mean_opening_angle_from_vertical_deg"],
                  "min_size": r["overall_size"]["min_largest_extent_fraction_of_box"]}
        facets = wm.score_wave(m, params)
        n_arcs = m[f"n_arcs_{params['min_arc_turn']}"]
        details = {
            "basic_shape": f"{n_arcs} arc{'s' if n_arcs != 1 else ''} found (2 needed, turning in opposite directions"
                           f"{'' if m['opposite_arcs'] else ': not the case'}), largest turn "
                           f"{m['largest_arc_turn']:.0f}° (at most {wm.MAX_ARC_TURN}°)",
            "edges": f"wider arc {m['width_ratio']:.2f} × the narrower (limit {params['max_width_ratio']:.2f})",
            "orientation": f"arcs open {m['opening_from_vertical']:.0f}° from vertical (limit "
                           f"{params['max_opening_angle']:.0f}°)",
            "overall_size": f"largest extent {m['size']:.0%} of the box (at least {params['min_size']:.0%})",
        }
        return ScoreResult(
            shape=self.shape, total=facets["total"], max_score=len(wm.FACETS),
            facets=[Facet(k, bool(facets[k]), details[k]) for k in wm.FACETS],
            method="Rules (centre line split into two arcs)", mask=mask)
