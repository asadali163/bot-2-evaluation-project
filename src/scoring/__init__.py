"""Per-shape scorers for the dashboard: the final settings and models in models/<shape>/ applied to one image.

    from src.scoring import load_scorer
    result = load_scorer("square").score("drawing.png")   # ScoreResult: total, facets with reasons, cleaned mask
"""
from importlib import import_module

SCORERS = {
    "circle": ("circle", "CircleScorer"),
    "square": ("square", "SquareScorer"),
    "triangle": ("triangle", "TriangleScorer"),
    "diamond": ("diamond", "DiamondScorer"),
    "star": ("star", "StarScorer"),
    "wave": ("wave", "WaveScorer"),
    "overlapped_circle": ("overlapped_circle", "OverlappedCircleScorer"),
    "overlapped_pencils": ("overlapped_pencils", "OverlappedPencilsScorer"),
}


def load_scorer(shape):
    """The scorer of one shape (shape keys as in models/ and src.shape_detection.SHAPE_NAMES)."""
    module, name = SCORERS[shape]
    return getattr(import_module(f"src.scoring.{module}"), name)()
