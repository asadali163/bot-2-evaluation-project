"""Square-only preprocessing (developed in notebooks/square/01_square_preprocessing.ipynb).

Kept separate so that Square-specific decisions never change the results of other shapes.
"""
from src import preprocessing as pp

# A square's line covers at least ~2% of the image; less than 1% left after frame removal means the drawing
# was erased together with the frame (the child drew on top of it, e.g. img2265).
MIN_DRAWING_INK = 0.01


def extract_ink_square(image):
    """Return (ink mask, colour-only fallback used, frame kept because removing it erased the drawing)."""
    mask, used_colour_fallback = pp.extract_ink(image)
    if mask.mean() < MIN_DRAWING_INK:
        mask, used_colour_fallback = pp.extract_ink(image, frame_removal=False)
        return mask, used_colour_fallback, True
    return mask, used_colour_fallback, False
