"""Star-only preprocessing (developed in notebooks/star/01_star_preprocessing.ipynb).

Kept separate so that Star-specific decisions never change the results of other shapes.

Problem: the shared stray-mark removal keeps a piece only if it lies close to the *largest* piece. Children often draw
the star in several strokes with small gaps between them, so the outline falls apart into a chain of pieces; a point
attached only to a neighbouring piece (not to the largest one) is then deleted (e.g. the top point of img401).
Fix: a piece is kept if it lies close to any piece already kept, repeated until nothing changes ("chaining").
Specks (< 1% of the drawing) and pieces far from every kept piece are still dropped, as in the shared rule.
"""
import cv2
import numpy as np

from src import preprocessing as pp


def remove_stray_marks_chained(mask, near=0.04, speck=0.01, large=0.25):
    """Keep the largest piece, large pieces, and every piece within `near` of a kept piece (repeatedly)."""
    n_pieces, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_pieces <= 2:
        return mask, 0
    areas = stats[:, cv2.CC_STAT_AREA]
    main = 1 + np.argmax(areas[1:])  # label 0 is the background
    keep = np.zeros(n_pieces, bool)
    keep[main] = True
    keep[1:] |= areas[1:] >= large * areas[main]
    candidates = [i for i in range(1, n_pieces) if not keep[i] and areas[i] >= speck * areas[main]]
    while candidates:
        distance_to_kept = cv2.distanceTransform((~keep[labels]).astype(np.uint8), cv2.DIST_L2, 5)
        joined = [i for i in candidates if distance_to_kept[labels == i].min() <= near * min(mask.shape)]
        if not joined:
            break
        keep[joined] = True
        candidates = [i for i in candidates if i not in joined]
    shared = pp.remove_stray_marks(mask)
    cleaned = keep[labels].astype(np.uint8)
    return cleaned, int((cleaned > shared).sum())  # pixels kept that the shared rule would have dropped


def extract_ink_star(image):
    """Return (ink mask, colour-only fallback used, pixels recovered by chaining).

    The shared pipeline (src/preprocessing.extract_ink) with the chained stray-mark removal.
    """
    image = pp.fill_white_patches(image)
    mask, _ = pp.remove_frame(pp.threshold_ink(pp.ink_strength(image)))
    used_fallback = not (pp.MIN_INK_FRACTION <= mask.mean() <= pp.MAX_INK_FRACTION)
    if used_fallback:
        mask, _ = pp.remove_frame(pp.threshold_ink(pp.ink_strength(image, colour_only=True)))
    mask, recovered = remove_stray_marks_chained(pp.remove_border_marks(mask))
    return pp.seal_pores(mask), used_fallback, recovered
