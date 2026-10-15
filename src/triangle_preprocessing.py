"""Triangle-only preprocessing (developed in notebooks/triangle/01_triangle_preprocessing.ipynb).

Kept separate so that Triangle-specific decisions never change the results of other shapes.

Problem: when a child draws the base of the triangle exactly on the printed frame line, the frame removal erases the
base, and only the two upper sides ("^") remain. Fix: if a stretch of erased frame runs between the two lowest loose
ends of the drawing, that stretch is the child's base and is put back.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

from src import preprocessing as pp

NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def loose_ends(mask):
    """(x, y) of the skeleton end points of an ink mask."""
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    neighbours = cv2.filter2D(skeleton, -1, NEIGHBOURS, borderType=cv2.BORDER_CONSTANT) * skeleton
    ys, xs = np.nonzero(neighbours == 1)
    return np.column_stack([xs, ys])


def restore_base_on_frame(mask, erased, tolerance=0.03):
    """Put back erased frame ink that joins the two lowest loose ends of the drawing.

    mask: cleaned drawing; erased: ink pixels removed as frame. Returns (mask, restored?).
    """
    ends = loose_ends(mask)
    if len(ends) < 2 or not erased.any():
        return mask, False
    side = min(mask.shape)
    near = int(tolerance * side)
    lowest = ends[np.argsort(ends[:, 1])[-2:]]  # the two lowest loose ends
    (x1, y1), (x2, y2) = sorted(lowest.tolist())
    if abs(y1 - y2) > 3 * near or x2 - x1 < 0.2 * side:  # not two ends of a missing base
        return mask, False
    band = np.zeros_like(mask)
    top, bottom = max(min(y1, y2) - near, 0), min(max(y1, y2) + near, mask.shape[0] - 1)
    band[top:bottom + 1, max(x1 - near, 0):x2 + near + 1] = 1
    base = erased * band
    columns = np.nonzero(base.any(axis=0))[0]
    if len(columns) < 0.8 * (x2 - x1):  # the erased line must run (almost) all the way between the ends
        return mask, False
    return np.maximum(mask, base), True


def extract_ink_triangle(image):
    """Return (ink mask, colour-only fallback used, base restored from the frame line)."""
    image = pp.fill_white_patches(image)
    raw = pp.threshold_ink(pp.ink_strength(image))
    used_fallback = False
    mask, frame = pp.remove_frame(raw)
    if not pp.MIN_INK_FRACTION <= mask.mean() <= pp.MAX_INK_FRACTION:
        used_fallback = True
        raw = pp.threshold_ink(pp.ink_strength(image, colour_only=True))
        mask, frame = pp.remove_frame(raw)
    mask = pp.seal_pores(pp.remove_stray_marks(pp.remove_border_marks(mask)))
    mask, restored = restore_base_on_frame(mask, raw * frame)
    if restored:
        mask = pp.seal_pores(mask)
    return mask, used_fallback, restored
