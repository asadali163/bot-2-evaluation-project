"""Overlapping-circles-only preprocessing (developed in notebooks/overlapped_circle/01_overlapped_circle_preprocessing.ipynb).

Kept separate so that decisions for this item never change the results of other shapes.

Problem: the shared stray-mark removal keeps any piece of at least 25% of the drawing's size, wherever it lies. In one
photo (img2537) an examiner's mark at the image edge (a circled letter) is that large and survives, far from the drawing.
Fix: a piece that lies far from the drawing (more than 10% of the image away) **and touches the image border** is
dropped, unless it is at least half the drawing's size. The border condition keeps the child's own separate strokes
(e.g. the loose lines of the score-0 drawing img2180).
"""
import cv2
import numpy as np

from src import preprocessing as pp


def remove_distant_border_pieces(mask, max_distance=0.10, border=0.03, keep_if_larger=0.5):
    """Drop pieces far from the largest piece that touch the image border, unless they are at least half its size."""
    n_pieces, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_pieces <= 2:
        return mask, 0
    areas = stats[:, cv2.CC_STAT_AREA]
    main = 1 + np.argmax(areas[1:])  # label 0 is the background
    distance_to_main = cv2.distanceTransform((labels != main).astype(np.uint8), cv2.DIST_L2, 5)
    h, w = mask.shape
    b = int(border * min(h, w))
    cleaned, dropped = mask.copy(), 0
    for i in range(1, n_pieces):
        x, y, bw, bh = stats[i, :4]
        touches_border = x <= b or y <= b or x + bw >= w - b or y + bh >= h - b
        far = distance_to_main[labels == i].min() > max_distance * min(h, w)
        if i != main and far and touches_border and areas[i] < keep_if_larger * areas[main]:
            cleaned[labels == i] = 0
            dropped += 1
    return cleaned, dropped


def extract_ink_overlapped_circle(image):
    """Return (ink mask, colour-only fallback used, number of border pieces dropped)."""
    mask, used_fallback = pp.extract_ink(image)
    mask, dropped = remove_distant_border_pieces(mask)
    return mask, used_fallback, dropped
