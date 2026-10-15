"""Diamond-only preprocessing (developed in notebooks/diamond/01_diamond_preprocessing.ipynb).

Kept separate so that Diamond-specific decisions never change the results of other shapes.

Problem: the shared stray-mark removal keeps every piece of at least 25% of the drawing's size, wherever it lies. In one
photo (img2148) a remnant of the box frame's corner is that large and survives, far from the drawing. Fix: a piece far
from the drawing (more than 10% of the image away) is dropped unless it is at least half the drawing's size. The pieces
of a genuine diamond (sides separated by small gaps) always lie close to each other.
"""
import cv2
import numpy as np

from src import preprocessing as pp


def remove_distant_pieces(mask, max_distance=0.10, keep_if_larger=0.5):
    """Drop pieces lying far from the largest piece, unless they are at least half its size."""
    n_pieces, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_pieces <= 2:
        return mask, 0
    areas = stats[:, cv2.CC_STAT_AREA]
    main = 1 + np.argmax(areas[1:])  # label 0 is the background
    distance_to_main = cv2.distanceTransform((labels != main).astype(np.uint8), cv2.DIST_L2, 5)
    cleaned, dropped = mask.copy(), 0
    for i in range(1, n_pieces):
        far = distance_to_main[labels == i].min() > max_distance * min(mask.shape)
        if i != main and far and areas[i] < keep_if_larger * areas[main]:
            cleaned[labels == i] = 0
            dropped += 1
    return cleaned, dropped


def extract_ink_diamond(image):
    """Return (ink mask, colour-only fallback used, number of distant pieces dropped)."""
    mask, used_fallback = pp.extract_ink(image)
    mask, dropped = remove_distant_pieces(mask)
    return mask, used_fallback, dropped
