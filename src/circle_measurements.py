"""Geometric measurements of a circle drawing, made on its cleaned 512 x 512 ink mask.

Developed and tested in notebooks/circle/03_circle_rule_based.ipynb, which explains each measurement with examples.
"""
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

MAX_GAP = 120  # close_gaps() gives up at this size: the drawing is very open
NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def load_mask(mask_dir, file):
    """Load a cleaned ink mask saved by notebook 02 (1 = ink, 0 = paper)."""
    mask = cv2.imread(str(Path(mask_dir) / f"{Path(file).stem}.png"), cv2.IMREAD_GRAYSCALE)
    return (mask > 0).astype(np.uint8)


def disk(radius):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


# ---------------------------------------------------------------- closure


def enclosed_region(mask, min_share=0.2):
    """The largest region of paper enclosed by ink, or None if there is no substantial one."""
    filled = ndimage.binary_fill_holes(mask)
    holes = (filled & ~mask.astype(bool)).astype(np.uint8)
    n, regions, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    if n <= 1:
        return None
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    region = regions == largest

    ys, xs = np.nonzero(mask)
    hull_area = cv2.contourArea(cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32)))
    return region if region.sum() >= min_share * hull_area else None


def close_gaps(mask, max_radius=MAX_GAP // 2):
    """Return (gap size in pixels, mask with the gap closed)."""
    for radius in range(max_radius + 1):
        closed = mask if radius == 0 else cv2.morphologyEx(mask, cv2.MORPH_CLOSE, disk(radius))
        if enclosed_region(closed) is not None:
            return 2 * radius, closed
    return 2 * max_radius, closed  # never encloses a region: a very open drawing


def count_neighbours(skeleton):
    s = skeleton.astype(np.uint8)
    return cv2.filter2D(s, -1, NEIGHBOURS, borderType=cv2.BORDER_CONSTANT) * s


def branch_lengths(skeleton):
    """Length of the branch from each skeleton endpoint to the first junction."""
    neighbours = count_neighbours(skeleton)
    h, w = skeleton.shape
    lengths = []
    for start in zip(*np.nonzero(neighbours == 1)):  # endpoints have exactly one neighbour
        visited = {start}
        y, x = start
        length = 0.0
        while neighbours[y, x] < 3:  # stop at a junction
            steps = [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                     if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w
                     and skeleton[y + dy, x + dx] and (y + dy, x + dx) not in visited]
            if len(steps) != 1:
                break
            ny, nx = steps[0]
            length += np.hypot(ny - y, nx - x)
            visited.add((ny, nx))
            y, x = ny, nx
        lengths.append(length)
    return lengths


def overlap_parts(closed):
    """Return (skeleton, stroke width, retraced skeleton pixels)."""
    skeleton = skeletonize(closed > 0)
    local_width = 2 * cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    stroke_width = float(np.median(local_width[skeleton]))
    retraced = skeleton & (local_width > 1.5 * stroke_width)
    return skeleton, stroke_width, retraced


def measure_overlap(closed):
    skeleton, stroke_width, retraced = overlap_parts(closed)
    tails = [length for length in branch_lengths(skeleton) if length >= 1.5 * stroke_width]  # ignore tiny spurs
    longest_tail = max(tails, default=0.0)
    retraced_length = float(retraced.sum())
    return {"stroke_width": stroke_width, "longest_tail": longest_tail,
            "retraced_length": retraced_length, "overlap": longest_tail + retraced_length}


# ---------------------------------------------------------------- shape


def shape_body(closed, radius=12):
    """The filled drawing without thin protrusions (tails, slits)."""
    filled = ndimage.binary_fill_holes(closed).astype(np.uint8)
    body = cv2.morphologyEx(filled, cv2.MORPH_OPEN, disk(radius))
    return body if body.sum() >= 0.5 * filled.sum() else filled  # very open drawings have no body


def outline_points(body, n_points=200):
    """n_points equally spaced points along the outline of the body."""
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=cv2.contourArea)[:, 0, :].astype(float)
    points = np.vstack([points, points[:1]])
    distance = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))])
    targets = np.linspace(0, distance[-1], n_points, endpoint=False)
    return np.column_stack([np.interp(targets, distance, points[:, 0]), np.interp(targets, distance, points[:, 1])])


def turning_angles(points, window=8):
    """Change of direction (degrees) at each outline point, over `window` points before and after."""
    before = points - np.roll(points, window, axis=0)
    after = np.roll(points, -window, axis=0) - points
    cross = before[:, 0] * after[:, 1] - before[:, 1] * after[:, 0]
    dot = (before * after).sum(axis=1)
    return np.abs(np.degrees(np.arctan2(cross, dot)))


def measure_ellipse(body):
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    (_, _), (axis_1, axis_2), _ = cv2.fitEllipse(max(contours, key=cv2.contourArea))
    long_axis, short_axis = max(axis_1, axis_2), min(axis_1, axis_2)
    diameter = (long_axis + short_axis) / 2
    return {"axis_ratio": long_axis / max(short_axis, 1), "diameter": diameter,
            "size_fraction": diameter / body.shape[1]}


def measure_drawing(mask):
    """All measurements used by the rule-based method."""
    gap, closed = close_gaps(mask)
    body = shape_body(closed)
    encloses_shape = gap < MAX_GAP
    return {
        "gap": gap,
        **measure_overlap(closed),
        "sharpest_turn": float(turning_angles(outline_points(body)).max()) if encloses_shape else np.nan,
        **measure_ellipse(body),
    }


# ---------------------------------------------------------------- extra features for machine learning
# Developed in notebooks/circle/04_circle_machine_learning.ipynb.

ML_FEATURES = [
    "gap", "longest_tail", "retraced_length", "overlap", "n_tails", "n_tips", "tip_pair_distance",
    "retraced_length_13", "retraced_length_18", "extra_length", "doubled_ink_length", "n_pieces",
    "sharpest_turn", "n_sharp_points", "turning_std", "circularity", "solidity", "ellipse_residual",
    "axis_ratio", "size_fraction", "width_fraction", "height_fraction", "stroke_width",
]


def skeleton_length(skeleton):
    """Length of a 1-pixel centre line: straight neighbour steps count 1, diagonal steps sqrt(2)."""
    s = skeleton.astype(np.uint8)
    straight = (s[:, 1:] & s[:, :-1]).sum() + (s[1:, :] & s[:-1, :]).sum()
    diagonal = (s[1:, 1:] & s[:-1, :-1]).sum() + (s[1:, :-1] & s[:-1, 1:]).sum()
    return float(straight + np.sqrt(2) * diagonal)


def ink_on_mask_grid(image, mask):
    """Ink strength of the original image, brought onto the mask grid, only around the drawing."""
    from src import preprocessing as pp

    strength = cv2.resize(pp.pad_to_square(pp.ink_strength(pp.fill_white_patches(image))), mask.shape[::-1],
                          interpolation=cv2.INTER_AREA)
    return strength * cv2.dilate(mask, np.ones((5, 5), np.uint8))


def closure_features(image, mask):
    gap, closed = close_gaps(mask)
    skeleton = skeletonize(closed > 0)
    local_width = 2 * cv2.distanceTransform(closed, cv2.DIST_L2, 5)
    width = float(np.median(local_width[skeleton]))
    relative_width = local_width[skeleton] / width

    branches = np.array(branch_lengths(skeleton))
    tips = np.column_stack(np.nonzero(count_neighbours(skeleton) == 1))
    tip_pair_distance = 0.0
    if len(tips) >= 2:
        distances = np.hypot(*(tips[:, None, :] - tips[None, :, :]).transpose(2, 0, 1))
        np.fill_diagonal(distances, np.inf)
        near = distances[distances <= 120]  # tips of an overlap lie close together
        tip_pair_distance = float(near.max()) if near.size else 0.0

    total_length = skeleton_length(skeleton)
    region = enclosed_region(closed)
    if region is None:
        loop_length = total_length
    else:
        contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        loop_length = cv2.arcLength(max(contours, key=len), True) + np.pi * width

    ink = ink_on_mask_grid(image, mask)
    r = max(int(round(width)), 3)
    ink_amount = cv2.blur(ink, (2 * r + 1, 2 * r + 1))[skeleton]  # ink around each point of the centre line
    ink_ratio = ink_amount / max(np.median(ink_amount), 1e-6)

    return {
        "n_tails": int((branches >= 1.5 * width).sum()),
        "n_tips": int(len(tips)),
        "tip_pair_distance": tip_pair_distance,
        "retraced_length_13": float((relative_width > 1.3).sum()),
        "retraced_length_18": float((relative_width > 1.8).sum()),
        "extra_length": max(total_length - loop_length, 0.0),
        "doubled_ink_length": float((ink_ratio > 1.5).sum()),
        "n_pieces": cv2.connectedComponents(mask)[0] - 1,
    }


def shape_features(mask):
    _, closed = close_gaps(mask)
    body = shape_body(closed)
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    outline = max(contours, key=cv2.contourArea)
    area, perimeter = cv2.contourArea(outline), cv2.arcLength(outline, True)
    angles = turning_angles(outline_points(body))

    # Distance of the outline from the fitted ellipse (0 = exactly on it), in ellipse-radius units
    (cx, cy), (axis_1, axis_2), angle = cv2.fitEllipse(outline)
    points = outline[:, 0, :].astype(float) - [cx, cy]
    t = np.radians(angle)
    u = points[:, 0] * np.cos(t) + points[:, 1] * np.sin(t)
    v = -points[:, 0] * np.sin(t) + points[:, 1] * np.cos(t)

    ys, xs = np.nonzero(mask)
    return {
        "n_sharp_points": int((angles > 60).sum()),
        "turning_std": float(angles.std()),
        "circularity": 4 * np.pi * area / perimeter**2,
        "solidity": area / cv2.contourArea(cv2.convexHull(outline)),
        "ellipse_residual": float(np.mean(np.abs(np.hypot(u / (axis_1 / 2), v / (axis_2 / 2)) - 1))),
        "width_fraction": (xs.max() - xs.min() + 1) / mask.shape[1],
        "height_fraction": (ys.max() - ys.min() + 1) / mask.shape[0],
    }


def all_features(image, mask):
    """Rule measurements plus the extra machine-learning features (original image and its 512 x 512 mask)."""
    return {**measure_drawing(mask), **closure_features(image, mask), **shape_features(mask)}
