"""Overlapping-circles measurements and rules, on the cleaned 512 x 512 ink mask.

Developed and explained in notebooks/overlapped_circle/02_overlapped_circle_rule_based.ipynb (copied here unchanged,
except that measure_circles takes the mask and the original image instead of a file name). The overlap facet is called
"overlap_facet" here, because "overlap" is the closure measurement (a line drawn twice).
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

from src import preprocessing as pp

SIZE = 512  # masks are 512 x 512 pixels (the whole answer box)
FACET_COLUMNS = ["basic_shape", "closure", "edges", "orientation", "overlap_facet", "overall_size"]


NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
def pruned_skeleton(mask, steps=6):
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    for _ in range(steps):
        ends = (cv2.filter2D(skeleton, -1, NEIGHBOURS, borderType=cv2.BORDER_CONSTANT) * skeleton) == 1
        if not ends.any():
            break
        skeleton[ends] = 0
    return skeleton


def fit_circle(points):
    """Least-squares circle through points (algebraic fit): returns centre (x, y) and radius."""
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([x, y, np.ones(len(x))])
    (a, b, c), *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    centre = np.array([a / 2, b / 2])
    return centre, float(np.sqrt(max(c + centre @ centre, 1.0)))


def fit_two_circles(points, rounds=15):
    """Split the points along their longest direction, fit a circle to each half, then repeatedly reassign every point
    to the circle it lies closest to (distance to the circle line) and refit."""
    centred = points - points.mean(axis=0)
    direction = np.linalg.svd(centred, full_matrices=False)[2][0]
    side = centred @ direction > 0
    circles = []
    for _ in range(rounds):
        if side.sum() < 10 or (~side).sum() < 10:
            break
        circles = [fit_circle(points[~side]), fit_circle(points[side])]
        distance = np.stack([np.abs(np.linalg.norm(points - c, axis=1) - r) for c, r in circles], 1)
        new_side = distance[:, 1] < distance[:, 0]
        if (new_side == side).all():
            break
        side = new_side
    if not circles:
        return None, None
    order = np.argsort([c[1] for c, _ in circles])  # upper circle first (smaller y)
    circles = [circles[i] for i in order]
    side = side if order[0] == 0 else ~side
    return circles, side.astype(int)  # side: index of each point's circle in the returned order


def ink_on_grid(image):
    image = pp.fill_white_patches(image)
    return cv2.resize(pp.pad_to_square(pp.ink_strength(image)), (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def longest_circular_run(flags):
    if not flags.any():
        return 0
    if flags.all():
        return len(flags)
    flags = np.roll(flags, -np.argmin(flags))
    edges = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    return int((np.where(edges == -1)[0] - np.where(edges == 1)[0]).max())


def measure_circles(mask, original):
    ys, xs = np.nonzero(pruned_skeleton(mask))
    points = np.column_stack([xs, ys]).astype(float)
    m = {"photo": int(np.median(original.reshape(-1, 3), axis=0).mean() <= 235)}
    my, mx = np.nonzero(mask)
    m["size"] = float(max(np.ptp(my), np.ptp(mx)) / SIZE)
    circles, side = fit_two_circles(points)
    if circles is None:
        return m, None
    (c1, r1), (c2, r2) = circles
    radial = np.stack([np.abs(np.linalg.norm(points - c, axis=1) - r) for c, r in circles], 1)
    residual = radial[np.arange(len(points)), side]
    mean_r = (r1 + r2) / 2

    # Basic shape: how well two circles explain the line; how round each part is
    m["fit_error"] = float(np.median(residual) / mean_r)
    m["on_circles"] = float((radial.min(axis=1) < 0.12 * mean_r).mean())
    ellipse_ratios = []
    for k in range(2):
        part = points[side == k].astype(np.float32)
        if len(part) >= 5:
            (_, _), (a, b), _ = cv2.fitEllipse(part)
            ellipse_ratios.append(max(a, b) / max(min(a, b), 1e-6))
    m["elongation"] = float(max(ellipse_ratios)) if ellipse_ratios else np.nan

    # Edges: sizes of the two circles
    m["radius_ratio"] = float(max(r1, r2) / min(r1, r2))
    m["area_ratio"] = m["radius_ratio"] ** 2

    # Orientation: direction from the upper to the lower centre, from vertical
    dx, dy = c2 - c1
    m["tilt"] = float(abs(np.degrees(np.arctan2(dx, dy))))  # 0 = directly above

    # Overlap: how far the circles overlap, relative to their size (0 = touching, 1 = the smaller lies on the centre)
    distance = float(np.linalg.norm(c2 - c1))
    m["overlap_depth"] = float((r1 + r2 - distance) / (2 * min(r1, r2)))
    m["centre_distance"] = distance / mean_r

    # Closure: walk around each circle in 2-degree steps; gap = longest stretch without ink (also not faint ink)
    faint = cv2.dilate(ink_on_grid(original), np.ones((7, 7), np.uint8))
    gaps, doubled = [], []
    for k, (c, r) in enumerate(circles):
        angles = np.degrees(np.arctan2(points[:, 1] - c[1], points[:, 0] - c[0])) % 360
        near = (side == k) & (radial[:, k] < 0.15 * r)  # this circle's own line only (not the other circle crossing it)
        bins = (angles[near] // 2).astype(int)
        covered = np.zeros(180, bool)
        covered[bins] = True
        empty = ~covered
        for b in np.where(empty)[0]:  # faint-ink check along the empty stretch
            a = np.radians(2 * b + 1)
            probe = [c + (r + o) * np.array([np.cos(a), np.sin(a)]) for o in np.linspace(-0.1 * r, 0.1 * r, 7)]
            values = [faint[int(y), int(x)] for x, y in probe if 0 <= int(x) < SIZE and 0 <= int(y) < SIZE]
            if values and max(values) >= 0.05:
                empty[b] = False
        gaps.append(np.radians(2 * longest_circular_run(empty)) * r)  # arc length in pixels
        # doubled line: two separate line passes at the same angle
        per_bin = [np.sort(np.linalg.norm(points[near][bins == b] - c, axis=1)) for b in range(180)]
        doubled_bins = np.array([len(v) >= 2 and np.diff(v).max() > 4 for v in per_bin])
        doubled.append(np.radians(2 * longest_circular_run(doubled_bins)) * r)
    m["gap"] = float(max(gaps))
    m["doubled_length"] = float(max(doubled))
    m["off_circles_length"] = float((radial.min(axis=1) > 0.15 * mean_r).sum())  # tails, lines past the circle
    loose = (cv2.filter2D(pruned_skeleton(mask), -1, NEIGHBOURS, borderType=cv2.BORDER_CONSTANT) * pruned_skeleton(mask)) == 1
    m["loose_ends"] = int(loose.sum())
    m["overlap"] = m["doubled_length"]  # the closure overlap: a line drawn twice (the off-circle length is kept for notebook 03)
    return m, np.array([[*c1, r1], [*c2, r2]])


def score_drawing(m, params):
    basic = int(not np.isnan(m["fit_error"]) and m["on_circles"] >= params["min_on_circles"])
    facets = {
        "basic_shape": basic,
        "closure": int(m["gap"] <= params["max_gap"] and m["overlap"] <= params["max_overlap"]),
        "edges": int(m["radius_ratio"] <= params["max_radius_ratio"] and m["elongation"] < 3),
        "orientation": int(m["tilt"] <= params["max_tilt"]),
        "overlap_facet": int(params["min_depth"] <= m["overlap_depth"] <= params["max_depth"]),
        "overall_size": int(m["size"] >= params["min_size"]),
    }
    if basic == 0:
        facets = {facet: 0 for facet in facets}
    facets["total"] = sum(facets[f] for f in FACET_COLUMNS)
    return facets
