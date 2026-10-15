"""Triangle measurements and rules, on the cleaned 512 x 512 ink mask.

Developed and explained in notebooks/triangle/02_triangle_rule_based.ipynb (copied here unchanged, except that
measure_triangle takes the mask and the original image instead of a file name).
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

from src import preprocessing as pp

SIZE = 512  # masks are 512 x 512 pixels (the whole answer box)
FACETS = ["basic_shape", "closure", "edges", "orientation", "overall_size"]


NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
K = 3  # number of sides
def pruned_skeleton(mask, steps=6):
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    for _ in range(steps):
        ends = (cv2.filter2D(skeleton, -1, NEIGHBOURS, borderType=cv2.BORDER_CONSTANT) * skeleton) == 1
        if not ends.any():
            break
        skeleton[ends] = 0
    return skeleton


def order_corners(corners):
    # Clockwise, starting at the top-most corner.
    corners = np.asarray(corners, float).reshape(-1, 2)
    centre = corners.mean(axis=0)
    corners = corners[np.argsort(np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0]))]
    return np.roll(corners, -np.argmin(corners[:, 1]), axis=0)


def first_guess(mask):
    hull = cv2.convexHull(cv2.findNonZero(mask))
    perimeter = cv2.arcLength(hull, True)
    for eps in np.linspace(0.005, 0.3, 120):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == K:
            return order_corners(approx)
        if len(approx) < K:
            break
    _, triangle = cv2.minEnclosingTriangle(hull.astype(np.float32))
    return order_corners(triangle)


def distance_to_segment(points, a, b):
    length = np.linalg.norm(b - a) + 1e-9
    direction = (b - a) / length
    t = (points - a) @ direction
    return np.linalg.norm(points - (a + np.outer(np.clip(t, 0, length), direction)), axis=1), t, length, direction


def intersect(p1, d1, p2, d2):
    matrix = np.array([d1, -d2]).T
    if abs(np.linalg.det(matrix)) < 1e-9:
        return (p1 + p2) / 2
    return p1 + np.linalg.solve(matrix, p2 - p1)[0] * d1


def fit_triangle(mask, points, rounds=10):
    corners = first_guess(mask)
    for round_ in range(rounds):
        distances = np.stack([distance_to_segment(points, corners[i], corners[(i + 1) % K])[0] for i in range(K)], 1)
        nearest = distances.argmin(axis=1)
        side = np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % K]) for i in range(K)])
        tolerance = max(0.10 - 0.01 * round_, 0.05) * side
        lines = []
        for i in range(K):
            a, b = corners[i], corners[(i + 1) % K]
            d, t, length, direction = distance_to_segment(points, a, b)
            selected = (nearest == i) & (d < tolerance) & (t > 0.12 * length) & (t < 0.88 * length)
            if selected.sum() < 10:
                lines.append((a, direction))
                continue
            vx, vy, x0, y0 = cv2.fitLine(points[selected].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            fitted = np.array([vx, vy])
            lines.append((np.array([x0, y0]), fitted if fitted @ direction >= 0 else -fitted))
        new = np.array([intersect(*lines[i - 1], *lines[i]) for i in range(K)])
        if not np.all(np.isfinite(new)) or np.abs(new - corners).max() > side:
            break
        corners = new
    return order_corners(corners)


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


def measure_triangle(mask, original):
    ys, xs = np.nonzero(pruned_skeleton(mask))
    points = np.column_stack([xs, ys]).astype(float)
    corners = fit_triangle(mask, points)
    sides = np.array([np.linalg.norm(corners[(i + 1) % K] - corners[i]) for i in range(K)])
    side = sides.mean()
    distances = np.stack([distance_to_segment(points, corners[i], corners[(i + 1) % K])[0] for i in range(K)], 1)
    nearest, distance = distances.argmin(axis=1), distances.min(axis=1)
    m = {"photo": int(np.median(original.reshape(-1, 3), axis=0).mean() <= 235)}

    # Basic shape and edges
    angles = []
    for i in range(K):
        a, b = corners[i - 1] - corners[i], corners[(i + 1) % K] - corners[i]
        angles.append(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
    m.update({"on_outline": float((distance < 0.06 * side).mean()), "smallest_angle": min(angles),
              "side_ratio": float(sides.max() / sides.min())})

    # Orientation and size
    middles = [(corners[i] + corners[(i + 1) % K]) / 2 for i in range(K)]
    lowest = int(np.argmax([p[1] for p in middles]))
    apex = corners[(lowest + 2) % K]
    centre = corners.mean(axis=0)
    directions = np.degrees(np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0]))  # upright apex: -90
    m.update({"rotation": float(np.abs((directions + 90 + 60) % 120 - 60).min()),
              "apex_above": float((middles[lowest][1] - apex[1]) / side),
              "size": float(sides.max() / SIZE)})

    # Closure 1: overshoot past each corner (ignoring the other sides' pixels near the corner)
    tolerance = max(4, 0.04 * side)
    overshoots = []
    for i in range(K):
        a, b = corners[i], corners[(i + 1) % K]
        length = np.linalg.norm(b - a)
        direction = (b - a) / length
        normal = np.array([-direction[1], direction[0]])
        t, offset = (points - a) @ direction, np.abs((points - a) @ normal)
        other = np.minimum(distance_to_segment(points, corners[(i + 1) % K], corners[(i + 2) % K])[0],
                           distance_to_segment(points, corners[(i + 2) % K], corners[i])[0])
        for beyond in (-t, t - length):
            on_line = (offset < tolerance) & (beyond > 0) & (other > tolerance)
            run, previous = 0.0, tolerance  # the first `tolerance` pixels are hidden by the other side
            for value in np.sort(beyond[on_line]):
                if value - previous > 6:
                    break
                run = previous = value
            overshoots.append(run)
    m["overshoot"] = float(max(overshoots))

    # Closure 2: walk along the outline in 2-pixel steps (as for the Square)
    steps = int(sides.sum() // 2)
    start = np.concatenate([[0], np.cumsum(sides)])
    covered, offsets, width = np.zeros(steps, bool), [[] for _ in range(steps)], np.zeros(steps)
    half_width = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    for i in range(K):
        a, b = corners[i], corners[(i + 1) % K]
        direction = (b - a) / sides[i]
        normal = np.array([-direction[1], direction[0]])
        t, offset = (points - a) @ direction, (points - a) @ normal
        selected = (nearest == i) & (np.abs(offset) < 0.12 * side) & (t >= -2) & (t <= sides[i] + 2)
        positions = ((start[i] + np.clip(t[selected], 0, sides[i])) / 2).astype(int) % steps
        covered[positions] = True
        for p, o, w in zip(positions, offset[selected], 2 * half_width[points[selected, 1].astype(int), points[selected, 0].astype(int)]):
            offsets[p].append(o)
            width[p] = max(width[p], w)
    for shift in (-2, -1, 1, 2):
        covered |= np.roll(covered, shift)
    faint = cv2.dilate(ink_on_grid(original), np.ones((7, 7), np.uint8))
    empty = np.zeros(steps, bool)
    for k in np.where(~covered)[0]:
        position = 2.0 * k
        i = min(int(np.searchsorted(start, position, side="right") - 1), K - 1)
        direction = (corners[(i + 1) % K] - corners[i]) / sides[i]
        normal = np.array([-direction[1], direction[0]])
        point = corners[i] + (position - start[i]) * direction
        values = [faint[int(y), int(x)] for x, y in (point + o * normal for o in np.linspace(-0.06 * side, 0.06 * side, 13))
                  if 0 <= int(x) < SIZE and 0 <= int(y) < SIZE]
        empty[k] = (max(values) if values else 0) < 0.05
    m["gap"] = 2.0 * longest_circular_run(empty)
    doubled = np.array([len(o) >= 2 and np.diff(np.sort(o)).max() > 4 for o in offsets])
    doubled |= np.roll(doubled, 1) & np.roll(doubled, -1)
    m["doubled_length"] = 2.0 * longest_circular_run(doubled)
    m["thick_length"] = 2.0 * longest_circular_run(width > 1.8 * np.median(width[width > 0]))
    m["overlap"] = max(m["overshoot"], m["doubled_length"], 0 if m["photo"] else m["thick_length"])
    return m, corners


def score_triangle(m, params):
    basic = int(m["smallest_angle"] >= 25 and m["side_ratio"] <= 3 and m["on_outline"] >= 0.75)
    facets = {
        "basic_shape": basic,
        "closure": int(m["gap"] <= params["pixels_per_inch"] / 8 and m["overlap"] <= params["pixels_per_inch"] / 4
                       and m["doubled_length"] <= params["max_doubled"]),
        "edges": int(m["side_ratio"] <= params["max_side_ratio"]),
        "orientation": int(m["rotation"] <= params["max_rotation"] and m["apex_above"] > 0),
        "overall_size": int(m["size"] >= params["min_size"]),
    }
    if basic == 0:
        facets = {facet: 0 for facet in facets}
    facets["total"] = sum(facets[f] for f in FACETS)
    return facets
