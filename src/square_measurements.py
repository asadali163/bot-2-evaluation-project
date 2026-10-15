"""Square measurements and rules, on the cleaned 512 x 512 ink mask.

Developed and explained in notebooks/square/02_square_rule_based.ipynb (copied here unchanged, except that
measure_square takes the mask and the original image instead of a file name).
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize

from src import preprocessing as pp

SIZE = 512  # masks are 512 x 512 pixels (the whole answer box)
FACETS = ["basic_shape", "closure", "edges", "orientation", "overall_size"]


NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
def pruned_skeleton(mask, steps=6):
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    for _ in range(steps):  # remove short spurs, one end pixel per step
        neighbours = cv2.filter2D(skeleton, -1, NEIGHBOURS, borderType=cv2.BORDER_CONSTANT) * skeleton
        ends = neighbours == 1
        if not ends.any():
            break
        skeleton[ends] = 0
    return skeleton


def order_corners(corners):
    # Order the 4 corners clockwise, starting at the top-left one.
    corners = np.asarray(corners, float).reshape(-1, 2)
    centre = corners.mean(axis=0)
    corners = corners[np.argsort(np.arctan2(corners[:, 1] - centre[1], corners[:, 0] - centre[0]))]
    return np.roll(corners, -np.argmin(corners.sum(axis=1)), axis=0)


def first_guess(mask):
    hull = cv2.convexHull(cv2.findNonZero(mask))
    perimeter = cv2.arcLength(hull, True)
    for eps in np.linspace(0.005, 0.15, 60):
        approx = cv2.approxPolyDP(hull, eps * perimeter, True)
        if len(approx) == 4:
            return order_corners(approx)
        if len(approx) < 4:
            break
    return order_corners(cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(mask))))


def distance_to_segment(points, a, b):
    # Distance of points to segment a-b, their position t along it, its length and direction.
    length = np.linalg.norm(b - a) + 1e-9
    direction = (b - a) / length
    t = (points - a) @ direction
    projection = a + np.outer(np.clip(t, 0, length), direction)
    return np.linalg.norm(points - projection, axis=1), t, length, direction


def intersect(p1, d1, p2, d2):
    matrix = np.array([d1, -d2]).T
    if abs(np.linalg.det(matrix)) < 1e-9:
        return (p1 + p2) / 2
    return p1 + np.linalg.solve(matrix, p2 - p1)[0] * d1


def fit_quadrilateral(mask, points, rounds=10):
    corners = first_guess(mask)
    for round_ in range(rounds):
        distances = np.stack([distance_to_segment(points, corners[i], corners[(i + 1) % 4])[0] for i in range(4)], 1)
        nearest = distances.argmin(axis=1)
        side = np.mean([np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)])
        tolerance = max(0.10 - 0.01 * round_, 0.05) * side
        lines = []
        for i in range(4):
            a, b = corners[i], corners[(i + 1) % 4]
            d, t, length, direction = distance_to_segment(points, a, b)
            selected = (nearest == i) & (d < tolerance) & (t > 0.12 * length) & (t < 0.88 * length)
            if selected.sum() < 10:
                lines.append((a, direction))
                continue
            vx, vy, x0, y0 = cv2.fitLine(points[selected].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
            fitted = np.array([vx, vy])
            lines.append((np.array([x0, y0]), fitted if fitted @ direction >= 0 else -fitted))
        new = np.array([intersect(*lines[i - 1], *lines[i]) for i in range(4)])
        if not np.all(np.isfinite(new)) or np.abs(new - corners).max() > side:
            break
        corners = new
    return order_corners(corners)


def ink_on_grid(image):
    # Ink strength (0..1) of the original image on the 512 x 512 mask grid, to check for faint ink.
    image = pp.fill_white_patches(image)
    return cv2.resize(pp.pad_to_square(pp.ink_strength(image)), (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def is_photo(image):
    # Grey paper = phone photo; white paper = scan (median paper colour, as in notebook 01).
    return bool(np.median(image.reshape(-1, 3), axis=0).mean() <= 235)


def longest_circular_run(flags):
    # Longest run of True values, treating the array as a loop (the outline is closed).
    if not flags.any():
        return 0
    if flags.all():
        return len(flags)
    flags = np.roll(flags, -np.argmin(flags))
    edges = np.diff(np.concatenate([[0], flags.astype(int), [0]]))
    return int((np.where(edges == -1)[0] - np.where(edges == 1)[0]).max())


def measure_square(mask, original):
    skeleton = pruned_skeleton(mask)
    ys, xs = np.nonzero(skeleton)
    points = np.column_stack([xs, ys]).astype(float)
    corners = fit_quadrilateral(mask, points)
    sides = np.array([np.linalg.norm(corners[(i + 1) % 4] - corners[i]) for i in range(4)])
    side = sides.mean()
    distances = np.stack([distance_to_segment(points, corners[i], corners[(i + 1) % 4])[0] for i in range(4)], 1)
    nearest, distance = distances.argmin(axis=1), distances.min(axis=1)
    m = {"photo": int(is_photo(original))}

    # Basic shape
    m["on_outline"] = float((distance < 0.06 * side).mean())
    angles = []
    for i in range(4):
        a, b = corners[i - 1] - corners[i], corners[(i + 1) % 4] - corners[i]
        angles.append(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))
    m["corner_deviation"] = float(np.abs(np.array(angles) - 90).max())
    m["convex"] = int(cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2)))
    m["side_ratio"] = float(sides.max() / sides.min())

    # Orientation and size
    side_angles = [((np.degrees(np.arctan2(*(corners[(i + 1) % 4] - corners[i])[::-1])) + 45) % 90) - 45 for i in range(4)]
    m["tilt"] = float(np.mean(np.abs(side_angles)))
    m["rotation"] = float(abs(np.mean(side_angles)))
    m["side_fraction"] = float(side / SIZE)

    # Closure 1: overshoot past each corner, along each side's line
    overshoots = []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        length = np.linalg.norm(b - a)
        direction = (b - a) / length
        normal = np.array([-direction[1], direction[0]])
        t, offset = (points - a) @ direction, np.abs((points - a) @ normal)
        for beyond in (-t, t - length):  # past the start corner, past the end corner
            on_line = (offset < max(4, 0.04 * side)) & (beyond > 0)
            run = previous = 0.0
            for value in np.sort(beyond[on_line]):  # continuous run of ink starting at the corner
                if value - previous > 4:
                    break
                run = previous = value
            overshoots.append(run)
    m["overshoot"] = float(max(overshoots))

    # Closure 2: walk along the outline in 2-pixel steps
    perimeter_steps = int(sides.sum() // 2)
    start = np.concatenate([[0], np.cumsum(sides)])
    covered = np.zeros(perimeter_steps, bool)
    offsets = [[] for _ in range(perimeter_steps)]
    width = np.zeros(perimeter_steps)
    half_width = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        direction = (b - a) / sides[i]
        normal = np.array([-direction[1], direction[0]])
        t, offset = (points - a) @ direction, (points - a) @ normal
        selected = (nearest == i) & (np.abs(offset) < 0.12 * side) & (t >= -2) & (t <= sides[i] + 2)
        steps = ((start[i] + np.clip(t[selected], 0, sides[i])) / 2).astype(int) % perimeter_steps
        covered[steps] = True
        local_width = 2 * half_width[points[selected, 1].astype(int), points[selected, 0].astype(int)]
        for step, o, w in zip(steps, offset[selected], local_width):
            offsets[step].append(o)
            width[step] = max(width[step], w)
    for shift in (-2, -1, 1, 2):  # skeleton steps on diagonals can skip a position
        covered |= np.roll(covered, shift)

    faint = cv2.dilate(ink_on_grid(original), np.ones((7, 7), np.uint8))
    empty = np.zeros(perimeter_steps, bool)
    for k in np.where(~covered)[0]:
        position = 2.0 * k
        i = min(int(np.searchsorted(start, position, side="right") - 1), 3)
        direction = (corners[(i + 1) % 4] - corners[i]) / sides[i]
        normal = np.array([-direction[1], direction[0]])
        point = corners[i] + (position - start[i]) * direction
        across = [point + o * normal for o in np.linspace(-0.06 * side, 0.06 * side, 13)]
        values = [faint[int(y), int(x)] for x, y in across if 0 <= int(x) < SIZE and 0 <= int(y) < SIZE]
        empty[k] = (max(values) if values else 0) < 0.05  # not even faint ink
    m["gap"] = 2.0 * longest_circular_run(empty)

    doubled = np.array([len(o) >= 2 and np.diff(np.sort(o)).max() > 4 for o in offsets])
    doubled |= np.roll(doubled, 1) & np.roll(doubled, -1)  # close one-step holes
    m["doubled_length"] = 2.0 * longest_circular_run(doubled)
    m["thick_length"] = 2.0 * longest_circular_run(width > 1.8 * np.median(width[width > 0]))
    # Line thickness is only trusted in scans: in photos, focus and lighting change the apparent width (Section 2b)
    m["overlap"] = max(m["overshoot"], m["doubled_length"], 0 if m["photo"] else m["thick_length"])
    m["off_outline_length"] = float((distance > 0.06 * side).sum())
    return m, corners


def score_square(m, params):
    is_quadrilateral = (m["convex"] == 1 and m["side_ratio"] <= 3 and m["corner_deviation"] <= 45
                        and m["on_outline"] >= 0.75)
    facets = {
        "basic_shape": int(is_quadrilateral),
        "closure": int(m["gap"] <= params["pixels_per_inch"] / 8 and m["overlap"] <= params["pixels_per_inch"] / 4),
        "edges": int(m["side_ratio"] <= 1.5),  # the rubric's value
        "orientation": int(m["tilt"] <= params["max_tilt"]),
        "overall_size": int(m["side_fraction"] >= params["min_side_fraction"]),
    }
    if facets["basic_shape"] == 0:  # manual: basic shape 0 -> everything 0
        facets = {facet: 0 for facet in facets}
    facets["total"] = sum(facets[f] for f in FACETS)
    return facets
