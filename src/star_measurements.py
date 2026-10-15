"""Star measurements and rules, on the cleaned 512 x 512 ink mask.

Developed and explained in notebooks/star/02_star_rule_based.ipynb (copied here unchanged, except that measure_star
takes the mask and the original image instead of a file name).
"""
import cv2
import numpy as np
from scipy.signal import find_peaks
from skimage.morphology import skeletonize

from src import preprocessing as pp

SIZE = 512  # masks are 512 x 512 pixels (the whole answer box)
FACETS = ["basic_shape", "closure", "edges", "orientation", "overall_size"]


NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
BINS = 360  # one direction per degree
def pruned_skeleton(mask, steps=6):
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    for _ in range(steps):
        ends = (cv2.filter2D(skeleton, -1, NEIGHBOURS, borderType=cv2.BORDER_CONSTANT) * skeleton) == 1
        if not ends.any():
            break
        skeleton[ends] = 0
    return skeleton


def circular_smooth(values, width=5):
    padded = np.concatenate([values[-width:], values, values[:width]])
    return np.convolve(padded, np.ones(width) / width, mode="same")[width:-width]


def outer_envelope(points, centre):
    # For each direction (1 degree), the distance of the farthest ink from the centre; y axis pointing up.
    dx, dy = points[:, 0] - centre[0], centre[1] - points[:, 1]
    radius, angle = np.hypot(dx, dy), np.degrees(np.arctan2(dy, dx)) % 360
    envelope = np.full(BINS, np.nan)
    np.fmax.at(envelope, (angle * BINS / 360).astype(int) % BINS, radius)
    known = ~np.isnan(envelope)
    index = np.arange(BINS)
    envelope = np.interp(index, index[known], envelope[known], period=BINS)
    return circular_smooth(envelope), known.mean()


def find_points(envelope, prominence):
    # Tips = peaks of the envelope (searched on three copies so peaks at 0 degrees are found).
    tripled = np.concatenate([envelope] * 3)
    peaks, properties = find_peaks(tripled, prominence=prominence, distance=BINS // 30)
    middle = (peaks >= BINS) & (peaks < 2 * BINS)
    return peaks[middle] - BINS, properties["prominences"][middle]


def star_points(mask):
    """Return the centre, the envelope, the 5 tips and 5 inner corners (as envelope directions) and the tip count."""
    ys, xs = np.nonzero(mask)
    points = np.column_stack([xs, ys]).astype(float)
    hull = cv2.convexHull(points.astype(np.float32))
    moments = cv2.moments(hull)
    centre = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    envelope, coverage = outer_envelope(points, centre)
    mean_radius = envelope.mean()
    peaks, prominences = find_points(envelope, 0.08 * mean_radius)
    tips = np.sort(peaks[np.argsort(prominences)[::-1][:5]]) if len(peaks) >= 5 else None
    corners = None
    if tips is not None:  # inner corner i lies between tip i and tip i + 1
        corners = []
        for i in range(5):
            a, b = tips[i], tips[(i + 1) % 5]
            span = np.arange(a, b if b > a else b + BINS) % BINS
            corners.append(span[np.argmin(envelope[span])])
        corners = np.array(corners)
    counts = {f"tips_{int(100 * p)}": len(find_points(envelope, p * mean_radius)[0]) for p in (0.08, 0.15, 0.25)}
    return centre, envelope, tips, corners, counts, coverage, prominences


def to_image(centre, envelope, directions):
    angle = np.radians(np.asarray(directions) * 360 / BINS)
    return np.column_stack([centre[0] + envelope[directions] * np.cos(angle), centre[1] - envelope[directions] * np.sin(angle)])


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


def distance_to_segment(points, a, b):
    length = np.linalg.norm(b - a) + 1e-9
    direction = (b - a) / length
    t = (points - a) @ direction
    return np.linalg.norm(points - (a + np.outer(np.clip(t, 0, length), direction)), axis=1)


def measure_star(mask, original):
    centre, envelope, tips, corners, counts, coverage, prominences = star_points(mask)
    ys, xs = np.nonzero(mask)
    m = {"photo": int(np.median(original.reshape(-1, 3), axis=0).mean() <= 235), **counts,
         "size": float(max(np.ptp(xs), np.ptp(ys)) / SIZE)}
    mean_radius = envelope.mean()
    m["sixth_tip_prominence"] = float(np.sort(prominences)[::-1][5] / mean_radius) if len(prominences) > 5 else 0.0
    if tips is None:
        return m, None
    m["fifth_tip_prominence"] = float(np.sort(prominences)[::-1][4] / mean_radius)
    T, V = to_image(centre, envelope, tips), to_image(centre, envelope, corners)

    # Edges: point length (tip to the middle of its two inner corners) and width (distance between them)
    lengths = np.array([np.linalg.norm(T[i] - (V[i - 1] + V[i]) / 2) for i in range(5)])
    widths = np.array([np.linalg.norm(V[i] - V[i - 1]) for i in range(5)])
    m.update({"length_ratio": float(lengths.max() / lengths.min()), "width_ratio": float(widths.max() / widths.min()),
              "tip_radius_ratio": float(envelope[tips].max() / envelope[tips].min()),
              "corner_radius_ratio": float(envelope[corners].max() / envelope[corners].min()),
              "inner_outer": float(envelope[corners].mean() / envelope[tips].mean())})
    m["edges_ratio"] = max(m["length_ratio"], m["width_ratio"])

    # Orientation: rotation of the whole five-point pattern from "one point straight up" (0 - 36 degrees). The five
    # tip directions are combined (a regular star repeats every 72 degrees, so 5 x angle is the same for all tips).
    tip_angles = tips * 360 / BINS
    pattern = np.degrees(np.angle(np.mean(np.exp(5j * np.radians(tip_angles))))) / 5
    m["rotation"] = float(abs((pattern - 90 + 36) % 72 - 36))
    m["nearest_tip_rotation"] = float(np.min(np.abs((tip_angles - 90 + 36) % 72 - 36)))
    m["tip_irregularity"] = float(np.abs(((tip_angles - pattern) % 72 + 36) % 72 - 36).mean())
    angles = np.sort(tip_angles)
    m["tip_spacing_sd"] = float(np.diff(np.concatenate([angles, [angles[0] + 360]])).std())
    mirror = envelope[(BINS // 2 - np.arange(BINS)) % BINS]  # the envelope mirrored about the vertical axis
    m["mirror_asymmetry"] = float(np.abs(envelope - mirror).mean() / mean_radius)
    low = T[np.argsort(T[:, 1])[-2:]]
    m["bottom_tilt"] = float(abs(np.degrees(np.arctan2(low[1, 1] - low[0, 1], abs(low[1, 0] - low[0, 0]) + 1e-6))))

    # Closure: walk along the 10-sided outline tip - corner - tip ... in 2-pixel steps (as for the other shapes)
    outline = np.empty((10, 2))
    outline[0::2], outline[1::2] = T, V
    sk_y, sk_x = np.nonzero(pruned_skeleton(mask))
    points = np.column_stack([sk_x, sk_y]).astype(float)
    distances = np.stack([distance_to_segment(points, outline[i], outline[(i + 1) % 10]) for i in range(10)], 1)
    nearest, distance = distances.argmin(axis=1), distances.min(axis=1)
    sides = np.array([np.linalg.norm(outline[(i + 1) % 10] - outline[i]) for i in range(10)])
    side = sides.mean()
    m["on_outline"] = float((distance < 0.15 * side).mean())
    steps = int(sides.sum() // 2)
    start = np.concatenate([[0], np.cumsum(sides)])
    covered, offsets = np.zeros(steps, bool), [[] for _ in range(steps)]
    for i in range(10):
        a, b = outline[i], outline[(i + 1) % 10]
        direction = (b - a) / sides[i]
        normal = np.array([-direction[1], direction[0]])
        t, offset = (points - a) @ direction, (points - a) @ normal
        selected = (nearest == i) & (np.abs(offset) < 0.25 * side) & (t >= -2) & (t <= sides[i] + 2)
        positions = ((start[i] + np.clip(t[selected], 0, sides[i])) / 2).astype(int) % steps
        covered[positions] = True
        for p, o in zip(positions, offset[selected]):
            offsets[p].append(o)
    for shift in (-2, -1, 1, 2):
        covered |= np.roll(covered, shift)
    faint = cv2.dilate(ink_on_grid(original), np.ones((7, 7), np.uint8))
    empty = np.zeros(steps, bool)
    for k in np.where(~covered)[0]:
        position = 2.0 * k
        i = min(int(np.searchsorted(start, position, side="right") - 1), 9)
        direction = (outline[(i + 1) % 10] - outline[i]) / sides[i]
        normal = np.array([-direction[1], direction[0]])
        point = outline[i] + (position - start[i]) * direction
        values = [faint[int(y), int(x)] for x, y in (point + o * normal for o in np.linspace(-0.1 * side, 0.1 * side, 11))
                  if 0 <= int(x) < SIZE and 0 <= int(y) < SIZE]
        empty[k] = (max(values) if values else 0) < 0.05
    m["gap"] = 2.0 * longest_circular_run(empty)
    doubled = np.array([len(o) >= 2 and np.diff(np.sort(o)).max() > 4 for o in offsets])
    doubled |= np.roll(doubled, 1) & np.roll(doubled, -1)
    m["doubled_length"] = 2.0 * longest_circular_run(doubled)
    # Overshoot: skeleton ink far beyond the outline near a tip or corner (a line drawn past the point)
    m["off_outline_length"] = float((distance > 0.15 * side).sum())
    m["overlap"] = max(m["doubled_length"], m["off_outline_length"])
    loose = (cv2.filter2D(pruned_skeleton(mask), -1, NEIGHBOURS, borderType=cv2.BORDER_CONSTANT) * pruned_skeleton(mask)) == 1
    m["loose_ends"] = int(loose.sum())
    return m, np.vstack([T, V])


def score_star(m, params):
    basic = int(not np.isnan(m["length_ratio"]) and m["on_outline"] >= params["min_on_outline"])
    facets = {
        "basic_shape": basic,
        "closure": int(m["gap"] <= params["max_gap"] and m["overlap"] <= params["max_overlap"]),
        "edges": int(m["edges_ratio"] <= params["max_edges_ratio"]),
        "orientation": int(m["rotation"] <= params["max_rotation"]),
        "overall_size": int(m["size"] >= params["min_size"]),
    }
    if basic == 0:
        facets = {facet: 0 for facet in facets}
    facets["total"] = sum(facets[f] for f in FACETS)
    return facets
