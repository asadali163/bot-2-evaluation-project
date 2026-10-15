"""Overlapping-pencils measurements and rules, on the cleaned 512 x 512 ink mask.

Developed and explained in notebooks/overlapped_pencils/02_overlapped_pencils_rule_based.ipynb (copied here unchanged,
except that measure_pencils takes the mask and the original image instead of a file name). The overlap facet is called
"overlap_facet" here, as in the notebook.
"""
import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from src import preprocessing as pp

SIZE = 512  # masks are 512 x 512 pixels (the whole answer box)
FACET_COLUMNS = ["basic_shape", "closure", "edges", "orientation", "overlap_facet", "overall_size"]


NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
def enclosed_regions(mask, radius):
    """Regions enclosed by the line after closing gaps up to `radius` pixels (largest first)."""
    closed = mask if radius == 0 else cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    holes = (ndimage.binary_fill_holes(closed) & ~closed.astype(bool)).astype(np.uint8)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(holes, connectivity=4)
    return [(labels == i, stats[i, cv2.CC_STAT_AREA], centroids[i]) for i in range(1, n)]


def split_pencils(mask):
    """Return (left pencil, right pencil, overlap, closing radius used) as filled masks, or None.

    The smallest closing radius that gives at least two sizeable enclosed regions is used. The leftmost and the
    rightmost sizeable regions are the outer parts of the two pencils; a region touching both of them is the overlap
    (the shared middle part, which can be small). Each pencil is its outer part plus the overlap. Every region is grown
    by the width of the line, so that it includes its outline and touches its neighbours across the line."""
    total = ndimage.binary_fill_holes(mask).sum()
    for radius in (0, 3, 6, 10, 16, 24):
        regions = [r for r in enclosed_regions(mask, radius) if r[1] > 0.003 * total]
        if sum(r[1] > 0.03 * total for r in regions) >= 2:
            break
    else:
        return None
    stroke = np.median(cv2.distanceTransform(mask, cv2.DIST_L2, 3)[skeletonize(mask > 0)])  # half the line width
    reach = radius + int(np.ceil(2 * stroke)) + 2  # far enough to cross the line between two regions
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1,) * 2)
    big = sorted([r for r in regions if r[1] > 0.03 * total], key=lambda r: r[2][0])
    outer_left, outer_right = (cv2.dilate(r[0].astype(np.uint8), grow) > 0 for r in (big[0], big[-1]))
    middle = [r for r in regions if r is not big[0] and r is not big[-1]]
    touching = [r for r in middle if ((cv2.dilate(r[0].astype(np.uint8), grow) > 0) & outer_left).any()
                and ((cv2.dilate(r[0].astype(np.uint8), grow) > 0) & outer_right).any()]
    if touching:
        overlap = cv2.dilate(max(touching, key=lambda r: r[1])[0].astype(np.uint8), grow) > 0
        left, right = outer_left | overlap, outer_right | overlap
    else:  # no enclosed shared part: the pencils only touch, or are drawn apart
        left, right = outer_left, outer_right
        overlap = left & right
    return left, right, overlap, radius


def pencil_geometry(region):
    """Axis angle (degrees from vertical, positive = top leans right), length, width, and which end is pointed."""
    ys, xs = np.nonzero(region)
    points = np.column_stack([xs - xs.mean(), ys - ys.mean()]).astype(float)
    axis = np.linalg.svd(points, full_matrices=False)[2][0]
    axis = axis if axis[1] > 0 else -axis  # pointing down the page
    normal = np.array([-axis[1], axis[0]])
    t, q = points @ axis, points @ normal
    lo, hi = np.percentile(t, [2, 98])
    length = hi - lo

    def width(a, b):
        sel = (t >= lo + a * length) & (t <= lo + b * length)
        return float(np.percentile(q[sel], 95) - np.percentile(q[sel], 5)) if sel.sum() > 5 else np.nan

    top_width, bottom_width, middle_width = width(0.0, 0.2), width(0.8, 1.0), width(0.3, 0.7)
    angle = float(np.degrees(np.arctan2(-axis[0], axis[1])))  # 0 = vertical; positive: top leans to the right
    return {"angle": angle, "length": float(length), "width": middle_width,
            "pointed_at_bottom": int(bottom_width < top_width),
            "end_contrast": float(min(top_width, bottom_width) / max(top_width, bottom_width, 1e-6)),
            "elongation": float(length / max(middle_width, 1e-6)), "area": float(region.sum())}


def ink_on_grid(image):
    image = pp.fill_white_patches(image)
    return cv2.resize(pp.pad_to_square(pp.ink_strength(image)), (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def skeleton_parts(mask):
    skeleton = skeletonize(mask > 0).astype(np.uint8)
    neighbours = cv2.filter2D(skeleton, -1, NEIGHBOURS, borderType=cv2.BORDER_CONSTANT) * skeleton
    return skeleton, np.argwhere(neighbours == 1), neighbours >= 3


def tail_lengths(skeleton, ends, junctions):
    """Walk from each loose end along the line to the first junction: the length of a tail / overshoot."""
    lengths = []
    for y, x in ends:
        previous, current, length = None, (y, x), 0
        while not junctions[current] and length < 200:
            nxt = [(current[0] + dy, current[1] + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                   if (dy or dx) and 0 <= current[0] + dy < SIZE and 0 <= current[1] + dx < SIZE
                   and skeleton[current[0] + dy, current[1] + dx] and (current[0] + dy, current[1] + dx) != previous]
            if not nxt:
                length = 0  # a free line end not attached to the drawing: not a tail
                break
            previous, current, length = current, nxt[0], length + 1
        lengths.append(length)
    return lengths


def measure_pencils(mask, original):
    ys, xs = np.nonzero(mask)
    m = {"photo": int(np.median(original.reshape(-1, 3), axis=0).mean() <= 235),
         "size": float(max(np.ptp(xs), np.ptp(ys)) / SIZE)}

    # Closure: tails (line drawn past a corner or tip), loose ends, gaps between line ends, thick (retraced) line
    skeleton, ends, junctions = skeleton_parts(mask)
    tails = sorted(tail_lengths(skeleton, ends, junctions), reverse=True)
    m["loose_ends"] = len(ends)
    m["longest_tail"] = float(tails[0]) if tails else 0.0
    m["second_tail"] = float(tails[1]) if len(tails) > 1 else 0.0
    faint = cv2.dilate(ink_on_grid(original), np.ones((7, 7), np.uint8))
    gaps = [0.0]
    if len(ends) >= 2:
        distance = np.linalg.norm(ends[:, None] - ends[None], axis=2)
        np.fill_diagonal(distance, np.inf)
        for i, j in enumerate(distance.argmin(axis=1)):
            if distance[j].argmin() == i and i < j:  # mutually nearest pair of loose ends
                (y1, x1), (y2, x2) = ends[i], ends[j]
                line = [faint[int(y1 + (y2 - y1) * s), int(x1 + (x2 - x1) * s)] for s in np.linspace(0.2, 0.8, 7)]
                if min(line) < 0.05:  # no ink at all between them, not even faint ink
                    gaps.append(float(distance[i, j]))
    m["gap"] = max(gaps)
    half_width = cv2.distanceTransform(mask, cv2.DIST_L2, 3)[skeleton > 0]
    m["thick_ratio"] = float(np.percentile(half_width, 98) / max(np.median(half_width), 1e-6))

    split = split_pencils(mask)
    m["split_radius"] = np.nan if split is None else float(split[3])
    if split is None:
        return m, None
    left, right, overlap, _ = split
    L, R = pencil_geometry(left), pencil_geometry(right)
    for side, g in (("left", L), ("right", R)):
        for key, value in g.items():
            m[f"{side}_{key}"] = value

    # Basic shape: both parts elongated, with one narrow and one wide end
    m["min_elongation"] = min(L["elongation"], R["elongation"])
    m["max_end_contrast"] = max(L["end_contrast"], R["end_contrast"])  # close to 1: no clear pointed end

    # Edges: the larger pencil vs the smaller, in length and in width
    m["length_ratio"] = max(L["length"], R["length"]) / max(min(L["length"], R["length"]), 1e-6)
    m["width_ratio"] = max(L["width"], R["width"]) / max(min(L["width"], R["width"]), 1e-6)
    m["edges_ratio"] = max(m["length_ratio"], m["width_ratio"])

    # Orientation: left pencil pointing down, right pencil pointing up (as in the stimulus); tilt of each
    m["points_correctly"] = int(L["pointed_at_bottom"] == 1 and R["pointed_at_bottom"] == 0)

    # Overlap: area shared by the two pencils relative to the smaller pencil
    m["overlap_fraction"] = float(overlap.sum() / max(min(left.sum(), right.sum()), 1))
    centre_l, centre_r = np.argwhere(left).mean(axis=0), np.argwhere(right).mean(axis=0)
    m["centre_distance"] = float(np.linalg.norm(centre_r - centre_l) / max((L["width"] + R["width"]) / 2, 1e-6))
    return m, (left, right, overlap)


def score_drawing(m, params):
    split = not np.isnan(m["left_angle"])
    basic = int(not split or m["min_elongation"] >= params["min_elongation"])  # an unsplittable drawing is not failed
    tilt = max(abs(m["left_angle"] - params["left_tilt"]), abs(m["right_angle"] - params["right_tilt"])) if split else 0
    facets = {
        "basic_shape": basic,
        "closure": int(m["thick_ratio"] <= params["max_thick"] and m["longest_tail"] <= params["max_tail"]),
        "edges": int(not split or m["edges_ratio"] <= params["max_edges_ratio"]),
        "orientation": int(not split or (m["points_correctly"] == 1 and tilt <= params["max_tilt_deviation"])),
        "overlap_facet": int(not split or params["min_overlap"] <= m["overlap_fraction"] <= params["max_overlap"]),
        "overall_size": int(m["size"] >= params["min_size"]),
    }
    if basic == 0:
        facets = {facet: 0 for facet in facets}
    facets["total"] = sum(facets[f] for f in FACET_COLUMNS)
    return facets
