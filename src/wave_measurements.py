"""Wavy-line measurements and rules, on the cleaned 512 x 512 ink mask.

Developed and explained in notebooks/wave/02_wave_rule_based.ipynb (copied here unchanged, except that measure_wave
takes the mask instead of a file name).
"""
import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path
from skimage.morphology import skeletonize

SIZE = 512  # masks are 512 x 512 pixels (the whole answer box)
FACETS = ["basic_shape", "edges", "orientation", "overall_size"]


def centre_line(mask):
    # Ordered points of the longest path through the skeleton (from one end of the line to the other).
    skeleton = skeletonize(mask > 0)
    ys, xs = np.nonzero(skeleton)
    index = -np.ones(skeleton.shape, int)
    index[ys, xs] = np.arange(len(ys))
    rows, cols, lengths = [], [], []
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:  # each neighbour pair once
        y2, x2 = ys + dy, xs + dx
        inside = (
            (y2 >= 0) & (y2 < skeleton.shape[0]) & (x2 >= 0) & (x2 < skeleton.shape[1])
        )
        neighbour = np.full(len(ys), -1)
        neighbour[inside] = index[y2[inside], x2[inside]]
        linked = neighbour >= 0
        rows += list(np.where(linked)[0])
        cols += list(neighbour[linked])
        lengths += [np.hypot(dy, dx)] * int(linked.sum())
    graph = coo_matrix((lengths, (rows, cols)), shape=(len(ys), len(ys))).tocsr()

    distance = shortest_path(graph, directed=False, indices=0)
    one_end = int(
        np.argmax(np.where(np.isfinite(distance), distance, -1))
    )  # farthest point from any point
    distance, previous = shortest_path(
        graph, directed=False, indices=one_end, return_predecessors=True
    )
    other_end = int(
        np.argmax(np.where(np.isfinite(distance), distance, -1))
    )  # farthest from that end
    path = [other_end]
    while path[-1] != one_end:
        path.append(previous[path[-1]])
    return np.column_stack([xs[path], ys[path]]).astype(float)


def resample(points, n=200):
    steps = np.hypot(*np.diff(points, axis=0).T)
    position = np.concatenate([[0], np.cumsum(steps)])
    targets = np.linspace(0, position[-1], n)
    return (
        np.column_stack([np.interp(targets, position, points[:, i]) for i in range(2)]),
        position[-1],
    )


def signed_turning(points, sigma=4):
    smooth = np.column_stack(
        [
            ndimage.gaussian_filter1d(points[:, i], sigma, mode="nearest")
            for i in range(2)
        ]
    )
    direction = np.unwrap(np.arctan2(*np.diff(smooth, axis=0).T[::-1]))
    return smooth, np.degrees(np.diff(direction))  # turning per step, in degrees


def find_arcs(turning, min_turn):
    # Arcs: stretches that keep turning the same way and turn at least min_turn degrees in total.
    # Returns a list of (start, end, total turning). Small wiggles in the other direction (< min_turn) do not split
    # an arc: neighbouring arcs turning the same way are merged.
    sign = np.sign(turning)
    sign[np.abs(turning) < 0.3] = 0  # almost straight: belongs to the previous stretch
    for i in range(1, len(sign)):
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    stretches, start = [], 0
    for i in range(1, len(sign) + 1):
        if i == len(sign) or sign[i] != sign[start]:
            stretches.append((start, i, float(turning[start:i].sum())))
            start = i
    arcs = []
    for start, end, turn in stretches:
        if abs(turn) < min_turn:
            continue  # a wiggle, not an arc
        if arcs and np.sign(arcs[-1][2]) == np.sign(turn):
            arcs[-1] = (
                arcs[-1][0],
                end,
                arcs[-1][2] + turn,
            )  # same direction after a small wiggle: same arc
        else:
            arcs.append((start, end, turn))
    return arcs


def split_into_arcs(points, turning):
    arcs = find_arcs(turning, min_turn=45)
    pairs = [(min(abs(a[2]), abs(b[2])), a, b) for a, b in zip(arcs, arcs[1:]) if np.sign(a[2]) != np.sign(b[2])]
    if pairs:
        _, a, b = max(pairs, key=lambda p: p[0])
        return (a[1] + b[0]) // 2, True
    return len(points) // 2, False  # no pair of opposite arcs: split in the middle


def describe_arc(arc):
    chord = arc[-1] - arc[0]
    normal = np.array([-chord[1], chord[0]]) / (np.linalg.norm(chord) + 1e-9)
    apex = arc[np.argmax(np.abs((arc - arc[0]) @ normal))]
    opening = (arc[0] + arc[-1]) / 2 - apex
    return {"length": float(np.hypot(*np.diff(arc, axis=0).T).sum()),
            "width": float(np.ptp(arc[:, 0])), "height": float(np.ptp(arc[:, 1])),
            "opening_from_vertical": float(np.degrees(np.arctan2(abs(opening[0]), abs(opening[1])))),
            "opens_down": int(opening[1] > 0)}  # image y points down


def ratio(a, b):
    return max(a, b) / max(min(a, b), 1e-6)


def measure_wave(mask):
    points, line_length = resample(centre_line(mask))
    smooth, turning = signed_turning(points)
    split, opposite = split_into_arcs(smooth, turning)
    first, second = describe_arc(smooth[:split + 1]), describe_arc(smooth[split:])
    ys, xs = np.nonzero(mask)
    axis = smooth[-1] - smooth[0]
    m = {f"n_arcs_{t}": len(find_arcs(turning, t)) for t in (45, 60, 90, 120)}
    m["largest_arc_turn"] = max((abs(a[2]) for a in find_arcs(turning, 45)), default=0.0)
    m.update({
        "opposite_arcs": int(opposite),
        "turn_first": abs(turning[:split].sum()), "turn_second": abs(turning[split:].sum()),
        "line_length": line_length / SIZE,
        "width_ratio": ratio(first["width"], second["width"]),
        "height_ratio": ratio(first["height"], second["height"]),
        "length_ratio": ratio(first["length"], second["length"]),
        "opening_from_vertical": (first["opening_from_vertical"] + second["opening_from_vertical"]) / 2,
        "axis_from_horizontal": float(np.degrees(np.arctan2(abs(axis[1]), abs(axis[0])))),
        "first_opens_down": first["opens_down"],
        "size": max(np.ptp(xs), np.ptp(ys)) / SIZE,
        "width_fraction": np.ptp(xs) / SIZE, "height_fraction": np.ptp(ys) / SIZE,
    })
    return m, smooth, split


MAX_ARC_TURN = 300  # an arc turns about a half circle (180 deg); a curl or loop (manual example 3) turns ~360 deg
def score_wave(m, params):
    basic = int(
        m[f"n_arcs_{params['min_arc_turn']}"] == 2
        and m["opposite_arcs"] == 1
        and m["largest_arc_turn"] <= MAX_ARC_TURN
    )
    facets = {
        "basic_shape": basic,
        "edges": int(m["width_ratio"] <= params["max_width_ratio"]),
        "orientation": int(m["opening_from_vertical"] <= params["max_opening_angle"]),
        "overall_size": int(m["size"] >= params["min_size"]),
    }
    if basic == 0:  # manual: basic shape 0 -> everything 0
        facets = {facet: 0 for facet in facets}
    facets["total"] = sum(facets[f] for f in FACETS)
    return facets
