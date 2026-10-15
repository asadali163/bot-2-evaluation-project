"""Label parsing + image preprocessing for Overlapped Pencils scoring.

Output of preprocess(): a float32 'ink map' in [0,1] (1 = ink), background,
paper colour, ink colour, redaction boxes and frame lines removed.
"""
import re, os, glob
import numpy as np
import cv2

FACETS = ["basic_shape", "closure", "edges", "orientation", "overlap", "overall_size"]
MAX_SIDE = 768


def parse_label(fname):
    """Return (total, facets[6] or None, note)."""
    b = os.path.basename(fname)
    m = re.match(r"img(\d+)-p-(\d)(?:\((\d+)\))?\.(png|jpe?g)$", b, re.I)
    if not m:
        return None, None, "unparsed"
    total = int(m.group(2))
    code = m.group(3)
    if code is None:
        if total == 6:
            return 6, [1] * 6, ""
        if total == 0:  # basic shape failed -> everything 0 by rule
            return 0, [0] * 6, ""
        return total, None, "no facet code"
    if len(code) != 6:
        return total, None, f"facet code has {len(code)} digits"
    f = [int(c) for c in code]
    if sum(f) != total:
        return total, None, f"facet code sums to {sum(f)} not {total}"
    return total, f, ""


def load_dataset(folder):
    rows = []
    for p in sorted(glob.glob(os.path.join(folder, "*"))):
        if not re.search(r"\.(png|jpe?g)$", p, re.I):
            continue
        t, f, note = parse_label(p)
        if t is None:
            print("skip", p)
            continue
        rows.append(dict(path=p, name=os.path.basename(p), total=t, facets=f, note=note))
    return rows


def _resize(img):
    h, w = img.shape[:2]
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def preprocess(path, return_debug=False):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    bgr = _resize(bgr)
    h, w = bgr.shape[:2]
    # 1) fill pure-white redaction boxes / glare with the median paper colour
    white = (bgr.min(axis=2) >= 245).astype(np.uint8)
    paper_med = np.median(bgr.reshape(-1, 3), axis=0)
    if white.mean() < 0.5:  # photo on grey paper: white blobs are redactions
        white = cv2.dilate(white, np.ones((7, 7), np.uint8))
        bgr = bgr.copy()
        bgr[white > 0] = paper_med.astype(np.uint8)

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0]
    a = lab[..., 1] - 128
    bb = lab[..., 2] - 128
    k = max(31, (min(h, w) // 12) | 1)
    # 2) local background estimate (strokes are thin -> median removes them)
    Lbg = cv2.medianBlur(L.astype(np.uint8), k).astype(np.float32)
    abg = cv2.medianBlur((a + 128).astype(np.uint8), k).astype(np.float32) - 128
    bbg = cv2.medianBlur((bb + 128).astype(np.uint8), k).astype(np.float32) - 128
    dark = np.clip(Lbg - L, 0, None)                      # darker than paper
    chroma = np.sqrt((a - abg) ** 2 + (bb - bbg) ** 2)    # colour vs paper
    score = dark + 1.5 * chroma
    # robust normalisation
    bh_, bw_ = int(0.08 * h), int(0.08 * w)
    interior = score[bh_:h - bh_, bw_:w - bw_]
    hi = np.percentile(interior, 99.5)
    noise = np.median(score) + 3 * (np.median(np.abs(score - np.median(score))) * 1.4826 + 1e-3)
    ink = np.clip((score - noise) / max(hi - noise, 1e-3), 0, 1)
    ink = cv2.GaussianBlur(ink, (3, 3), 0)

    # 3) binary mask, remove tiny specks and frame/border lines
    thr = 0.25
    strong = ink > thr
    weak = (ink > 0.10).astype(np.uint8)
    nw, lw = cv2.connectedComponents(weak, connectivity=8)
    good = np.unique(lw[strong]); good = good[good > 0]
    mask = np.isin(lw, good).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros(n, bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = areas.max() if len(areas) else 0
    for i in range(1, n):
        x, y, ww, hh, ar = stats[i]
        if ar < max(25, 0.01 * big):
            continue
        touches = x <= 2 or y <= 2 or x + ww >= w - 2 or y + hh >= h - 2
        line_h = ww > 0.5 * w and hh < 0.06 * h
        line_v = hh > 0.5 * h and ww < 0.06 * w
        frame = False
        if ww > 0.6 * w or hh > 0.6 * h:  # possibly the printed box: mostly along the borders?
            ys_, xs_ = np.where(lbl[y:y + hh, x:x + ww] == i)
            ys_ = ys_ + y; xs_ = xs_ + x
            b = 0.08
            inband = (xs_ < b * w) | (xs_ > (1 - b) * w) | (ys_ < b * h) | (ys_ > (1 - b) * h)
            frame = inband.mean() > 0.6
        near_edge = (min(x, y) < 0.06 * min(h, w)) or (x + ww > w * 0.94) or (y + hh > h * 0.94)
        if (line_h or line_v) and near_edge:
            continue
        if frame:
            continue
        if touches and ar < 0.05 * big:
            continue
        keep[i] = True
    clean = keep[lbl]
    # remove frame lines merged with drawing: thin straight lines hugging the borders
    m = int(0.035 * min(h, w))
    for side in range(4):
        sl = [slice(None), slice(None)]
        if side == 0: sl[0] = slice(0, m)
        if side == 1: sl[0] = slice(h - m, h)
        if side == 2: sl[1] = slice(0, m)
        if side == 3: sl[1] = slice(w - m, w)
        strip = clean[tuple(sl)]
        axis = 1 if side < 2 else 0
        frac = strip.mean(axis=axis)
        rows_line = frac > 0.5
        if side < 2:
            strip[rows_line, :] = False
        else:
            strip[:, rows_line] = False
    # 4) erase long, axis-aligned straight lines in the outer band (printed frame)
    cu = clean.astype(np.uint8) * 255
    lines = cv2.HoughLinesP(cu, 1, np.pi / 180, threshold=60,
                            minLineLength=int(0.2 * min(h, w)), maxLineGap=12)
    band = 0.16
    if lines is not None:
        for lx1, ly1, lx2, ly2 in lines.reshape(-1, 4):
            dx, dy = abs(lx2 - lx1), abs(ly2 - ly1)
            horiz = dy < 0.04 * max(dx, 1)
            vert = dx < 0.04 * max(dy, 1)
            if horiz and (max(ly1, ly2) < band * h or min(ly1, ly2) > (1 - band) * h):
                cv2.line(cu, (int(lx1), int(ly1)), (int(lx2), int(ly2)), 0, 9)
            elif vert and (max(lx1, lx2) < band * w or min(lx1, lx2) > (1 - band) * w):
                cv2.line(cu, (int(lx1), int(ly1)), (int(lx2), int(ly2)), 0, 9)
    clean = cu > 0
    # 5) keep only components belonging to the drawing cluster
    n, lbl, stats, cents = cv2.connectedComponentsWithStats(clean.astype(np.uint8), 8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        big = areas.max()
        main = [i + 1 for i, a in enumerate(areas) if a >= 0.08 * big]
        X0 = min(stats[i, 0] for i in main); Y0 = min(stats[i, 1] for i in main)
        X1 = max(stats[i, 0] + stats[i, 2] for i in main); Y1 = max(stats[i, 1] + stats[i, 3] for i in main)
        mg = 0.04 * max(h, w)
        keep = np.zeros(n, bool)
        for i in range(1, n):
            x, y, ww, hh, ar = stats[i]
            if i in main:
                keep[i] = True
            elif x >= X0 - mg and y >= Y0 - mg and x + ww <= X1 + mg and y + hh <= Y1 + mg:
                keep[i] = True
        clean = keep[lbl]
    ink = ink * cv2.dilate(clean.astype(np.uint8), np.ones((5, 5), np.uint8))
    if return_debug:
        return ink.astype(np.float32), clean, bgr
    return ink.astype(np.float32)


def crop_canvas(ink, size=384, pad=0.08, tight=True):
    """Crop to drawing bounding box (tight) and place on a square canvas.
    Returns uint8 image with black strokes on white, plus bbox info."""
    h, w = ink.shape
    ys, xs = np.where(ink > 0.2)
    if len(xs) < 10:
        ys, xs = np.array([0, h - 1]), np.array([0, w - 1])
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    if not tight:
        x0, y0, x1, y1 = 0, 0, w - 1, h - 1
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    side = int(max(bw, bh) * (1 + 2 * pad))
    canvas = np.zeros((side, side), np.float32)
    cx, cy = (side - bw) // 2, (side - bh) // 2
    canvas[cy:cy + bh, cx:cx + bw] = ink[y0:y1 + 1, x0:x1 + 1]
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    canvas = np.clip(canvas * 1.6, 0, 1)
    img = (255 * (1 - canvas)).astype(np.uint8)
    return img, (x0, y0, x1, y1, w, h)
