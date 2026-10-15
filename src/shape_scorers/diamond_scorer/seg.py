"""Ink segmentation for diamond drawings.
Returns a binary mask of the drawing (frame lines, specks, redaction patches removed)
at a standard working size of S x S-ish pixels (max side = S).
"""
import numpy as np, cv2
from skimage.filters import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects

S = 600  # working resolution (max side)


def load(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = im.shape[:2]
    f = S / max(h, w)
    im = cv2.resize(im, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)
    return im


def _norm_blur(x, w, k):
    num = cv2.GaussianBlur(x * w, (0, 0), k)
    den = cv2.GaussianBlur(w, (0, 0), k)
    return num / np.maximum(den, 1e-6)


def ink_score(im):
    """Per-pixel 'inkiness': perceptual (CIELAB) difference from the local paper colour,
    counting only darkening in lightness plus any chroma change.  Pale coloured pens on
    grey paper are much better separated this way than by grey level or HSV saturation."""
    f = im.astype(np.float32) / 255.0
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # redaction patches: near-pure white flat regions on non-white (photographed) paper
    patch = (f.min(2) > 0.97).astype(np.uint8)
    if np.median(gray) > 0.9 or patch.mean() > 0.5:
        patch[:] = 0
    patch = cv2.morphologyEx(patch, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    src = im.copy()
    if patch.any():
        src[patch > 0] = np.median(im[patch == 0], axis=0).astype(np.uint8)
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    bg = np.stack([cv2.medianBlur(lab[..., i], 31) for i in range(3)], -1).astype(np.float32)
    d = lab.astype(np.float32) - bg
    dE = np.sqrt(np.clip(-d[..., 0], 0, None) ** 2 + (2 * d[..., 1]) ** 2 + (2 * d[..., 2]) ** 2)
    score = dE / 100.0
    pb = cv2.dilate(patch, np.ones((7, 7), np.uint8))
    score[pb > 0] = 0
    return score, None, None


def remove_frame(mask):
    """Remove long, straight horizontal/vertical lines (the response-box frame)."""
    h, w = mask.shape
    m = mask.astype(np.uint8)
    L = int(0.25 * min(h, w))
    hor = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    ver = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    lines = cv2.dilate(hor | ver, np.ones((7, 7), np.uint8))
    # only remove lines in the outer band (frame lives near borders); drawings are inside
    band = np.zeros_like(m)
    b = int(0.12 * min(h, w))
    band[:b, :] = 1; band[-b:, :] = 1; band[:, :b] = 1; band[:, -b:] = 1
    out = mask.copy()
    out[(lines > 0) & (band > 0)] = False
    # thin border strip is always discarded (paper edge / shadow)
    e = int(0.015 * min(h, w))
    out[:e, :] = False; out[-e:, :] = False; out[:, :e] = False; out[:, -e:] = False
    return out, (hor | ver)


def frame_box(lines, shape):
    """Estimate the response-box extent from detected frame lines; fallback to image."""
    h, w = shape
    ys, xs = np.nonzero(lines)
    rows = np.nonzero(lines.sum(1) > 0.25 * w)[0]
    cols = np.nonzero(lines.sum(0) > 0.25 * h)[0]
    top = rows[rows < h * 0.3]; bot = rows[rows > h * 0.7]
    lef = cols[cols < w * 0.3]; rig = cols[cols > w * 0.7]
    y0 = top.max() if len(top) else 0
    y1 = bot.min() if len(bot) else h - 1
    x0 = lef.max() if len(lef) else 0
    x1 = rig.min() if len(rig) else w - 1
    return x0, y0, x1, y1


def segment(path, lo=None, hi=0.28):
    im = load(path)
    score, dark, col = ink_score(im)
    score = cv2.GaussianBlur(score, (0, 0), 0.8)
    if lo is None:
        # adaptive low threshold: just above paper-texture noise
        noise = np.percentile(score[score < 0.3], 95)
        lo = float(np.clip(2.5 * noise, 0.09, 0.16))
    mask = apply_hysteresis_threshold(score, lo, hi)
    # hysteresis may grow into speckle; a light opening removes hair-like noise
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
    mask, lines = remove_frame(mask)
    fb = frame_box(lines, mask.shape)
    mask = remove_small_objects(mask, max_size=40)
    # keep the main drawing: components whose area is meaningful relative to the largest,
    # and that lie near the largest component (merge across small gaps)
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return im, mask, fb, score
    areas = st[1:, cv2.CC_STAT_AREA]
    big = 1 + int(np.argmax(areas))
    # group components via dilation proximity
    grp = cv2.dilate((lab == big).astype(np.uint8), np.ones((3, 3), np.uint8))
    keep = np.zeros(n, bool); keep[big] = True
    # iterative: add comps within 30px of current group, with area >= 4% of largest
    D = cv2.distanceTransform((1 - grp).astype(np.uint8), cv2.DIST_L2, 5)
    changed = True
    while changed:
        changed = False
        cur = np.isin(lab, np.nonzero(keep)[0])
        D = cv2.distanceTransform((~cur).astype(np.uint8), cv2.DIST_L2, 5)
        for i in range(1, n):
            if keep[i]:
                continue
            a = st[i, cv2.CC_STAT_AREA]
            dmin = D[lab == i].min()
            if (a >= 0.04 * areas.max() and dmin < 60) or (a >= 0.15 * areas.max()):
                keep[i] = True; changed = True
    mask = np.isin(lab, np.nonzero(keep)[0])
    return im, mask, fb, score
