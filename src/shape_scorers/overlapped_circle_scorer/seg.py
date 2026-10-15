"""Ink segmentation for overlapping-circle drawings."""
import cv2
import numpy as np

import os
WORK = int(os.environ.get("OC_WORK", "512"))  # working resolution (longest side)


def load(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = WORK / max(h, w)
    img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


def ink_mask(img):
    """Return binary mask of pen strokes plus a continuous 'inkness' map."""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Background estimate: large median blur on a closed image (removes thin strokes)
    k = 31
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    bg = cv2.medianBlur(closed.astype(np.uint8), k).astype(np.float32)
    dark = np.clip(bg - gray, 0, None)
    # Colour cue: chroma relative to background colour (blue/purple/red pens)
    bgc = cv2.medianBlur(cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))), k).astype(np.float32)
    chroma_px = f.max(2) - f.min(2)
    chroma_bg = bgc.max(2) - bgc.min(2)
    chroma = np.clip(chroma_px - chroma_bg, 0, None)
    inkness = np.maximum(dark, 0.8 * chroma)
    # suppress white privacy boxes: pixels much brighter than local paper and near-white
    bg_paper = np.median(gray)
    white = ((gray > 235) & (gray > bg_paper + 18) & (chroma_px < 25)).astype(np.uint8)
    if white.any():
        n, lab, st, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
        boxes = np.zeros_like(white)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= 30:
                boxes[lab == i] = 1
        boxes = cv2.dilate(boxes, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
        inkness[boxes > 0] = 0
    # Ignore a thin margin where the paper frame line lives
    m = int(0.025 * min(h, w))
    inkness[:m, :] = 0; inkness[-m:, :] = 0; inkness[:, :m] = 0; inkness[:, -m:] = 0
    strong = np.percentile(inkness[inkness > 0], 99.7) if (inkness > 0).any() else 1
    thr = max(12.0, min(0.33 * strong, 44.0))
    mask = (inkness > thr).astype(np.uint8)
    # hysteresis: keep weak pixels connected to strong ones
    weak = (inkness > max(8.0, min(0.18 * strong, 0.62 * thr))).astype(np.uint8)
    # hue-guided recovery of faint strokes drawn with the same (coloured) pen
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0].astype(np.float32) * 2.0
    core = (inkness > thr) & (chroma_px > 40)
    if core.sum() > 200:
        hh_ = hue[core]
        ang = np.deg2rad(hh_)
        mh = np.rad2deg(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) % 360
        dh = np.abs(((hue - mh) + 180) % 360 - 180)
        faint = (inkness > max(10.0, 0.35 * thr)) & (chroma > 12) & (dh < 22)
        weak = weak | faint.astype(np.uint8)
    n, lab = cv2.connectedComponents(weak, connectivity=8)
    keep = np.zeros(n, bool)
    keep[np.unique(lab[mask > 0])] = True
    keep[0] = False
    mask = keep[lab].astype(np.uint8)
    # remove components touching image border (frame lines, edge of paper) and tiny specks
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    total = st[1:, cv2.CC_STAT_AREA].sum() if n > 1 else 0
    for i in range(1, n):
        x, y, ww, hh, a = st[i]
        touches = x <= m + 1 or y <= m + 1 or x + ww >= w - m - 1 or y + hh >= h - m - 1
        long_thin = max(ww, hh) > 0.6 * min(h, w) and min(ww, hh) < 0.06 * min(h, w)
        if a < max(25, 0.004 * total):
            continue
        if touches and long_thin:
            continue
        # straight line segments (frame edges, ruler lines): PCA on component pixels
        if max(ww, hh) > 0.2 * min(h, w):
            yy, xx = np.nonzero(lab == i)
            c = np.cov(np.stack([xx, yy]).astype(np.float32))
            ev = np.sort(np.linalg.eigvalsh(c))
            if ev[0] < 0.004 * ev[1] or np.sqrt(max(ev[0], 0)) < 3.0:
                continue
        out[lab == i] = 1
    return out, inkness, thr
