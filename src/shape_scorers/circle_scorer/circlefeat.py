"""Geometric feature extraction for hand-drawn circle scoring.

Pipeline: load -> normalise background -> ink mask -> remove response-box border
-> isolate main drawing -> skeleton / contour / radial-ray analysis -> features.
All length features are normalised by the response-box size (BOX), so they are
independent of scan resolution.
"""
import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy import ndimage as ndi

WORK = 900  # working resolution (long side of the box, px)


def _redactions(img):
    """Pure-white rectangles pasted over photos (anonymisation). Returns mask."""
    g = img.min(axis=2)
    if np.median(g) > 232:          # white scan: redactions invisible / irrelevant
        return np.zeros(g.shape, np.uint8)
    w = (g >= 250).astype(np.uint8)
    lab, n = ndi.label(w)
    out = np.zeros_like(w)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        reg = lab[sl] == i
        if reg.sum() >= 30 and reg.mean() > 0.6:
            out[sl][reg] = 1
    return cv2.dilate(out, np.ones((7, 7), np.uint8))


def _ink_mask(img, red=None):
    """Thin-dark-structure (black-hat) ink map normalised by local background."""
    img = img.copy()
    if red is not None and red.any():
        fill = np.median(img[red == 0], axis=0)
        img[red > 0] = fill
    mn = cv2.GaussianBlur(img.min(axis=2), (0, 0), 1.0).astype(np.float32)
    k = 25
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clo = cv2.morphologyEx(mn, cv2.MORPH_CLOSE, se)
    bh = np.clip((clo - mn) / np.maximum(clo, 1), 0, 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = cv2.GaussianBlur(hsv[..., 1], (0, 0), 1.0).astype(np.float32) / 255.0
    sat_th = np.clip(sat - cv2.morphologyEx(sat, cv2.MORPH_OPEN, se), 0, 1)
    score = np.maximum(bh, 0.8 * sat_th)
    if red is not None and red.any():
        score[red > 0] = 0
    s8 = np.clip(score * 255, 0, 255).astype(np.uint8)
    med = np.median(s8); mad = np.median(np.abs(s8.astype(np.float32) - med)) + 0.5
    t, _ = cv2.threshold(s8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    strong_t = max(t, 30)
    weak_t = min(max(9, med + 7 * 1.4826 * mad), strong_t * 0.6)
    mask = (s8 > weak_t).astype(np.uint8)
    strong = s8 > strong_t
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    keep = np.zeros(n + 1, bool)
    keep[np.unique(lab[strong])] = True
    keep[0] = False
    mask = keep[lab].astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return score, mask


def _remove_border(mask):
    """Detect long straight lines in the outer band (the response box) and erase them.
    Returns cleaned mask and the inner box rectangle (x0,y0,x1,y1)."""
    H, W = mask.shape
    m = mask.copy()
    lines = cv2.HoughLinesP(mask * 255, 1, np.pi / 360, threshold=int(min(H, W) * 0.25),
                            minLineLength=int(min(H, W) * 0.35), maxLineGap=int(min(H, W) * 0.03))
    band = 0.12
    erase = np.zeros_like(mask)
    cand = {'t': [], 'b': [], 'l': [], 'r': []}
    if lines is not None:
        for (a, b, c, d) in lines.reshape(-1, 4).tolist():
            dx, dy = c - a, d - b
            ang = abs(np.degrees(np.arctan2(dy, dx))) % 180
            horiz = ang < 6 or ang > 174
            vert = 84 < ang < 96
            my, mx = (b + d) / 2, (a + c) / 2
            if horiz and my < band * H: cand['t'].append((my, max(b, d), (a, b, c, d)))
            elif horiz and my > (1 - band) * H: cand['b'].append((H - my, min(b, d), (a, b, c, d)))
            elif vert and mx < band * W: cand['l'].append((mx, max(a, c), (a, b, c, d)))
            elif vert and mx > (1 - band) * W: cand['r'].append((W - mx, min(a, c), (a, b, c, d)))
    lim = {'t': 0, 'b': H - 1, 'l': 0, 'r': W - 1}
    th = max(3, int(min(H, W) * 0.012))
    for side, L in cand.items():
        if not L: continue
        L.sort()
        outer = L[0][0]
        # keep only lines belonging to the outermost border (within a few px of it)
        grp = [l for l in L if l[0] - outer < min(H, W) * 0.025]
        for _, inner, (a, b, c, d) in grp:
            cv2.line(erase, (a, b), (c, d), 1, th)
        inn = [g[1] for g in grp]
        lim[side] = max(inn) if side in 'tl' else min(inn)
    x0, y0, x1, y1 = lim['l'], lim['t'], lim['r'], lim['b']
    m[erase > 0] = 0
    # anything outside the inner box (plus a small margin) is not the drawing
    mg = int(min(H, W) * 0.012)
    outside = np.ones_like(m)
    outside[y0 + mg:y1 - mg, x0 + mg:x1 - mg] = 0
    m[outside > 0] = 0
    return m, (x0, y0, x1, y1)


def _main_drawing(m, box):
    """Group ink components (bridging small breaks) and keep the dominant group."""
    x0, y0, x1, y1 = box
    S = max(x1 - x0, y1 - y0)
    lab, n = ndi.label(m, structure=np.ones((3, 3)))
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    # drop specks
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= max(15, 0.02 * sizes.max())
    m2 = keep[lab].astype(np.uint8)
    r = max(2, int(S * 0.03))
    grp = cv2.dilate(m2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    glab, gn = ndi.label(grp, structure=np.ones((3, 3)))
    if gn == 0:
        return m2
    gsz = ndi.sum(m2, glab, range(1, gn + 1))
    # extent of each group: prefer big drawings over dense small marks
    best = None
    for gi in range(1, gn + 1):
        ys, xs = np.nonzero((glab == gi) & (m2 > 0))
        if len(xs) == 0: continue
        ext = max(xs.max() - xs.min(), ys.max() - ys.min())
        sc = gsz[gi - 1] * (1 + ext / S)
        if best is None or sc > best[0]:
            best = (sc, gi)
    return ((glab == best[1]) & (m2 > 0)).astype(np.uint8)


def _skeleton_graph(sk):
    """Endpoints, branchpoints and spur (endpoint->junction) lengths."""
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    nb = cv2.filter2D(sk.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    ends = np.argwhere((sk > 0) & (nb == 1))
    branch = (sk > 0) & (nb >= 3)
    # spur lengths: walk from each endpoint until a branch point
    H, W = sk.shape
    spurs = []
    skb = sk > 0
    for (y, x) in ends:
        L, py, px, cy, cx = 0, -1, -1, y, x
        while True:
            nxt = None
            cnt = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0: continue
                    yy, xx = cy + dy, cx + dx
                    if 0 <= yy < H and 0 <= xx < W and skb[yy, xx] and not (yy == py and xx == px):
                        cnt += 1
                        if nxt is None: nxt = (yy, xx)
            if cnt != 1 or branch[cy, cx] or L > 5000:
                break
            L += 1
            py, px, cy, cx = cy, cx, nxt[0], nxt[1]
        spurs.append((L, branch[cy, cx]))
    return ends, branch, spurs


def extract(path, return_debug=False):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    H0, W0 = img.shape[:2]
    sc = WORK / max(H0, W0)
    img = cv2.resize(img, (int(W0 * sc), int(H0 * sc)), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    red = _redactions(img)
    score, mask = _ink_mask(img, red)
    m, box = _remove_border(mask)
    x0, y0, x1, y1 = box
    BW, BH = max(x1 - x0, 1), max(y1 - y0, 1)
    BOX = (BW + BH) / 2.0
    d = _main_drawing(m, box)
    allink = m
    f = {}
    ys, xs = np.nonzero(d)
    if len(xs) < 30:
        f['empty'] = 1
        return (f, None) if return_debug else f
    f['empty'] = 0
    pts = np.stack([xs, ys], 1).astype(np.float32)
    ink = len(xs)

    mg2 = int(BOX * 0.03)
    f['touch_border'] = int(xs.min() <= x0 + mg2 or ys.min() <= y0 + mg2 or xs.max() >= x1 - mg2 or ys.max() >= y1 - mg2)
    # ---- size -------------------------------------------------------------
    f['bbox_w'] = (xs.max() - xs.min()) / BOX
    f['bbox_h'] = (ys.max() - ys.min()) / BOX
    f['bbox_max'] = max(f['bbox_w'], f['bbox_h'])
    f['bbox_min'] = min(f['bbox_w'], f['bbox_h'])
    f['box_aspect'] = BW / BH

    # ---- stroke width & skeleton -----------------------------------------
    dt = cv2.distanceTransform(d, cv2.DIST_L2, 3)
    sk = skeletonize(d > 0)
    sklen = sk.sum()
    sw = 2 * np.median(dt[sk]) if sklen else 1.0
    f['stroke_w'] = sw / BOX
    f['sk_len'] = sklen / BOX
    f['ink_per_len'] = ink / max(sklen, 1) / max(sw, 1)  # >1 means thick / doubled strokes
    f['stroke_w_p90'] = 2 * np.percentile(dt[sk], 90) / max(sw, 1e-3)

    ends, branch, spurs = _skeleton_graph(sk)
    f['n_end'] = len(ends)
    f['n_branch'] = int(ndi.label(branch, structure=np.ones((3, 3)))[1])

    # ---- faint-ink recovery near the drawing outline (very light pens) -------
    ss = cv2.GaussianBlur(score, (0, 0), 2.0)
    inner = np.zeros_like(d); mg = int(BOX * 0.02)
    inner[y0 + mg:y1 - mg, x0 + mg:x1 - mg] = 1
    far = cv2.distanceTransform((1 - (allink > 0)).astype(np.uint8), cv2.DIST_L2, 3) > 15
    bgv = ss[(inner > 0) & far & (red == 0)]
    low = max(0.008, 1.3 * np.percentile(bgv, 99.5)) if len(bgv) > 100 else 1.0
    hullm = np.zeros_like(d); cv2.drawContours(hullm, [cv2.convexHull(pts.astype(np.int32))], -1, 1, int(BOX * 0.12))
    if len(pts) >= 5:
        try:
            el_ = cv2.fitEllipse(pts)
            if max(el_[1]) < 2.5 * BOX:
                cv2.ellipse(hullm, el_, 1, int(BOX * 0.12))
        except cv2.error:
            pass
    faint = ((ss > low) & (inner > 0) & (hullm > 0)).astype(np.uint8)
    faint = cv2.morphologyEx(faint, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) & (hullm > 0)
    fl, fn = ndi.label(faint, structure=np.ones((3, 3)))
    seed = np.unique(fl[(cv2.dilate(d, np.ones((7, 7), np.uint8)) > 0) & (fl > 0)])
    keepf = np.zeros(fn + 1, bool); keepf[seed] = True; keepf[0] = False
    faint = keepf[fl]
    dF = np.maximum(d, faint.astype(np.uint8))
    f['faint_added'] = (dF.sum() - d.sum()) / max(d.sum(), 1)
    f['faint_low'] = low

    # redaction blocks touching the stroke may hide part of it: give benefit of doubt
    def add_red(mm):
        out = mm.copy()
        if red.any():
            rl, rn = ndi.label(red)
            touch = np.unique(rl[(cv2.dilate(mm, np.ones((5, 5), np.uint8)) > 0) & (rl > 0)])
            for t in touch:
                if t: out[rl == t] = 1
        return out
    dR0 = add_red(d)
    dR = add_red(dF)
    f['red_touch'] = int(red.any() and (dR0.sum() > d.sum()))

    # ---- closure: smallest dilation radius that makes a big enclosed hole ----
    hull_area = cv2.contourArea(cv2.convexHull(pts))
    def gap_radius(mm):
        dtb = cv2.distanceTransform((1 - mm).astype(np.uint8), cv2.DIST_L2, 5)
        for r in list(range(1, 20)) + list(range(20, 160, 3)):
            cur = dtb <= r
            hole = ndi.binary_fill_holes(cur) & ~cur
            if hole.sum() > 0.3 * hull_area * (1 - r / 200.0) ** 2 * 0 + 0.15 * hull_area:
                return r
        return 160
    gap_r = gap_radius(dR)
    f['gap_close'] = 2 * gap_r / BOX
    f['gap_close_raw'] = 2 * gap_radius(d) / BOX

    # ---- overlap: tree parts of the skeleton after bridging small gaps -------
    br = max(2, int(round(sw)))
    dc = cv2.morphologyEx(dR0, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * br + 1, 2 * br + 1)))
    sk2 = skeletonize(dc > 0).astype(np.uint8)
    core = sk2.copy()
    kk = np.ones((3, 3), np.uint8); kk[1, 1] = 0
    for _ in range(3000):
        nb = cv2.filter2D(core, -1, kk, borderType=cv2.BORDER_CONSTANT)
        e = (core > 0) & (nb <= 1)
        if not e.any(): break
        core[e] = 0
    tree = (sk2 > 0) & (core == 0)
    f['loop_len'] = core.sum() / BOX
    f['tree_len'] = tree.sum() / BOX
    tl, tn = ndi.label(tree, structure=np.ones((3, 3)))
    tsz = np.sort(ndi.sum(tree, tl, range(1, tn + 1)))[::-1] if tn else np.array([0.0])
    f['spur_max'] = tsz[0] / BOX
    f['spur_2nd'] = (tsz[1] / BOX) if len(tsz) > 1 else 0.0
    f['n_spur_big'] = int((tsz / BOX > 0.03).sum())
    cl, cn = ndi.label(core, structure=np.ones((3, 3)))
    f['n_loops_comp'] = cn
    # number of independent cycles (holes) in the closed drawing
    holes = ndi.binary_fill_holes(dc > 0) & ~(dc > 0)
    hl, hn = ndi.label(holes)
    hs = ndi.sum(holes, hl, range(1, hn + 1)) if hn else np.array([0.0])
    f['n_holes_big'] = int((hs > 0.002 * BOX ** 2).sum())
    f['hole2_frac'] = (np.sort(hs)[::-1][1] / max(hs.max(), 1)) if hn > 1 else 0.0

    # ---- shape from filled outline ---------------------------------------
    # shape outline from the ink itself (not redaction blocks) when gaps are modest
    gF = gap_radius(dF)
    f['gap_close_ink'] = 2 * gF / BOX
    src, gg = (dF, gF) if gF <= 0.08 * BOX else (dR, gap_r)
    rr_ = max(gg, int(sw)) + 1
    ker_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr_ + 1, 2 * rr_ + 1))
    filled = ndi.binary_fill_holes(cv2.dilate(src, ker_)).astype(np.uint8)
    filled = cv2.erode(filled, ker_)
    filled = np.maximum(filled, ndi.binary_fill_holes(src).astype(np.uint8))
    if filled.sum() == 0: filled = src.copy()
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    A = cv2.contourArea(c); P = cv2.arcLength(c, True)
    hull = cv2.convexHull(c); HA = cv2.contourArea(hull); HP = cv2.arcLength(hull, True)
    f['circularity'] = 4 * np.pi * A / max(P * P, 1)
    f['hull_circularity'] = 4 * np.pi * HA / max(HP * HP, 1)
    f['solidity'] = A / max(HA, 1)
    f['area_frac'] = A / BOX ** 2
    f['equiv_diam'] = np.sqrt(4 * A / np.pi) / BOX
    (cx, cy), (ra, rb), ang = cv2.minAreaRect(c)
    f['rect_aspect'] = max(ra, rb) / max(min(ra, rb), 1)
    f['rect_fill'] = A / max(ra * rb, 1)      # circle ~0.785, square ~1.0
    # Feret diameters (longest / shortest caliper width) on hull
    hp = hull[:, 0, :].astype(np.float64)
    angs = np.linspace(0, np.pi, 180, endpoint=False)
    proj = hp @ np.stack([np.cos(angs), np.sin(angs)])
    widths = proj.max(0) - proj.min(0)
    f['feret_max'] = widths.max() / BOX
    f['feret_min'] = widths.min() / BOX
    f['feret_ratio'] = widths.max() / max(widths.min(), 1)
    f['feret_max_frac_box'] = f['feret_max']
    if len(c) >= 5:
        (ex, ey), (ea, eb), eang = cv2.fitEllipse(c)
        f['ellipse_ratio'] = max(ea, eb) / max(min(ea, eb), 1)
        # deviation of contour from fitted ellipse
        em = np.zeros_like(filled); cv2.ellipse(em, ((ex, ey), (ea, eb), eang), 1, -1)
        f['ellipse_iou'] = (em & filled).sum() / max((em | filled).sum(), 1)
    else:
        f['ellipse_ratio'] = 1; f['ellipse_iou'] = 0

    # corners: turning angle along a smoothed contour
    cc = c[:, 0, :].astype(np.float64)
    n = len(cc)
    if n > 30:
        # resample uniformly
        seg = np.sqrt((np.diff(np.vstack([cc, cc[:1]]), axis=0) ** 2).sum(1))
        s = np.concatenate([[0], np.cumsum(seg)])
        N = 256
        t = np.linspace(0, s[-1], N, endpoint=False)
        ccl = np.vstack([cc, cc[:1]])
        rx = np.interp(t, s, ccl[:, 0]); ry = np.interp(t, s, ccl[:, 1])
        for nm, sig in (('', 2.0), ('_s', 4.0)):
            sx = ndi.gaussian_filter1d(rx, sig, mode='wrap'); sy = ndi.gaussian_filter1d(ry, sig, mode='wrap')
            dx = np.gradient(sx); dy = np.gradient(sy)
            w = 6
            ax, ay = np.roll(dx, w), np.roll(dy, w)
            bx, by = np.roll(dx, -w), np.roll(dy, -w)
            turn = np.abs(np.arctan2(ax * by - ay * bx, ax * bx + ay * by))
            f['turn_max' + nm] = turn.max()
            f['turn_p95' + nm] = np.percentile(turn, 95)
            f['turn_std' + nm] = turn.std()
            # corner count: local maxima with turning > 60 deg
            pk = (turn > np.roll(turn, 1)) & (turn >= np.roll(turn, -1)) & (turn > np.radians(60))
            f['n_corner' + nm] = int(pk.sum())
        # corners after removing thin protrusions (tails / overlaps)
        ro = max(3, int(BOX * 0.035))
        fo = cv2.morphologyEx(filled, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ro + 1,) * 2))
        co_ = cv2.findContours(fo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if co_:
            co = max(co_, key=cv2.contourArea)[:, 0, :].astype(np.float64)
            sg = np.sqrt((np.diff(np.vstack([co, co[:1]]), axis=0) ** 2).sum(1)); s2 = np.concatenate([[0], np.cumsum(sg)])
            t2 = np.linspace(0, s2[-1], 256, endpoint=False); cl2 = np.vstack([co, co[:1]])
            ox = ndi.gaussian_filter1d(np.interp(t2, s2, cl2[:, 0]), 2, mode='wrap'); oy = ndi.gaussian_filter1d(np.interp(t2, s2, cl2[:, 1]), 2, mode='wrap')
            gx, gy = np.gradient(ox), np.gradient(oy)
            for w in (4, 8):
                ax2, ay2, bx2, by2 = np.roll(gx, w), np.roll(gy, w), np.roll(gx, -w), np.roll(gy, -w)
                tn = np.arctan2(ax2 * by2 - ay2 * bx2, ax2 * bx2 + ay2 * by2)   # signed turning
                f[f'oturn_max{w}'] = float(np.abs(tn).max())
                pk2 = (tn > np.roll(tn, 1)) & (tn >= np.roll(tn, -1)) & (tn > np.radians(50 if w == 4 else 70))
                f[f'ocorner_n{w}'] = int(pk2.sum())
                f[f'oturn_p90_{w}'] = float(np.percentile(np.abs(tn), 90))
            Ao = cv2.contourArea(co.astype(np.float32)); Po = cv2.arcLength(co.astype(np.float32), True)
            f['ocirc'] = 4 * np.pi * Ao / max(Po * Po, 1)
            (_, _), (oa, ob), _ = cv2.minAreaRect(co.astype(np.float32)); f['orect_fill'] = Ao / max(oa * ob, 1)
        # radius profile from centroid
        M = cv2.moments(c); mx, my = M['m10'] / M['m00'], M['m01'] / M['m00']
        rr = np.hypot(rx - mx, ry - my)
        f['radius_cv'] = rr.std() / rr.mean()
        # fit with Fourier: energy in harmonics 3..8 relative to mean radius (cornerness)
        F = np.abs(np.fft.rfft(rr)) / len(rr)
        f['harm2'] = F[2] / F[0]
        f['harm3_8'] = np.sqrt((F[3:9] ** 2).sum()) / F[0]
        f['harm4'] = F[4] / F[0]
        f['harm9p'] = np.sqrt((F[9:40] ** 2).sum()) / F[0]
        # polygon approximation vertex count
        for eps in (0.02, 0.04):
            ap = cv2.approxPolyDP(c, eps * P, True)
            f[f'poly_v{int(eps*100)}'] = len(ap)
    else:
        for k in ['turn_max', 'turn_p95', 'turn_std', 'n_corner', 'turn_max_s', 'turn_p95_s', 'turn_std_s',
                  'n_corner_s', 'radius_cv', 'harm2', 'harm3_8', 'harm4', 'harm9p', 'poly_v2', 'poly_v4']:
            f[k] = 0

    # ---- extra strokes near the drawing (re-tracing, second attempts) -------
    near = cv2.dilate(d, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(BOX * 0.2) | 1,) * 2))
    extra = (allink > 0) & (d == 0) & (near > 0)
    el, en_ = ndi.label(extra, structure=np.ones((3, 3)))
    elen = 0.0
    if en_:
        for i, sl in enumerate(ndi.find_objects(el), 1):
            comp = el[sl] == i
            if comp.sum() < 20: continue
            elen += skeletonize(comp).sum()
    f['extra_len'] = elen / BOX

    # ---- small holes inside the stroke (lens between two parallel strokes) ---
    holes_all = ndi.binary_fill_holes(dR0 > 0) & ~(dR0 > 0)
    hl2, hn2 = ndi.label(holes_all)
    small_area = 0; small_n = 0; small_len = 0
    if hn2:
        hs2 = ndi.sum(holes_all, hl2, range(1, hn2 + 1))
        big = hs2.max()
        for i, sl in enumerate(ndi.find_objects(hl2), 1):
            a_ = hs2[i - 1]
            if a_ >= 0.5 * big or a_ < 4: continue
            small_n += 1; small_area += a_
            hh, ww = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
            small_len = max(small_len, np.hypot(hh, ww))
    f['lens_n'] = small_n; f['lens_area'] = small_area / BOX ** 2; f['lens_len'] = small_len / BOX

    # ---- local stroke width along the main loop -----------------------------
    dtR = cv2.distanceTransform(dc, cv2.DIST_L2, 5)
    cw = dtR[core > 0] if core.sum() > 20 else dtR[sk2 > 0]
    if len(cw):
        medw = np.median(cw)
        f['wide_frac16'] = float((cw > 1.6 * medw).mean())
        f['wide_frac22'] = float((cw > 2.2 * medw).mean())
        f['w_p98'] = float(np.percentile(cw, 98) / max(medw, 0.5))
    else:
        f['wide_frac16'] = f['wide_frac22'] = f['w_p98'] = 0.0

    # ---- polar unwrap of the ink score around the centroid ------------------
    Mf = cv2.moments(filled); pcx, pcy = Mf['m10'] / Mf['m00'], Mf['m01'] / Mf['m00']
    sc_m = score * (cv2.dilate(np.maximum(dF, extra.astype(np.uint8)), np.ones((9, 9), np.uint8)) > 0)
    rmean = np.sqrt(filled.sum() / np.pi)
    NA = 720
    th_ = np.linspace(0, 2 * np.pi, NA, endpoint=False)
    # coarse polar to locate the stroke radius r0(theta)
    rc = np.linspace(0.3, 1.9, 240) * rmean
    mx_ = (pcx + np.outer(rc, np.cos(th_))).astype(np.float32); my_ = (pcy + np.outer(rc, np.sin(th_))).astype(np.float32)
    pc = cv2.remap(cv2.GaussianBlur(sc_m, (0, 0), 2), mx_, my_, cv2.INTER_LINEAR, borderValue=0)
    # contour-based r0: outer boundary of filled shape gives a robust centreline radius
    pf = cv2.remap(filled.astype(np.float32), mx_, my_, cv2.INTER_NEAREST, borderValue=0)
    idx = np.where(pf.any(0), pf.shape[0] - 1 - np.argmax(pf[::-1] > 0, axis=0), 0)
    r0 = rc[idx] - sw / 2
    r0 = ndi.median_filter(r0, 15, mode='wrap'); r0 = ndi.gaussian_filter1d(r0, 4, mode='wrap')
    NR = 128
    off = np.linspace(-0.25, 0.25, NR)
    RR = r0[None, :] * (1 + off[:, None])
    mapx = (pcx + RR * np.cos(th_)[None]).astype(np.float32); mapy = (pcy + RR * np.sin(th_)[None]).astype(np.float32)
    polar = cv2.remap(sc_m, mapx, mapy, cv2.INTER_LINEAR, borderValue=0)   # NR x NA (straightened)
    polar_d = polar
    from scipy.signal import find_peaks
    npk = np.zeros(NA); sep_ = np.zeros(NA)
    gmax = np.percentile(polar.max(0), 50) + 1e-6
    dr = (off[1] - off[0]) * rmean / BOX
    for j in range(NA):
        col = ndi.gaussian_filter1d(polar[:, j], 0.7)
        pk, pr = find_peaks(col, height=0.3 * gmax, prominence=0.15 * gmax, distance=2)
        npk[j] = len(pk)
        sep_[j] = (pk.max() - pk.min()) * dr if len(pk) > 1 else 0
    f['pk_multi_frac'] = float((npk >= 2).mean())
    f['pk_zero_frac'] = float((npk == 0).mean())
    def longest_(b):
        if b.all(): return len(b)
        b2 = np.concatenate([b, b]); best = cur = 0
        for v in b2:
            cur = cur + 1 if v else 0; best = max(best, cur)
        return min(best, len(b))
    sm = ndi.uniform_filter1d((npk >= 2).astype(float), 7, mode='wrap') > 0.5
    f['pk_multi_longest'] = longest_(sm) / NA
    f['pk_zero_longest'] = longest_(ndi.uniform_filter1d((npk == 0).astype(float), 5, mode='wrap') > 0.5) / NA
    f['pk_sep_max'] = float(np.sort(sep_)[-5])
    f['pk_sep_p95'] = float(np.percentile(sep_, 95))
    # radial spread of ink around centreline (2nd moment) per angle
    w_ = polar / (polar.sum(0, keepdims=True) + 1e-6)
    mu = (w_ * off[:, None]).sum(0); var = (w_ * (off[:, None] - mu) ** 2).sum(0)
    spr = np.sqrt(var); meds = np.median(spr) + 1e-6
    f['spr_hi_frac'] = float((spr > 2.0 * meds).mean())
    f['spr_hi_longest'] = longest_(ndi.uniform_filter1d((spr > 2.0 * meds).astype(float), 7, mode='wrap') > 0.5) / NA
    f['spr_p98'] = float(np.percentile(spr, 98) / meds)
    # radial ink mass per angle relative to median (thick / doubled)
    mass = polar.sum(0); medm = np.median(mass) + 1e-6
    f['mass_hi_frac'] = float((mass > 1.7 * medm).mean())
    f['mass_hi_longest'] = longest_(ndi.uniform_filter1d((mass > 1.7 * medm).astype(float), 5, mode='wrap') > 0.5) / NA
    f['mass_lo_longest'] = longest_(ndi.uniform_filter1d((mass < 0.25 * medm).astype(float), 3, mode='wrap') > 0.5) / NA
    f['mass_cv'] = float(mass.std() / medm)

    # ---- radial ray analysis (gaps / overlaps / double strokes) ----------
    M = cv2.moments(filled); mx, my = M['m10'] / M['m00'], M['m01'] / M['m00']
    R = int(np.hypot(*d.shape))
    nA = 360
    runs = np.zeros(nA); thick = np.zeros(nA); spread = np.zeros(nA)
    rmax = np.zeros(nA)
    rs = np.arange(0, R, 0.7)
    for i, a in enumerate(np.linspace(0, 2 * np.pi, nA, endpoint=False)):
        px = (mx + rs * np.cos(a)).astype(int); py = (my + rs * np.sin(a)).astype(int)
        ok = (px >= 0) & (py >= 0) & (px < d.shape[1]) & (py < d.shape[0])
        v = d[py[ok], px[ok]] > 0
        if v.sum() == 0:
            continue
        dv = np.diff(np.concatenate([[0], v.astype(int), [0]]))
        st = np.nonzero(dv == 1)[0]; en = np.nonzero(dv == -1)[0]
        lens = (en - st) * 0.7
        # merge runs separated by < 1px
        runs[i] = len(st)
        thick[i] = lens.sum()
        spread[i] = (en.max() - st.min()) * 0.7
        rmax[i] = en.max() * 0.7
    medth = np.median(thick[runs > 0]) if (runs > 0).any() else 1
    f['ray_zero_frac'] = (runs == 0).mean()
    f['ray_multi_frac'] = (runs >= 2).mean()
    f['ray_thick_frac'] = (thick > 1.8 * medth).mean()
    f['ray_spread_frac'] = (spread > 2.5 * medth).mean()
    f['ray_spread_max'] = spread.max() / BOX
    # longest contiguous angular run of "doubled" rays
    dbl = (runs >= 2) | (spread > 2.5 * medth)
    def longest(b):
        if b.all(): return len(b)
        b2 = np.concatenate([b, b]); best = cur = 0
        for v in b2:
            cur = cur + 1 if v else 0; best = max(best, cur)
        return min(best, len(b))
    f['ray_dbl_longest'] = longest(dbl) / nA
    f['ray_zero_longest'] = longest(runs == 0) / nA
    # radial separation of the two strokes where doubled (overlap distance)
    sep = spread[dbl] / BOX
    f['ray_dbl_sep_max'] = sep.max() if len(sep) else 0.0
    f['ray_dbl_sep_mean'] = sep.mean() if len(sep) else 0.0
    # gap arc length (in box units)
    meanr = np.median(rmax[rmax > 0]) if (rmax > 0).any() else 1
    f['ray_gap_arc'] = f['ray_zero_longest'] * 2 * np.pi * meanr / BOX

    # position of drawing within box (for sanity)
    f['cx_rel'] = (mx - x0) / BW; f['cy_rel'] = (my - y0) / BH

    if return_debug:
        dbg = dict(dF=dF, ss=ss, low=low, polar=polar, polar_d=polar_d, red=red, core=core, tree=tree, img=img, mask=mask, drawing=d, skel=sk, box=box, contour=c, ends=ends, center=(mx, my))
        return f, dbg
    return f
