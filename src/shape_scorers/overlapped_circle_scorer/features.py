"""Two-ellipse fitting and rubric-aligned geometric features."""
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .seg import load, ink_mask

NS = 360  # samples around each ellipse


def clean_components(mask):
    """Keep the main drawing; drop far-away stray marks."""
    n, lab, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    areas = st[1:, cv2.CC_STAT_AREA]
    big = np.argmax(areas) + 1
    x, y, w, h = st[big, :4]
    out = np.zeros_like(mask)
    total = areas.sum()
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        xi, yi, wi, hi = st[i, :4]
        # distance between bounding boxes
        dx = max(0, max(x, xi) - min(x + w, xi + wi))
        dy = max(0, max(y, yi) - min(y + h, yi + hi))
        d = np.hypot(dx, dy)
        if a >= 0.15 * total or d < 25 or (a >= 0.04 * total and d < 60):
            out[lab == i] = 1
    return out


def prune(skel, L):
    sk = skel.copy().astype(np.uint8)
    k = np.ones((3, 3), np.uint8)
    for _ in range(L):
        nb = cv2.filter2D(sk, -1, k, borderType=cv2.BORDER_CONSTANT) - sk
        ends = (sk == 1) & (nb <= 1)
        if not ends.any():
            break
        sk[ends] = 0
    return sk


def endpoints(sk):
    k = np.ones((3, 3), np.uint8)
    nb = cv2.filter2D(sk.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT) - sk
    ys, xs = np.nonzero((sk == 1) & (nb == 1))
    return np.stack([xs, ys], 1).astype(np.float32)


def ellipse_pts(e, n=NS):
    (cx, cy), (w, h), ang = e
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a, b = w / 2, h / 2
    c, s = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    x = a * np.cos(t); y = b * np.sin(t)
    return np.stack([cx + c * x - s * y, cy + s * x + c * y], 1)


def fit(pts):
    if len(pts) < 6:
        return None
    try:
        e = cv2.fitEllipseDirect(pts.astype(np.float32))
    except cv2.error:
        return None
    (cx, cy), (w, h), ang = e
    if not np.isfinite([cx, cy, w, h]).all() or min(w, h) < 4 or max(w, h) > 3000:
        return None
    return e


def two_ellipse_fit(P, tol):
    """k-ellipses EM. P: Nx2 skeleton points. Returns (e1, e2, assign, dists)."""
    best = None
    inits = []
    ymed = np.median(P[:, 1]); xmed = np.median(P[:, 0])
    inits.append(P[:, 1] < ymed)
    inits.append(P[:, 1] < np.percentile(P[:, 1], 40))
    inits.append(P[:, 1] < np.percentile(P[:, 1], 60))
    inits.append(P[:, 0] < xmed)
    d = (P - P.mean(0))
    u, s, vt = np.linalg.svd(d, full_matrices=False)
    inits.append(d @ vt[0] < 0)
    for g in inits:
        e1, e2 = fit(P[g]), fit(P[~g])
        if e1 is None or e2 is None:
            continue
        for it in range(25):
            t1, t2 = cKDTree(ellipse_pts(e1, 720)), cKDTree(ellipse_pts(e2, 720))
            d1, _ = t1.query(P); d2, _ = t2.query(P)
            a = d1 <= d2
            dm = np.minimum(d1, d2)
            inl = dm < max(3 * tol, np.percentile(dm, 90)) if it < 5 else dm < 3 * tol
            n1, n2 = fit(P[a & inl]), fit(P[~a & inl])
            if n1 is None or n2 is None:
                break
            e1, e2 = n1, n2
        t1, t2 = cKDTree(ellipse_pts(e1, 720)), cKDTree(ellipse_pts(e2, 720))
        d1, _ = t1.query(P); d2, _ = t2.query(P)
        dm = np.minimum(d1, d2)
        cost = np.mean(np.minimum(dm, 4 * tol) ** 2)
        if best is None or cost < best[0]:
            best = (cost, e1, e2, d1 <= d2, dm, d1, d2)
    if best is None:
        return None
    _, e1, e2, a, dm, d1, d2 = best
    # order: e1 = upper
    if e1[0][1] > e2[0][1]:
        e1, e2, a, d1, d2 = e2, e1, ~a, d2, d1
    return e1, e2, a, dm, d1, d2


def ell_axes(e):
    (cx, cy), (w, h), ang = e
    return max(w, h), min(w, h)


def ellipse_overlap_area(e1, e2, shape):
    H, W = shape
    pad = 200
    m1 = np.zeros((H + 2 * pad, W + 2 * pad), np.uint8); m2 = m1.copy()
    off = lambda e: ((e[0][0] + pad, e[0][1] + pad), e[1], e[2])
    cv2.ellipse(m1, off(e1), 1, -1); cv2.ellipse(m2, off(e2), 1, -1)
    a1, a2, i = m1.sum(), m2.sum(), (m1 & m2).sum()
    return a1, a2, i


def coverage_features(e, other, P_own, mask, sw, tol, inkness=None):
    """Gap / double-stroke analysis around one ellipse."""
    E = ellipse_pts(e, NS)
    seg_len = np.linalg.norm(np.roll(E, -1, 0) - E, axis=1)
    perim = seg_len.sum()
    # coverage by own skeleton points
    if len(P_own):
        tr = cKDTree(P_own)
        dcov, _ = tr.query(E)
    else:
        dcov = np.full(NS, 1e9)
    covered = dcov < max(tol * 1.6, 0.06 * ell_axes(e)[0] / 2 + sw)
    # longest uncovered run (circular)
    unc = ~covered
    if unc.all():
        max_gap = perim
    else:
        r = np.roll(unc, -np.argmax(~unc))
        runs, cur = [], 0
        L = np.roll(seg_len, -np.argmax(~unc))
        for u, l in zip(r, L):
            if u:
                cur += l
            else:
                if cur: runs.append(cur)
                cur = 0
        if cur: runs.append(cur)
        max_gap = max(runs) if runs else 0.0
    tot_gap = seg_len[unc].sum()
    # double stroke: count ink runs along the normal through each sample
    (cx, cy), _, _ = e
    nrm = E - np.array([cx, cy]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
    other_t = cKDTree(ellipse_pts(other, 720))
    H, W = mask.shape
    band = max(4 * sw, 0.12 * ell_axes(e)[1] / 2)
    offs = np.arange(-band, band + 0.5, 0.5)
    runs_n = np.zeros(NS, int)
    near_other = np.zeros(NS, bool)
    do, _ = other_t.query(E)
    near_other = do < band + 2 * sw
    width = np.full(NS, np.nan); dark = np.full(NS, np.nan); roff = np.full(NS, np.nan)
    zero = np.argmin(np.abs(offs))
    for i in range(NS):
        q = E[i][None, :] + offs[:, None] * nrm[i][None, :]
        xi = np.clip(np.round(q[:, 0]).astype(int), 0, W - 1)
        yi = np.clip(np.round(q[:, 1]).astype(int), 0, H - 1)
        v = mask[yi, xi] > 0
        runs_n[i] = np.sum(v[1:] & ~v[:-1]) + int(v[0])
        if v.any() and inkness is not None:
            # run nearest to the ellipse
            lab_, nl = ndi.label(v)
            idx = np.nonzero(v)[0]
            j = idx[np.argmin(np.abs(idx - zero))]
            run = lab_ == lab_[j]
            width[i] = run.sum() * 0.5
            dark[i] = inkness[yi[run], xi[run]].mean()
            roff[i] = offs[run].mean()
    dbl = (runs_n >= 2) & ~near_other
    dbl_len = seg_len[dbl].sum()
    # longest contiguous double-run arc
    max_dbl = 0.0; cur = 0.0
    for k in range(2 * NS):
        i = k % NS
        if dbl[i]:
            cur += seg_len[i]; max_dbl = max(max_dbl, min(cur, perim))
        else:
            cur = 0.0
    ok = ~near_other & ~np.isnan(width)
    wr = dr = jump = thick_len = 0.0
    if ok.sum() > 30:
        wmed = np.nanmedian(width[ok]); dmed = np.nanmedian(dark[ok])
        ws = width.copy(); ws[~ok] = wmed
        dk = dark.copy(); dk[~ok] = dmed
        ws_s = ndi.uniform_filter1d(ws, 7, mode='wrap'); dk_s = ndi.uniform_filter1d(dk, 7, mode='wrap')
        wr = float(ws_s.max() / max(wmed, 1e-3)); dr = float(dk_s.max() / max(dmed, 1e-3))
        thick_len = float(seg_len[(ws_s > 1.35 * wmed) & ok].sum())
        ro = roff.copy(); ro[~ok] = np.nan
        dif = np.abs(np.diff(np.concatenate([ro, ro[:1]])))
        dif = dif[~np.isnan(dif)]
        jump = float(dif.max() / max(sw, 1)) if dif.size else 0.0
    return dict(max_gap=max_gap, tot_gap=tot_gap, dbl_len=dbl_len, max_dbl=max_dbl, perim=perim,
                wratio=wr, dratio=dr, jump=jump, thick_len=thick_len)


def radial_residual(e, P_own):
    """Robust shape-irregularity: deviation of own points from ellipse, by angle."""
    if len(P_own) < 10:
        return 1.0, 1.0
    E = ellipse_pts(e, 720)
    d, _ = cKDTree(E).query(P_own)
    s = ell_axes(e)[0] / 2 + ell_axes(e)[1] / 2
    return float(np.mean(d) / s * 2), float(np.percentile(d, 95) / s * 2)


def corner_score(sk_own_pts, e):
    """Max turning-angle based corner detection along ellipse-angle-sorted points."""
    if len(sk_own_pts) < 30:
        return 0.0
    (cx, cy), _, _ = e
    ang = np.arctan2(sk_own_pts[:, 1] - cy, sk_own_pts[:, 0] - cx)
    r = np.hypot(sk_own_pts[:, 1] - cy, sk_own_pts[:, 0] - cx)
    bins = np.linspace(-np.pi, np.pi, 73)
    idx = np.digitize(ang, bins) - 1
    rr = np.array([np.median(r[idx == i]) if (idx == i).any() else np.nan for i in range(72)])
    ok = ~np.isnan(rr)
    if ok.sum() < 36:
        return 0.0
    # interpolate missing, compute local peakiness of radius (corners -> sharp local maxima)
    xs = np.arange(72)
    rr = np.interp(xs, xs[ok], rr[ok], period=72)
    rr_s = ndi.uniform_filter1d(rr, 9, mode='wrap')
    peak = (rr - rr_s) / np.median(rr)
    # second difference (curvature in polar radius)
    d2 = (np.roll(rr, 2) + np.roll(rr, -2) - 2 * rr) / np.median(rr)
    return float(max(peak.max(), -d2.min()))


def extract(path, return_debug=False):
    img, s = load(path)
    H, W = img.shape[:2]
    mask, inkness, thr = ink_mask(img)
    mask = clean_components(mask)
    f = {}
    ink_area = int(mask.sum())
    f['ink_area'] = ink_area / (H * W)
    if ink_area < 50:
        f['fit_ok'] = 0
        return (f, None) if return_debug else f
    sk = skeletonize(mask > 0).astype(np.uint8)
    skl = int(sk.sum())
    sw = ink_area / max(skl, 1)  # stroke width estimate
    f['stroke_w'] = sw
    ys, xs = np.nonzero(sk)
    P = np.stack([xs, ys], 1).astype(np.float64)
    tol = max(2.0, sw)
    res = two_ellipse_fit(P, tol)
    # drawing bbox
    my, mx = np.nonzero(mask)
    bw, bh = mx.max() - mx.min() + 1, my.max() - my.min() + 1
    f['bbox_w'] = bw / W; f['bbox_h'] = bh / H
    f['bbox_diag'] = np.hypot(bw, bh) / np.hypot(W, H)
    f['bbox_aspect'] = bh / max(bw, 1)
    # components and endpoints
    ncc, _ = cv2.connectedComponents(mask, connectivity=8)
    f['n_cc'] = ncc - 1
    skp = prune(sk, int(round(max(4, 1.5 * sw))))
    ep = endpoints(skp)
    f['n_end'] = len(ep)
    # Euler number / holes: overlapping circles closed -> 3 holes
    inv = (1 - cv2.dilate(mask, np.ones((3, 3), np.uint8))).astype(np.uint8)
    nh, lab_h, st_h, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    holes = [st_h[i, cv2.CC_STAT_AREA] for i in range(1, nh)
             if not (st_h[i, 0] == 0 or st_h[i, 1] == 0 or st_h[i, 0] + st_h[i, 2] >= W or st_h[i, 1] + st_h[i, 3] >= H)]
    holes = sorted(holes, reverse=True)
    f['n_holes_big'] = sum(h > 0.002 * H * W for h in holes)
    f['n_holes'] = len([h for h in holes if h > 30])
    if res is None:
        f['fit_ok'] = 0
        return (f, None) if return_debug else f
    f['fit_ok'] = 1
    e1, e2, a, dm, d1, d2 = res
    P1, P2 = P[a & (dm < 3 * tol)], P[~a & (dm < 3 * tol)]
    f['frac_inlier'] = float(np.mean(dm < 2 * tol))
    f['frac_far'] = float(np.mean(dm > 4 * tol))
    f['far_len'] = float(np.sum(dm > 3 * tol)) / W
    f['res_mean'] = float(np.mean(np.minimum(dm, 6 * tol)) / tol)
    f['frac_split'] = min(len(P1), len(P2)) / max(len(P1) + len(P2), 1)
    # per-ellipse geometry
    A1, B1 = ell_axes(e1); A2, B2 = ell_axes(e2)
    f['elong1'] = A1 / B1; f['elong2'] = A2 / B2
    f['elong_max'] = max(A1 / B1, A2 / B2)
    # size of each: use mean diameter and area
    D1, D2 = (A1 + B1) / 2, (A2 + B2) / 2
    f['size_ratio_diam'] = max(D1, D2) / min(D1, D2)
    f['size_ratio_area'] = max(A1 * B1, A2 * B2) / min(A1 * B1, A2 * B2)
    f['size_ratio_major'] = max(A1, A2) / min(A1, A2)
    f['size_ratio_height'] = max(e1[1][0], e1[1][1], 1) and None
    # heights/widths in image axes via sampled points
    E1, E2 = ellipse_pts(e1), ellipse_pts(e2)
    h1, h2 = np.ptp(E1[:, 1]), np.ptp(E2[:, 1]); w1, w2 = np.ptp(E1[:, 0]), np.ptp(E2[:, 0])
    f['size_ratio_height'] = max(h1, h2) / min(h1, h2)
    f['size_ratio_width'] = max(w1, w2) / min(w1, w2)
    f['upper_bigger'] = float(D1 > D2)
    f['diam_mean'] = (D1 + D2) / 2 / W
    f['diam_max'] = max(D1, D2) / W
    f['diam_min'] = min(D1, D2) / W
    # orientation
    (x1, y1), (x2, y2) = e1[0], e2[0]
    dx, dy = x2 - x1, y2 - y1
    f['tilt_deg'] = float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-9)))
    f['dx_norm'] = abs(dx) / ((D1 + D2) / 2)
    f['dy_norm'] = dy / ((D1 + D2) / 2)
    f['ell_tilt1'] = abs(((e1[2] + 90) % 180) - 90) if A1 / B1 > 1.15 else 0.0
    f['ell_tilt2'] = abs(((e2[2] + 90) % 180) - 90) if A2 / B2 > 1.15 else 0.0
    for nm, e, A, B in (('1', e1, A1, B1), ('2', e2, A2, B2)):
        (_, _), (w_, h_), ang_ = e
        major_ang = ang_ if w_ >= h_ else ang_ + 90  # direction of major axis (deg)
        dev = abs(((major_ang % 180) + 45) % 90 - 45)  # deviation from nearest of horizontal/vertical (0..45)
        f['axis_dev' + nm] = dev
        f['axis_tiltw' + nm] = dev * (A / B - 1)
    f['axis_tiltw_max'] = max(f['axis_tiltw1'], f['axis_tiltw2'])
    f['axis_tiltw_sum'] = f['axis_tiltw1'] + f['axis_tiltw2']
    # overlap
    dist = np.hypot(dx, dy)
    r1, r2 = (h1 / 2), (h2 / 2)
    f['ov_depth_v'] = (y1 + h1 / 2 - (y2 - h2 / 2)) / ((h1 + h2) / 2)  # vertical overlap / mean height
    f['ov_depth_min'] = (y1 + h1 / 2 - (y2 - h2 / 2)) / min(h1, h2)
    f['center_dist'] = dist / ((D1 + D2) / 2)
    a1, a2, inter = ellipse_overlap_area(e1, e2, (H, W))
    f['ov_area_small'] = inter / max(min(a1, a2), 1)
    f['ov_area_union'] = inter / max(a1 + a2 - inter, 1)
    # overall size (relative to image frame)
    f['total_h'] = (max(E1[:, 1].max(), E2[:, 1].max()) - min(E1[:, 1].min(), E2[:, 1].min())) / H
    f['total_w'] = (max(E1[:, 0].max(), E2[:, 0].max()) - min(E1[:, 0].min(), E2[:, 0].min())) / W
    # shape quality
    rm1, r951 = radial_residual(e1, P1); rm2, r952 = radial_residual(e2, P2)
    f['res_rel_max'] = max(rm1, rm2); f['res95_max'] = max(r951, r952)
    f['corner_max'] = max(corner_score(P1, e1), corner_score(P2, e2))
    # closure
    c1 = coverage_features(e1, e2, P1, mask, sw, tol, inkness)
    c2 = coverage_features(e2, e1, P2, mask, sw, tol, inkness)
    for k in ['max_gap', 'tot_gap', 'dbl_len', 'max_dbl']:
        f[k + '_max'] = max(c1[k], c2[k]) / W
        f[k + '_sum'] = (c1[k] + c2[k]) / W
    for k in ['wratio', 'dratio', 'jump']:
        f[k + '_max'] = max(c1[k], c2[k])
    f['thick_len_sum'] = (c1['thick_len'] + c2['thick_len']) / W
    # endpoint counts at several prune lengths
    for L in (8, 14):
        f['n_end_p%d' % L] = len(endpoints(prune(sk, L)))
    f['max_gap_rel'] = max(c1['max_gap'] / c1['perim'], c2['max_gap'] / c2['perim'])
    f['dbl_rel'] = max(c1['dbl_len'] / c1['perim'], c2['dbl_len'] / c2['perim'])
    # robust gap: coverage of each ellipse by ANY skeleton point (assignment-free), looser tolerance
    Tall = cKDTree(P)
    for nm, e in (('g1', e1), ('g2', e2)):
        E = ellipse_pts(e, NS)
        seg = np.linalg.norm(np.roll(E, -1, 0) - E, axis=1)
        dd, _ = Tall.query(E)
        tolc = 0.10 * ell_axes(e)[1] / 2 + sw
        unc = dd > tolc
        if unc.all():
            mg = seg.sum()
        elif not unc.any():
            mg = 0.0
        else:
            st = np.argmax(~unc); u = np.roll(unc, -st); L = np.roll(seg, -st)
            mg = cur = 0.0
            for uu, l in zip(u, L):
                cur = cur + l if uu else 0.0
                mg = max(mg, cur)
        f['gap2_' + nm] = mg / W
    f['gap2_max'] = max(f['gap2_g1'], f['gap2_g2'])
    # skeleton junctions (degree>=3) after pruning; two clean crossings give ~2-4
    kk = np.ones((3, 3), np.uint8)
    nbj = cv2.filter2D(skp.astype(np.uint8), -1, kk, borderType=cv2.BORDER_CONSTANT) - skp
    jun = ((skp == 1) & (nbj >= 3)).astype(np.uint8)
    nj, labj, _, cenj = cv2.connectedComponentsWithStats(cv2.dilate(jun, np.ones((5, 5), np.uint8)))
    f['n_junc'] = nj - 1
    # junctions away from the two expected ellipse crossings -> forks from overshoots / retracing
    E1d, E2d = ellipse_pts(e1, 1440), ellipse_pts(e2, 1440)
    d12, _ = cKDTree(E2d).query(E1d)
    cross = E1d[d12 < 2.0]
    dmean_ = (D1 + D2) / 2
    if nj > 1:
        cj = cenj[1:]
        if len(cross):
            dj, _ = cKDTree(cross).query(cj)
        else:
            dj = np.full(len(cj), 1e9)
        f['n_junc_extra'] = int(np.sum(dj > 0.08 * dmean_))
    else:
        f['n_junc_extra'] = 0
    # endpoints away from crossings
    if len(ep) and len(cross):
        de, _ = cKDTree(cross).query(ep)
        f['n_end_far'] = int(np.sum(de > 0.08 * dmean_))
    else:
        f['n_end_far'] = len(ep)
    # endpoint pairs that face each other across a small gap
    if len(ep) >= 2:
        T = cKDTree(ep)
        dd2, _ = T.query(ep, k=2)
        dmean = (D1 + D2) / 2
        f['n_ep_close'] = int(np.sum(dd2[:, 1] < 0.3 * dmean))
    else:
        f['n_ep_close'] = 0
    # endpoint pair distance (gap between stroke ends)
    if len(ep) >= 2:
        T = cKDTree(ep)
        dd, _ = T.query(ep, k=2)
        f['ep_nn_min'] = float(dd[:, 1].min()) / W
        f['ep_nn_med'] = float(np.median(dd[:, 1])) / W
    else:
        f['ep_nn_min'] = 1.0; f['ep_nn_med'] = 1.0
    if return_debug:
        return f, dict(img=img, mask=mask, sk=sk, e1=e1, e2=e2, P=P, a=a, dm=dm, ep=ep, tol=tol)
    return f
