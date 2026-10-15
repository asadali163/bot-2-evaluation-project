"""Geometric measurements for each scoring facet of the Diamond item.

All lengths are expressed as a fraction of the response-box width (fw), so they are
independent of photo resolution.  Physical rule thresholds (1/8", 1/4") are calibrated
against the labelled data in calibrate.py.
"""
import numpy as np, cv2
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from .seg import segment
from skimage.filters import apply_hysteresis_threshold


def _disk(r):
    r = max(int(round(r)), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def interior_at(mask, r, min_frac):
    """Interior region enclosed by the drawing after closing gaps with radius r."""
    m = mask.astype(np.uint8)
    d = cv2.dilate(m, _disk(r)) if r > 0 else m
    filled = ndi.binary_fill_holes(d)
    holes = filled & ~d.astype(bool)
    if not holes.any():
        return None
    lab, n = ndi.label(holes)
    sizes = ndi.sum(holes, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    hole = lab == k
    if hole.sum() < min_frac:
        return None
    # undo the dilation so the interior reaches the stroke centre-line again
    hole = cv2.dilate(hole.astype(np.uint8), _disk(r)) > 0 if r > 0 else hole
    return hole


def skeleton_endpoints(sk):
    k = np.ones((3, 3), np.uint8); k[1, 1] = 0
    nb = cv2.filter2D(sk.astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    ends = sk & (nb == 1)
    junc = sk & (nb >= 3)
    return ends, junc


def prune(sk, length):
    """Remove skeleton spurs shorter than `length` px (iterative end erosion + regrowth)."""
    sk = sk.copy()
    orig = sk.copy()
    for _ in range(length):
        e, _ = skeleton_endpoints(sk)
        if not e.any():
            break
        sk &= ~e
    # regrow from remaining endpoints along original skeleton so main branches keep length
    for _ in range(length):
        e, _ = skeleton_endpoints(sk)
        grow = cv2.dilate(e.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        new = grow & orig & ~sk
        if not new.any():
            break
        sk |= new
    return sk


def turning_corners(poly_pts, ang_thr=35):
    """Count corners on a closed contour using turning angle over a window."""
    p = poly_pts.astype(np.float64)
    n = len(p)
    if n < 20:
        return 0, []
    # resample uniformly
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1))]
    L = d[-1]
    t = np.linspace(0, L, 200, endpoint=False)
    xs = np.interp(t, d, np.r_[p[:, 0], p[0, 0]]); ys = np.interp(t, d, np.r_[p[:, 1], p[0, 1]])
    q = np.c_[xs, ys]
    w = 8  # window (4% of perimeter each side)
    a = q - np.roll(q, w, 0); b = np.roll(q, -w, 0) - q
    ang = np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1)))
    # local maxima of |turn|
    absang = np.abs(ang)
    peaks = []
    for i in np.argsort(-absang):
        if absang[i] < ang_thr:
            break
        if all(min(abs(i - j), 200 - abs(i - j)) > w for j in peaks):
            peaks.append(i)
    return len(peaks), sorted(absang[peaks].tolist(), reverse=True)


def quad_metrics(quad):
    q = quad.reshape(-1, 2).astype(np.float64)
    sides = np.linalg.norm(q - np.roll(q, -1, 0), axis=1)
    d1 = q[2] - q[0]; d2 = q[3] - q[1]

    def axis_dev(v):  # angle (deg) between v and the nearest of horizontal/vertical
        a = np.degrees(np.arctan2(abs(v[1]), abs(v[0])))  # 0..90
        return min(a, 90 - a)

    diag_dev = (axis_dev(d1) + axis_dev(d2)) / 2
    # side directions vs. 45 degrees
    sd = []
    for i in range(4):
        v = q[(i + 1) % 4] - q[i]
        a = np.degrees(np.arctan2(abs(v[1]), abs(v[0])))
        sd.append(abs(a - 45))
    # interior angles
    angs = []
    for i in range(4):
        u = q[i - 1] - q[i]; v = q[(i + 1) % 4] - q[i]
        angs.append(np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9), -1, 1))))
    return sides, diag_dev, float(np.mean(sd)), angs


def tail_lengths(sk, ends, junc, cap):
    """For every skeleton endpoint walk along the skeleton until a junction is hit.
    Returns lengths of branches that end in a junction (tails / overshoots) and of
    those that do not (free curve ends, i.e. gap ends)."""
    H, W = sk.shape
    tails, free = [], []
    eys, exs = np.nonzero(ends)
    for y, x in zip(eys, exs):
        prev = None; cur = (y, x); n = 0; hit = False
        while n < cap:
            yy, xx = cur
            nbrs = [(yy + dy, xx + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx) and 0 <= yy + dy < H and 0 <= xx + dx < W and sk[yy + dy, xx + dx]]
            if junc[yy, xx] and n > 0:
                hit = True; break
            nxt = [p for p in nbrs if p != prev and (prev is None or max(abs(p[0] - prev[0]), abs(p[1] - prev[1])) > 1)]
            if not nxt:
                break
            prev = cur; cur = nxt[0]; n += 1
        (tails if hit else free).append(n)
    return tails, free


def signed_rotation(quad, contour):
    """Rotation of the drawn figure away from a diamond (vertex-up) pose, in degrees.
    Signed averages cancel the opposite skews of a tall/flat rhombus, so only a real
    rotation of the whole figure is measured."""
    q = quad.reshape(-1, 2).astype(np.float64)
    out = {}
    devs, ws = [], []
    for i in range(4):
        v = q[(i + 1) % 4] - q[i]
        a = np.degrees(np.arctan2(v[1], v[0])) % 90.0  # 0..90
        devs.append(a - 45.0); ws.append(np.linalg.norm(v))
    devs = np.array(devs); ws = np.array(ws)
    out['rot_side'] = float(abs(np.sum(devs * ws) / ws.sum()))
    d1 = q[2] - q[0]; d2 = q[3] - q[1]

    def sdev(v):
        a = np.degrees(np.arctan2(v[1], v[0])) % 90.0
        return a if a < 45 else a - 90
    out['rot_diag'] = float(abs((sdev(d1) + sdev(d2)) / 2))
    out['diag_max_dev'] = float(max(abs(sdev(d1)), abs(sdev(d2))))
    # principal axes of the filled contour (second moments)
    m = cv2.moments(contour)
    if m['m00'] > 0:
        mu20, mu02, mu11 = m['mu20'] / m['m00'], m['mu02'] / m['m00'], m['mu11'] / m['m00']
        th = 0.5 * np.degrees(np.arctan2(2 * mu11, mu20 - mu02))
        ecc = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11 ** 2) / (mu20 + mu02)
        out['axis_dev'] = float(min(abs(th) % 90, 90 - abs(th) % 90))
        out['elong'] = float(ecc)
    else:
        out['axis_dev'] = 0.0; out['elong'] = 0.0
    # where are the extreme points?  For a diamond the top-most vertex sits mid-width.
    xs, ys = q[:, 0], q[:, 1]
    w = xs.max() - xs.min() + 1e-6; h = ys.max() - ys.min() + 1e-6
    top = q[np.argmin(ys)]; bot = q[np.argmax(ys)]; lef = q[np.argmin(xs)]; rig = q[np.argmax(xs)]
    out['vertex_offcentre'] = float(np.mean([abs(top[0] - xs.min()) / w - 0.5, abs(bot[0] - xs.min()) / w - 0.5,
                                             abs(lef[1] - ys.min()) / h - 0.5, abs(rig[1] - ys.min()) / h - 0.5]
                                            ) if False else np.mean(np.abs([(top[0] - xs.min()) / w - 0.5,
                                                                            (bot[0] - xs.min()) / w - 0.5,
                                                                            (lef[1] - ys.min()) / h - 0.5,
                                                                            (rig[1] - ys.min()) / h - 0.5])))
    return out


def features(path, return_debug=False):
    im, mask, fb, score = segment(path)
    H, W = mask.shape
    x0, y0, x1, y1 = fb
    fw = float(max(x1 - x0, y1 - y0, 0.6 * max(H, W)))
    f = {}
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        f['empty'] = 1
        return f
    f['empty'] = 0
    # stroke width
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    sk0 = skeletonize(mask)
    sw = float(2 * np.median(dt[sk0])) if sk0.any() else 3.0
    f['stroke_w'] = sw / fw
    hull = cv2.convexHull(np.c_[xs, ys].astype(np.int32))
    hull_area = cv2.contourArea(hull)
    f['n_comp'] = int(ndi.label(mask, np.ones((3, 3)))[1])

    # ---------------- closure: smallest closing radius that creates an interior -------------
    min_frac = 0.25 * hull_area
    r_star = None
    for r in list(range(0, 12)) + list(range(12, 60, 3)) + list(range(60, 160, 10)):
        hole = interior_at(mask, r, min_frac)
        if hole is not None:
            r_star = r
            break
    if r_star is None:
        r_star = 160
        hole = None
    # gap width ≈ 2r (dilation from both sides) minus nothing; express relative to fw
    f['gap'] = 2 * r_star / fw
    f['closed'] = int(r_star == 0)

    # faint-ink corrected gap: same test on a lower threshold mask restricted near drawing
    # grow from confident ink into connected faint ink on a smoothed score map
    sm = cv2.GaussianBlur(score, (0, 0), 1.5)
    noise = np.percentile(sm[sm < 0.3], 95)
    cand = sm > max(0.04, 1.4 * noise)
    lab_c, _ = ndi.label(cand, np.ones((3, 3)))
    ids = np.unique(lab_c[mask & cand]); ids = ids[ids > 0]
    faint = np.isin(lab_c, ids)
    faint |= mask
    r2 = 160
    for r in list(range(0, 12)) + list(range(12, 60, 3)) + list(range(60, 160, 10)):
        h2 = interior_at(faint, r, min_frac)
        if h2 is not None:
            r2 = r
            break
    f['gap_faint'] = 2 * r2 / fw
    # if pale ink closes the figure better, use it for the geometry of the figure
    if r2 < r_star:
        hole = h2
        r_star = r2
        f['used_faint'] = 1
    else:
        f['used_faint'] = 0

    # endpoints of the pruned skeleton (each open end is a gap or a tail)
    sk = prune(sk0, max(int(1.5 * sw), 6))
    ends, junc = skeleton_endpoints(sk)
    f['n_ends'] = int(ends.sum())
    f['n_junc'] = int(cv2.connectedComponents(cv2.dilate(junc.astype(np.uint8), np.ones((3, 3), np.uint8)))[0] - 1)
    tails, gaps_e = tail_lengths(sk, ends, junc, cap=int(0.3 * fw))
    tl = sorted([t for t in tails], reverse=True)
    f['tail_max'] = (tl[0] if tl else 0) / fw
    f['tail_2'] = (tl[1] if len(tl) > 1 else 0) / fw
    f['n_tails'] = int(sum(t > 2 * sw for t in tl))
    f['tail_sum'] = sum(tl) / fw
    # fine skeleton (light pruning) to catch short crossings
    skf = prune(sk0, max(int(0.8 * sw), 3))
    ef, jf = skeleton_endpoints(skf)
    f['n_ends_fine'] = int(ef.sum())
    f['n_junc_fine'] = int(cv2.connectedComponents(cv2.dilate(jf.astype(np.uint8), np.ones((3, 3), np.uint8)))[0] - 1)
    tf, _ = tail_lengths(skf, ef, jf, cap=int(0.3 * fw))
    f['tail_max_fine'] = (max(tf) if tf else 0) / fw
    # width irregularity (doubled / retraced strokes are locally wider)
    wv = dt[sk0]
    f['width_ratio'] = float(np.percentile(wv, 95) / max(np.median(wv), 1)) if len(wv) else 1
    f['width_p99'] = float(np.percentile(wv, 99) * 2 / fw) if len(wv) else 0
    # small enclosed loops in the ink (crossings, doubled lines)
    holes_all = ndi.binary_fill_holes(mask) & ~mask
    hl, hn = ndi.label(holes_all)
    hs = sorted(ndi.sum(holes_all, hl, range(1, hn + 1)), reverse=True) if hn else []
    f['n_small_holes'] = int(sum(1 for a in hs[1:] if a >= 4)) if hs else 0
    f['small_hole_area'] = float(sum(hs[1:]) / fw ** 2) if len(hs) > 1 else 0.0
    f['skel_len'] = float(sk.sum() / fw)

    f['no_interior'] = int(hole is None)
    if hole is None:
        # no enclosed interior found (very pale ink or open scribble): use the convex hull
        # of the visible ink for the geometry; closure/shape decide on the other evidence
        hole = np.zeros_like(mask, dtype=np.uint8)
        cv2.fillPoly(hole, [hull.reshape(-1, 1, 2)], 1)
        hole = cv2.erode(hole, _disk(max(sw / 2, 1))) > 0

    # filled shape bounded by the stroke centre-line
    region = hole.astype(np.uint8)
    cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    chull = cv2.convexHull(c)
    f['solidity'] = area / max(cv2.contourArea(chull), 1)
    f['fill_ratio'] = area / max(hull_area, 1)  # interior vs whole ink hull (low => lots of ink outside loop)

    def poly_iou(n):
        try:
            p = cv2.approxPolyN(chull, n)
        except cv2.error:
            return 0.0, None
        pm = np.zeros_like(region); cv2.fillPoly(pm, [p.reshape(-1, 1, 2).astype(np.int32)], 1)
        inter = (pm & region).sum(); uni = (pm | region).sum()
        return inter / max(uni, 1), p

    f['skel_ratio'] = float(sk.sum() / max(cv2.arcLength(c, True), 1))
    f['quad_iou'], quad = poly_iou(4)
    f['tri_iou'], _ = poly_iou(3)
    f['pent_iou'], _ = poly_iou(5)
    f['hex_iou'], _ = poly_iou(6)
    nc, cang = turning_corners(c.reshape(-1, 2))
    f['n_corners'] = nc
    f['top_corners'] = (cang + [0] * 5)[:5]

    # ---------------- overlap: ink (skeleton) lying outside the closed loop -------------------
    filled = ndi.binary_fill_holes(hole | cv2.dilate(hole.astype(np.uint8), _disk(1)).astype(bool))
    dist_out = cv2.distanceTransform((~filled).astype(np.uint8), cv2.DIST_L2, 5)
    tol = sw * 1.0 + r_star + 2
    outside = sk & (dist_out > tol)
    # longest protruding piece: measure as max over components of (skeleton length + tol)
    lab, n = ndi.label(outside, np.ones((3, 3)))
    if n:
        lens = ndi.sum(outside, lab, range(1, n + 1))
        # include the part inside the tolerance band: geometric extent from loop
        ext = [dist_out[lab == i].max() for i in range(1, n + 1)]
        f['overlap'] = float(max(max(lens) * 1.0 + tol, 0)) / fw
        f['overlap_ext'] = float(max(ext)) / fw
        f['overlap_total'] = float(sum(lens)) / fw
    else:
        f['overlap'] = 0.0; f['overlap_ext'] = 0.0; f['overlap_total'] = 0.0
    # inward tails: skeleton inside the interior farther than tol from the loop boundary
    dist_in = cv2.distanceTransform(filled.astype(np.uint8), cv2.DIST_L2, 5)
    inner = sk & (dist_in > tol + sw)
    f['inner_ink'] = float(inner.sum()) / fw

    # ---------------- edges / orientation / size ----------------------------------------------
    if quad is None or len(quad.reshape(-1, 2)) != 4:
        quad = cv2.boxPoints(cv2.minAreaRect(c)).reshape(-1, 1, 2)
    sides, diag_dev, side_dev, angs = quad_metrics(quad)
    f['edge_ratio'] = float(sides.max() / max(sides.min(), 1e-6))
    f['diag_dev'] = float(diag_dev)
    f['side_dev'] = side_dev
    f['min_angle'] = float(min(angs)); f['max_angle'] = float(max(angs))
    f.update(signed_rotation(quad, c))
    f['angle_sum_dev'] = float(abs(sum(angs) - 360))
    # orientation from principal axes / extreme vertex placement
    q = quad.reshape(-1, 2).astype(float)
    bx0, by0 = q.min(0); bx1, by1 = q.max(0)
    f['size_w'] = (bx1 - bx0) / fw
    f['size_h'] = (by1 - by0) / fw
    f['size_max'] = max(f['size_w'], f['size_h'])
    f['size_area'] = area / fw ** 2
    # ---------------- corner-local closure evidence ------------------------------------------
    # smoothed skeleton (fewer hair spurs) for endpoint / junction counting
    ms = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(sw / 3, 1)) > 0.5
    sks = prune(skeletonize(ms), max(int(sw), 4))
    es, js = skeleton_endpoints(sks)
    f['n_ends_s'] = int(es.sum())
    f['n_junc_s'] = int(cv2.connectedComponents(cv2.dilate(js.astype(np.uint8), np.ones((3, 3), np.uint8)))[0] - 1)
    ts, fr = tail_lengths(sks, es, js, cap=int(0.3 * fw))
    f['tail_max_s'] = (max(ts) if ts else 0) / fw
    f['free_ends_s'] = len(fr)
    qm = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillPoly(qm, [quad.reshape(-1, 1, 2).astype(np.int32)], 1)
    dq = cv2.distanceTransform((1 - qm).astype(np.uint8), cv2.DIST_L2, 5)
    R = 0.09 * fw
    yy, xx = np.mgrid[:H, :W]
    c_out, c_ends, c_junc, c_len = [], [], [], []
    for (vx, vy) in quad.reshape(-1, 2):
        disk = (xx - vx) ** 2 + (yy - vy) ** 2 <= R * R
        skd = sks & disk
        c_out.append(float(dq[skd].max()) if skd.any() else 0.0)
        c_ends.append(int((es & disk).sum())); c_junc.append(int((js & disk).sum() > 0))
        c_len.append(float(skd.sum()) / (2 * R))
    f['corner_out_max'] = max(c_out) / fw
    f['corner_out_sum'] = sum(c_out) / fw
    f['corner_ends'] = int(sum(c_ends))
    f['corner_junc'] = int(sum(c_junc))
    f['corner_len_max'] = float(max(c_len))
    # ink outside the fitted quad anywhere (tails / doubled strokes), excluding the stroke width
    f['quad_out_max'] = float(dq[sks].max()) / fw if sks.any() else 0.0
    f['quad_out_len'] = float((sks & (dq > sw)).sum()) / fw

    # ---------------- ring / transect crossing counts (robust to skeleton noise) ---------------
    md = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    qv = quad.reshape(-1, 2).astype(float)
    cen = qv.mean(0)

    def runs(vals, circular):
        v = vals.astype(int)
        if circular:
            if v.all():
                return 1
            k = int(np.argmin(v)); v = np.roll(v, -k)
        return int(((v[1:] == 1) & (v[:-1] == 0)).sum() + (v[0] == 1 and not circular))

    ring_ex, ring_max, ring_def = [], [], []
    for (vx, vy) in qv:
        # move the centre slightly inward so the ring is centred on the stroke corner
        for rho in (0.035, 0.05, 0.07):
            rr = rho * fw
            th = np.linspace(0, 2 * np.pi, 720, endpoint=False)
            px = np.clip((vx + rr * np.cos(th)).round().astype(int), 0, W - 1)
            py = np.clip((vy + rr * np.sin(th)).round().astype(int), 0, H - 1)
            n = runs(md[py, px], True)
            ring_ex.append(max(n - 2, 0)); ring_max.append(n); ring_def.append(max(2 - n, 0))
    f['ring_excess'] = int(sum(ring_ex)); f['ring_max'] = int(max(ring_max)); f['ring_deficit'] = int(sum(ring_def))
    f['ring_corners_bad'] = int(sum(1 for i in range(4) if max(ring_ex[3 * i:3 * i + 3]) > 0))
    tr_ex = []
    for i in range(4):
        a, b = qv[i], qv[(i + 1) % 4]
        v = b - a; L = np.linalg.norm(v) + 1e-6; nrm = np.array([-v[1], v[0]]) / L
        for t in np.linspace(0.2, 0.8, 7):
            p0 = a + t * v
            ss = np.linspace(-0.07 * fw, 0.07 * fw, 120)
            px = np.clip((p0[0] + ss * nrm[0]).round().astype(int), 0, W - 1)
            py = np.clip((p0[1] + ss * nrm[1]).round().astype(int), 0, H - 1)
            tr_ex.append(max(runs(md[py, px], False) - 1, 0))
    f['transect_excess'] = int(sum(tr_ex)); f['transect_frac'] = float(np.mean(np.array(tr_ex) > 0))

    # tilt of the paper itself (frame) is ignored; drawings are scored relative to the page
    if return_debug:
        return f, dict(im=im, mask=mask, sk=sk, hole=hole, quad=quad, fb=fb, outside=outside)
    return f
