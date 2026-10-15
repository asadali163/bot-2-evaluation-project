#!/usr/bin/env python3
"""Automatic scoring of the Diamond copy item (Item 6, max 5 points).

Facets (1 = pass, 0 = fail):
  Basic shape  : exactly four sides and four distinct corners          -> geometric rule
  Closure      : no gap > 1/8", no overlap > 1/4"                       -> trained classifier on gap/overlap measurements
  Edges        : longest edge <= 1.5 x shortest edge                     -> trained classifier on edge measurements
  Orientation  : overall orientation similar to stimulus (vertex-up)     -> geometric rule (rotation > 20 deg fails)
  Overall size : at least half the size of the stimulus                  -> geometric rule (area < 1/4 of reference)
If Basic shape = 0, every facet and the total are 0.

Usage
  python scorer.py score  <image_folder> [--out results.csv] [--overlays overlay_dir]
  python scorer.py train  <labelled_folder> [--model model.joblib]
Labels for training are read from file names such as  imgN-D-5.png  or  imgN-D-4(10111).png
(the 5-digit code is Shape, Closure, Edges, Orientation, Size).
"""
import os, re, sys, json, argparse, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np, pandas as pd, joblib
from multiprocessing import Pool

# ---------------------------------------------------------------- configuration -------------
CLOSURE_F = ['gap', 'gap_faint', 'overlap', 'overlap_ext', 'overlap_total', 'skel_ratio', 'small_hole_area',
             'n_small_holes', 'width_p99', 'width_ratio', 'tail_max', 'tail_2', 'n_tails', 'tail_sum',
             'tail_max_fine', 'n_ends', 'n_ends_fine', 'n_junc', 'n_junc_fine', 'skel_len', 'stroke_w',
             'n_comp', 'fill_ratio', 'used_faint']
EDGE_F = ['edge_ratio', 'angle_sum_dev', 'min_angle', 'max_angle', 'elong', 'vertex_offcentre', 'side_dev',
          'diag_dev', 'tri_iou', 'quad_iou', 'corner1', 'corner4']
ORIENT_DEG = 20.0          # rotation away from the vertex-up diamond pose that counts as "noticeably different"
SHAPE_QUAD_IOU = 0.80      # below this the figure is not a quadrilateral (stars, circles, scribbles)
SHAPE_PENT_GAIN = 0.06     # a 5th corner improves the fit this much -> not four distinct corners
SIZE_AREA_FRAC = 0.25      # half the linear size of the stimulus == one quarter of its area
FACETS = ['shape', 'closure', 'edges', 'orient', 'size']
FACET_NAMES = {'shape': 'Basic Shape', 'closure': 'Closure', 'edges': 'Edges', 'orient': 'Orientation',
               'size': 'Overall Size'}


# ---------------------------------------------------------------- helpers -------------------
def parse_label(fn):
    m = re.search(r'-D-(\d)(?:\((\d+)\))?\.', fn)
    if not m:
        return {'total': None, **{k: None for k in FACETS}}
    tot = int(m.group(1)); code = m.group(2)
    r = {'total': tot}
    if tot == 5:
        r.update({k: 1 for k in FACETS})
    elif tot == 0:
        r.update({k: 0 for k in FACETS})
    elif code and len(code) == 5:
        r.update({k: int(c) for k, c in zip(FACETS, code)})
    else:
        r.update({k: None for k in FACETS})
    return r


def _feat(path):
    from .features import features
    try:
        f = features(path)
    except Exception as e:  # never crash a batch on one bad image
        f = {'error': repr(e)}
    tc = f.pop('top_corners', None)
    if tc:
        for i, v in enumerate(tc):
            f[f'corner{i + 1}'] = v
    f['file'] = os.path.basename(path)
    return f


def extract(folder, workers=2):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    with Pool(workers) as p:
        rows = p.map(_feat, [os.path.join(folder, f) for f in files])
    return pd.DataFrame(rows)


def shape_fail(d):
    return ((d.quad_iou < SHAPE_QUAD_IOU) | ((d.pent_iou - d.quad_iou) > SHAPE_PENT_GAIN)).astype(int)


# ---------------------------------------------------------------- training ------------------
def _models():
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    closure = lambda: ExtraTreesClassifier(400, min_samples_leaf=3, max_features=0.5,
                                           class_weight='balanced_subsample', random_state=0, n_jobs=2)
    edges = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.3, class_weight='balanced', max_iter=3000))
    return closure, edges


def _f1_threshold(y, p):
    best, bt = -1, 0.5
    for t in np.unique(np.quantile(p, np.linspace(0, 1, 401))):
        pred = p >= t
        tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
        sc = 2 * tp / max(2 * tp + fp + fn, 1)
        if sc > best:
            best, bt = sc, t
    return float(bt)


def train(df, exclude=()):
    from sklearn.model_selection import StratifiedKFold
    d = df[~df.file.isin(exclude)].copy()
    closure_fn, edges_fn = _models()
    out = {'config': dict(ORIENT_DEG=ORIENT_DEG, SHAPE_QUAD_IOU=SHAPE_QUAD_IOU, SHAPE_PENT_GAIN=SHAPE_PENT_GAIN,
                          SIZE_AREA_FRAC=SIZE_AREA_FRAC),
           'ref_area': float(d[d.total == 5].size_area.median())}
    for facet, feats, fn in [('closure', CLOSURE_F, closure_fn), ('edges', EDGE_F, edges_fn)]:
        dd = d[d[facet].notna() & (d['shape'] == 1)]
        X = dd[feats].fillna(0).values; y = (1 - dd[facet]).astype(int).values
        oof = np.zeros(len(y)); reps = 3
        for r in range(reps):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(X, y):
                oof[te] += fn().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] / reps
        out[facet] = {'model': fn().fit(X, y), 'threshold': _f1_threshold(y, oof), 'features': feats}
    return out


# ---------------------------------------------------------------- scoring -------------------
def predict(df, model):
    d = df.copy()
    ok = d.get('error', pd.Series(index=d.index, dtype=object)).isna() & (d.get('empty', 0) == 0)
    fail = pd.DataFrame(index=d.index)
    fail['shape'] = shape_fail(d)
    fail['size'] = (d.size_area < SIZE_AREA_FRAC * model['ref_area']).astype(int)
    fail['orient'] = (d.rot_diag > ORIENT_DEG).astype(int)
    for facet in ['closure', 'edges']:
        m = model[facet]
        p = m['model'].predict_proba(d[m['features']].fillna(0).values)[:, 1]
        d[f'p_fail_{facet}'] = p
        fail[facet] = (p >= m['threshold']).astype(int)
    res = pd.DataFrame({'file': d.file})
    for f in FACETS:
        res[FACET_NAMES[f]] = np.where(fail['shape'] == 1, 0, 1 - fail[f])
    res['Total'] = res[[FACET_NAMES[f] for f in FACETS]].sum(1)
    # transparency: key measurements behind each decision
    res['edge_ratio'] = d.edge_ratio.round(2)
    res['rotation_deg'] = d.rot_diag.round(1)
    res['size_vs_typical'] = (d.size_area / model['ref_area']).round(2)
    res['closure_fail_prob'] = d.p_fail_closure.round(3)
    res['edges_fail_prob'] = d.p_fail_edges.round(3)
    # flags for human review
    flags = []
    for i in d.index:
        fl = []
        if d.loc[i, 'quad_iou'] < 0.6:
            fl.append('NOT A DIAMOND? (possibly a different test item)')
        elif fail.loc[i, 'shape']:
            fl.append('basic shape failed - check')
        for facet in ['closure', 'edges']:
            p, t = d.loc[i, f'p_fail_{facet}'], model[facet]['threshold']
            lg = lambda x: np.log(np.clip(x, 1e-4, 1 - 1e-4) / (1 - np.clip(x, 1e-4, 1 - 1e-4)))
            if abs(lg(p) - lg(t)) < 0.6 and not fail.loc[i, 'shape']:
                fl.append(f'{facet} borderline')
        if d.loc[i, 'no_interior'] == 1:
            fl.append('very faint ink')
        flags.append('; '.join(fl))
    res['review_flags'] = flags
    res.loc[~ok, 'review_flags'] = 'could not process image'
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score'); s.add_argument('folder'); s.add_argument('--out', default='diamond_scores.csv')
    s.add_argument('--model', default=os.path.join(HERE, 'model.joblib'))
    s.add_argument('--with-labels', action='store_true', help='compare with labels in file names')
    t = sub.add_parser('train'); t.add_argument('folder'); t.add_argument('--model', default=os.path.join(HERE, 'model.joblib'))
    t.add_argument('--exclude', nargs='*', default=[])
    a = ap.parse_args()
    df = extract(a.folder)
    if a.cmd == 'train':
        lab = pd.DataFrame([parse_label(f) for f in df.file])
        df = pd.concat([df, lab], axis=1)
        joblib.dump(train(df, exclude=a.exclude), a.model)
        print('saved', a.model)
    else:
        res = predict(df, joblib.load(a.model))
        if a.with_labels:
            lab = pd.DataFrame([parse_label(f) for f in res.file])
            res['label_total'] = lab.total.values
            res['agree'] = res.Total == res.label_total
            print('exact agreement with file-name labels: %.3f' % res.agree.mean())
        res.to_csv(a.out, index=False)
        print(res.Total.value_counts().sort_index().to_string())
        print('saved', a.out)


if __name__ == '__main__':
    main()
