"""Overlapping-circles rubric scorer: rule-based shape gate + per-facet calibrated decisions."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FACETS = ['shape', 'closure', 'edges', 'orient', 'overlap', 'size']
CLOSURE_FEATS = ['dbl_len_sum', 'dbl_len_max', 'max_dbl_max', 'dbl_rel', 'wratio_max', 'jump_max', 'thick_len_sum',
                 'n_end', 'n_end_p8', 'n_end_p14', 'corner_max', 'res_mean', 'max_gap_max', 'frac_inlier', 'far_len', 'stroke_w', 'n_junc', 'gap2_max', 'n_ep_close']


def shape_ok(r):
    """Basic shape: two roughly circular, corner-free closed shapes. Gross-failure gate."""
    if not r.get('fit_ok', 0):
        return 0
    if r['frac_inlier'] < 0.87:          # much of the ink is not on two ellipses (scribbles, extra shapes)
        return 0
    if r['corner_max'] > 0.6:            # pronounced corner (e.g. triangle)
        return 0
    if r['elong_max'] > 4.0:             # degenerate fit / line, not a circle
        return 0
    gap_rel = r['gap2_max'] / (np.pi * max(r['diam_min'], 1e-6))
    if r['n_junc'] <= 1 and gap_rel > 0.25:  # strokes never cross and arcs are half-missing (e.g. an S-curve)
        return 0
    return 1


# Images whose content contradicts their label (wrong item filed under Overlapped Circle).
# Excluded from training only; still reported in evaluation.
SUSPECT_LABELS = {}  # misfiled training files: listed in the full project only


def _acc_threshold(x, y, direction):
    """Pick t maximising accuracy for rule fail = x>t (direction=+1) or x<t (-1). Midpoint tie-break."""
    xs = np.sort(np.unique(x))
    cand = np.concatenate([[xs[0] - 1e-9], (xs[:-1] + xs[1:]) / 2, [xs[-1] + 1e-9]])
    accs = np.array([((direction * (x - t) > 0).astype(int) == y).mean() for t in cand])
    best = np.flatnonzero(accs >= accs.max() - 1e-12)
    # middle of the widest run of best thresholds
    return float(cand[best[len(best) // 2]]), float(accs.max())


class Stump:
    def __init__(self, feat, direction):
        self.feat, self.direction = feat, direction

    def fit(self, df, y):
        self.t, _ = _acc_threshold(df[self.feat].values, y, self.direction)
        return self

    def predict_fail(self, df):
        return (self.direction * (df[self.feat].values - self.t) > 0).astype(int)


class OrStumps:
    """fail if f1 > t1 OR f2 > t2 (joint grid search, accuracy; tie-break toward stricter-to-fail)."""
    def __init__(self, f1, f2):
        self.f1, self.f2 = f1, f2

    def fit(self, df, y):
        a, b = df[self.f1].values, df[self.f2].values
        ca = np.unique(np.quantile(a, np.linspace(0.7, 1, 61))); cb = np.unique(np.quantile(b, np.linspace(0.7, 1, 61)))
        ca = np.append(ca, a.max() + 1e-9); cb = np.append(cb, b.max() + 1e-9)
        best = (-1, 0, 0)
        for t1 in ca:
            for t2 in cb:
                acc = ((((a > t1) | (b > t2)).astype(int)) == y).mean()
                if acc > best[0] + 1e-12 or (abs(acc - best[0]) < 1e-12 and (t1 + 1e-9) * (t2 + 1e-9) > (best[1] + 1e-9) * (best[2] + 1e-9)):
                    best = (acc, t1, t2)
        _, self.t1, self.t2 = best
        return self

    def predict_fail(self, df):
        return ((df[self.f1].values > self.t1) | (df[self.f2].values > self.t2)).astype(int)


class TwoSided:
    """fail if x < lo or x > hi (overlap too large / too small)."""
    def __init__(self, feat):
        self.feat = feat

    def fit(self, df, y):
        x = df[self.feat].values
        qs = np.quantile(x, np.linspace(0, 0.25, 51)); hs = np.quantile(x, np.linspace(0.75, 1, 51))
        best = (-1, None, None)
        for lo in np.concatenate([[x.min() - 1e-9], qs]):
            for hi in np.concatenate([hs, [x.max() + 1e-9]]):
                acc = (((x < lo) | (x > hi)).astype(int) == y).mean()
                # tie-break: wider pass band (more conservative)
                if acc > best[0] + 1e-12 or (abs(acc - best[0]) < 1e-12 and hi - lo > best[2] - best[1]):
                    best = (acc, lo, hi)
        _, self.lo, self.hi = best
        return self

    def predict_fail(self, df):
        x = df[self.feat].values
        return ((x < self.lo) | (x > self.hi)).astype(int)


class LogitFacet:
    def __init__(self, feats, C=0.5):
        self.feats = feats
        self.est = make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight='balanced', max_iter=5000))

    def fit(self, df, y):
        X = df[self.feats].values
        p = cross_val_predict(clone(self.est), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=7), method='predict_proba')[:, 1]
        ts = np.unique(np.quantile(p, np.linspace(0, 1, 401)))
        accs = np.array([((p >= t).astype(int) == y).mean() for t in ts])
        best = np.flatnonzero(accs >= accs.max() - 1e-12)
        self.t = float(ts[best[len(best) // 2]])
        self.est.fit(X, y)
        return self

    def proba(self, df):
        return self.est.predict_proba(df[self.feats].values)[:, 1]

    def predict_fail(self, df):
        return (self.proba(df) >= self.t).astype(int)


def default_models():
    return {
        'closure': LogitFacet(CLOSURE_FEATS),
        'edges': Stump('size_ratio_diam', +1),
        'orient': OrStumps('dx_norm', 'axis_tiltw_sum'),
        'overlap': TwoSided('center_dist'),
        'size': Stump('diam_mean', -1),
    }


class OverlapCircleScorer:
    def __init__(self, models=None):
        self.models = models or default_models()
        self.medians = None

    def _prep(self, df):
        df = df.copy()
        cols = [c for c in self.medians.index if c in df.columns]
        df[cols] = df[cols].fillna(self.medians[cols])
        return df

    def fit(self, df):
        """df: feature rows with facet labels (1=credit, 0=no credit). Uses rows with shape credit."""
        num = df.select_dtypes('number')
        self.medians = num.median()
        t = self._prep(df[(df.facets_ok == 1) & (df['shape'] == 1) & ~df['file'].isin(SUSPECT_LABELS)])
        for fac, m in self.models.items():
            m.fit(t, (t[fac] == 0).astype(int).values)
        return self

    def predict(self, df):
        df = self._prep(df)
        out = pd.DataFrame({'file': df['file'].values})
        sh = np.array([shape_ok(r) for r in df.to_dict('records')])
        out['shape'] = sh
        for fac in ['closure', 'edges', 'orient', 'overlap', 'size']:
            out[fac] = (1 - self.models[fac].predict_fail(df)) * sh
        if 'closure' in self.models and hasattr(self.models['closure'], 'proba'):
            out['closure_fail_prob'] = np.round(self.models['closure'].proba(df), 3)
        out['score'] = out[FACETS].sum(1)
        return out

    def describe(self):
        m = self.models
        return {
            'edges': f"fail if larger/smaller mean diameter > {m['edges'].t:.3f}",
            'orient': f"fail if horizontal centre offset / mean diameter > {m['orient'].t1:.3f} or summed ellipse-axis tilt score > {m['orient'].t2:.1f}",
            'overlap': f"fail if centre distance / mean diameter < {m['overlap'].lo:.3f} or > {m['overlap'].hi:.3f}",
            'size': f"fail if mean circle diameter / image width < {m['size'].t:.3f}",
            'closure': f"logistic model on {len(m['closure'].feats)} stroke-end features, fail if p >= {m['closure'].t:.3f}",
        }
