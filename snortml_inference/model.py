"""
snortml_inference/model.py
===========================
The anomaly detector. It is fitted on a baseline of this network's own
traffic and answers one question per flow (or per host-window): how often
would something at least this unusual show up in traffic like the baseline?

Pipeline, per context group:

1. Rank-Gaussian transform. Each feature is mapped through its own baseline
   distribution to a z-score, so heavy tails, discreteness and odd shapes
   all become "how far into the tail is this value". Beyond the baseline
   range the z-score keeps growing linearly.
2. Shrinkage Mahalanobis distance on those z-scores. This learns which
   features move together and flags combinations that do not.
3. Optional autoencoder residual (a small MLP trained with scikit-learn and
   run with plain numpy). It adds non-linear structure the distance misses.
4. Conformal calibration. The combined score is compared with held-out
   baseline scores from the same group to give a p-value; a generalized
   Pareto tail extends that below 1/n. A threshold on the p-value is
   therefore a false-alarm rate, not a magic number.

Scoring needs only numpy. scikit-learn and scipy are used when fitting.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

P_FLOOR = 1e-12


@dataclass
class DetectorParams:
    min_fit: int = 300  # rows needed to model a group
    min_cal: int = 120  # held-out rows needed to calibrate it
    # The last group in a fallback chain is modeled on less data, so that
    # traffic in rare contexts is still scored instead of ignored.
    min_fit_last: int = 100
    min_cal_last: int = 40
    max_knots: int = 96  # knots per feature in the rank-Gaussian map
    scale_floor: float = 0.15  # smallest feature change counted as one sigma
    eig_floor: float = 0.15  # floor on correlation eigenvalues
    tail_fraction: float = 0.05  # share of calibration scores used for the tail fit
    tail_min: int = 30
    tail_max: int = 2000
    # A group calibrated on n held-out rows never reports a p-value below
    # 1 / (max_extrapolation * n): rarity is not claimed far beyond the evidence.
    max_extrapolation: float = 100.0
    # Features where only "more than usual" counts. A quiet minute or a tiny
    # flow is never an alert; values below the typical level are treated as
    # typical. An unusually low value next to an unusually high one (many
    # flows, few ports) still breaks the learned correlation and scores.
    one_sided: list = field(default_factory=list)
    cal_grid: int = 1024
    autoencoder: bool = True
    ae_weight: float = 0.5
    # "log": score = log1p(distance) + ae_weight * log1p(autoencoder error).
    # The autoencoder reconstructs normal traffic so tightly that its raw
    # error ratio can reach the hundreds for a flow that is only slightly
    # off-pattern; on a log scale it informs the score without drowning out
    # the distance term. "sum" adds the raw terms.
    combine: str = "log"
    ae_max_rows: int = 150_000
    ae_clip: float = 8.0
    seed: int = 7


def fit_gpd(excess: np.ndarray):
    """Generalized Pareto fit by probability-weighted moments.
    Returns (xi, sigma). xi is limited to a range where the fit is stable."""
    y = np.sort(excess[np.isfinite(excess)])
    k = y.size
    if k < 5 or y[-1] <= 0:
        return 0.0, max(float(y.mean()) if k else 1.0, 1e-6)
    a0 = float(y.mean())
    weights = (k - np.arange(1, k + 1)) / max(k - 1, 1)
    a1 = float((y * weights).mean())
    denom = a0 - 2.0 * a1
    if denom <= 1e-12:
        return 0.0, max(a0, 1e-6)
    xi = 2.0 - a0 / denom
    # xi < 0 would mean a hard upper limit on benign scores, and anything past
    # it would get p = 0. Never claim that: use an exponential tail at least.
    xi_c = min(max(xi, 0.0), 0.5)
    if xi_c != xi:
        sigma = a0 * (1.0 - xi_c)  # keep the mean excess
    else:
        sigma = 2.0 * a0 * a1 / denom
    return xi_c, max(sigma, 1e-6)


def gpd_sf(y: np.ndarray, xi: float, sigma: float) -> np.ndarray:
    """Survival function of the generalized Pareto distribution."""
    y = np.maximum(y, 0.0)
    if abs(xi) < 1e-6:
        return np.exp(-y / sigma)
    base = 1.0 + xi * y / sigma
    with np.errstate(invalid="ignore", divide="ignore"):
        sf = np.where(base > 0, np.power(np.maximum(base, 1e-300), -1.0 / xi), 0.0)
    return sf


class GroupedDetector:
    """Per-group anomaly model with shared autoencoder and calibrated p-values."""

    def __init__(self, numeric: list, categorical: list,
                 parent: Callable[[str], Optional[str]],
                 params: Optional[DetectorParams] = None):
        self.all_numeric = list(numeric)
        self.categorical = list(categorical)
        self.parent = parent
        self.params = params or DetectorParams()
        self.groups: list = []
        self._gidx: dict = {}
        self._resolve_cache: dict = {}
        self.num_idx: list = []
        self.ae_layers: list = []
        self.ae_norm = 1.0

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    @property
    def dn(self) -> int:
        return len(self.num_idx)

    @property
    def dc(self) -> int:
        return len(self.categorical)

    @property
    def d(self) -> int:
        return self.dn + self.dc

    @property
    def feature_names(self) -> list:
        return [self.all_numeric[i] for i in self.num_idx] + self.categorical

    def _one_sided_columns(self) -> list:
        wanted = set(self.params.one_sided)
        return [j for j, i in enumerate(self.num_idx) if self.all_numeric[i] in wanted]

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, keys: np.ndarray, Xnum: np.ndarray, Xcat: list,
            split: np.ndarray) -> dict:
        """split: 0 = fit, 1 = calibration, 2 = validation.
        Returns a report including validation p-values per row."""
        from scipy.special import ndtri

        p = self.params
        keys = keys.astype(str)
        n = keys.shape[0]
        fit_rows = split == 0

        # 1. numeric columns that actually carry data
        finite_share = np.isfinite(Xnum[fit_rows]).mean(axis=0) if fit_rows.any() else np.zeros(Xnum.shape[1])
        self.num_idx = [i for i, s in enumerate(finite_share) if s >= 0.5]
        Xn = Xnum[:, self.num_idx]

        # 2. which groups have enough data; small ones fall back to a parent
        mapping = self._resolve_groups(keys, split)
        self.groups = sorted({v for v in mapping.values() if v is not None})
        self._gidx = {k: i for i, k in enumerate(self.groups)}
        self._resolve_cache = {}
        g = self._group_index(keys)
        G, dn, dc, d = len(self.groups), self.dn, self.dc, self.d
        if G == 0:
            raise ValueError(
                f"not enough baseline data: no context has {p.min_fit} fit rows "
                f"and {p.min_cal} held-out rows"
            )

        K = p.max_knots
        self.knots_x = np.zeros((G, dn, K))
        self.knots_z = np.zeros((G, dn, K))
        self.knots_n = np.zeros((G, dn), dtype=np.int64)
        self.scale_lo = np.ones((G, dn))
        self.scale_hi = np.ones((G, dn))
        self.expl = np.zeros((G, dn, 3))
        self.edge = np.zeros(G)
        self.cats = [[None] * dc for _ in range(G)]
        self.mu = np.zeros((G, d))
        self.prec = np.zeros((G, d, d))
        self.n_fit = np.zeros(G, dtype=np.int64)
        self.ae_layers = []
        self.ae_norm = 1.0

        # 3. per-group transforms and correlation structure
        Z_fit_parts, g_fit_parts = [], []
        for gi in range(G):
            rows = np.nonzero(fit_rows & (g == gi))[0]
            nf = rows.size
            self.n_fit[gi] = nf
            self.edge[gi] = float(ndtri(1.0 - 0.5 / nf))
            for j in range(dn):
                self._fit_feature(gi, j, Xn[rows, j], ndtri)
            for c in range(dc):
                self._fit_category(gi, c, Xcat[c][rows], nf, ndtri)
            Z = self._transform(gi, Xn[rows], [xc[rows] for xc in Xcat], centered=False)
            self._fit_structure(gi, Z)
            Z_fit_parts.append(Z - self.mu[gi])
            g_fit_parts.append(np.full(nf, gi, dtype=np.int64))

        # 4. autoencoder over all groups
        if p.autoencoder and p.ae_weight > 0:
            self._fit_autoencoder(np.vstack(Z_fit_parts), np.concatenate(g_fit_parts))

        # 5. scores for every row, then calibration
        raw = np.full(n, np.nan)
        modeled = g >= 0
        raw[modeled] = self._raw_scores(g[modeled], Xn[modeled],
                                        [xc[modeled] for xc in Xcat])[0]

        Kc = p.cal_grid
        self.cal_q = np.zeros((G, Kc))
        self.cal_n = np.zeros(G, dtype=np.int64)
        self.tail_u = np.zeros(G)
        self.tail_frac = np.zeros(G)
        self.tail_xi = np.zeros(G)
        self.tail_sigma = np.ones(G)

        # Out-of-fold p-values for the held-out rows: calibrate on one held-out
        # half, score the other, and swap. These are honest estimates of how
        # baseline-like traffic will score, used to check the alert budget.
        oof_p = np.full(n, np.nan)
        for cal_part, score_part in ((1, 2), (2, 1)):
            for gi in range(G):
                cal = raw[(split == cal_part) & (g == gi)]
                rows = np.nonzero((split == score_part) & (g == gi))[0]
                if cal.size >= 20 and rows.size:
                    self._calibrate(gi, cal)
                    oof_p[rows] = np.clip(self._pvalues(gi, raw[rows]), P_FLOOR, 1.0)
        # final: calibrate on every held-out row
        for gi in range(G):
            self._calibrate(gi, raw[(split != 0) & (g == gi)])

        return {
            "groups": [
                {"key": str(k), "n_fit": int(self.n_fit[i]), "n_cal": int(self.cal_n[i])}
                for i, k in enumerate(self.groups)
            ],
            "features": self.feature_names,
            "rows": int(n),
            "rows_unmodeled": int((~modeled).sum()),
            "oof_p": oof_p,
            "group_index": g,
        }

    def _resolve_groups(self, keys: np.ndarray, split: np.ndarray) -> dict:
        p = self.params
        uniq, inv = np.unique(keys, return_inverse=True)
        cnt_fit = np.bincount(inv[split == 0], minlength=uniq.size)
        cnt_cal = np.bincount(inv[split != 0], minlength=uniq.size)
        mapping = {str(k): str(k) for k in uniq}
        uniq = [str(k) for k in uniq]
        while True:
            tot_fit, tot_cal = {}, {}
            for k, cf, cc in zip(uniq, cnt_fit, cnt_cal):
                t = mapping[k]
                if t is None:
                    continue
                tot_fit[t] = tot_fit.get(t, 0) + int(cf)
                tot_cal[t] = tot_cal.get(t, 0) + int(cc)
            small = set()
            for t in tot_fit:
                last = self.parent(t) is None
                need_fit = p.min_fit_last if last else p.min_fit
                need_cal = p.min_cal_last if last else p.min_cal
                if tot_fit[t] < need_fit or tot_cal[t] < need_cal:
                    small.add(t)
            if not small:
                break
            for k in uniq:
                if mapping[k] in small:
                    mapping[k] = self.parent(mapping[k])
        return mapping

    def _fit_feature(self, gi: int, j: int, v: np.ndarray, ndtri) -> None:
        v = v[np.isfinite(v)]
        nv = v.size
        if nv < 5:
            self.knots_n[gi, j] = 0
            return
        xs, counts = np.unique(v, return_counts=True)
        cum = np.cumsum(counts)
        mid = np.clip((cum - 0.5 * counts) / nv, 0.5 / nv, 1.0 - 0.5 / nv)
        z = ndtri(mid)
        K = self.params.max_knots
        if xs.size > K:
            target = np.linspace(z[0], z[-1], K)
            idx = np.searchsorted(z, target).clip(0, xs.size - 1)
            idx = np.unique(np.concatenate([[0], idx, [xs.size - 1]]))
            if idx.size > K:
                idx = idx[np.unique(np.linspace(0, idx.size - 1, K).astype(int))]
            xs, z = xs[idx], z[idx]
        k = xs.size
        self.knots_x[gi, j, :k] = xs
        self.knots_z[gi, j, :k] = z
        self.knots_n[gi, j] = k
        q05, q50, q95 = np.quantile(v, [0.05, 0.5, 0.95])
        floor = self.params.scale_floor
        self.scale_lo[gi, j] = max((q50 - q05) / 1.645, floor)
        self.scale_hi[gi, j] = max((q95 - q50) / 1.645, floor)
        self.expl[gi, j] = (q05, q50, q95)

    def _fit_category(self, gi: int, c: int, values: np.ndarray, nf: int, ndtri) -> None:
        uniq, counts = np.unique(values.astype(str), return_counts=True)
        table = {}
        for u, cnt in zip(uniq, counts):
            share = min(cnt / nf, 1.0 - 1e-9)
            table[str(u)] = float(ndtri(1.0 - share / 2.0))
        self.cats[gi][c] = {
            "z": table,
            "unseen": float(ndtri(1.0 - 0.25 / nf)),
            "top": str(uniq[int(np.argmax(counts))]),
        }

    def _fit_structure(self, gi: int, Z: np.ndarray) -> None:
        """Mean and regularized inverse covariance of the z-scores."""
        from sklearn.covariance import LedoitWolf

        d = self.d
        Zc = np.clip(Z, -6.0, 6.0)
        mu = Zc.mean(axis=0)
        sd = Zc.std(axis=0)
        live = sd > 1e-6
        C = np.eye(d)
        if live.sum() >= 2:
            std = (Zc[:, live] - mu[live]) / sd[live]
            cov = LedoitWolf(assume_centered=True).fit(std).covariance_
            dg = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
            C[np.ix_(live, live)] = cov / dg[:, None] / dg[None, :]
        # A feature whose z-scores vary less than 1 (discrete, mostly one
        # value) keeps unit variance so a rare value is not over-counted.
        S = np.maximum(sd, 1.0)
        w, V = np.linalg.eigh(C)
        w = np.maximum(w, self.params.eig_floor)
        C_inv = (V / w) @ V.T
        self.mu[gi] = mu
        self.prec[gi] = C_inv / S[:, None] / S[None, :]

    def _fit_autoencoder(self, Zc: np.ndarray, g: np.ndarray) -> None:
        import warnings

        from sklearn.exceptions import ConvergenceWarning
        from sklearn.neural_network import MLPRegressor

        p = self.params
        d, G = self.d, len(self.groups)
        rng = np.random.default_rng(p.seed)
        if Zc.shape[0] > p.ae_max_rows:
            pick = rng.choice(Zc.shape[0], p.ae_max_rows, replace=False)
            Zc, g = Zc[pick], g[pick]
        target = np.clip(Zc, -p.ae_clip, p.ae_clip)
        onehot = np.zeros((target.shape[0], G))
        onehot[np.arange(target.shape[0]), g] = 1.0
        hidden = max(32, 4 * d)
        bottleneck = max(3, d // 2)
        mlp = MLPRegressor(
            hidden_layer_sizes=(hidden, bottleneck, hidden),
            activation="relu", solver="adam", alpha=1e-4, batch_size=256,
            learning_rate_init=1e-3, max_iter=60, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=6, tol=1e-4,
            random_state=p.seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            mlp.fit(np.hstack([target, onehot]), target)
        self.ae_layers = [
            (np.asarray(W, dtype=np.float64), np.asarray(b, dtype=np.float64))
            for W, b in zip(mlp.coefs_, mlp.intercepts_)
        ]
        mse = self._ae_residual(target, g)[0]
        self.ae_norm = max(float(np.median(mse)), 1e-3)

    def _calibrate(self, gi: int, scores: np.ndarray) -> None:
        p = self.params
        scores = np.sort(scores[np.isfinite(scores)])
        n = scores.size
        self.cal_n[gi] = n
        if n == 0:
            self.cal_q[gi] = 0.0
            self.tail_u[gi], self.tail_frac[gi] = np.inf, 1.0
            return
        self.cal_q[gi] = np.quantile(scores, np.linspace(0.0, 1.0, p.cal_grid))
        k = int(min(max(round(p.tail_fraction * n), p.tail_min), p.tail_max, n - 1))
        if k < 5:
            self.tail_u[gi], self.tail_frac[gi] = np.inf, 1.0
            return
        u = scores[n - k - 1]
        xi, sigma = fit_gpd(scores[n - k:] - u)
        self.tail_u[gi] = u
        self.tail_frac[gi] = k / n
        self.tail_xi[gi] = xi
        self.tail_sigma[gi] = sigma

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _group_index(self, keys: np.ndarray) -> np.ndarray:
        uniq, inv = np.unique(keys.astype(str), return_inverse=True)
        idx = np.empty(uniq.size, dtype=np.int64)
        for i, k in enumerate(uniq):
            hit = self._resolve_cache.get(k)
            if hit is None:
                cur = k
                while cur is not None and cur not in self._gidx:
                    cur = self.parent(cur)
                hit = self._gidx[cur] if cur is not None else -1
                if len(self._resolve_cache) > 100_000:
                    self._resolve_cache.clear()
                self._resolve_cache[k] = hit
            idx[i] = hit
        return idx[inv]

    def _transform(self, gi: int, Xn: np.ndarray, Xcat: list,
                   centered: bool = True) -> np.ndarray:
        m = Xn.shape[0]
        Z = np.zeros((m, self.d))
        edge = self.edge[gi]
        for j in range(self.dn):
            k = self.knots_n[gi, j]
            if k == 0:
                Z[:, j] = self.mu[gi, j]
                continue
            x = Xn[:, j]
            xs = self.knots_x[gi, j, :k]
            zs = self.knots_z[gi, j, :k]
            with np.errstate(invalid="ignore"):
                z = np.interp(x, xs, zs)
                hi = x > xs[-1]
                lo = x < xs[0]
            if hi.any():
                z[hi] = max(zs[-1], edge) + (x[hi] - xs[-1]) / self.scale_hi[gi, j]
            if lo.any():
                z[lo] = min(zs[0], -edge) - (xs[0] - x[lo]) / self.scale_lo[gi, j]
            missing = ~np.isfinite(x)
            if missing.any():
                z[missing] = np.nan
            Z[:, j] = z
        for c in range(self.dc):
            table = self.cats[gi][c]
            uniq, inv = np.unique(Xcat[c].astype(str), return_inverse=True)
            zu = np.array([table["z"].get(u, table["unseen"]) for u in uniq])
            Z[:, self.dn + c] = zu[inv]
        cols = self._one_sided_columns()
        if cols:
            with np.errstate(invalid="ignore"):
                Z[:, cols] = np.maximum(Z[:, cols], 0.0)  # NaN stays NaN
        if centered:
            Z -= self.mu[gi]
            Z[~np.isfinite(Z)] = 0.0  # missing value: neutral
        else:
            col_mean = np.nanmean(Z, axis=0) if m else np.zeros(self.d)
            col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
            bad = ~np.isfinite(Z)
            if bad.any():
                Z[bad] = np.take(col_mean, np.nonzero(bad)[1])
        return Z

    def _ae_residual(self, Zc: np.ndarray, g: np.ndarray, chunk: int = 50_000):
        """Zc: centered z-scores already clipped. Returns (mse, squared error)."""
        m, G = Zc.shape[0], len(self.groups)
        err = np.zeros_like(Zc)
        for start in range(0, m, chunk):
            z = Zc[start:start + chunk]
            onehot = np.zeros((z.shape[0], G))
            onehot[np.arange(z.shape[0]), g[start:start + chunk]] = 1.0
            a = np.hstack([z, onehot])
            for W, b in self.ae_layers[:-1]:
                a = np.maximum(a @ W + b, 0.0)
            W, b = self.ae_layers[-1]
            err[start:start + chunk] = (z - (a @ W + b)) ** 2
        return err.mean(axis=1), err

    def _raw_scores(self, g: np.ndarray, Xn: np.ndarray, Xcat: list):
        """g must contain modeled group indexes only. Returns (score, contributions)."""
        m, d = g.shape[0], self.d
        Zc = np.zeros((m, d))
        contrib = np.zeros((m, d))
        for gi in np.unique(g):
            rows = np.nonzero(g == gi)[0]
            Z = self._transform(int(gi), Xn[rows], [xc[rows] for xc in Xcat])
            Zc[rows] = Z
            contrib[rows] = Z * (Z @ self.prec[gi]) / d
        dist = contrib.sum(axis=1)
        ae_part = None
        if self.ae_layers:
            clip = self.params.ae_clip
            _, err = self._ae_residual(np.clip(Zc, -clip, clip), g)
            ae_part = err / (d * self.ae_norm)
        w = self.params.ae_weight
        if self.params.combine == "log":
            score = np.log1p(np.maximum(dist, 0.0))
            if ae_part is not None:
                ae_total = ae_part.sum(axis=1)
                ae_log = w * np.log1p(ae_total)
                # share each term's (log-scale) weight among its features
                scale_d = score / np.maximum(dist, 1e-12)
                scale_a = ae_log / np.maximum(ae_total, 1e-12)
                contrib = contrib * scale_d[:, None] + ae_part * scale_a[:, None]
                score = score + ae_log
        else:
            score = dist
            if ae_part is not None:
                contrib = contrib + w * ae_part
                score = score + w * ae_part.sum(axis=1)
        return score, contrib

    def _pvalues(self, gi: int, s: np.ndarray, floored: bool = True) -> np.ndarray:
        n = self.cal_n[gi]
        if n == 0:
            return np.ones_like(s)
        grid = np.linspace(0.0, 1.0, self.cal_q.shape[1])
        F = np.interp(s, self.cal_q[gi], grid)
        p_emp = ((1.0 - F) * n + 1.0) / (n + 1.0)
        u = self.tail_u[gi]
        if not np.isfinite(u):
            return p_emp
        frac = self.tail_frac[gi]
        tail = frac * gpd_sf(s - u, self.tail_xi[gi], self.tail_sigma[gi])
        p = np.where(s > u, tail, np.maximum(p_emp, frac))
        if not floored:
            return p
        return np.maximum(p, self.p_floor(gi))

    def p_floor(self, gi: int) -> float:
        """Smallest p-value this group can report."""
        n = max(int(self.cal_n[gi]), 1)
        return 1.0 / (self.params.max_extrapolation * n)

    def score(self, keys: np.ndarray, Xnum: np.ndarray, Xcat: list) -> dict:
        """Returns per-row group index (-1 = no model for this context),
        p-value, surprise (-log10 p) and per-feature contributions."""
        n = keys.shape[0]
        g = self._group_index(keys) if n else np.zeros(0, dtype=np.int64)
        p = np.ones(n)
        deep = np.ones(n)  # tail estimate without the evidence floor
        raw = np.full(n, np.nan)
        contrib = np.zeros((n, self.d))
        modeled = g >= 0
        if modeled.any():
            Xn = Xnum[:, self.num_idx]
            rows = np.nonzero(modeled)[0]
            raw[rows], contrib[rows] = self._raw_scores(
                g[rows], Xn[rows], [xc[rows] for xc in Xcat])
            for gi in np.unique(g[rows]):
                sel = rows[g[rows] == gi]
                deep[sel] = self._pvalues(int(gi), raw[sel], floored=False)
                p[sel] = np.maximum(deep[sel], self.p_floor(int(gi)))
        p = np.clip(p, P_FLOOR, 1.0)
        deep = np.clip(deep, P_FLOOR, 1.0)
        return {"group": g, "p": p, "surprise": -np.log10(p),
                "depth": -np.log10(deep), "raw": raw, "contrib": contrib}

    # ------------------------------------------------------------------
    # Explanations
    # ------------------------------------------------------------------

    def typical_range(self, gi: int, feature: str):
        """(q05, q50, q95) of a numeric feature in its transformed units, or
        the most common category for a categorical feature."""
        names = self.feature_names
        j = names.index(feature)
        if j < self.dn:
            return tuple(float(v) for v in self.expl[gi, j])
        return self.cats[gi][j - self.dn]["top"]

    # ------------------------------------------------------------------
    # Persistence (JSON + npz; no pickle)
    # ------------------------------------------------------------------

    def to_state(self):
        meta = {
            "numeric": self.all_numeric,
            "categorical": self.categorical,
            "num_idx": self.num_idx,
            "groups": self.groups,
            "cats": self.cats,
            "params": asdict(self.params),
            "ae_norm": self.ae_norm,
            "ae_depth": len(self.ae_layers),
        }
        arrays = {
            name: getattr(self, name)
            for name in ("knots_x", "knots_z", "knots_n", "scale_lo", "scale_hi",
                         "expl", "edge", "mu", "prec", "n_fit", "cal_q", "cal_n",
                         "tail_u", "tail_frac", "tail_xi", "tail_sigma")
        }
        for i, (W, b) in enumerate(self.ae_layers):
            arrays[f"ae_W{i}"] = W
            arrays[f"ae_b{i}"] = b
        return meta, arrays

    @classmethod
    def from_state(cls, meta: dict, arrays: dict,
                   parent: Callable[[str], Optional[str]]) -> "GroupedDetector":
        det = cls(meta["numeric"], meta["categorical"], parent,
                  DetectorParams(**meta["params"]))
        det.num_idx = list(meta["num_idx"])
        det.groups = list(meta["groups"])
        det._gidx = {k: i for i, k in enumerate(det.groups)}
        det._resolve_cache = {}
        det.cats = meta["cats"]
        det.ae_norm = float(meta["ae_norm"])
        for name in ("knots_x", "knots_z", "knots_n", "scale_lo", "scale_hi",
                     "expl", "edge", "mu", "prec", "n_fit", "cal_q", "cal_n",
                     "tail_u", "tail_frac", "tail_xi", "tail_sigma"):
            setattr(det, name, np.asarray(arrays[name]))
        det.ae_layers = [
            (np.asarray(arrays[f"ae_W{i}"]), np.asarray(arrays[f"ae_b{i}"]))
            for i in range(int(meta["ae_depth"]))
        ]
        return det
