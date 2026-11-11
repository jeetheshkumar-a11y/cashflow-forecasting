"""Point-forecast and probabilistic metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def point_metrics(W, pi, pe) -> dict:
    """Errors in currency units (INR for the simulator)."""
    ti, te = W.y_inc, W.y_exp
    net_t, net_p = ti - te, pi - pe
    cum_t, cum_p = np.cumsum(net_t, 1), np.cumsum(net_p, 1)
    return {
        "MAE 30d income": np.abs(ti.sum(1) - pi.sum(1)).mean(),
        "MAE 30d expense": np.abs(te.sum(1) - pe.sum(1)).mean(),
        "MAE 30d net": np.abs(net_t.sum(1) - net_p.sum(1)).mean(),
        "MAE balance path": np.abs(cum_t - cum_p).mean(),
        "MAE daily income": np.abs(ti - pi).mean(),
        "MAE daily expense": np.abs(te - pe).mean(),
    }


def per_window_errors(W, pi, pe) -> np.ndarray:
    return np.abs((W.y_inc - W.y_exp).sum(1) - (pi - pe).sum(1))


def paired_bootstrap(err_a, err_b, groups, n_boot=2000, seed=0):
    """Mean difference err_a - err_b with a 95% CI, resampling *users*
    (windows from the same user are correlated, so we must not treat them as
    independent)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"g": groups, "d": err_a - err_b})
    per_user = df.groupby("g")["d"].agg(["sum", "count"])
    s, c = per_user["sum"].to_numpy(), per_user["count"].to_numpy()
    idx = rng.integers(0, len(s), (n_boot, len(s)))
    boots = s[idx].sum(1) / c[idx].sum(1)
    return float((s.sum() / c.sum())), float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def interval_metrics(y, lo, hi) -> dict:
    return {"coverage": float(np.mean((y >= lo) & (y <= hi))), "mean width": float(np.mean(hi - lo))}


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def reliability_table(p, y, bins=10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": p, "y": y})
    return df.groupby("bin").agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size")).reset_index()


def auc(p, y) -> float:
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")
