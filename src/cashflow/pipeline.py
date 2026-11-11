"""Shared experiment plumbing: splits, fitting, uncertainty evaluation."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .conformal import ShortfallRisk, mapie_crosscheck, min_balance_path, split_conformal
from .features import Series, Windows, make_windows
from .metrics import auc, brier, interval_metrics, point_metrics


@dataclass
class Split:
    L: int = 90
    H: int = 30
    train_end: int = 610      # last training origin (targets end before val)
    train_step: int = 3
    val: tuple = (640, 700, 3)
    cal: tuple = (730, 900, 7)
    test: tuple = (930, None, 7)   # None -> until the end of the data

    def origins(self, S: Series, which: str):
        out = {}
        for u in S.users:
            n = len(S.inc[u])
            if which == "train":
                r = range(self.L, self.train_end + 1, self.train_step)
            else:
                a, b, s = getattr(self, which)
                b = n - self.H if b is None else b
                r = range(a, min(b, n - self.H) + 1, s)
            out[u] = list(r)
        return out

    def windows(self, S: Series, which: str) -> Windows:
        return make_windows(S, self.origins(S, which), self.L, self.H)


def fit_predict(model, Wtr, Wval, S, eval_sets: dict):
    t0 = time.time()
    if getattr(model, "global_model", False):
        if hasattr(model, "kind"):
            model.fit(Wtr, S, W_val=Wval)
        else:
            model.fit(Wtr, S)
    fit_s = time.time() - t0
    preds = {}
    for k, (W, SS) in eval_sets.items():
        t1 = time.time()
        preds[k] = model.predict(W, SS)
        preds[k + "_seconds"] = time.time() - t1
    preds["fit_seconds"] = fit_s
    return preds


def net_total(W, p=None):
    if p is None:
        return (W.y_inc - W.y_exp).sum(1)
    return (p[0] - p[1]).sum(1)


def uncertainty_report(name, Wc, pc, Wt, pt, alphas=(0.1, 0.2), check_mapie=False):
    """Conformal intervals for the 30-day net flow + shortfall probability."""
    rows, per_persona = [], []
    yc, yt = net_total(Wc), net_total(Wt)
    fc, ft = net_total(Wc, pc), net_total(Wt, pt)
    for a in alphas:
        for kind in ("standard", "normalized"):
            if kind == "standard":
                lo, hi, q = split_conformal(yc, fc, ft, a)
            else:
                lo, hi, q = split_conformal(yc, fc, ft, a, Wc.sigma, Wt.sigma)
            m = interval_metrics(yt, lo, hi)
            rows.append({"model": name, "nominal": 1 - a, "method": kind, **m})
            if check_mapie and kind == "standard":
                mlo, mhi = mapie_crosscheck(yc, fc, ft, a)
                rows[-1]["mapie_max_abs_diff"] = float(max(np.abs(mlo - lo).max(), np.abs(mhi - hi).max()))
            for p in np.unique(Wt.persona):
                mk = Wt.persona == p
                per_persona.append({"model": name, "nominal": 1 - a, "method": kind, "persona": p,
                                    **interval_metrics(yt[mk], lo[mk], hi[mk])})

    # shortfall: lowest balance in the next H days below zero?
    mc_true = min_balance_path(Wc.b0, Wc.y_inc - Wc.y_exp)
    mc_pred = min_balance_path(Wc.b0, pc[0] - pc[1])
    mt_true = min_balance_path(Wt.b0, Wt.y_inc - Wt.y_exp)
    mt_pred = min_balance_path(Wt.b0, pt[0] - pt[1])
    event = (mt_true < 0).astype(float)
    sf = {}
    risk = ShortfallRisk(normalized=True).fit(mc_true, mc_pred, Wc.sigma)
    p = risk.predict_proba(mt_pred, Wt.sigma)
    sf = {"model": name, "Brier (conformal P)": brier(p, event), "AUC": auc(p, event),
          "Brier (point forecast 0/1)": brier((mt_pred < 0).astype(float), event),
          "base rate": float(event.mean())}
    # "hard" cases: people whose current balance is below one month of spending
    hard = Wt.b0 < Wt.y_exp.sum(1)
    sf["Brier hard cases"] = brier(p[hard], event[hard])
    sf["n hard"] = int(hard.sum())
    return pd.DataFrame(rows), pd.DataFrame(per_persona), sf, p, event


def point_table(preds, W, key="test"):
    rows = []
    for name, P in preds.items():
        rows.append({"model": name, **point_metrics(W, *P[key])})
    return pd.DataFrame(rows)
