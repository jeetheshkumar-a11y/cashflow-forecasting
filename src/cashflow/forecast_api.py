"""One-call forecasting for a single person -- used by the demo app.

    from cashflow.forecast_api import forecast
    res = forecast(daily_df, model="GRU")
    res["p_short"], res["net_interval"], res["balance_point"], ...
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .conformal import ShortfallRisk, balance_fan, min_balance_path, split_conformal
from .features import Series, make_windows

ART = Path(__file__).resolve().parents[2] / "artifacts"
L, H = 90, 30
DEMO_MODELS = ("GRU", "N-BEATS", "LSTM", "Seasonal avg (last 3 months)", "ETS")
_KIND = {"GRU": "gru", "N-BEATS": "nbeats", "LSTM": "lstm"}


@lru_cache(maxsize=None)
def _calibration():
    z = np.load(ART / "calibration.npz")
    out = {}
    for k in z.files:
        model, field = k.split("|")
        out.setdefault(model, {})[field] = z[k]
    return out


@lru_cache(maxsize=None)
def _model(name):
    from .models.baselines import ETS, SeasonalNaive
    from .models.deep import KerasForecaster
    if name in _KIND:
        return KerasForecaster.load(ART / f"{_KIND[name]}.keras", kind=_KIND[name])
    if name == "ETS":
        return ETS(n_jobs=1)
    return SeasonalNaive(3)


def available_models():
    cal = _calibration() if (ART / "calibration.npz").exists() else {}
    ok = []
    for m in DEMO_MODELS:
        if m not in cal:
            continue
        if m in _KIND and not (ART / f"{_KIND[m]}.keras").exists():
            continue
        ok.append(m)
    return ok


def forecast(daily: pd.DataFrame, model: str = "GRU", levels=(0.8, 0.5)):
    """``daily`` needs columns date, income, expense, balance (one person,
    at least 90 days).  Forecasts the 30 days after the last date."""
    daily = daily.sort_values("date").reset_index(drop=True)
    if len(daily) < L + 1:
        raise ValueError(f"need at least {L + 1} days of history, got {len(daily)}")
    # pad H future days with zeros so the window builder can be reused
    fut_dates = pd.date_range(daily["date"].iloc[-1] + pd.Timedelta(days=1), periods=H, freq="D")
    pad = pd.DataFrame({"date": fut_dates, "income": 0.0, "expense": 0.0,
                        "balance": daily["balance"].iloc[-1]})
    full = pd.concat([daily[["date", "income", "expense", "balance"]], pad], ignore_index=True)
    full["user_id"] = 0
    S = Series.from_frame(full)
    t = len(daily)
    W = make_windows(S, {0: [t]}, L, H)
    pi, pe = _model(model).predict(W, S)
    pi, pe = np.clip(pi[0], 0, None), np.clip(pe[0], 0, None)

    c = _calibration()[model]
    alphas = tuple(round(1 - lv, 3) for lv in levels)
    point, bands = balance_fan(W.b0[0], pi - pe, c["cal_true_net"], c["cal_pred_net"], c["cal_sigma"],
                               W.sigma[0], alphas=alphas)
    lo, hi, _ = split_conformal(c["cal_true_net"].sum(1), c["cal_pred_net"].sum(1),
                                np.array([(pi - pe).sum()]), alphas[0], c["cal_sigma"], W.sigma[:1])
    risk = ShortfallRisk().fit(min_balance_path(c["cal_b0"], c["cal_true_net"]),
                               min_balance_path(c["cal_b0"], c["cal_pred_net"]), c["cal_sigma"])
    m_pred = min_balance_path(W.b0, (pi - pe)[None])
    p_short = float(risk.predict_proba(m_pred, W.sigma)[0])
    return {
        "dates": fut_dates, "income": pi, "expense": pe, "net_total": float((pi - pe).sum()),
        "net_interval": (float(lo[0]), float(hi[0])), "level": levels[0],
        "balance_point": point, "bands": bands, "b0": float(W.b0[0]),
        "p_short": p_short, "min_balance": float(m_pred[0]),
        "min_balance_q": risk.quantiles(m_pred, W.sigma, (0.1, 0.5, 0.9))[0],
    }
