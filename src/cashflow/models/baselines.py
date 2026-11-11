"""Simple baselines.  Each model exposes

    fit(train_windows, series)          (may be a no-op)
    predict(windows, series) -> (pred_income[N,H], pred_expense[N,H])

Local statistical models (ETS, Prophet) are re-fitted for every user and
every forecast origin using only data before that origin.
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import Ridge

from ..features import Series, Windows


class Model:
    name = "model"
    global_model = False

    def fit(self, W: Windows, S: Series):
        return self

    def predict(self, W: Windows, S: Series):
        raise NotImplementedError


# ---------------------------------------------------------------- naive family
class HistoricMean(Model):
    """Every future day = average day of the last ``L`` days."""
    name = "Mean (90d)"

    def __init__(self, L=90):
        self.L = L

    def predict(self, W, S):
        H = W.y_inc.shape[1]
        pi = np.array([S.inc[u][t - self.L:t].mean() for u, t in zip(W.user, W.origin)])
        pe = np.array([S.exp[u][t - self.L:t].mean() for u, t in zip(W.user, W.origin)])
        return np.repeat(pi[:, None], H, 1), np.repeat(pe[:, None], H, 1)


def _lag_index(dates: pd.DatetimeIndex, t: int, H: int, months: int, lag: int | None):
    """Index of the 'same' period ``months`` months (or ``lag`` steps) earlier."""
    if lag is not None:
        # repeat the season that started ``months`` seasons ago
        return t - lag * months + (np.arange(H) % lag)
    future = dates[t:t + H] if t + H <= len(dates) else pd.date_range(dates[t], periods=H, freq="D")
    past = future - pd.DateOffset(months=months)
    idx = dates.get_indexer(past)
    return idx


class SeasonalNaive(Model):
    """Same calendar day last month (daily data) or same hour ``lag`` steps
    earlier (hourly data).  With ``k>1`` averages the last k seasons -- a
    'monthly profile' forecast that captures month-end bills well."""

    def __init__(self, k=1, lag=None):
        self.k, self.lag = k, lag
        self.name = "Seasonal naive (last month)" if k == 1 else f"Seasonal avg (last {k} months)"
        if lag:
            self.name = "Seasonal naive" if k == 1 else f"Seasonal avg (last {k})"

    def predict(self, W, S):
        H = W.y_inc.shape[1]
        PI, PE = np.zeros((len(W), H)), np.zeros((len(W), H))
        for n, (u, t) in enumerate(zip(W.user, W.origin)):
            acc_i, acc_e = np.zeros(H), np.zeros(H)
            for m in range(1, self.k + 1):
                idx = _lag_index(S.dates[u], t, H, m, self.lag)
                idx = np.clip(idx, 0, t - 1)
                acc_i += S.inc[u][idx]
                acc_e += S.exp[u][idx]
            PI[n], PE[n] = acc_i / self.k, acc_e / self.k
        return PI, PE


# ---------------------------------------------------------------- ETS
def _ets_one(y, H, period):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = ExponentialSmoothing(y, trend=None, seasonal="add", seasonal_periods=period,
                                     initialization_method="estimated").fit(optimized=True)
            f = m.forecast(H)
        except Exception:  # pragma: no cover - fall back to the mean
            f = np.full(H, y.mean())
    return np.clip(f, 0, None)


class ETS(Model):
    """Holt-Winters exponential smoothing, ETS(A,N,A), weekly seasonality,
    fitted per user and per origin on the last ``history`` days."""
    name = "ETS"

    def __init__(self, history=365, period=7, n_jobs=-1):
        self.history, self.period, self.n_jobs = history, period, n_jobs

    def predict(self, W, S):
        H = W.y_inc.shape[1]

        def job(u, t):
            lo = max(0, t - self.history)
            return (_ets_one(S.inc[u][lo:t], H, self.period),
                    _ets_one(S.exp[u][lo:t], H, self.period))
        out = Parallel(n_jobs=self.n_jobs)(delayed(job)(u, t) for u, t in zip(W.user, W.origin))
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])


# ---------------------------------------------------------------- Prophet
def _prophet_one(dates, y, H, freq):
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)
    from prophet import Prophet
    df = pd.DataFrame({"ds": dates, "y": y})
    if freq == "D":
        m = Prophet(yearly_seasonality=len(df) >= 730, weekly_seasonality=True,
                    daily_seasonality=False, uncertainty_samples=0)
        m.add_seasonality("monthly", period=30.5, fourier_order=6)

        def regs(d):
            d = d.copy()
            d["month_start"] = (d.ds.dt.day <= 5).astype(float)
            d["month_end"] = (d.ds.dt.days_in_month - d.ds.dt.day < 4).astype(float)
            return d
        m.add_regressor("month_start"); m.add_regressor("month_end")
        df = regs(df)
        m.fit(df)
        fut = regs(pd.DataFrame({"ds": pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=H, freq="D")}))
    else:
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True,
                    uncertainty_samples=0)
        m.fit(df)
        fut = pd.DataFrame({"ds": pd.date_range(dates[-1] + pd.Timedelta(hours=1), periods=H, freq="h")})
    return np.clip(m.predict(fut)["yhat"].values, 0, None)


class ProphetModel(Model):
    """Meta's Prophet with weekly + monthly seasonality and month-start /
    month-end regressors (so it *knows* about rent and card bills)."""
    name = "Prophet"

    def __init__(self, history=365, n_jobs=-1):
        self.history, self.n_jobs = history, n_jobs

    def predict(self, W, S):
        H = W.y_inc.shape[1]

        def job(u, t):
            lo = max(0, t - self.history)
            d = S.dates[u][lo:t]
            return (_prophet_one(d, S.inc[u][lo:t], H, S.freq), _prophet_one(d, S.exp[u][lo:t], H, S.freq))
        out = Parallel(n_jobs=self.n_jobs)(delayed(job)(u, t) for u, t in zip(W.user, W.origin))
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])


# ---------------------------------------------------------------- linear
class RidgeDirect(Model):
    """Global linear model on the flattened (scaled) input window.  A strong
    'simple ML' baseline that deep models must beat to justify themselves."""
    name = "Ridge (linear)"
    global_model = True

    def __init__(self, alpha=10.0):
        self.m = Ridge(alpha=alpha)

    def fit(self, W, S, **_):
        self.m.fit(W.X.reshape(len(W), -1), W.y)
        return self

    def predict(self, W, S):
        H = W.y_inc.shape[1]
        p = np.clip(self.m.predict(W.X.reshape(len(W), -1)), 0, None) * W.scale[:, None]
        return p[:, :H], p[:, H:]
