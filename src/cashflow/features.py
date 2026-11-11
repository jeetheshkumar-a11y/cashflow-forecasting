"""Turn daily (or hourly) series into supervised windows.

At a forecast *origin* ``t`` we know everything up to the end of period
``t-1`` and must forecast income and expense for ``t .. t+H-1``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def calendar_features(dates: pd.DatetimeIndex, freq: str = "D") -> np.ndarray:
    dates = pd.DatetimeIndex(dates)
    if freq == "H":
        h = dates.hour.values
        dw = dates.dayofweek.values
        return np.stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                         np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7)], 1).astype("float32")
    dom = dates.day.values
    dim = dates.days_in_month.values
    dw = dates.dayofweek.values
    frac = (dom - 1) / dim
    return np.stack([
        np.sin(2 * np.pi * frac), np.cos(2 * np.pi * frac),
        np.sin(2 * np.pi * dw / 7), np.cos(2 * np.pi * dw / 7),
        (dom <= 5).astype(float), (dim - dom < 4).astype(float),
    ], 1).astype("float32")


@dataclass
class Series:
    """All series of a dataset, as arrays per user."""
    users: list
    dates: dict
    inc: dict
    exp: dict
    bal: dict
    persona: dict
    cal: dict
    freq: str = "D"

    @classmethod
    def from_frame(cls, df: pd.DataFrame, freq: str = "D") -> "Series":
        users, dates, inc, exp, bal, per, cal = [], {}, {}, {}, {}, {}, {}
        if "user_id" not in df:
            df = df.assign(user_id=0)
        for u, g in df.sort_values("date").groupby("user_id", sort=True):
            users.append(u)
            dates[u] = pd.DatetimeIndex(g["date"].values)
            inc[u] = g["income"].to_numpy(float)
            exp[u] = g["expense"].to_numpy(float)
            bal[u] = g["balance"].to_numpy(float) if "balance" in g else np.cumsum(inc[u] - exp[u])
            per[u] = g["persona"].iloc[0] if "persona" in g else "all"
            cal[u] = calendar_features(dates[u], freq)
        return cls(users, dates, inc, exp, bal, per, cal, freq)


@dataclass
class Windows:
    X: np.ndarray        # (N, L, F) scaled inputs
    y: np.ndarray        # (N, 2H) scaled targets  [income..., expense...]
    scale: np.ndarray    # (N,)
    y_inc: np.ndarray    # (N, H) raw
    y_exp: np.ndarray    # (N, H) raw
    user: np.ndarray
    origin: np.ndarray   # integer index of first forecast period
    origin_date: np.ndarray
    persona: np.ndarray
    b0: np.ndarray       # balance at the start of the forecast window
    sigma: np.ndarray    # recent volatility of H-period net flow (for normalized conformal)

    def __len__(self):
        return len(self.user)

    def subset(self, mask):
        return Windows(**{k: getattr(self, k)[mask] for k in self.__dataclass_fields__})


def _net_volatility(inc, exp, t, H, s, lookback=180):
    """Typical size of an H-day net cash flow swing for this user, from the
    std of rolling H-day net sums over the last ``lookback`` days (plus a
    small floor).  Used to scale conformal scores."""
    net = (inc - exp)[max(0, t - lookback):t]
    roll = np.convolve(net, np.ones(H), mode="valid")
    return float(roll.std() + 0.1 * s * H)


def make_windows(S: Series, origins: dict, L: int, H: int) -> Windows:
    """``origins``: dict user -> iterable of origin indices."""
    X, y, sc, yi, ye, us, org, od, per, b0, sig = ([] for _ in range(11))
    for u in S.users:
        inc, exp, bal, cal = S.inc[u], S.exp[u], S.bal[u], S.cal[u]
        for t in origins.get(u, []):
            if t - L < 0 or t + H > len(inc):
                continue
            pi, pe = inc[t - L:t], exp[t - L:t]
            s = 0.5 * (pi.mean() + pe.mean()) + 1.0
            b = bal[t - 1]
            feats = np.column_stack([
                pi / s, pe / s,
                np.full(L, np.tanh(b / (s * H))),  # cushion in "months of flow"
                cal[t - L:t],
            ]).astype("float32")
            X.append(feats)
            yi.append(inc[t:t + H]); ye.append(exp[t:t + H])
            y.append(np.concatenate([inc[t:t + H], exp[t:t + H]]) / s)
            sc.append(s); us.append(u); org.append(t); od.append(S.dates[u][t])
            per.append(S.persona[u]); b0.append(b)
            sig.append(_net_volatility(inc, exp, t, H, s))
    return Windows(np.array(X, "float32"), np.array(y, "float32"), np.array(sc), np.array(yi),
                   np.array(ye), np.array(us), np.array(org), np.array(od), np.array(per),
                   np.array(b0), np.array(sig))
