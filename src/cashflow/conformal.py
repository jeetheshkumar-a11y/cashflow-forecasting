"""Uncertainty: conformal prediction and shortfall probability.

The question a freelancer actually asks is
    "What's the chance I run short this month?"
A single point forecast cannot answer that.  We use *split conformal
prediction*: hold out a calibration period, measure how wrong the model
was there, and use the distribution of those errors to build
    * prediction intervals with a guaranteed coverage level, and
    * a predictive distribution for the lowest balance in the next 30 days,
      from which P(balance < 0) follows directly.

The guarantee only needs the calibration and test errors to be
exchangeable -- it works on top of *any* model (ETS, LSTM, ...), which is
why it is a good fit for a model-comparison project.

Two variants:
    standard   : score = |y - y_hat|            (same width for everybody)
    normalized : score = |y - y_hat| / sigma     (sigma = user's recent
                 volatility, so stable earners get narrow intervals and
                 volatile freelancers get wide ones)
"""
from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample corrected (1-alpha) quantile of nonconformity scores."""
    # the k-th smallest score with k = ceil((n+1)(1-alpha)) -- the textbook
    # split-conformal rule (identical to MAPIE)
    s = np.sort(np.asarray(scores))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.inf) if k > n else float(s[k - 1])


def split_conformal(cal_true, cal_pred, test_pred, alpha=0.1, cal_sigma=None, test_sigma=None):
    """Symmetric split-conformal interval.  Returns (lower, upper, qhat)."""
    cal_true, cal_pred, test_pred = map(np.asarray, (cal_true, cal_pred, test_pred))
    if cal_sigma is None:
        q = conformal_quantile(np.abs(cal_true - cal_pred), alpha)
        return test_pred - q, test_pred + q, q
    q = conformal_quantile(np.abs(cal_true - cal_pred) / cal_sigma, alpha)
    return test_pred - q * test_sigma, test_pred + q * test_sigma, q


def min_balance_path(b0, net_daily):
    """Lowest balance over the horizon given a starting balance and daily net flows."""
    return np.asarray(b0) + np.cumsum(net_daily, axis=1).min(axis=1)


class ShortfallRisk:
    """Conformal predictive distribution for the minimum balance over the
    horizon.  ``P(short) = P(min balance < threshold)``.

    For a test window with predicted minimum balance m_hat and scale s, the
    predictive distribution is  { m_hat + s * r_i }  over the calibration
    residuals r_i = (m_i - m_hat_i) / s_i.  Its CDF at the threshold is the
    shortfall probability (smoothed with the usual (k+U)/(n+1) rule)."""

    def __init__(self, normalized=True):
        self.normalized = normalized

    def fit(self, true_min, pred_min, sigma=None):
        s = sigma if (self.normalized and sigma is not None) else np.ones_like(true_min)
        self.r = np.sort((np.asarray(true_min) - np.asarray(pred_min)) / s)
        return self

    def predict_proba(self, pred_min, sigma=None, threshold=0.0):
        s = sigma if (self.normalized and sigma is not None) else np.ones_like(pred_min)
        need = (threshold - np.asarray(pred_min)) / s  # residual needed to fall below
        k = np.searchsorted(self.r, need, side="left")
        return (k + 0.5) / (len(self.r) + 1)

    def quantiles(self, pred_min, sigma=None, qs=(0.05, 0.5, 0.95)):
        s = sigma if (self.normalized and sigma is not None) else np.ones_like(pred_min)
        rq = np.quantile(self.r, qs)
        return np.asarray(pred_min)[:, None] + np.asarray(s)[:, None] * rq[None, :]


def balance_fan(b0, pred_net_daily, cal_true_net, cal_pred_net, cal_sigma, sigma, alphas=(0.2, 0.5)):
    """Per-day conformal bands for the cumulative balance path of ONE window.

    For each horizon day h, scores are |cum_true_h - cum_pred_h| / sigma on the
    calibration set; intervals are the cumulative prediction +/- q_h * sigma.
    Returns dict alpha -> (lower[H], upper[H]) and the point path."""
    cum_cal_t = np.cumsum(cal_true_net, 1)
    cum_cal_p = np.cumsum(cal_pred_net, 1)
    point = b0 + np.cumsum(pred_net_daily)
    bands = {}
    for a in alphas:
        q = np.array([conformal_quantile(np.abs(cum_cal_t[:, h] - cum_cal_p[:, h]) / cal_sigma, a)
                      for h in range(cum_cal_t.shape[1])])
        bands[a] = (point - q * sigma, point + q * sigma)
    return point, bands


def mapie_crosscheck(cal_true, cal_pred, test_pred, alpha=0.1):
    """Same split-conformal interval computed with MAPIE, used as a sanity
    check of our implementation.  The point forecast is passed in as the only
    feature to an identity 'prefit' regressor."""
    from mapie.regression import SplitConformalRegressor
    from sklearn.base import BaseEstimator, RegressorMixin

    class Identity(BaseEstimator, RegressorMixin):
        def fit(self, X, y):
            self.fitted_ = True
            return self

        def predict(self, X):
            return np.asarray(X)[:, 0]

    est = Identity().fit(np.c_[cal_pred], cal_true)
    m = SplitConformalRegressor(est, confidence_level=1 - alpha, prefit=True)
    m.conformalize(np.c_[cal_pred], np.asarray(cal_true))
    _, iv = m.predict_interval(np.c_[test_pred])
    return iv[:, 0, 0], iv[:, 1, 0]
