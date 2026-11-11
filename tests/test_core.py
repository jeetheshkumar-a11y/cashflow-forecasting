"""Fast unit tests (no deep-model training).  Run with:  pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cashflow.conformal import ShortfallRisk, conformal_quantile, min_balance_path, split_conformal
from cashflow.features import Series, make_windows
from cashflow.metrics import paired_bootstrap, point_metrics
from cashflow.models.baselines import HistoricMean, SeasonalNaive
from cashflow.pipeline import Split
from cashflow.simulate import SimConfig, shifted_config, simulate, transactions_to_daily


@pytest.fixture(scope="module")
def small():
    return simulate(SimConfig(n_users=6, n_days=400, seed=1), return_transactions=True)


def test_simulator_shapes_and_balance(small):
    daily, tx = small
    assert set(daily.columns) >= {"user_id", "persona", "date", "income", "expense", "balance"}
    assert (daily.income >= 0).all() and (daily.expense >= 0).all()
    for _, g in daily.groupby("user_id"):
        # balance is the running sum of the flows
        np.testing.assert_allclose(np.diff(g.balance.values), (g.income - g.expense).values[1:], rtol=1e-9, atol=1e-6)


def test_simulator_has_irregular_income_and_month_start_rent(small):
    daily, tx = small
    # income arrives on a minority of days (irregular gaps)
    assert (daily.income > 0).mean() < 0.6
    rent = tx[tx.category == "rent"]
    assert len(rent) > 0 and (pd.to_datetime(rent.date).dt.day <= 10).all()


def test_shifted_world_is_different():
    a = simulate(SimConfig(n_users=20, n_days=365))
    b = simulate(shifted_config(n_users=20, n_days=365))
    assert not np.allclose(a.income.sum(), b.income.sum())


def test_windows_do_not_leak_future(small):
    daily, _ = small
    S = Series.from_frame(daily)
    u = S.users[0]
    W = make_windows(S, {u: [200]}, 90, 30)
    s = W.scale[0]
    # first input feature = scaled income of the 90 days BEFORE the origin
    np.testing.assert_allclose(W.X[0, :, 0] * s, S.inc[u][110:200], rtol=1e-5)
    np.testing.assert_allclose(W.y_inc[0], S.inc[u][200:230])
    assert W.b0[0] == S.bal[u][199]


def test_split_periods_are_ordered():
    sp = Split()
    assert sp.train_end + sp.H <= sp.val[0]
    assert sp.val[1] + sp.H <= sp.cal[0]
    assert sp.cal[1] + sp.H <= sp.test[0]


def test_conformal_quantile_rule():
    s = np.arange(1, 10)  # 9 scores
    # k = ceil(10 * 0.9) = 9 -> 9th smallest
    assert conformal_quantile(s, 0.1) == 9
    assert conformal_quantile(s, 0.01) == np.inf


def test_split_conformal_coverage_is_valid():
    rng = np.random.default_rng(0)
    hits = []
    for _ in range(200):
        y = rng.standard_t(3, 600)
        lo, hi, _ = split_conformal(y[:300], np.zeros(300), np.zeros(300), alpha=0.1)
        hits.append(np.mean((y[300:] >= lo) & (y[300:] <= hi)))
    assert 0.88 < np.mean(hits) < 0.92


def test_matches_mapie():
    pytest.importorskip("mapie")
    from cashflow.conformal import mapie_crosscheck
    rng = np.random.default_rng(1)
    yc, fc, ft = rng.normal(size=200), rng.normal(size=200), rng.normal(size=50)
    lo, hi, _ = split_conformal(yc, fc, ft, 0.1)
    mlo, mhi = mapie_crosscheck(yc, fc, ft, 0.1)
    np.testing.assert_allclose(lo, mlo, atol=1e-8)
    np.testing.assert_allclose(hi, mhi, atol=1e-8)


def test_shortfall_probability_monotone():
    rng = np.random.default_rng(2)
    r = ShortfallRisk(normalized=False).fit(rng.normal(size=500), np.zeros(500))
    p = r.predict_proba(np.array([-2.0, 0.0, 2.0]))
    assert p[0] > p[1] > p[2]
    assert abs(p[1] - 0.5) < 0.08


def test_min_balance_path():
    m = min_balance_path(np.array([100.0]), np.array([[-30, -80, 50]]))
    assert m[0] == -10


def test_baselines_and_metrics(small):
    daily, _ = small
    S = Series.from_frame(daily)
    W = make_windows(S, {u: [150, 200, 250] for u in S.users}, 90, 30)
    for m in (HistoricMean(), SeasonalNaive(1), SeasonalNaive(3)):
        pi, pe = m.predict(W, S)
        assert pi.shape == W.y_inc.shape and np.isfinite(pi).all()
        met = point_metrics(W, pi, pe)
        assert all(v >= 0 for v in met.values())
    # the 'last month' copy must equal the data exactly one month earlier
    pi, _ = SeasonalNaive(1).predict(W, S)
    u, t = W.user[0], W.origin[0]
    idx = S.dates[u].get_indexer(S.dates[u][t:t + 30] - pd.DateOffset(months=1))
    np.testing.assert_allclose(pi[0], S.inc[u][idx])


def test_paired_bootstrap_detects_difference():
    rng = np.random.default_rng(3)
    g = np.repeat(np.arange(40), 10)
    a = rng.normal(1.0, 1, 400)
    b = rng.normal(0.0, 1, 400)
    d, lo, hi = paired_bootstrap(a, b, g)
    assert lo > 0 and 0.7 < d < 1.3


def test_transactions_to_daily():
    tx = pd.DataFrame({"date": ["2024-01-01", "2024-01-01", "2024-01-03"], "amount": [100.0, -40.0, -10.0]})
    d = transactions_to_daily(tx, start_balance=5)
    assert list(d.income) == [100, 0, 0]
    assert list(d.expense) == [40, 0, 10]
    assert d.balance.iloc[-1] == 55
