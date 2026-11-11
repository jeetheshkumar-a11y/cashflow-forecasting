"""Second dataset: PaySim (aggregate hourly inflow / outflow).

    1. Download PS_20174392719_1491204439457_log.csv from
       https://www.kaggle.com/datasets/ealaxi/paysim1 into data/
    2. python scripts/run_paysim.py

Forecast the next 24 hours of platform inflow and outflow from the last 7
days, with the same models and the same conformal evaluation as the main
experiment.  PaySim has only ~743 hours, and activity becomes very sparse
after roughly hour 400, so by default only the first 400 hours are used
(``--max-step``).  With so little data the deep models are at a clear
disadvantage -- that is a real finding, not a bug: on short single series,
simple baselines are usually the right tool.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from cashflow.features import Series
from cashflow.models.baselines import ETS, HistoricMean, ProphetModel, RidgeDirect, SeasonalNaive
from cashflow.models.deep import KerasForecaster
from cashflow.paysim import DEFAULT_PATH, load_paysim_hourly
from cashflow.pipeline import Split, fit_predict, net_total, point_table
from cashflow.conformal import split_conformal
from cashflow.metrics import interval_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_PATH))
    ap.add_argument("--max-step", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--no-prophet", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "paysim"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    hourly = load_paysim_hourly(args.csv)
    hourly = hourly[hourly.step <= args.max_step].reset_index(drop=True)
    # work in millions so numbers are readable
    hourly[["income", "expense"]] /= 1e6
    S = Series.from_frame(hourly, freq="H")
    n = len(hourly)
    L, H = 168, 24
    # chronological split of forecast origins, with gaps so that the 24h targets of
    # one block never overlap the targets of the next evaluated block
    o0, o1 = L, n - H
    span = o1 - o0
    cut = lambda f: o0 + int(f * span)
    sp = Split(L=L, H=H, train_end=cut(0.42), train_step=1, val=(cut(0.47), cut(0.55), 1),
               cal=(cut(0.60), cut(0.75), 1), test=(cut(0.75) + H, o1, 1))
    Wtr, Wval, Wc, Wt = (sp.windows(S, k) for k in ("train", "val", "cal", "test"))
    print(f"hours={n} windows train={len(Wtr)} val={len(Wval)} cal={len(Wc)} test={len(Wt)}")

    models = [HistoricMean(L=L), SeasonalNaive(1, lag=24), SeasonalNaive(3, lag=24),
              ETS(history=L * 2, period=24), RidgeDirect(alpha=50.0)]
    models += [KerasForecaster(k, epochs=args.epochs, batch_size=32, patience=8) for k in ("lstm", "gru", "nbeats")]
    if not args.no_prophet:
        models.append(ProphetModel(history=L * 2))

    preds = {}
    for m in models:
        print("running", m.name, flush=True)
        P = fit_predict(m, Wtr, Wval, S, {"cal": (Wc, S), "test": (Wt, S)})
        preds[m.name] = {"cal": P["cal"], "test": P["test"]}

    tab = point_table(preds, Wt).rename(columns=lambda c: c.replace("30d", "24h").replace("daily", "hourly"))
    rows = []
    for name, P in preds.items():
        lo, hi, _ = split_conformal(net_total(Wc), net_total(Wc, P["cal"]), net_total(Wt, P["test"]), 0.1)
        rows.append({"model": name, **interval_metrics(net_total(Wt), lo, hi)})
    tab = tab.merge(pd.DataFrame(rows).rename(columns={"coverage": "90% coverage", "mean width": "90% width"}))
    tab = tab.sort_values("MAE 24h net")
    tab.to_csv(out / "paysim_metrics.csv", index=False)
    print("(amounts in millions per 24h)")
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
