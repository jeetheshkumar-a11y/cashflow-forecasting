"""Full experiment: simulate -> train -> backtest -> uncertainty -> figures.

    python scripts/run_experiment.py              # full run (~1-2 h on a laptop CPU)
    python scripts/run_experiment.py --quick      # ~5 min smoke run
    python scripts/run_experiment.py --with-chronos   # also Chronos (needs internet + torch)

Outputs go to results/ (tables + figures) and artifacts/ (models used by the demo).
Predictions are cached in results/cache/ so re-running only redoes what changed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from cashflow import plots
from cashflow.conformal import ShortfallRisk, balance_fan, min_balance_path, split_conformal
from cashflow.features import Series
from cashflow.metrics import paired_bootstrap, per_window_errors, reliability_table
from cashflow.models.baselines import ETS, HistoricMean, ProphetModel, RidgeDirect, SeasonalNaive
from cashflow.models.deep import KerasForecaster
from cashflow.pipeline import Split, fit_predict, net_total, point_table, uncertainty_report
from cashflow.simulate import SimConfig, shifted_config, simulate


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--prophet-every", type=int, default=4, help="run Prophet on every k-th user (it is slow)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--with-chronos", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    if args.quick:
        args.users, args.seeds, args.epochs, args.prophet_every = 30, 1, 3, 3
        args.out = str(ROOT / "results_quick")

    out = Path(args.out); figs = out / "figures"; cache = out / "cache"
    for p in (out, figs, cache, ROOT / "artifacts"):
        p.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    log("simulating", args.users, "users (base world + shifted world)")
    base = simulate(SimConfig(n_users=args.users))
    shift = simulate(shifted_config(n_users=args.users))
    S, S_shift = Series.from_frame(base), Series.from_frame(shift)
    sp = Split()
    Wtr, Wval = sp.windows(S, "train"), sp.windows(S, "val")
    Wc, Wt = sp.windows(S, "cal"), sp.windows(S, "test")
    Wc_s, Wt_s = sp.windows(S_shift, "cal"), sp.windows(S_shift, "test")
    log(f"windows: train={len(Wtr)} val={len(Wval)} cal={len(Wc)} test={len(Wt)}")

    eval_sets = {"cal": (Wc, S), "test": (Wt, S), "cal_shift": (Wc_s, S_shift), "test_shift": (Wt_s, S_shift)}

    # ------------------------------------------------------------------ models
    models = [HistoricMean(), SeasonalNaive(1), SeasonalNaive(3), ETS(), RidgeDirect()]
    for kind in ("lstm", "gru", "nbeats"):
        for seed in range(args.seeds):
            m = KerasForecaster(kind, epochs=args.epochs, seed=seed)
            m.name_seed = f"{m.name}#s{seed}"
            models.append(m)
    if args.with_chronos:
        from cashflow.models.chronos_model import ChronosModel
        models.append(ChronosModel())

    preds, timing = {}, {}
    for m in models:
        key = getattr(m, "name_seed", m.name)
        f = cache / (key.replace(" ", "_").replace("/", "_").replace("#", "_") + ".npz")
        if f.exists():
            z = np.load(f)
            preds[key] = {k: (z[k + "_i"], z[k + "_e"]) for k in eval_sets}
            timing[key] = float(z["fit_seconds"])
            log("cached", key)
            continue
        log("running", key)
        P = fit_predict(m, Wtr, Wval, S, eval_sets)
        preds[key] = {k: P[k] for k in eval_sets}
        timing[key] = P["fit_seconds"] + sum(P[k + "_seconds"] for k in eval_sets)
        np.savez_compressed(f, fit_seconds=timing[key],
                            **{f"{k}_i": P[k][0] for k in eval_sets}, **{f"{k}_e": P[k][1] for k in eval_sets})
        if hasattr(m, "kind") and getattr(m, "seed", 1) == 0:
            m.save(ROOT / "artifacts" / f"{m.kind}.keras")
        log(f"  done {key} in {timing[key]:.0f}s")

    # Prophet on a subset of users (slow: one fit per user, origin and series)
    pusers = set(S.users[:: args.prophet_every])
    sub = {k: (W.subset(np.isin(W.user, list(pusers))), SS) for k, (W, SS) in eval_sets.items()}
    f = cache / "Prophet.npz"
    if f.exists():
        z = np.load(f)
        prophet = {k: (z[k + "_i"], z[k + "_e"]) for k in ("cal", "test")}
    else:
        log(f"running Prophet on {len(pusers)} users")
        t0 = time.time()
        pm = ProphetModel()
        prophet = {k: pm.predict(sub[k][0], sub[k][1]) for k in ("cal", "test")}
        timing["Prophet"] = time.time() - t0
        np.savez_compressed(f, **{f"{k}_i": prophet[k][0] for k in prophet}, **{f"{k}_e": prophet[k][1] for k in prophet})
        log(f"  done Prophet in {timing['Prophet']:.0f}s")

    # ------------------------------------------------------------------ point accuracy
    deep_names = ["LSTM", "GRU", "N-BEATS"]
    simple = [m.name for m in models if not hasattr(m, "kind")]
    rows = []
    for name in simple:
        rows.append(point_table({name: preds[name]}, Wt).iloc[0].to_dict() | {"seed std (net)": 0.0})
    for dn in deep_names:
        seeds = [k for k in preds if k.startswith(dn + "#")]
        tabs = pd.concat([point_table({k: preds[k]}, Wt) for k in seeds])
        r = tabs.drop(columns="model").mean().to_dict()
        r["model"] = dn
        r["seed std (net)"] = tabs["MAE 30d net"].std(ddof=0) if len(seeds) > 1 else 0.0
        rows.append(r)
    point = pd.DataFrame(rows)
    ref = "Seasonal avg (last 3 months)"
    point["skill vs " + ref] = 1 - point["MAE 30d net"] / point.loc[point.model == ref, "MAE 30d net"].iloc[0]
    point["train+predict seconds"] = [timing.get(n, np.nan) if n in timing else
                                      np.mean([timing[k] for k in timing if k.startswith(n + "#")]) for n in point.model]
    point = point.sort_values("MAE 30d net")
    point.to_csv(out / "point_metrics.csv", index=False)
    log("\n" + point.round(3).to_string(index=False))

    # one representative prediction per deep model (seed 0) for the remaining analyses
    P0 = {n: preds[n] for n in simple}
    for dn in deep_names:
        P0[dn] = preds[dn + "#s0"]

    # Prophet subset comparison (all models evaluated on the same users)
    mk = np.isin(Wt.user, list(pusers))
    Wt_sub = Wt.subset(mk)
    sub_rows = [{"model": "Prophet", **point_table({"p": {"test": prophet["test"]}}, Wt_sub).iloc[0].drop("model").to_dict()}]
    for n, P in P0.items():
        sub_rows.append({"model": n, **point_table({n: {"test": (P["test"][0][mk], P["test"][1][mk])}}, Wt_sub)
                         .iloc[0].drop("model").to_dict()})
    sub_tab = pd.DataFrame(sub_rows).sort_values("MAE 30d net")
    sub_tab.to_csv(out / "point_metrics_prophet_subset.csv", index=False)

    # ------------------------------------------------------------------ bootstrap CIs
    boot = []
    e_ref = per_window_errors(Wt, *P0[ref]["test"])
    for n, P in P0.items():
        e = per_window_errors(Wt, *P["test"])
        d, lo, hi = paired_bootstrap(e, e_ref, Wt.user)
        # CI of the MAE itself (resampling users)
        _, mlo, mhi = paired_bootstrap(e, np.zeros_like(e), Wt.user)
        boot.append({"model": n, "mae": e.mean(), "lo": mlo, "hi": mhi,
                     "diff vs " + ref: d, "diff lo": lo, "diff hi": hi,
                     "significantly better": hi < 0, "significantly worse": lo > 0})
    boot = pd.DataFrame(boot).sort_values("mae")
    boot.to_csv(out / "bootstrap_vs_reference.csv", index=False)
    plots.model_comparison(boot, figs / "model_comparison.png", ref)

    # best simple vs best deep
    best_simple = point[~point.model.isin(deep_names)].iloc[0].model
    best_deep = point[point.model.isin(deep_names)].iloc[0].model
    e_s = per_window_errors(Wt, *P0[best_simple]["test"])
    e_d = per_window_errors(Wt, *P0[best_deep]["test"])
    d, lo, hi = paired_bootstrap(e_d, e_s, Wt.user)
    headline = {"best_simple": best_simple, "best_deep": best_deep,
                "mae_diff_deep_minus_simple": d, "ci95": [lo, hi]}

    # ------------------------------------------------------------------ uncertainty
    unc, per_p, sf_rows, probs = [], [], [], {}
    for i, (n, P) in enumerate(P0.items()):
        u, pp, sf, p, event = uncertainty_report(n, Wc, P["cal"], Wt, P["test"], check_mapie=(i == 0))
        unc.append(u); per_p.append(pp); sf_rows.append(sf); probs[n] = p
    unc, per_p, sf_tab = pd.concat(unc), pd.concat(per_p), pd.DataFrame(sf_rows).sort_values("Brier (conformal P)")
    unc.to_csv(out / "conformal_intervals.csv", index=False)
    per_p.to_csv(out / "conformal_by_persona.csv", index=False)
    sf_tab.to_csv(out / "shortfall_risk.csv", index=False)
    log("\n" + unc[unc.nominal == 0.9].round(3).to_string(index=False))
    log("\n" + sf_tab.round(4).to_string(index=False))
    plots.coverage_plot(unc, figs / "conformal_coverage.png")
    plots.persona_coverage(per_p, best_deep, figs / "coverage_by_persona.png")
    best_risk = sf_tab.iloc[0].model
    rel = reliability_table(probs[best_risk], event)
    rel.to_csv(out / "reliability_table.csv", index=False)
    plots.reliability(rel, figs / "shortfall_reliability.png", f"Shortfall risk calibration ({best_risk})")

    # ------------------------------------------------------------------ distribution shift
    shift_rows = []
    for n, P in P0.items():
        e_in = per_window_errors(Wt, *P["test"]).mean()
        e_sh = per_window_errors(Wt_s, *P["test_shift"]).mean()
        yc, ys = net_total(Wc), net_total(Wt_s)
        # interval calibrated on the *old* world vs recalibrated on the new world
        lo, hi, _ = split_conformal(net_total(Wc), net_total(Wc, P["cal"]), net_total(Wt_s, P["test_shift"]), 0.1,
                                    Wc.sigma, Wt_s.sigma)
        cov_old = np.mean((ys >= lo) & (ys <= hi))
        lo, hi, _ = split_conformal(net_total(Wc_s), net_total(Wc_s, P["cal_shift"]), net_total(Wt_s, P["test_shift"]),
                                    0.1, Wc_s.sigma, Wt_s.sigma)
        cov_new = np.mean((ys >= lo) & (ys <= hi))
        # the shifted world has different money levels, so compare errors relative to
        # the average 30-day spending in each world (normalized MAE)
        n_in, n_sh = e_in / Wt.y_exp.sum(1).mean(), e_sh / Wt_s.y_exp.sum(1).mean()
        shift_rows.append({"model": n, "MAE in-distribution": e_in, "MAE shifted": e_sh,
                           "nMAE in-distribution": n_in, "nMAE shifted": n_sh,
                           "degradation % (nMAE)": 100 * (n_sh / n_in - 1),
                           "90% coverage, calibrated on old world": cov_old,
                           "90% coverage, re-calibrated on new world": cov_new})
    shift_tab = pd.DataFrame(shift_rows).sort_values("nMAE shifted")
    shift_tab.to_csv(out / "distribution_shift.csv", index=False)
    plots.shift_plot(shift_tab, figs / "distribution_shift.png")
    log("\n" + shift_tab.round(3).to_string(index=False))

    # ------------------------------------------------------------------ example figures
    ex_users = [u for u in S.users if S.persona[u] == "project"][:1] + [u for u in S.users if S.persona[u] == "gig"][:1]
    plots.user_history(S, ex_users, figs / "example_users.png")
    for n in [best_deep, best_simple]:
        P = P0[n]
        idx = int(np.where((Wt.user == ex_users[0]))[0][len(np.where(Wt.user == ex_users[0])[0]) // 2])
        pi, pe = P["test"][0][idx], P["test"][1][idx]
        point_path, bands = balance_fan(Wt.b0[idx], pi - pe, Wc.y_inc - Wc.y_exp, P["cal"][0] - P["cal"][1],
                                        Wc.sigma, Wt.sigma[idx], alphas=(0.1, 0.5))
        u, t = Wt.user[idx], Wt.origin[idx]
        risk = ShortfallRisk().fit(min_balance_path(Wc.b0, Wc.y_inc - Wc.y_exp),
                                   min_balance_path(Wc.b0, P["cal"][0] - P["cal"][1]), Wc.sigma)
        ps = float(risk.predict_proba(min_balance_path(Wt.b0[idx:idx + 1], (pi - pe)[None]), Wt.sigma[idx:idx + 1])[0])
        plots.fan_chart(S.dates[u][t - 90:t], S.bal[u][t - 90:t], S.dates[u][t:t + 30], point_path, bands,
                        Wt.b0[idx] + np.cumsum(Wt.y_inc[idx] - Wt.y_exp[idx]),
                        figs / f"fan_{n.replace(' ', '_').replace('(', '').replace(')', '')}.png",
                        f"User {u} ({S.persona[u]}), {n}", ps)

    # ------------------------------------------------------------------ demo artifacts
    demo = {}
    for n in ["GRU", "N-BEATS", "LSTM", ref, "ETS"]:
        P = P0[n]
        demo[n] = dict(cal_true_net=(Wc.y_inc - Wc.y_exp).astype("float32"),
                       cal_pred_net=(P["cal"][0] - P["cal"][1]).astype("float32"),
                       cal_sigma=Wc.sigma.astype("float32"), cal_b0=Wc.b0.astype("float32"))
    np.savez_compressed(ROOT / "artifacts" / "calibration.npz",
                        **{f"{n}|{k}": v for n, d in demo.items() for k, v in d.items()})
    summary = {"n_users": args.users, "seeds": args.seeds, "windows": {"train": len(Wtr), "val": len(Wval),
               "cal": len(Wc), "test": len(Wt)}, "headline": headline, "reference": ref,
               "prophet_users": len(pusers), "best_shortfall_model": best_risk}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log("summary", json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
