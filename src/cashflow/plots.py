"""Matplotlib figures for the README / report."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

# reference categorical palette, fixed order (see docs)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GRAY = "#9a9893"
INK, INK2 = "#0b0b0b", "#52514e"
DEEP = {"LSTM", "GRU", "N-BEATS"}

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": "#d6d4ce", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False,
})


def _inr(x, _=None):
    a = abs(x)
    if a >= 1e5:
        return f"₹{x/1e5:.1f}L"
    if a >= 1e3:
        return f"₹{x/1e3:.0f}k"
    return f"₹{x:.0f}"


def _inr_axis(ax, which="y"):
    """Axis formatter with ONE unit for all ticks (lakh if the axis reaches 2 lakh, else thousands)."""
    lo, hi = ax.get_ylim() if which == "y" else ax.get_xlim()
    big = max(abs(lo), abs(hi)) >= 2e5

    def f(x, _=None):
        return f"₹{x/1e5:.1f}L" if big else f"₹{x/1e3:,.0f}k"
    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(f)


def user_history(S, users, path):
    """Two panels per user (flows, balance) -- no dual axes."""
    fig, axes = plt.subplots(2 * len(users), 1, figsize=(10, 2.6 * len(users)),
                             gridspec_kw={"height_ratios": [1.3, 1] * len(users)})
    for i, u in enumerate(users):
        a1, a2 = axes[2 * i], axes[2 * i + 1]
        d = S.dates[u][-365:]
        a1.bar(d, S.inc[u][-365:], color=C[2], width=1.0, label="income")
        a1.bar(d, -S.exp[u][-365:], color=C[1], width=1.0, label="expense (negative)")
        _inr_axis(a1)
        a1.set_title(f"User {u} - persona: {S.persona[u]} (last 12 months)", loc="left")
        a1.legend(loc="upper left", ncol=2, fontsize=8)
        a1.set_xticklabels([])
        a2.plot(d, S.bal[u][-365:], color=INK, lw=1.5, label="balance")
        a2.axhline(0, color=C[7], lw=0.8, ls="--")
        _inr_axis(a2)
        a2.set_ylabel("balance")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def model_comparison(boot_df, path, ref_name):
    """Horizontal bars: MAE of 30-day net flow with 95% bootstrap CI."""
    df = boot_df.sort_values("mae")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(df) + 1.2))
    y = np.arange(len(df))
    colors = [C[0] if m in DEEP else GRAY for m in df.model]
    ax.barh(y, df.mae, color=colors, height=0.6)
    ax.errorbar(df.mae, y, xerr=[df.mae - df.lo, df.hi - df.mae], fmt="none", ecolor=INK, lw=1, capsize=3)
    ax.set_yticks(y, df.model)
    ax.invert_yaxis()
    _inr_axis(ax, "x")
    ax.set_xlabel("Mean absolute error of 30-day net cash flow (lower is better)")
    for yi, v, h in zip(y, df.mae, df.hi):
        ax.text(h, yi, "  " + _inr(v), va="center", ha="left", color=INK2, fontsize=9)
    ax.set_xlim(0, df.hi.max() * 1.15)
    ax.set_title("Point accuracy on held-out months (blue = deep learning)", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def coverage_plot(unc_df, path):
    df = unc_df[unc_df.nominal == 0.9]
    models = list(dict.fromkeys(df.model))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.42 * len(models) + 1.6), sharey=True)
    y = np.arange(len(models))
    for j, (meth, col) in enumerate([("standard", C[0]), ("normalized", C[1])]):
        d = df[df.method == meth].set_index("model").loc[models]
        axes[0].scatter(d.coverage, y + (j - 0.5) * 0.25, color=col, s=40, label=meth, zorder=3)
        axes[1].barh(y + (j - 0.5) * 0.3, d["mean width"], height=0.28, color=col, label=meth)
    axes[0].axvline(0.9, color=INK, ls="--", lw=1)
    axes[0].set_xlim(0.8, 1.0)
    axes[0].set_xlabel("empirical coverage of 90% interval (target 0.90)")
    axes[0].set_yticks(y, models)
    axes[0].invert_yaxis()
    _inr_axis(axes[1], "x")
    axes[1].set_xlabel("mean interval width (narrower is better, if coverage holds)")
    axes[0].legend(loc="lower left")
    axes[0].set_title("Conformal 90% intervals for 30-day net cash flow", loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def persona_coverage(pp_df, model, path):
    df = pp_df[(pp_df.model == model) & (pp_df.nominal == 0.9)]
    personas = sorted(df.persona.unique())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(personas))
    for j, (meth, col) in enumerate([("standard", C[0]), ("normalized", C[1])]):
        d = df[df.method == meth].set_index("persona").loc[personas]
        ax.bar(x + (j - 0.5) * 0.36, d.coverage, width=0.34, color=col, label=meth)
    ax.axhline(0.9, color=INK, ls="--", lw=1)
    ax.set_ylim(0.6, 1.0)
    ax.set_xticks(x, personas)
    ax.set_ylabel("coverage of 90% interval")
    ax.set_title(f"Coverage by freelancer type ({model})", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def reliability(rel, path, title):
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot([0, 1], [0, 1], color=GRAY, ls="--", lw=1, label="perfect calibration")
    ax.plot(rel.mean_pred, rel.frac_pos, color=C[0], lw=2, marker="o", ms=6, label="model")
    for _, r in rel.iterrows():
        ax.annotate(f"n={int(r.n)}", (r.mean_pred, r.frac_pos), textcoords="offset points", xytext=(4, -10),
                    fontsize=7, color=INK2)
    ax.set_xlabel("predicted P(balance goes below 0 in next 30 days)")
    ax.set_ylabel("observed frequency")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title, loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def fan_chart(dates_hist, bal_hist, dates_fut, point, bands, actual, path, title, p_short=None):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(dates_hist, bal_hist, color=INK, lw=1.5, label="balance (history)")
    for a, alpha in sorted(bands.items(), key=lambda kv: -kv[0]):
        lo, hi = alpha
        ax.fill_between(dates_fut, lo, hi, color=C[0], alpha=0.18 if a < 0.3 else 0.30,
                        lw=0, label=f"{int((1-a)*100)}% interval")
    ax.plot(dates_fut, point, color=C[0], lw=2, label="forecast")
    if actual is not None:
        ax.plot(dates_fut, actual, color=C[1], lw=1.5, ls="--", label="what actually happened")
    ax.axhline(0, color=C[7], lw=1)
    _inr_axis(ax)
    t = title + (f"   |   P(run short) = {p_short:.0%}" if p_short is not None else "")
    ax.set_title(t, loc="left")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def shift_plot(shift_df, path):
    df = shift_df.copy()
    models = list(df.model)
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(models) + 1.8))
    y = np.arange(len(models))
    ax.barh(y - 0.18, df["nMAE in-distribution"], height=0.34, color=C[0], label="same simulator as training")
    ax.barh(y + 0.18, df["nMAE shifted"], height=0.34, color=C[1], label="shifted world")
    ax.set_yticks(y, models)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("MAE of 30-day net flow, as % of average monthly spending")
    ax.set_title("Robustness: what happens when the world changes", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
