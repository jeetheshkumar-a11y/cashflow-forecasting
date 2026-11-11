"""Cash-flow forecast demo.

    streamlit run app/streamlit_app.py

Pick a simulated freelancer or upload your own bank statement (CSV with
``date,amount`` -- positive = money in, negative = money out).  The app
forecasts the next 30 days of income and expenses, draws the balance with
conformal uncertainty bands and answers the real question:
"What's the chance I run short this month?"

Needs the trained models in artifacts/ (created by scripts/run_experiment.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from cashflow.forecast_api import available_models, forecast
from cashflow.simulate import SimConfig, simulate, transactions_to_daily

st.set_page_config(page_title="Freelancer cash-flow forecast", layout="wide")
st.title("Freelancer cash-flow forecast (next 30 days)")
st.caption("Deep learning vs simple baselines, with conformal prediction intervals. "
           "Amounts are in ₹ for the simulated users.")

models = available_models()
if not models:
    st.error("No trained models found in artifacts/. Run `python scripts/run_experiment.py --quick` first.")
    st.stop()


@st.cache_data
def demo_users(n=40):
    df, tx = simulate(SimConfig(n_users=n, seed=2024), return_transactions=True)
    return df, tx


def inr(x):
    return f"₹{x:,.0f}"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Simulated freelancer", "Upload my statement (CSV)"])
    model = st.selectbox("Model", models, index=0)
    level = st.select_slider("Interval level", options=[0.5, 0.8, 0.9], value=0.8)
    compare = st.checkbox("Compare all models", value=False)

if source == "Simulated freelancer":
    df, _ = demo_users()
    personas = sorted(df.persona.unique())
    with st.sidebar:
        persona = st.selectbox("Persona", personas, index=personas.index("project") if "project" in personas else 0)
        ids = sorted(df.loc[df.persona == persona, "user_id"].unique())
        uid = st.selectbox("User", ids)
        g = df[df.user_id == uid].reset_index(drop=True)
        cut = st.slider("Forecast as of day", 120, len(g) - 30, len(g) - 30,
                        help="Move back in time to compare the forecast with what really happened.")
    hist, actual = g.iloc[:cut], g.iloc[cut:cut + 30]
else:
    with st.sidebar:
        f = st.file_uploader("CSV with columns date, amount", type="csv")
        b0 = st.number_input("Balance before the first transaction", value=0.0, step=1000.0)
        st.download_button("Download a sample statement", (ROOT / "data" / "sample_statement.csv").read_bytes()
                           if (ROOT / "data" / "sample_statement.csv").exists() else b"date,amount\n",
                           file_name="sample_statement.csv")
    if f is None:
        st.info("Upload a CSV (at least ~4 months of transactions) or switch to a simulated freelancer.")
        st.stop()
    tx = pd.read_csv(f)
    tx.columns = [c.strip().lower() for c in tx.columns]
    if not {"date", "amount"} <= set(tx.columns):
        st.error("The CSV needs columns `date` and `amount`.")
        st.stop()
    hist, actual = transactions_to_daily(tx, start_balance=b0), None

try:
    res = forecast(hist, model=model, levels=(level, 0.5))
except ValueError as e:
    st.error(str(e))
    st.stop()

# ------------------------------------------------------------------ headline
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected income (30d)", inr(res["income"].sum()))
c2.metric("Expected expenses (30d)", inr(res["expense"].sum()))
lo, hi = res["net_interval"]
c3.metric("Net cash flow (30d)", inr(res["net_total"]), help=f"{int(level*100)}% interval: {inr(lo)} to {inr(hi)}")
c4.metric("Chance balance goes below ₹0", f"{100 * res['p_short']:.0f}%")
st.write(f"**{int(level*100)}% prediction interval for the 30-day net flow:** {inr(lo)} to {inr(hi)}.  "
         f"Lowest balance expected in the next 30 days: {inr(res['min_balance'])} "
         f"(80% range {inr(res['min_balance_q'][0])} to {inr(res['min_balance_q'][2])}).")
if res["p_short"] >= 0.3:
    st.warning("High risk of running short. Consider chasing unpaid invoices or delaying discretionary spending.")

# ------------------------------------------------------------------ fan chart
fig, ax = plt.subplots(figsize=(11, 4.2))
h = hist.tail(120)
ax.plot(h["date"], h["balance"], color="#333", lw=1.4, label="balance (history)")
a_out, a_in = round(1 - level, 3), 0.5
for a, alpha_fill in ((a_out, 0.18), (a_in, 0.35)):
    b_lo, b_hi = res["bands"][a]
    ax.fill_between(res["dates"], b_lo, b_hi, color="#2a6fdb", alpha=alpha_fill, lw=0,
                    label=f"{int(round((1 - a) * 100))}% band")
ax.plot(res["dates"], res["balance_point"], color="#2a6fdb", lw=2, label=f"forecast ({model})")
if actual is not None and len(actual) == 30:
    ax.plot(actual["date"], actual["balance"], color="#d9480f", lw=1.6, ls="--", label="what really happened")
ax.axhline(0, color="#c92a2a", lw=1, alpha=0.7)
ax.yaxis.set_major_formatter(lambda x, _: f"₹{x/1000:,.0f}k")
ax.grid(alpha=0.25)
ax.legend(loc="upper left", fontsize=8, ncols=3)
ax.set_title("Account balance: last 120 days and next 30 days")
st.pyplot(fig)

c5, c6 = st.columns(2)
with c5:
    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.bar(res["dates"], res["income"], color="#2b8a3e", label="income")
    ax2.bar(res["dates"], -res["expense"], color="#c92a2a", label="expense")
    ax2.set_title("Expected daily flows")
    ax2.legend(fontsize=8)
    ax2.tick_params(axis="x", rotation=45, labelsize=7)
    st.pyplot(fig2)
with c6:
    st.markdown("**How to read this**")
    st.markdown(
        "* Daily freelance income is lumpy; the model predicts the *expected* amount per day, so "
        "single days look smoother than reality. The 30-day totals and the bands are what matter.\n"
        "* Bands come from **split conformal prediction**: the model's errors on a held-out calibration "
        "period are used to size the interval, scaled by this person's own recent volatility.\n"
        "* The shortfall probability is the conformal predictive distribution of the lowest balance "
        "over the next 30 days, evaluated at ₹0.")

if compare:
    rows = []
    for m in models:
        r = forecast(hist, model=m, levels=(level, 0.5))
        row = {"model": m, "income 30d": r["income"].sum(), "expense 30d": r["expense"].sum(),
               "net 30d": r["net_total"], "interval low": r["net_interval"][0],
               "interval high": r["net_interval"][1], "P(short)": r["p_short"]}
        if actual is not None and len(actual) == 30:
            row["actual net 30d"] = float((actual.income - actual.expense).sum())
        rows.append(row)
    t = pd.DataFrame(rows)
    fmt = {c: "₹{:,.0f}" for c in t.columns if c not in ("model", "P(short)")}
    fmt["P(short)"] = "{:.0%}"
    st.subheader("All models on this person")
    st.dataframe(t.style.format(fmt), hide_index=True, use_container_width=True)
