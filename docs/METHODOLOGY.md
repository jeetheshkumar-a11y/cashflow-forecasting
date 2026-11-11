# Methodology

This document explains *how* the experiment is set up and *why*, so the results in the README
can be checked and reproduced.

## 1. The forecasting task

At a forecast **origin** (a day *t*) we know the person's history up to the end of day *t−1*
and forecast **daily income and daily expenses for days t … t+29** (horizon H = 30), using the
last **90 days** as input (lookback L = 90).

What we evaluate, because it is what a freelancer cares about:

| Quantity | Why |
|---|---|
| 30-day total income, expense and **net** flow | "How much will I earn / spend this month?" Main metric: **MAE of 30-day net flow**. |
| Balance path (cumulative net flow, day by day) | "When does the money arrive relative to rent?" Timing matters, not just totals. |
| Daily MAE | Reported, but on lumpy freelance income it is nearly meaningless (most days are 0). |
| Interval coverage and width | "How sure is the forecast?" |
| P(balance < 0 in the next 30 days) | "What's the chance I run short this month?" |

## 2. Data

### 2.1 Simulator (`src/cashflow/simulate.py`)

The faculty warning, *"with simulated data your model may only learn your simulator"*, shaped the
design. The simulator deliberately contains the messy features of real freelance finances so
that every model faces the same hard problems:

* **Four personas** with different income rhythms: `retainer` (monthly fees from 1–2 clients,
  who can churn), `gig` (many small jobs), `project` (few big projects, invoiced at the end),
  `seasonal` (wedding / festive-season peak, Nov–Feb).
* **Irregular gaps**: busy / dry regimes switch at random (Markov switching), giving
  multi-week droughts with almost no income.
* **Payment delays**: invoices on net-7/15/30/45 terms plus log-normal lateness; 30% of projects
  are paid in two instalments; 2% of invoices are never paid. No bank credits on Sundays.
* **Month-end / month-start bills**: rent on days 1–5 (sometimes late), EMI, phone, internet,
  subscriptions on fixed days, utilities with a summer (AC) peak.
* **Credit-card lumpiness**: card spending is paid as one lump on the due date, sometimes
  only partly, with interest on the rest. Today's spending becomes next month's expense.
* **Rare big items**: quarterly advance tax (15 Jun/Sep/Dec/Mar), an annual insurance premium,
  Diwali spending, random shocks (medical, repairs).
* **Behavioural feedback**: people spend less when their balance is low and more after a good
  quarter, so expenses depend on income history.

200 users × 3 years (2023–2025), amounts in ₹. Average monthly income is ≈ ₹65k with large
spread between users.

### 2.2 A *shifted* world (robustness test)

`shifted_config()` generates a second population that the models never see during training:
payment delays ×1.8, dry spells ×2, shocks ×2, unpaid invoices 6%, and more project workers.
Every model is evaluated there with *no retraining*. This directly tests "did the model learn
freelancing, or just my simulator?"

### 2.3 PaySim (`src/cashflow/paysim.py`, `scripts/run_paysim.py`)

PaySim is a synthetic mobile-money log produced by an agent-based simulator **calibrated on real
transaction logs**, so it is a dataset our simulator did not create. Limitations, stated honestly:
it covers only ~743 hourly steps (about 31 days), and almost every customer appears only once or
twice, so per-person 30-day forecasting is impossible with it. We therefore use it at platform
level: forecast the next **24 hours** of total inflow (`CASH_IN`) and outflow
(`CASH_OUT, PAYMENT, TRANSFER, DEBIT`) from the previous 7 days, with the same models and the same
conformal evaluation. The CSV (470 MB) must be downloaded from Kaggle by the user; it could not be
downloaded in the environment where the committed results were produced.

## 3. Time-respecting splits (`src/cashflow/pipeline.py`, class `Split`)

All users share the same calendar. Forecast origins (day index from 2023-01-01):

| Set | Origins | Step | Purpose |
|---|---|---|---|
| train | 90 … 610 | every 3 days | fit deep / linear models |
| validation | 640 … 700 | 3 days | early stopping, LR schedule |
| calibration | 730 … 900 | 7 days | conformal calibration |
| test | 930 … end | 7 days | all reported numbers |

The 30-day targets of one set end before the next set starts, so no target is ever used as
an input or calibration point for a later set. Local models (ETS, Prophet, naive) are refitted
at every origin using only data before that origin. A unit test checks that windows never leak
future values.

## 4. Inputs and scaling (`src/cashflow/features.py`)

Each window has 9 features per day: income and expense divided by the window's own average daily
flow (so a ₹30k/month and a ₹3 lakh/month freelancer look alike to the network), a "cushion"
feature `tanh(balance / one month of flow)`, and 6 calendar features (day-of-month and
day-of-week as sin/cos, "first 5 days of month", "last 4 days of month"). Targets are scaled by
the same factor and un-scaled after prediction.

## 5. Models

| Model | Type | Notes |
|---|---|---|
| Mean (90d) | naive | every future day = average day of the last 90 days |
| Seasonal naive (last month) | naive | same calendar day last month |
| Seasonal avg (last 3 months) | naive | average of the same calendar day over the last 3 months; **reference model** for skill scores |
| ETS | statistical | Holt-Winters ETS(A,N,A), weekly seasonality, refit per user and origin (statsmodels) |
| Prophet | statistical | weekly + monthly seasonality + month-start / month-end regressors. Slow, so run on every 4th user; compared on that subset |
| Ridge (linear) | global ML | linear map from the flattened 90×9 window to the 60 outputs; the "simple ML" bar to clear |
| LSTM | deep | 2 stacked layers (64→32) + dense head, direct 60-value output |
| GRU | deep | same as LSTM with GRU cells |
| N-BEATS | deep, "newer" | generic N-BEATS (Oreshkin et al., ICLR 2020): 6 fully-connected blocks with backcast/forecast residual links, written in Keras |
| Chronos-Bolt | foundation model, zero-shot | optional wrapper (`--with-chronos`); needs Hugging Face access, which was blocked here, so no committed numbers |

Deep models are **global**: one network trained on windows from all 200 users. They use MSE
loss, Adam, early stopping on the validation set, and are trained with **3 random seeds**. The
tables report the mean over seeds plus the seed standard deviation, so a lucky seed cannot
decide the comparison. Extreme spikes (>60× the usual daily flow) are clipped in the *training
targets only*.

Why N-BEATS in Keras rather than Darts: the project brief asks for TensorFlow/Keras; Darts'
N-BEATS is PyTorch. The architecture is the same. Darts would be a natural way to add more models
(TFT, TiDE, NHiTS) later.

## 6. Comparing models honestly (`src/cashflow/metrics.py`)

Test windows from the same person are correlated, so treating 4,000 windows as independent would
give far-too-narrow error bars. The **paired bootstrap resamples users** (2,000 resamples), giving
a 95% CI for each model's MAE and for the difference to the reference. A model is called
"better" only if the whole CI of the difference is below zero.

## 7. Uncertainty (`src/cashflow/conformal.py`)

### 7.1 Split conformal prediction intervals

For any point model: compute nonconformity scores on the calibration windows, take the
⌈(n+1)(1−α)⌉-th smallest score q̂, and output `prediction ± q̂` for test windows. Two scores:

* **standard**: `|y − ŷ|`. Same width for everyone.
* **normalized**: `|y − ŷ| / σ`, where σ is the person's own recent volatility (std of rolling
  30-day net sums over the last 180 days). Steady earners get narrow intervals, volatile ones
  get wide ones. This matters for **conditional** coverage (per persona), not just the average.

The implementation is cross-checked against **MAPIE** (`SplitConformalRegressor`, prefit mode).
The maximum difference is stored in `results/conformal_intervals.csv` (column
`mapie_max_abs_diff`) and is tested in `tests/test_core.py`.

### 7.2 Shortfall probability

Define m = lowest balance over the next 30 days = `b0 + min(cumsum(net_daily))`. On the
calibration set we collect normalized residuals `r = (m − m̂)/σ`. For a new window, the predictive
distribution is `{m̂ + σ·rᵢ}`, and `P(short) = P(m < 0)` is read off its CDF. This is a conformal
predictive distribution, so it is distribution-free and works on top of any model.

It is evaluated with the **Brier score** (lower is better), **AUC**, and a **reliability
diagram**. As a baseline we also score the naive "point forecast says the balance dips below 0 →
P = 1, else 0" rule.

### 7.3 Caveat

Conformal guarantees assume calibration and test errors are exchangeable. Time series are not
exactly exchangeable, so the guarantee is approximate. That is why we *measure* the empirical
coverage on later test months and on the shifted world instead of assuming it.

## 8. Limitations

* All per-person results are on simulated data. The shifted-world test and PaySim reduce, but do
  not remove, the "you only learned your simulator" risk. The real test is a user's own bank
  statement, which the demo accepts.
* PaySim and Chronos code paths were written and unit-checked for interfaces but could not be
  executed here (no access to Kaggle or Hugging Face). Run them locally to add those rows.
* Prophet is evaluated on a 50-user subset because it is slow (one fit per user, origin and series).
* Hyper-parameters of the deep models were set once and not tuned extensively. A tuned model
  might do somewhat better, but the same is true for the baselines.
