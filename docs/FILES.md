# What each file contains

```
cashflow-forecasting/
├── README.md                     project overview, headline results, how to run
├── requirements.txt              Python packages (pip install -r requirements.txt)
├── LICENSE                       MIT licence
├── .gitignore                    keeps big/regenerable files (PaySim CSV, caches) out of git
│
├── src/cashflow/                 the Python package with all the logic
│   ├── simulate.py               realistic freelancer simulator + shifted world + CSV→daily converter
│   ├── features.py               turns daily series into 90-day-in / 30-day-out training windows
│   ├── pipeline.py               time-ordered train/val/calibration/test split, fit/predict loop,
│   │                             uncertainty report per model
│   ├── metrics.py                MAE metrics, user-level paired bootstrap, Brier, AUC, reliability
│   ├── conformal.py              split conformal intervals (standard + normalized), shortfall
│   │                             probability, balance fan bands, MAPIE cross-check
│   ├── forecast_api.py           one-call forecast for a single person (used by the demo)
│   ├── paysim.py                 loads the Kaggle PaySim CSV into hourly inflow/outflow
│   ├── plots.py                  all figures in results/figures/
│   └── models/
│       ├── baselines.py          Mean, Seasonal naive, Seasonal average, ETS, Prophet, Ridge
│       ├── deep.py               LSTM, GRU and N-BEATS in TensorFlow/Keras
│       └── chronos_model.py      optional zero-shot Chronos-Bolt foundation model
│
├── scripts/
│   ├── run_experiment.py         MAIN SCRIPT: simulate → train all models → backtest →
│   │                             conformal intervals → shortfall risk → shift test → figures
│   ├── run_paysim.py             same comparison on the PaySim dataset (needs the Kaggle CSV)
│   └── make_sample_statement.py  writes data/sample_statement.csv for the demo
│
├── app/
│   └── streamlit_app.py          interactive demo: pick a simulated freelancer or upload a CSV
│                                 statement → 30-day forecast, balance fan chart, P(run short)
│
├── artifacts/                    trained models used by the demo
│   ├── lstm.keras, gru.keras, nbeats.keras   trained networks (seed 0)
│   └── calibration.npz           calibration-set errors needed for the conformal bands
│
├── data/
│   └── sample_statement.csv      example bank statement (date, amount, category) to try in the demo
│
├── results/                      everything the README reports (produced by run_experiment.py)
│   ├── point_metrics.csv         accuracy of every model (MAE of 30-day income/expense/net, …)
│   ├── point_metrics_prophet_subset.csv   all models vs Prophet on the same 50 users
│   ├── bootstrap_vs_reference.csv          95% CIs and significance vs the reference model
│   ├── conformal_intervals.csv   coverage and width of 80%/90% intervals (+ MAPIE check)
│   ├── conformal_by_persona.csv  the same, split by freelancer type
│   ├── shortfall_risk.csv        Brier score / AUC of P(balance < 0)
│   ├── reliability_table.csv     data behind the reliability diagram
│   ├── distribution_shift.csv    accuracy and coverage in the shifted world
│   ├── summary.json              headline numbers
│   ├── training_log.txt          console log of the full training run (timings, raw tables)
│   ├── run_log.txt               console log of the final analysis pass that wrote these files
│   └── figures/                  PNG charts shown in the README
│
├── docs/
│   ├── METHODOLOGY.md            how and why: data, splits, models, metrics, conformal, limits
│   ├── FILES.md                  this file
│   └── GITHUB_SETUP.md           create account → repo → upload → add collaborator
│
└── tests/
    ├── test_core.py              unit tests: simulator, no-leakage windows, conformal coverage,
    │                             MAPIE equality, baselines, bootstrap, CSV conversion
    └── test_app.py               smoke test that the Streamlit demo runs end to end
```
