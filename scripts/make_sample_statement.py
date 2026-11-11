"""Write data/sample_statement.csv: one simulated freelancer's transactions in
the same format the demo accepts for uploads (date, amount, category).

    python scripts/make_sample_statement.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cashflow.simulate import SimConfig, simulate  # noqa: E402

daily, tx = simulate(SimConfig(n_users=12, seed=7, n_days=365), return_transactions=True)
uid = int(daily.loc[daily.persona == "project", "user_id"].iloc[0])
out = tx[tx.user_id == uid][["date", "amount", "category"]].copy()
out["date"] = out["date"].dt.strftime("%Y-%m-%d")
path = ROOT / "data" / "sample_statement.csv"
path.parent.mkdir(exist_ok=True)
out.to_csv(path, index=False)
start = daily[daily.user_id == uid].iloc[0]
print(f"wrote {len(out)} transactions to {path} (starting balance {start.balance_start:,.0f})")
