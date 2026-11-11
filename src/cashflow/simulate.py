"""Realistic freelancer cash-flow simulator.

Why a careful simulator?  A deep model trained on a toy simulator only learns
that simulator.  This one deliberately includes the messy parts of real
freelance finances so that simple baselines and deep models face the same
hard problems:

Income side
  * four personas (retainer, gig, project, seasonal) with different rhythms
  * busy / dry regimes (Markov switching) -> long irregular gaps in income
  * invoices paid with random delays (net-15/30/45 + lognormal lateness)
  * clients that churn, occasional unpaid invoices, split payments
Expense side
  * rent paid at the start of the month (sometimes a few days late)
  * EMI, phone, internet, subscriptions on fixed days
  * utilities with an Indian summer peak (AC bills in Apr-Jun)
  * credit-card statement: discretionary card spend of the previous cycle is
    paid as one lump around the due date (month-end lumpiness that depends
    on the past)
  * quarterly advance tax (15 Jun / 15 Sep / 15 Dec / 15 Mar) for some users
  * annual insurance premium, festival (Diwali) spending, random shocks
  * behavioural feedback: people spend less when their balance is low and
    more after a good month (lifestyle effect)

``ShiftConfig`` produces a deliberately *different* world (longer payment
delays, more dry spells, more shocks, different persona mix) to test how
models behave when the data no longer matches the training simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PERSONAS = ("retainer", "gig", "project", "seasonal")


@dataclass
class SimConfig:
    n_users: int = 200
    start: str = "2023-01-01"
    n_days: int = 1095  # 3 years
    persona_mix: tuple = (0.3, 0.3, 0.25, 0.15)
    payment_delay_mult: float = 1.0
    dry_spell_mult: float = 1.0
    shock_rate_mult: float = 1.0
    unpaid_prob: float = 0.02
    seed: int = 42
    extra: dict = field(default_factory=dict)


def shifted_config(**kw) -> SimConfig:
    """A deliberately shifted world for robustness testing."""
    base = dict(
        persona_mix=(0.15, 0.25, 0.45, 0.15),
        payment_delay_mult=1.8,
        dry_spell_mult=2.0,
        shock_rate_mult=2.0,
        unpaid_prob=0.06,
        seed=999,
    )
    base.update(kw)
    return SimConfig(**base)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def _lognormal(rng, mean, sigma):
    """Lognormal with the given *arithmetic* mean."""
    mu = np.log(mean) - 0.5 * sigma**2
    return float(rng.lognormal(mu, sigma))


def _payment_delay(rng, mult):
    terms = rng.choice([7, 15, 30, 45], p=[0.2, 0.35, 0.35, 0.1])
    late = rng.lognormal(np.log(4), 0.9) if rng.random() < 0.55 else 0.0
    return int(round((terms + late) * mult))


def _season_factor(day_of_year: int, persona: str) -> float:
    # Indian wedding / festive season peak (Nov-Feb) for seasonal workers
    if persona == "seasonal":
        return 1.0 + 0.9 * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)
    return 1.0 + 0.08 * np.cos(2 * np.pi * (day_of_year - 300) / 365.25)


def _diwali_window(year: int):
    # approximate Diwali dates (good enough for a simulator)
    d = {2022: "2022-10-24", 2023: "2023-11-12", 2024: "2024-11-01",
         2025: "2025-10-21", 2026: "2026-11-08", 2027: "2027-10-29"}
    c = pd.Timestamp(d.get(year, f"{year}-11-01"))
    return c - pd.Timedelta(days=12), c + pd.Timedelta(days=2)


# ----------------------------------------------------------------------------
# one user
# ----------------------------------------------------------------------------

def _simulate_user(uid: int, persona: str, cfg: SimConfig, rng: np.random.Generator):
    dates = pd.date_range(cfg.start, periods=cfg.n_days, freq="D")
    n = len(dates)
    inc = np.zeros(n)
    exp = np.zeros(n)
    tx = []  # (day_index, amount, category)

    def add_income(t, amt, cat):
        if 0 <= t < n and amt > 0:
            inc[t] += amt
            tx.append((t, round(amt, 2), cat))

    def add_expense(t, amt, cat):
        if 0 <= t < n and amt > 0:
            exp[t] += amt
            tx.append((t, -round(amt, 2), cat))

    # ---- user level parameters -------------------------------------------
    income_level = _lognormal(rng, 65000, 0.45)  # expected monthly income (INR)
    spend_ratio = rng.uniform(0.72, 0.98)       # expenses as share of income
    monthly_budget = income_level * spend_ratio
    rent = monthly_budget * rng.uniform(0.25, 0.35)
    rent_day = int(rng.integers(1, 6))
    has_emi = rng.random() < 0.4
    emi = monthly_budget * rng.uniform(0.08, 0.15) if has_emi else 0.0
    emi_day = int(rng.choice([5, 7, 10]))
    phone = rng.uniform(300, 900)
    internet = rng.uniform(600, 1200)
    subs = rng.uniform(200, 1500)
    subs_day = int(rng.integers(1, 29))
    util_base = monthly_budget * rng.uniform(0.03, 0.06)
    card_due = int(rng.integers(18, 29))
    pays_tax = rng.random() < 0.5
    insurance_day = int(rng.integers(0, 365))
    insurance_amt = monthly_budget * rng.uniform(0.3, 0.9)
    # what is left of the budget goes to groceries + discretionary
    fixed = rent + emi + phone + internet + subs + util_base
    flexible = max(monthly_budget - fixed, monthly_budget * 0.2)
    grocery_trip = flexible * 0.45 / 6.0
    discr_mean = flexible * 0.55 / 12.0
    card_share = rng.uniform(0.3, 0.8)  # share of discretionary on credit card
    balance0 = monthly_budget * rng.uniform(0.3, 2.5)

    # ---- income process ---------------------------------------------------
    p_dry_enter = {"retainer": 0.004, "gig": 0.012, "project": 0.010, "seasonal": 0.008}[persona]
    p_dry_enter *= cfg.dry_spell_mult
    p_dry_exit = {"retainer": 0.05, "gig": 0.06, "project": 0.04, "seasonal": 0.05}[persona]
    dry = False
    pending = []  # (pay_day, amount, category)

    # retainer clients
    n_ret = int(rng.integers(1, 3)) if persona == "retainer" else 0
    ret_clients = [dict(fee=income_level * rng.uniform(0.5, 0.8) / max(n_ret, 1),
                        day=int(rng.integers(1, 8))) for _ in range(n_ret)]
    gig_rate = {"gig": 0.55, "seasonal": 0.35, "retainer": 0.07, "project": 0.05}[persona]
    gig_size = income_level * {"gig": 0.06, "seasonal": 0.09, "retainer": 0.08, "project": 0.05}[persona]
    proj_rate = {"project": 1 / 28, "retainer": 1 / 200, "gig": 0.0, "seasonal": 1 / 120}[persona]
    active_projects = []  # (finish_day, amount)

    card_cycle_spend = 0.0
    month_income = []  # rolling record for lifestyle effect
    quarter_income = 0.0
    balance = balance0
    bal_path = np.zeros(n)

    for t, d in enumerate(dates):
        doy = d.dayofyear
        # regime switching
        if dry and rng.random() < p_dry_exit:
            dry = False
        elif not dry and rng.random() < p_dry_enter:
            dry = True
        activity = (0.15 if dry else 1.0) * _season_factor(doy, persona)

        # --- retainers: invoice at month start, paid after a short delay
        if d.day == 1:
            for c in list(ret_clients):
                if rng.random() < 0.035:  # client churns
                    ret_clients.remove(c)
                    continue
                pay = t + c["day"] + int(rng.poisson(2 * cfg.payment_delay_mult))
                amt = c["fee"] * rng.uniform(0.95, 1.05)
                if rng.random() > cfg.unpaid_prob:
                    pending.append((pay, amt, "retainer"))
            if persona == "retainer" and len(ret_clients) < 2 and rng.random() < 0.08:
                ret_clients.append(dict(fee=income_level * rng.uniform(0.3, 0.6),
                                        day=int(rng.integers(1, 8))))

        # --- gigs: small jobs, paid within a few days to weeks
        k = rng.poisson(gig_rate * activity)
        for _ in range(k):
            amt = _lognormal(rng, gig_size, 0.7)
            delay = int(rng.integers(0, 4)) if rng.random() < 0.6 else _payment_delay(rng, cfg.payment_delay_mult) // 2
            if rng.random() > cfg.unpaid_prob:
                pending.append((t + delay, amt, "gig"))

        # --- projects: start, finish after weeks, invoice, paid after terms
        if rng.random() < proj_rate * activity:
            size = _lognormal(rng, income_level * 1.1, 0.6)
            active_projects.append((t + int(rng.integers(7, 45)), size))
        for p in list(active_projects):
            if p[0] == t:
                active_projects.remove(p)
                if rng.random() < cfg.unpaid_prob:
                    continue
                delay = _payment_delay(rng, cfg.payment_delay_mult)
                if rng.random() < 0.3:  # paid in two instalments
                    pending.append((t + delay, p[1] * 0.5, "project"))
                    pending.append((t + delay + int(rng.integers(10, 40)), p[1] * 0.5, "project"))
                else:
                    pending.append((t + delay, p[1], "project"))

        # no bank credits on Sundays: shift to Monday
        for item in [x for x in pending if x[0] == t]:
            pending.remove(item)
            if d.dayofweek == 6:
                pending.append((t + 1, item[1], item[2]))
            else:
                add_income(t, item[1], item[2])

        # --- expenses -------------------------------------------------------
        recent = np.mean(month_income[-3:]) if month_income else income_level
        lifestyle = np.clip(0.75 + 0.25 * recent / income_level, 0.6, 1.4)
        cushion = balance / monthly_budget
        austerity = np.clip(0.45 + 0.45 * cushion, 0.25, 1.2)

        if d.day == rent_day:
            late = int(rng.integers(1, 5)) if rng.random() < 0.12 else 0
            add_expense(t + late, rent, "rent")
        if has_emi and d.day == emi_day:
            add_expense(t, emi, "emi")
        if d.day == 3:
            add_expense(t, phone, "phone")
        if d.day == 8:
            add_expense(t, internet, "internet")
        if d.day == subs_day:
            add_expense(t, subs, "subscriptions")
        if d.day == 12:
            summer = 1.0 + 0.6 * np.exp(-((doy - 135) / 35) ** 2)
            add_expense(t + int(rng.integers(0, 6)), util_base * summer * rng.uniform(0.8, 1.2), "utilities")
        if rng.random() < 6 / 30:
            add_expense(t, _lognormal(rng, grocery_trip * austerity, 0.35), "groceries")

        weekend = d.dayofweek >= 5
        diwali = False
        lo, hi = _diwali_window(d.year)
        if lo <= d <= hi:
            diwali = True
        p_discr = (0.55 if weekend else 0.32) * (1.6 if diwali else 1.0)
        if rng.random() < p_discr:
            amt = _lognormal(rng, discr_mean * lifestyle * austerity * (1.8 if diwali else 1.0), 0.9)
            if rng.random() < card_share:
                card_cycle_spend += amt  # paid later as card bill
            else:
                add_expense(t, amt, "discretionary")
        if d.day == card_due and card_cycle_spend > 0:
            # most people pay in full, some pay minimum and roll over
            pay_frac = 1.0 if rng.random() < 0.8 else rng.uniform(0.3, 0.7)
            add_expense(t, card_cycle_spend * pay_frac, "credit_card")
            card_cycle_spend *= (1 - pay_frac) * 1.035  # interest on rollover

        if pays_tax and (d.month, d.day) in [(6, 15), (9, 15), (12, 15), (3, 15)]:
            add_expense(t, 0.12 * quarter_income, "advance_tax")
            quarter_income = 0.0
        if doy == insurance_day + 1:
            add_expense(t, insurance_amt, "insurance")
        if rng.random() < (0.7 / 365) * cfg.shock_rate_mult:
            add_expense(t, _lognormal(rng, monthly_budget * 0.5, 0.7), "shock")

        balance += inc[t] - exp[t]
        bal_path[t] = balance
        quarter_income += inc[t]
        if d.day == d.days_in_month:
            month_income.append(inc[max(0, t - d.days_in_month + 1): t + 1].sum())

    daily = pd.DataFrame({
        "user_id": uid, "persona": persona, "date": dates,
        "income": inc, "expense": exp, "balance": bal_path,
    })
    daily["balance_start"] = daily["balance"] - daily["income"] + daily["expense"]
    txdf = pd.DataFrame(tx, columns=["t", "amount", "category"])
    txdf["date"] = dates[txdf["t"].clip(upper=n - 1)]
    txdf["user_id"] = uid
    return daily, txdf.drop(columns="t"), balance0


def simulate(cfg: SimConfig | None = None, return_transactions: bool = False):
    """Simulate ``cfg.n_users`` freelancers.

    Returns a daily frame with columns
    ``user_id, persona, date, income, expense, balance, balance_start``
    (and optionally the raw transaction log).
    """
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    personas = rng.choice(PERSONAS, size=cfg.n_users, p=np.array(cfg.persona_mix) / sum(cfg.persona_mix))
    dailies, txs = [], []
    for uid in range(cfg.n_users):
        urng = np.random.default_rng(rng.integers(1 << 31))
        d, tx, _ = _simulate_user(uid, str(personas[uid]), cfg, urng)
        dailies.append(d)
        if return_transactions:
            txs.append(tx)
    daily = pd.concat(dailies, ignore_index=True)
    if return_transactions:
        return daily, pd.concat(txs, ignore_index=True).sort_values(["user_id", "date"])
    return daily


def transactions_to_daily(tx: pd.DataFrame, start_balance: float = 0.0) -> pd.DataFrame:
    """Turn a transaction list (``date``, ``amount``; +income / -expense) into
    the daily income / expense / balance frame used by the models.  Used by
    the demo for uploaded bank statements."""
    tx = tx.copy()
    tx["date"] = pd.to_datetime(tx["date"]).dt.normalize()
    idx = pd.date_range(tx["date"].min(), tx["date"].max(), freq="D")
    inc = tx.loc[tx.amount > 0].groupby("date")["amount"].sum().reindex(idx, fill_value=0.0)
    exp = (-tx.loc[tx.amount < 0].groupby("date")["amount"].sum()).reindex(idx, fill_value=0.0)
    out = pd.DataFrame({"date": idx, "income": inc.values, "expense": exp.values})
    out["balance"] = start_balance + (out.income - out.expense).cumsum()
    out["balance_start"] = out["balance"] - out["income"] + out["expense"]
    out["user_id"] = 0
    out["persona"] = "uploaded"
    return out
