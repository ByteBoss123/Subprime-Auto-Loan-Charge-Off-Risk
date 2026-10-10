"""Shared feature construction. Origination-time fields only, so nothing leaks from loan performance."""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CUT1 = pd.Timestamp("2024-05-31")   # pool cutoff, first SEC filing
CUT2 = pd.Timestamp("2025-05-31")   # out-of-time snapshot 12 months later
END2 = pd.Timestamp("2026-05-31")

NUM = ["score", "orig_apr", "orig_term", "pti", "ltv", "log_amount", "age_months"]
CAT = ["used", "income_verified_3", "employment_verified_3", "subvented_any", "score_missing"]
FEATURES = NUM + CAT


def months_between(a, b):
    return (b.dt.year - a.dt.year) * 12 + (b.dt.month - a.dt.month)


def load():
    d = pd.read_parquet(ROOT / "data/amcar-2024-1.parquet")
    for c in ["orig_month", "first_period", "last_period", "exit_period"]:
        d[c] = pd.to_datetime(d[c])
    d["exit_type"] = d.exit_type.replace({"repurchase": "censored"})  # 6 servicer repurchases: not a credit event
    d["used"] = (d.new_used == "2").astype(int)
    d["income_verified_3"] = (d.income_verified == "3").astype(int)
    d["employment_verified_3"] = (d.employment_verified == "3").astype(int)
    d["subvented_any"] = (d.subvented != "0").astype(int)
    d["score_missing"] = d.score.isna().astype(int)
    d["score"] = d.score.fillna(d.score.median())
    d["log_amount"] = np.log(d.orig_amount)
    end = d.exit_period.fillna(d.last_period)
    d["end_period"] = end
    return d


def snapshot(d, cut, horizon_end):
    """Loans active at `cut`; label = charged off in (cut, horizon_end]."""
    active = d[(d.first_period <= cut) & (d.end_period > cut)].copy()
    active["age_months"] = months_between(active.orig_month, pd.Series(cut, index=active.index))
    active["y"] = ((active.exit_type == "chargeoff") & (active.exit_period > cut)
                   & (active.exit_period <= horizon_end)).astype(int)
    return active
