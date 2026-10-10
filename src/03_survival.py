"""Cause-specific Cox proportional hazards on loan age with delayed entry (left truncation), competing risks.

Loans enter the pool already seasoned, so each loan is at risk only from its age at the pool cutoff onward.
Development data is censored at 2025-05-31 so the out-of-time window (Jun 2025 - May 2026) stays unseen.
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.duration.hazard_regression import PHReg

from features import CUT1, CUT2, END2, FEATURES, NUM, ROOT, load, months_between, snapshot

d = load()
w1 = snapshot(d, CUT1, CUT2)
train_ids, _ = train_test_split(w1.asset_id, test_size=0.3, random_state=42, stratify=w1.y)
dev = d[d.asset_id.isin(set(train_ids))].copy()

# Survival setup on the loan-age time scale, censored at CUT2
end = dev.end_period.where(dev.end_period <= CUT2, CUT2)
dev["entry"] = months_between(dev.orig_month, dev.first_period)   # at risk after the pool-cutoff report (left truncation)
dev["time"] = months_between(dev.orig_month, end)
in_window = dev.exit_period.notna() & (dev.exit_period <= CUT2)
dev["co"] = (in_window & (dev.exit_type == "chargeoff")).astype(int)
dev["pp"] = (in_window & (dev.exit_type == "prepay")).astype(int)
dev = dev[dev.time > dev.entry]
dev["age_months"] = 0  # age is the time scale in the Cox model, not a covariate
cov = [c for c in FEATURES if c != "age_months"]
num = [c for c in NUM if c != "age_months"]
sc = StandardScaler().fit(dev[num])
def Z(df):
    z = df[cov].astype(float).copy()
    z[num] = sc.transform(df[num])
    return z

fits = {}
for cause in ["co", "pp"]:
    fits[cause] = PHReg(dev.time.values, Z(dev).values, status=dev[cause].values,
                        entry=dev.entry.values, ties="efron").fit()

co = fits["co"]
hr = pd.DataFrame({"hazard_ratio_per_sd": np.exp(co.params), "p_value": co.pvalues,
                   "ci_low": np.exp(co.conf_int()[:, 0]), "ci_high": np.exp(co.conf_int()[:, 1])},
                  index=cov).round(4)

# Proportional-hazards check: correlation of scaled Schoenfeld residuals with event time
res = co.schoenfeld_residuals
ev = dev.co.values == 1
t_ev = dev.time.values[ev]
ph = {c: round(float(np.corrcoef(res[ev, j], t_ev)[0, 1]), 4) for j, c in enumerate(cov)}

# Baseline hazards by month of age -> discrete monthly hazards
def monthly_baseline(fit, max_age=130):
    t, H, _ = fit.baseline_cumulative_hazard[0]
    grid = np.arange(0, max_age + 1)
    Hg = np.interp(grid, t, H, left=0.0, right=H[-1])
    return np.diff(Hg, prepend=0.0)          # increment of baseline cumulative hazard at each age

b_co, b_pp = monthly_baseline(fits["co"]), monthly_baseline(fits["pp"])

def pd12(df):
    """12-month charge-off probability with prepayment as a competing exit (discrete-time Aalen-Johansen)."""
    a0 = months_between(df.orig_month, pd.Series(df.snap.iloc[0], index=df.index)).values
    lp_co = np.exp(Z(df).values @ fits["co"].params)
    lp_pp = np.exp(Z(df).values @ fits["pp"].params)
    S, P = np.ones(len(df)), np.zeros(len(df))
    for k in range(1, 13):
        age = np.minimum(a0 + k, len(b_co) - 1)
        h_co = 1 - np.exp(-b_co[age] * lp_co)
        h_pp = 1 - np.exp(-b_pp[age] * lp_pp)
        P += S * h_co
        S *= np.clip(1 - h_co - h_pp, 0, 1)
    return P

_, test_ids = train_test_split(w1.asset_id, test_size=0.3, random_state=42, stratify=w1.y)
test = w1[w1.asset_id.isin(set(test_ids))].copy(); test["snap"] = CUT1
oot = snapshot(d, CUT2, END2); oot["snap"] = CUT2
p_test, p_oot = pd12(test), pd12(oot)

# Charge-off hazard shape by age (seasoning curve) from the baseline, at average covariates
age_curve = pd.DataFrame({"age_months": np.arange(len(b_co)), "baseline_monthly_co_hazard": b_co}).iloc[6:73:6]

out = {
    "development_loans": int(len(dev)), "dev_chargeoffs": int(dev.co.sum()), "dev_prepays": int(dev.pp.sum()),
    "test_in_time": {"auc": round(roc_auc_score(test.y, p_test), 4),
                     "mean_pd": round(float(p_test.mean()), 4), "actual": round(float(test.y.mean()), 4)},
    "out_of_time": {"auc": round(roc_auc_score(oot.y, p_oot), 4),
                    "mean_pd": round(float(p_oot.mean()), 4), "actual": round(float(oot.y.mean()), 4)},
    "ph_check_schoenfeld_corr_with_time": ph,
    "pool_age_at_cutoff_months": {k: float(v) for k, v in
        months_between(w1.orig_month, pd.Series(CUT1, index=w1.index)).describe().round(1).items()},
}
print(json.dumps(out, indent=1)); print(hr.to_string()); print(age_curve.to_string(index=False))
(ROOT / "results/03_cox.json").write_text(json.dumps(out, indent=1))
hr.to_csv(ROOT / "results/03_cox_hazard_ratios.csv"); age_curve.to_csv(ROOT / "results/03_seasoning_curve.csv", index=False)
