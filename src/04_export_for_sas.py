"""Export the exact modeling tables used in Python so the SAS programs run on identical rows."""
import pandas as pd
from sklearn.model_selection import train_test_split

from features import CUT1, CUT2, END2, FEATURES, ROOT, load, months_between, snapshot

d = load()
w1 = snapshot(d, CUT1, CUT2)
tr, te = train_test_split(w1.asset_id, test_size=0.3, random_state=42, stratify=w1.y)
w1["split"] = w1.asset_id.isin(set(tr)).map({True: "train", False: "test"})
w2 = snapshot(d, CUT2, END2); w2["split"] = "oot"
cols = ["asset_id", "split", "y"] + FEATURES
pd.concat([w1[cols], w2[cols]]).to_csv(ROOT / "sas/pd_snapshots.csv", index=False)

dev = d[d.asset_id.isin(set(tr))].copy()
end = dev.end_period.where(dev.end_period <= CUT2, CUT2)
dev["entry"] = months_between(dev.orig_month, dev.first_period)
dev["time"] = months_between(dev.orig_month, end)
win = dev.exit_period.notna() & (dev.exit_period <= CUT2)
dev["co"] = (win & (dev.exit_type == "chargeoff")).astype(int)
dev["pp"] = (win & (dev.exit_type == "prepay")).astype(int)
dev = dev[dev.time > dev.entry]
dev[["asset_id", "entry", "time", "co", "pp"] + [c for c in FEATURES if c != "age_months"]] \
    .to_csv(ROOT / "sas/survival_dev.csv", index=False)
print(len(w1), len(w2), len(dev))
