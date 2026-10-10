"""Independently re-parse raw SEC ABS-EE Exhibit 102 records and check them against the loan panel.

The panel (data/amcar-2024-1.parquet) was built by a third-party MIT-licensed repo from 27 monthly
AmeriCredit Automobile Receivables Trust 2024-1 filings. Before modeling on it, this script re-reads
300 verbatim records from the July 2026 filing with its own parser and confirms the origination fields match.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NS = {"a": "http://www.sec.gov/edgar/document/absee/autoloan/assetdata"}

panel = pd.read_parquet(ROOT / "data/amcar-2024-1.parquet")
tree = ET.parse(ROOT / "data/raw/amcar-2024-1_2026-07_first300.xml")
rows = []
for a in tree.getroot().findall("a:assets", NS):
    g = lambda t: (a.find(f"a:{t}", NS).text if a.find(f"a:{t}", NS) is not None else None)
    rows.append({
        "asset_id": g("assetNumber"),
        "orig_amount": float(g("originalLoanAmount")),
        "orig_apr": float(g("originalInterestRatePercentage")),
        "orig_term": int(g("originalLoanTerm")),
        "score_raw": g("obligorCreditScore"),
        "state": g("obligorGeographicLocation"),
        "originator": g("originatorName"),
    })
raw = pd.DataFrame(rows)
m = raw.merge(panel, on="asset_id", how="left", suffixes=("_raw", ""))
score_raw = pd.to_numeric(m.score_raw, errors="coerce")

checks = {
    "raw_records": len(raw),
    "matched_in_panel": int(m.deal.notna().sum()),
    "orig_amount_match": int((abs(m.orig_amount_raw - m.orig_amount) < 0.01).sum()),
    "orig_apr_match": int((abs(m.orig_apr_raw - m.orig_apr) < 1e-9).sum()),
    "orig_term_match": int((m.orig_term_raw == m.orig_term).sum()),
    "score_match": int(((score_raw == m.score) | (score_raw.isna() & m.score.isna())).sum()),
    "state_match": int((m.state_raw == m.state).sum()),
    "originators": sorted(raw.originator.unique().tolist()),
}

# Panel-level integrity checks
p = panel
checks.update({
    "panel_loans": len(p),
    "duplicate_asset_ids": int(p.asset_id.duplicated().sum()),
    "exit_type_counts": p.exit_type.value_counts().to_dict(),
    "chargeoff_without_exit_period": int(((p.exit_type == "chargeoff") & p.exit_period.isna()).sum()),
    "apr_out_of_range": int(((p.orig_apr <= 0) | (p.orig_apr > 1)).sum()),
    "pti_out_of_range": int(((p.pti <= 0) | (p.pti > 1)).sum()),
    "score_missing": int(p.score.isna().sum()),
    "orig_month_range": [str(p.orig_month.min()), str(p.orig_month.max())],
})
print(json.dumps(checks, indent=1, default=str))
(ROOT / "results/01_validation.json").write_text(json.dumps(checks, indent=1, default=str))
