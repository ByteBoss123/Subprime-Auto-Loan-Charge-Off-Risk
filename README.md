# Subprime Auto Loan Charge-Off Risk from Public SEC Filings

Predicts 12-month charge-off risk for 72,665 real subprime auto loans. The loans come from the AmeriCredit Automobile Receivables Trust 2024-1 loan-level SEC filings (Form ABS-EE, 27 monthly reports, May 2024 to July 2026). The project uses logistic regression and a competing-risks Cox survival model on loan age, validated out-of-time, and includes model documentation and a monitoring protocol.

## Results
| | In-time AUC | Out-of-time AUC | OOT calibration error |
|---|---|---|---|
| Bureau score only | 0.561 | 0.550 | n/a |
| Logistic regression | 0.687 | 0.656 | -3.08 pp (6.01% predicted vs 9.09% actual) |
| Cox survival (loan-age time scale) | 0.703 | 0.653 | +1.40 pp (10.49% vs 9.09%) |

- APR is the strongest predictor (odds ratio 1.56 per SD). Bureau score alone gives only AUC 0.561 within this subprime pool.
- The 12-month charge-off rate rose 56% (5.84% to 9.09%) as the pool aged. PSI on every input except loan age (which shifts by construction) and on the model's own PD stayed below 0.012 and did not detect it; the calibration backtest did.
- Modeling seasoning with the Cox model cut out-of-time calibration error 55%.
- PCA (5 components, 80% variance) gave no accuracy gain (0.685 vs 0.687).

Full write-up, limitations and monitoring triggers: [MODEL_CARD.md](MODEL_CARD.md).

## Run
```
pip install pandas pyarrow statsmodels scikit-learn
bash run.sh
```

| Script | What it does |
|---|---|
| `src/01_validate.py` | Re-parses 300 raw SEC XML records and checks them against the panel; runs integrity checks |
| `src/02_pd_model.py` | Logistic PD model, score-only and PCA benchmarks, out-of-time backtest, PSI |
| `src/03_survival.py` | Cause-specific Cox models with delayed entry, competing-risks 12-month PD, proportional-hazards check |
| `src/04_export_for_sas.py` | Writes the exact modeling tables for SAS |
| `sas/auto_loan_credit_risk.sas` | `PROC LOGISTIC`, `PROC PRINCOMP` and `PROC PHREG` (with `ENTRY=`) replication. Not yet executed; runs in SAS OnDemand for Academics |

Outputs are in `results/`.

## Data provenance
- **Source:** SEC EDGAR, CIK 0002020251.
- **Parsing:** the loan-level panel was parsed by the MIT-licensed public repo [SamuelJWebber/abs-risk](https://github.com/SamuelJWebber/abs-risk) (commit pinned in `data/SOURCE_COMMIT`), which logs a SHA-256 for every filing in `data/amcar-2024-1.diagnostics.json`.
- **Independent check:** this project re-parses raw records with its own code before using the panel.
