# Model Documentation: 12-Month Auto Loan Charge-Off Models (AmeriCredit 2024-1)

Written in the style of a bank model-risk document (SR 11-7 structure: purpose, data, methodology, performance, limitations, monitoring). This is a portfolio study on public data, not a production model.

## 1. Purpose
Estimate each active loan's probability of charge-off within the next 12 months, for loss forecasting and risk-tiering of a subprime auto loan pool.

## 2. Data
| Item | Detail |
|---|---|
| Source | SEC Form ABS-EE, Exhibit 102 (Reg AB II asset-level data), AmeriCredit Automobile Receivables Trust 2024-1 (CIK 0002020251). |
| Coverage | 27 monthly filings, May 2024 to July 2026. 72,665 loans and 1,434,537 loan-months. |
| Exits observed | 9,550 charge-offs, 28,547 prepayments, 6 repurchases (treated as censored), 34,562 still active. |
| Parsing | Loan panel built by the MIT-licensed public repo SamuelJWebber/abs-risk (commit in `data/SOURCE_COMMIT`), which records a SHA-256 for every source filing (`data/amcar-2024-1.diagnostics.json`). |
| Independent check | `src/01_validate.py` re-parses 300 verbatim July 2026 records. Original amount, APR, term and credit score match the panel for 300/300 loans. State matches 290/300: the panel keeps the first filing's state, and the 10 differences are consistent with borrower relocation. State is not a model input. |
| Inputs | Origination fields only: bureau score (492 missing, median-imputed with a flag), APR, term, payment-to-income, LTV, log amount, new/used, income and employment verification, subvention, and loan age at snapshot. No protected-class attributes exist in the filing. |

## 3. Methodology
**Windows.** Development: loans active at 2024-05-31 (71,652), outcome = charge-off by 2025-05-31. Out-of-time (OOT): loans active at 2025-05-31 (51,231), outcome = charge-off by 2026-05-31. Development is split 70/30, stratified, seed 42.

**Model A: logistic regression.** Standardized inputs; coefficients, Wald 95% CIs and VIFs reported (max VIF 3.8).

**Model B: cause-specific Cox proportional hazards.**
- Time scale is loan age in months, with delayed entry: loans join the pool already seasoned, so each loan is at risk only from its age at the pool cutoff onward.
- Charge-off and prepayment hazards are fit separately.
- The 12-month PD combines them in a discrete-time competing-risks calculation, so a prepaid loan cannot later charge off.
- Development data is censored at 2025-05-31 so no OOT outcome is seen.

**Benchmarks.**
- Bureau score alone.
- Logistic regression on the first 5 principal components (80% of numeric-input variance) plus the binary flags.

## 4. Performance
| Model | In-time test AUC | OOT AUC | OOT mean PD vs actual |
|---|---|---|---|
| Bureau score only | 0.561 | 0.550 | n/a |
| PCA (5 components) logistic | 0.685 | 0.650 | n/a |
| Logistic regression | 0.687 (KS 0.284) | 0.656 (KS 0.229) | 6.01% vs 9.09% |
| Cox, loan-age time scale | 0.703 | 0.653 | 10.49% vs 9.09% |

**Findings**
1. **APR drives the ranking; the bureau score barely does.** APR is the strongest driver (odds ratio 1.56 per SD; Cox hazard ratio 1.70). Bureau score alone reaches only AUC 0.561. Within one subprime pool, the score range is narrow and the lender's own pricing already carries its internal risk view.
2. **Ranking held out-of-time; calibration did not.** All 10 OOT deciles stay monotonic. But the realized 12-month charge-off rate rose from 5.84% to 9.09% (+56% relative) as the pool aged. The logistic model under-predicted it by 3.08 pp (Hosmer-Lemeshow p = 0.50 in-time, p < 1e-190 OOT).
3. **Modeling loan age fixed most of the calibration gap.** The Cox model learns the seasoning curve from older loans already in the pool. That cut absolute OOT calibration error 55% (3.08 pp to 1.40 pp), now over-predicting, with similar ranking (AUC 0.653 vs 0.656).
4. **Input and score PSI missed the shift.** PSI stayed below 0.012 on every input except loan age, and was 0.0045 on the model's own PD. Loan age shifts by construction as the pool seasons. Only the calibration backtest caught the 56% loss-rate rise.
5. **PCA added nothing.** It did not improve accuracy (0.685 vs 0.687) and loses coefficient-level interpretability. The interpretable logistic model is preferred.

## 5. Limitations
- **One trust.** A single deal, one securitized pool. Securitized loans are selected by the issuer and are not the issuer's full loan book.
- **Inputs are origination fields only.** Behavioral data (payment history, delinquency at snapshot) is excluded from this version.
- **Proportional hazards are mildly violated for APR.** The Schoenfeld residual correlation with time is -0.107, so the APR effect weakens with age. Other covariates have |r| < 0.07.
- **Seasoning curve at older ages is noisy.** The baseline hazard beyond about 24 months rests on fewer, older-vintage loans (median age at cutoff is 8 months). That likely explains the Cox model's over-prediction.
- **Macro and calendar effects are not separated from seasoning.** The model attributes the loss rise to age alone.
- **The SAS programs are written but not yet executed** (see `sas/`).

## 6. Monitoring Protocol
| Check | Frequency | Trigger | Action |
|---|---|---|---|
| Calibration backtest: actual vs predicted charge-off rate, overall and by decile | Monthly | Actual outside the predicted rate's 95% binomial interval for 2 consecutive months | Recalibrate intercept or baseline hazard; escalate if repeated |
| Discrimination (AUC, KS) on matured 12-month cohorts | Quarterly | AUC drop > 0.03 vs development | Root-cause review; redevelopment if persistent |
| PSI on inputs (excluding loan age) and on model PD | Monthly | > 0.10 investigate, > 0.25 action | Input data review |
| Proportional-hazards check (Schoenfeld residuals) | At each refit | \|r\| > 0.10 | Add time-varying effect or stratify |
| Data integrity: record counts, exit codes, raw-vs-parsed field match | Each filing | Any field mismatch, or loans absent without an exit code | Block scoring until resolved |

PSI is kept as a secondary check, not the primary one. Finding 4 shows that stable inputs can coexist with a large loss-rate shift.
