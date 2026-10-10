/* Subprime auto loan (AmeriCredit 2024-1) credit risk: SAS replication of the Python models.
   Status: NOT YET EXECUTED. Written to run in SAS OnDemand for Academics (free).
   1. Upload pd_snapshots.csv and survival_dev.csv to a folder named "autoloan" in your SAS OnDemand home folder.
   2. Set the path below, run, and compare to results/02_pd_model.json and results/03_cox_hazard_ratios.csv.
      PROC LOGISTIC c-statistics should match the Python AUCs (0.6869 test, 0.6557 OOT) to ~3 decimals.
      PROC PHREG reports per-unit hazard ratios; Python reports per-SD, so convert before comparing. */

%let path = /home/&sysuserid./autoloan;

proc import datafile="&path./pd_snapshots.csv" out=pd dbms=csv replace; guessingrows=max; run;
proc import datafile="&path./survival_dev.csv" out=surv dbms=csv replace; guessingrows=max; run;

/* Standardize numeric inputs on the training split only, then apply to test and OOT */
proc stdize data=pd(where=(split="train")) out=_null_ outstat=stdstat method=std;
  var score orig_apr orig_term pti ltv log_amount age_months;
run;
proc stdize data=pd out=pd_z method=in(stdstat);
  var score orig_apr orig_term pti ltv log_amount age_months;
run;

/* 1. 12-month charge-off PD: logistic regression, scored on in-time test and out-of-time windows */
proc logistic data=pd_z(where=(split="train")) descending outmodel=pd_model;
  model y = score orig_apr orig_term pti ltv log_amount age_months
            used income_verified_3 employment_verified_3 subvented_any score_missing
            / lackfit clodds=wald;
  title "12-month charge-off PD - development";
run;

proc logistic inmodel=pd_model;
  score data=pd_z(where=(split="test")) out=sc_test fitstat;
  score data=pd_z(where=(split="oot"))  out=sc_oot  fitstat;
run;

/* Calibration by decile on the out-of-time window */
proc rank data=sc_oot groups=10 out=sc_oot_r; var p_1; ranks decile; run;
proc means data=sc_oot_r mean n; class decile; var p_1 y;
  title "OOT calibration: predicted vs actual by decile"; run;

/* 2. Principal components of the numeric origination features */
proc princomp data=pd(where=(split="train")) out=pcs n=5 std;
  var score orig_apr orig_term pti ltv log_amount age_months;
  title "PCA on origination features";
run;

/* 3. Cause-specific Cox model for charge-off on the loan-age time scale with delayed entry.
      Prepayment (pp=1) is treated as censoring for the charge-off hazard. */
proc phreg data=surv;
  model time*co(0) = score orig_apr orig_term pti ltv log_amount
                     used income_verified_3 employment_verified_3 subvented_any score_missing
                     / entry=entry ties=efron risklimits;
  output out=sch ressch=r_score r_apr r_term r_pti r_ltv r_amt r_used r_inc r_emp r_sub r_miss;
  title "Cause-specific Cox: charge-off (delayed entry)";
run;

/* Proportional-hazards check: Schoenfeld residuals vs event time (Python: results/03_cox.json) */
proc corr data=sch(where=(co=1)) pearson;
  var r_score r_apr r_term r_pti r_ltv r_amt r_used r_inc r_emp r_sub r_miss; with time;
  title "Schoenfeld residual correlation with time";
run;
