/* SAS replication of the current single-snapshot logistic PD model (src/02_pd_model.py).
   Reads sas/pd_snapshots.csv (written by src/04_export_for_sas.py) straight from GitHub,
   with a DATA step instead of PROC IMPORT (much faster on 122K rows).
   Expected (Python, results/02_pd_model.json): test AUC 0.6869, out-of-time AUC 0.6557. */

filename pdcsv temp;
proc http
  url="https://raw.githubusercontent.com/ByteBoss123/Subprime-Auto-Loan-Charge-Off-Risk/main/sas/pd_snapshots.csv"
  out=pdcsv;
run;

data pd;
  infile pdcsv dsd firstobs=2 truncover;
  length asset_id $ 32 split $ 8;
  input asset_id $ split $ y score orig_apr orig_term pti ltv log_amount age_months
        used income_verified_3 employment_verified_3 subvented_any score_missing;
run;

title "Row counts by split (expected train 50,156 / test 21,496 / oot 51,231)";
proc freq data=pd; tables split*y / norow nocol nopercent; run;

/* Standardize numeric inputs on the training split only, as in Python */
proc stdize data=pd(where=(split="train")) out=_null_ outstat=stdstat method=std;
  var score orig_apr orig_term pti ltv log_amount age_months;
run;
proc stdize data=pd out=pdz method=in(stdstat);
  var score orig_apr orig_term pti ltv log_amount age_months;
run;

title "Logistic PD model - development (train split)";
proc logistic data=pdz(where=(split="train")) descending outmodel=pd_model;
  model y = score orig_apr orig_term pti ltv log_amount age_months
            used income_verified_3 employment_verified_3 subvented_any score_missing;
run;

title "Test AUC (Python 0.6869)";
proc logistic inmodel=pd_model;
  score data=pdz(where=(split="test")) out=sc_test fitstat;
run;

title "Out-of-time AUC (Python 0.6557)";
proc logistic inmodel=pd_model;
  score data=pdz(where=(split="oot")) out=sc_oot fitstat;
run;
title;
