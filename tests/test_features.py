import pandas as pd
import pytest

import features as f


def frame(rows):
    d = pd.DataFrame(rows, columns=["first_period", "end_period", "exit_period", "exit_type", "orig_month"])
    for c in ["first_period", "end_period", "exit_period", "orig_month"]:
        d[c] = pd.to_datetime(d[c])
    return d


CUT, HORIZON = pd.Timestamp("2024-05-31"), pd.Timestamp("2025-05-31")


def test_label_counts_only_chargeoffs_inside_the_horizon():
    d = frame([
        ["2024-05-31", "2024-12-31", "2024-12-31", "chargeoff", "2023-01-01"],  # inside window -> 1
        ["2024-05-31", "2025-09-30", "2025-09-30", "chargeoff", "2023-01-01"],  # after horizon -> 0
        ["2024-05-31", "2024-10-31", "2024-10-31", "prepay", "2023-01-01"],     # prepay -> 0
        ["2024-05-31", "2026-07-31", None, "censored", "2023-01-01"],           # still open -> 0
    ])
    s = f.snapshot(d, CUT, HORIZON)
    assert s.y.tolist() == [1, 0, 0, 0]


def test_snapshot_excludes_loans_not_active_at_cutoff():
    d = frame([
        ["2024-05-31", "2024-03-31", "2024-03-31", "chargeoff", "2022-01-01"],  # ended before cut
        ["2024-08-31", "2025-01-31", "2025-01-31", "chargeoff", "2024-07-01"],  # entered after cut
        ["2024-05-31", "2025-01-31", "2025-01-31", "chargeoff", "2024-01-01"],  # active
    ])
    s = f.snapshot(d, CUT, HORIZON)
    assert len(s) == 1 and s.y.iloc[0] == 1


def test_loan_age_is_measured_at_the_cutoff():
    d = frame([["2024-05-31", "2026-01-31", None, "censored", "2023-05-15"]])
    assert f.snapshot(d, CUT, HORIZON).age_months.iloc[0] == 12


def test_features_use_no_performance_fields():
    performance = {"exit_period", "exit_type", "exit_code", "last_period", "max_dpd", "first_30_period",
                   "first_60_period", "first_90_period", "chargeoff_amount", "recovered_amount", "bal_last",
                   "months_observed", "end_period", "y"}
    assert not performance & set(f.FEATURES)


@pytest.fixture(scope="module")
def loans():
    return f.load()


def test_pool_has_72665_unique_loans(loans):
    assert len(loans) == 72665
    assert loans.asset_id.is_unique


def test_repurchases_are_censored_not_credit_events(loans):
    assert set(loans.exit_type.unique()) <= {"chargeoff", "prepay", "censored"}
