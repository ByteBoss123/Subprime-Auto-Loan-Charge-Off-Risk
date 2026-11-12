"""Every headline number in README.md must match the committed results files."""
import pytest


def test_out_of_time_auc_beats_bureau_score(results):
    m, _ = results
    assert round(m["full_model"]["out_of_time"]["auc"], 3) == 0.656
    assert round(m["score_only"]["out_of_time"]["auc"], 3) == 0.550
    assert round(m["full_model"]["test_in_time"]["auc"], 3) == 0.687


def test_chargeoff_rate_rose_56_percent(results):
    m, _ = results
    dev, oot = m["full_model"]["test_in_time"]["bad_rate"], m["full_model"]["out_of_time"]["bad_rate"]
    assert (dev, oot) == (0.0584, 0.0909)
    assert round(oot / dev - 1, 2) == 0.56


def test_psi_missed_the_shift(results):
    m, _ = results
    stable = {k: v for k, v in m["psi"].items() if k != "age_months"}
    assert max(stable.values()) < 0.012


def test_cox_cut_calibration_error_55_percent(results):
    m, cox = results
    logit_err = abs(m["full_model"]["mean_pd_vs_actual_oot"][0] - m["full_model"]["mean_pd_vs_actual_oot"][1])
    cox_err = abs(cox["out_of_time"]["mean_pd"] - cox["out_of_time"]["actual"])
    assert 1 - cox_err / logit_err == pytest.approx(0.55, abs=0.01)


def test_pca_gave_no_gain(results):
    m, _ = results
    assert m["pca_model"]["test_in_time"]["auc"] < m["full_model"]["test_in_time"]["auc"]
