import numpy as np

from conftest import load_functions

ks, psi, hosmer_lemeshow = load_functions("02_pd_model.py", ["ks", "psi", "hosmer_lemeshow"])
rng = np.random.default_rng(0)


def test_psi_is_zero_for_identical_samples():
    x = rng.normal(size=5000)
    assert psi(x, x) == 0


def test_psi_flags_a_large_shift():
    x = rng.normal(size=5000)
    assert psi(x, x + 1.0) > 0.25


def test_ks_is_one_for_perfect_separation_and_low_for_noise():
    y = np.r_[np.zeros(500), np.ones(500)]
    assert ks(y, y + 0.0) == 1.0
    assert ks(y, rng.random(1000)) < 0.15


def test_hosmer_lemeshow_accepts_a_calibrated_model():
    p = rng.uniform(0.02, 0.3, 20000)
    y = rng.random(20000) < p
    _, pvalue = hosmer_lemeshow(y.astype(int), p)
    assert pvalue > 0.01
