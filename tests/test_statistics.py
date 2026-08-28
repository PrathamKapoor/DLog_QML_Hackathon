"""
Statistics tests, checked against closed-form values and against scipy where
available. Every function here feeds a number that appears in the README, so
these are correctness tests, not smoke tests.
"""

import math

import numpy as np
import pytest

from dlog_qml.statistics import (
    AccuracySummary,
    bootstrap_diff_ci,
    describe_runs,
    exact_binomial_p,
    mcnemar_exact,
    wilson_interval,
)


# ------------------------------------------------------------ binomial test

def test_chance_accuracy_gives_p_one():
    assert exact_binomial_p(0.5, 50) == pytest.approx(1.0)


def test_perfect_accuracy_matches_closed_form():
    # Two-sided p for 50/50 successes is 2 * 2^-50.
    assert exact_binomial_p(1.0, 50) == pytest.approx(2 * 2.0 ** -50, rel=1e-9)


def test_known_value_47_of_50():
    # This is the coset kernel's reference result. Two-sided exact p =
    # 2 * sum_{k=47}^{50} C(50,k) / 2^50 = 2 * 20876 / 2^50.
    expected = 2 * (1 + 50 + 1225 + 19600) / 2 ** 50
    assert exact_binomial_p(47 / 50, 50) == pytest.approx(expected, rel=1e-9)
    assert exact_binomial_p(47 / 50, 50) < 4e-11


def test_known_value_48_of_50():
    expected = 2 * (1 + 50 + 1225) / 2 ** 50
    assert exact_binomial_p(48 / 50, 50) == pytest.approx(expected, rel=1e-9)


def test_symmetry_about_one_half():
    for k in range(0, 51):
        assert exact_binomial_p(k / 50, 50) == pytest.approx(
            exact_binomial_p((50 - k) / 50, 50), rel=1e-12)


def test_sixty_percent_is_not_significant():
    """The claim-discipline test: 0.60 on n=50 must NOT clear p<0.01."""
    p = exact_binomial_p(0.60, 50)
    assert p > 0.1
    assert not AccuracySummary.build(0.60, 50).significant


def test_p_value_is_monotone_away_from_chance():
    values = [exact_binomial_p(k / 50, 50) for k in range(25, 51)]
    assert all(a >= b - 1e-15 for a, b in zip(values, values[1:]))


@pytest.mark.parametrize("k,n", [(30, 50), (40, 50), (47, 50), (12, 20), (100, 200)])
def test_agrees_with_scipy(k, n):
    scipy_stats = pytest.importorskip("scipy.stats")
    expected = scipy_stats.binomtest(k, n, 0.5, alternative="two-sided").pvalue
    assert exact_binomial_p(k / n, n) == pytest.approx(expected, rel=1e-9)


# ----------------------------------------------------------- Wilson interval

def test_wilson_contains_the_estimate():
    for k in range(0, 51):
        lo, hi = wilson_interval(k / 50, 50)
        assert lo <= k / 50 <= hi


def test_wilson_stays_inside_unit_interval():
    for accuracy in (0.0, 0.02, 0.5, 0.98, 1.0):
        lo, hi = wilson_interval(accuracy, 50)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_known_value():
    # 47/50 = 0.94 -> [0.838, 0.979] to three decimals (quoted in the README).
    lo, hi = wilson_interval(0.94, 50)
    assert lo == pytest.approx(0.838, abs=1e-3)
    assert hi == pytest.approx(0.979, abs=1e-3)


def test_wilson_narrows_with_more_data():
    narrow = wilson_interval(0.9, 1000)
    wide = wilson_interval(0.9, 50)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


def test_wilson_centre_is_shrunk_toward_one_half():
    lo, hi = wilson_interval(1.0, 20)
    assert hi == pytest.approx(1.0)
    assert lo < 1.0            # unlike Wald, which would give [1,1]


# ------------------------------------------------------------------ McNemar

def test_mcnemar_identical_models():
    y = np.array([1, -1, 1, -1, 1])
    result = mcnemar_exact(y, y, y)
    assert result["n_discordant"] == 0
    assert result["p_value"] == 1.0


def test_mcnemar_counts_discordant_pairs():
    y = np.array([1, 1, 1, 1])
    a = np.array([1, 1, 1, -1])       # 3 correct
    b = np.array([1, -1, -1, -1])     # 1 correct
    result = mcnemar_exact(y, a, b)
    assert result["n_a_correct_b_wrong"] == 2
    assert result["n_a_wrong_b_correct"] == 0
    assert result["n_discordant"] == 2
    assert result["p_value"] == pytest.approx(0.5)     # 2 * 2^-2


def test_mcnemar_known_value():
    """19 vs 3 discordant -> two-sided exact p = 2 * sum_{k>=19} C(22,k)/2^22."""
    y = np.ones(50, dtype=int)
    a = np.ones(50, dtype=int)
    b = np.ones(50, dtype=int)
    a[:3] = -1                       # A wrong on 3 where B is right
    b[3:22] = -1                     # B wrong on 19 where A is right
    result = mcnemar_exact(y, a, b)
    assert result["n_a_correct_b_wrong"] == 19
    assert result["n_a_wrong_b_correct"] == 3
    tail = sum(math.comb(22, k) for k in range(19, 23))
    assert result["p_value"] == pytest.approx(2 * tail / 2 ** 22, rel=1e-9)


def test_mcnemar_agrees_with_scipy_binomtest():
    scipy_stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(0)
    y = rng.choice([-1, 1], size=60)
    a = np.where(rng.random(60) < 0.85, y, -y)
    b = np.where(rng.random(60) < 0.55, y, -y)
    result = mcnemar_exact(y, a, b)
    expected = scipy_stats.binomtest(
        result["n_a_correct_b_wrong"], result["n_discordant"], 0.5).pvalue
    assert result["p_value"] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------- bootstrap

def test_bootstrap_difference_matches_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.choice([-1, 1], size=100)
    a = np.where(rng.random(100) < 0.9, y, -y)
    b = np.where(rng.random(100) < 0.6, y, -y)
    result = bootstrap_diff_ci(y, a, b, n_boot=2000, seed=0)
    assert result["difference"] == pytest.approx((a == y).mean() - (b == y).mean())
    assert result["ci_low"] < result["difference"] < result["ci_high"]


def test_bootstrap_is_deterministic_under_seed():
    y = np.array([1, -1] * 25)
    a = y.copy()
    b = -y
    first = bootstrap_diff_ci(y, a, b, n_boot=500, seed=3)
    second = bootstrap_diff_ci(y, a, b, n_boot=500, seed=3)
    assert first == second


# ----------------------------------------------------------------- summaries

def test_accuracy_summary_fields():
    summary = AccuracySummary.build(0.94, 50)
    assert summary.accuracy == 0.94
    assert summary.n == 50
    assert summary.p_value < 1e-9
    assert summary.significant
    assert set(summary.to_dict()) == {
        "accuracy", "n", "ci_low", "ci_high", "p_value", "significant_at_1pct"}


def test_summary_not_significant_below_chance():
    """High-confidence BELOW chance must not be reported as 'learns'."""
    summary = AccuracySummary.build(0.06, 50)
    assert summary.p_value < 1e-9
    assert not summary.significant


def test_describe_runs():
    stats = describe_runs([0.9, 0.92, 0.94])
    assert stats["mean"] == pytest.approx(0.92)
    assert stats["median"] == pytest.approx(0.92)
    assert stats["min"] == pytest.approx(0.90)
    assert stats["max"] == pytest.approx(0.94)
    assert stats["n_runs"] == 3
    assert stats["std"] == pytest.approx(np.std([0.9, 0.92, 0.94], ddof=1))


def test_describe_runs_handles_edge_cases():
    assert describe_runs([])["n_runs"] == 0
    assert describe_runs([0.5])["std"] == 0.0
