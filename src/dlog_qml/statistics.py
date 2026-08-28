"""
Statistical evaluation.

Every test here has an explicitly stated null hypothesis, because the single
easiest way to oversell this project would be to quote a tiny p-value without
saying what it is a p-value *for*.

  exact_binomial_p  H0: the classifier's accuracy on the held-out set equals
                    0.5 (chance, for a balanced binary task). Rejecting H0 says
                    the model learned SOMETHING. It says nothing whatsoever
                    about computational complexity, and in particular it is NOT
                    evidence of a complexity-theoretic quantum speed-up.

  wilson_interval   A 95% coverage interval for the true accuracy. Preferred
                    over Wald because it stays inside [0,1] and behaves near
                    the boundaries, where our quantum results live.

  mcnemar_exact     H0: two classifiers have equal error rates ON THE SAME test
                    items. This is the right paired test for "is model A better
                    than model B", which the two separate one-sample binomial
                    tests do not actually answer.

  bootstrap_ci      Distribution-free interval for any statistic, used for the
                    accuracy DIFFERENCE between two models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

Z_95 = 1.959963984540054


# ---------------------------------------------------------------------------
# One-sample tests against chance
# ---------------------------------------------------------------------------

def exact_binomial_p(accuracy: float, n: int, p0: float = 0.5) -> float:
    """
    Two-sided exact binomial p-value for `accuracy` against chance rate `p0`.

    H0: true accuracy == p0. Uses the "sum of all outcomes at most as likely as
    the observed one" definition, which is the standard two-sided construction
    and, for p0 = 0.5, coincides with twice the smaller tail.

    Context for interpretation: with n = 50 the standard error at chance is
    1/(2*sqrt(50)) = 0.0707, so an accuracy of 0.60 sits 1.4 sigma out and is
    NOT evidence of learning. This function exists to keep such scores from
    being read as signal.
    """
    if n <= 0:
        return float("nan")
    k = int(round(accuracy * n))
    k = max(0, min(n, k))
    probs = np.array([math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i)
                      for i in range(n + 1)])
    observed = probs[k]
    return float(min(1.0, probs[probs <= observed * (1 + 1e-12)].sum()))


def wilson_interval(accuracy: float, n: int, z: float = Z_95) -> Tuple[float, float]:
    """95% Wilson score interval. Well behaved near 0 and 1, unlike Wald."""
    if n <= 0:
        return (float("nan"), float("nan"))
    denom = 1.0 + z * z / n
    centre = (accuracy + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------
# Paired model-vs-model comparison
# ---------------------------------------------------------------------------

def mcnemar_exact(y_true: Sequence[int], pred_a: Sequence[int],
                  pred_b: Sequence[int]) -> Dict[str, float]:
    """
    Exact (binomial) McNemar test on the SAME held-out items.

    H0: the two classifiers have equal error rates, i.e. among items where
    exactly one is correct, model A is the correct one with probability 1/2.

    The exact binomial form is used rather than the chi-squared approximation
    because our discordant counts are small (tens), where chi-squared is known
    to be anti-conservative.
    """
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true

    n01 = int(np.sum(a_ok & ~b_ok))     # A right, B wrong
    n10 = int(np.sum(~a_ok & b_ok))     # A wrong, B right
    n_disc = n01 + n10

    if n_disc == 0:
        p_value = 1.0
    else:
        probs = np.array([math.comb(n_disc, i) * 0.5 ** n_disc
                          for i in range(n_disc + 1)])
        observed = probs[n01]
        p_value = float(min(1.0, probs[probs <= observed * (1 + 1e-12)].sum()))

    return {
        "n_a_correct_b_wrong": n01,
        "n_a_wrong_b_correct": n10,
        "n_discordant": n_disc,
        "p_value": p_value,
        "accuracy_a": float(a_ok.mean()),
        "accuracy_b": float(b_ok.mean()),
    }


def bootstrap_diff_ci(y_true: Sequence[int], pred_a: Sequence[int],
                      pred_b: Sequence[int], n_boot: int = 10000,
                      seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """
    Percentile bootstrap CI for accuracy(A) - accuracy(B), resampling test items.

    Paired: the same resampled index set is applied to both models, so the
    interval reflects the difference rather than two independent noises.
    """
    y_true = np.asarray(y_true)
    a_ok = (np.asarray(pred_a) == y_true).astype(float)
    b_ok = (np.asarray(pred_b) == y_true).astype(float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = a_ok[idx].mean(axis=1) - b_ok[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "difference": float(a_ok.mean() - b_ok.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": int(n_boot),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

@dataclass
class AccuracySummary:
    """Accuracy plus its uncertainty, in one reportable object."""

    accuracy: float
    n: int
    ci_low: float
    ci_high: float
    p_value: float

    @classmethod
    def build(cls, accuracy: float, n: int) -> "AccuracySummary":
        lo, hi = wilson_interval(accuracy, n)
        return cls(accuracy=float(accuracy), n=int(n), ci_low=lo, ci_high=hi,
                   p_value=exact_binomial_p(accuracy, n))

    @property
    def significant(self) -> bool:
        """Above chance at the 1% level (one-sided in direction, two-sided p)."""
        return self.p_value < 0.01 and self.accuracy > 0.5

    def to_dict(self) -> Dict[str, float]:
        return {"accuracy": self.accuracy, "n": self.n,
                "ci_low": self.ci_low, "ci_high": self.ci_high,
                "p_value": self.p_value, "significant_at_1pct": self.significant}


def describe_runs(values: Sequence[float]) -> Dict[str, float]:
    """Mean/std/median/min/max over repeated seeds. std is the sample std (ddof=1)."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        stats = {k: float("nan") for k in ("mean", "std", "median", "min", "max")}
        stats["n_runs"] = 0          # a count, not a statistic -- never NaN
        return stats
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n_runs": int(arr.size),
    }
