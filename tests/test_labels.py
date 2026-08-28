"""
Tests for the labelling rules, with particular attention to the off-by-one
behaviour at the interval boundaries -- p-1 is even, so a single-index slip
would silently unbalance the concept.
"""

import hashlib

import numpy as np
import pytest

from dlog_qml.group import DlogGroup
from dlog_qml.labels import (
    HalfIntervalLabel,
    ParityLabel,
    build_label_rule,
    derive_secret,
    legendre_predict,
)


@pytest.fixture(scope="module")
def group():
    return DlogGroup.from_qubits(10).build_tables()      # p = 1019, q = 509


# ------------------------------------------------------------------ secret

def test_secret_is_deterministic_and_in_range(group):
    a = derive_secret(20260828, group.p, group.g)
    b = derive_secret(20260828, group.p, group.g)
    assert a == b
    assert 0 <= a < group.order


def test_secret_matches_documented_formula(group):
    seed = 12345
    material = f"dlog-concept-class|p={group.p}|g={group.g}|seed={seed}".encode()
    expected = int.from_bytes(hashlib.sha256(material).digest(), "big") % (group.p - 1)
    assert derive_secret(seed, group.p, group.g) == expected


def test_secret_is_bound_to_the_group(group):
    # Same seed, different group -> different secret. Otherwise a size sweep
    # would silently reuse one secret across every p.
    other = DlogGroup.from_qubits(9)
    assert derive_secret(7, group.p, group.g) != derive_secret(7, other.p, other.g)


# ----------------------------------------------------------- half interval

def test_positive_interval_endpoints(group):
    rule = HalfIntervalLabel(secret_s=0, order=group.order, q=group.q)
    lo, hi = rule.positive_interval
    assert lo == 0
    assert hi == (group.p - 3) // 2
    assert hi == group.q - 1


@pytest.mark.parametrize("secret", [0, 1, 666, 1017])
def test_boundary_values(group, secret):
    rule = HalfIntervalLabel(secret_s=secret, order=group.order, q=group.q)
    order, q = group.order, group.q

    # shifted == 0            -> first positive
    assert rule((0 + secret) % order) == 1
    # shifted == q-1 = (p-3)/2 -> last positive
    assert rule((q - 1 + secret) % order) == 1
    # shifted == q             -> first negative
    assert rule((q + secret) % order) == -1
    # shifted == order-1       -> last negative
    assert rule((order - 1 + secret) % order) == -1


@pytest.mark.parametrize("secret", [0, 3, 509, 666, 1017])
def test_exact_balance_over_whole_group(group, secret):
    rule = HalfIntervalLabel(secret_s=secret, order=group.order, q=group.q)
    labels = rule.vectorised(np.arange(group.order))
    assert int((labels == 1).sum()) == group.q
    assert int((labels == -1).sum()) == group.q


@pytest.mark.parametrize("secret", [0, 666, 1017])
def test_closed_interval_definition_matches_implementation(group, secret):
    """The `< q` fast path must equal the documented closed interval [0,(p-3)/2]."""
    rule = HalfIntervalLabel(secret_s=secret, order=group.order, q=group.q)
    lo, hi = rule.positive_interval
    for x in range(group.order):
        shifted = (x - secret) % group.order
        expected = 1 if lo <= shifted <= hi else -1
        assert rule(x) == expected


def test_wraparound(group):
    """With a large secret the positive interval wraps past 0."""
    secret = group.order - 5
    rule = HalfIntervalLabel(secret_s=secret, order=group.order, q=group.q)
    for offset in range(5):
        assert rule((secret + offset) % group.order) == 1      # wraps through 0
    assert rule((secret - 1) % group.order) == -1


def test_vectorised_matches_scalar(group):
    rule = HalfIntervalLabel(secret_s=666, order=group.order, q=group.q)
    xs = np.arange(group.order)
    assert np.array_equal(rule.vectorised(xs), np.array([rule(int(x)) for x in xs]))


def test_rejects_inconsistent_parameters(group):
    with pytest.raises(ValueError):
        HalfIntervalLabel(secret_s=0, order=group.order, q=group.q + 1)
    with pytest.raises(ValueError):
        HalfIntervalLabel(secret_s=group.order, order=group.order, q=group.q)


# ------------------------------------------------------------------ parity

def test_parity_rule(group):
    rule = ParityLabel(order=group.order)
    assert rule(0) == 1 and rule(2) == 1
    assert rule(1) == -1 and rule(3) == -1
    labels = rule.vectorised(np.arange(group.order))
    assert int((labels == 1).sum()) == group.order // 2


# ------------------------------------------------------- Legendre attack

def test_legendre_recovers_dlog_parity_exactly(group):
    """
    Euler's criterion must equal dlog parity for EVERY group element.

    This is the mathematical fact that makes the parity concept unusable as a
    quantum-advantage instance.
    """
    h_values = np.arange(1, group.p)
    predicted = legendre_predict(group, h_values)
    actual = np.array([1 if group.dlog(int(h)) % 2 == 0 else -1 for h in h_values])
    assert np.array_equal(predicted, actual)


def test_legendre_solves_parity_and_fails_on_half_interval(group):
    from dlog_qml.data import generate_dataset

    parity = generate_dataset(n_qubits=10, n_train=200, n_test=50,
                              seed=5, labeling="parity")
    predictions = legendre_predict(parity.group, parity.group_elements("test"))
    assert (predictions == parity.labels("test")).mean() == 1.0

    hard = generate_dataset(n_qubits=10, n_train=200, n_test=50,
                            seed=5, labeling="half_interval")
    predictions = legendre_predict(hard.group, hard.group_elements("test"))
    accuracy = (predictions == hard.labels("test")).mean()
    assert 0.25 < accuracy < 0.75, f"expected chance level, got {accuracy}"


# ------------------------------------------------------------------ factory

def test_build_label_rule(group):
    assert build_label_rule("half_interval", group, 1).name == "half_interval"
    assert build_label_rule("parity", group, 1).name == "parity"
    with pytest.raises(ValueError):
        build_label_rule("nonsense", group, 1)
