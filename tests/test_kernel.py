"""
Quantum feature-map and kernel tests.

The important ones check that the fast BLAS Gram matrix really is the
circuit-defined fidelity kernel, and that the coset kernel matches its
closed-form triangular-overlap expression.
"""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from dlog_qml.data import generate_dataset
from dlog_qml.group import DlogGroup
from dlog_qml.kernels import (
    build_quantum_kernels,
    fidelity_gram,
    kernel_diagnostics,
    verify_against_circuits,
    verify_kernel_properties,
)
from dlog_qml.quantum_coset import CosetStateFeatureMap, default_window_bits
from dlog_qml.quantum_generic import LadderFeatureMap


@pytest.fixture(scope="module")
def small_group():
    return DlogGroup.from_qubits(8).build_tables()       # p = 227


@pytest.fixture(scope="module")
def coset(small_group):
    return CosetStateFeatureMap(small_group, window_bits=4)


# ------------------------------------------------------------- coset states

def test_statevector_is_normalised(coset, small_group):
    for h in (1, 2, 17, small_group.p - 1):
        psi = coset.statevector(h)
        assert psi.shape == (2 ** small_group.n_bits,)
        assert np.isclose(np.linalg.norm(psi), 1.0)


def test_statevector_support_is_the_coset(coset, small_group):
    h = 5
    psi = coset.statevector(h)
    support = set(np.flatnonzero(np.abs(psi) > 1e-12).tolist())
    expected = set()
    element = h
    for _ in range(coset.window):
        expected.add(element)
        element = (element * small_group.g) % small_group.p
    assert support == expected
    assert len(support) == coset.window


def test_amplitudes_are_uniform(coset):
    psi = coset.statevector(11)
    amplitudes = np.abs(psi[np.abs(psi) > 1e-12])
    assert np.allclose(amplitudes, 1.0 / np.sqrt(coset.window))


def test_window_too_large_is_rejected(small_group):
    with pytest.raises(ValueError, match="too large"):
        CosetStateFeatureMap(small_group, window_bits=8)   # 2*256 > 226


def test_default_window_respects_the_2M_le_N_condition():
    for n_qubits in (6, 7, 8, 9, 10):
        group = DlogGroup.from_qubits(n_qubits)
        m = default_window_bits(group)
        assert 2 * (2 ** m) <= group.order


# ------------------------------------------------- closed form vs simulation

def test_coset_kernel_matches_closed_form(coset, small_group):
    rng = np.random.default_rng(0)
    h_values = [int(small_group.power(int(x)))
                for x in rng.integers(0, small_group.order, size=25)]
    psi = coset.statevectors(h_values)
    simulated = fidelity_gram(psi)
    analytic = coset.analytic_gram(h_values, h_values)
    assert np.abs(simulated - analytic).max() < 1e-12


def test_triangular_overlap_formula_by_hand(coset, small_group):
    """Spot-check the formula against a directly counted set intersection."""
    N, M, g, p = small_group.order, coset.window, small_group.g, small_group.p
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = int(rng.integers(N)), int(rng.integers(N))
        h_a, h_b = pow(g, a, p), pow(g, b, p)

        support_a, support_b = set(), set()
        element = h_a
        for _ in range(M):
            support_a.add(element)
            element = (element * g) % p
        element = h_b
        for _ in range(M):
            support_b.add(element)
            element = (element * g) % p

        counted = (len(support_a & support_b) / M) ** 2
        assert np.isclose(counted, coset.analytic_kernel(h_a, h_b))


def test_kernel_depends_only_on_dlog_difference(coset, small_group):
    """K(h, h') is a function of (log h' - log h) mod N and nothing else."""
    g, p, N = small_group.g, small_group.p, small_group.order
    delta = 3
    values = set()
    for a in (0, 17, 101, 200):
        h_a = pow(g, a, p)
        h_b = pow(g, (a + delta) % N, p)
        values.add(round(coset.analytic_kernel(h_a, h_b), 12))
    assert len(values) == 1


def test_self_kernel_is_one(coset, small_group):
    for h in (1, 3, 50, small_group.p - 2):
        assert np.isclose(coset.analytic_kernel(h, h), 1.0)


# --------------------------------------------------- compute-uncompute audit

@pytest.mark.parametrize("kind", ["coset", "ladder"])
def test_gram_matches_compute_uncompute_circuits(kind, small_group):
    rng = np.random.default_rng(2)
    if kind == "coset":
        feature_map = CosetStateFeatureMap(small_group, window_bits=3)
        inputs = [int(small_group.power(int(x)))
                  for x in rng.integers(0, small_group.order, size=8)]
    else:
        feature_map = LadderFeatureMap(small_group.n_bits, reps=2)
        inputs = list(rng.integers(0, 2, size=(8, small_group.n_bits)))

    K = fidelity_gram(feature_map.statevectors(inputs))
    report = verify_against_circuits(feature_map, inputs, K, n_pairs=6, seed=0)
    assert report["max_compute_uncompute_error"] < 1e-9


def test_compute_uncompute_by_hand(small_group):
    """Explicitly build U(x_i) U(x_j)^dag and read P(|0^n>)."""
    feature_map = LadderFeatureMap(small_group.n_bits, reps=1)
    rng = np.random.default_rng(3)
    a = rng.integers(0, 2, size=small_group.n_bits)
    b = rng.integers(0, 2, size=small_group.n_bits)

    circuit = feature_map.circuit(a).compose(feature_map.circuit(b).inverse())
    fidelity = abs(Statevector(circuit).data[0]) ** 2
    direct = abs(np.vdot(feature_map.statevector(b), feature_map.statevector(a))) ** 2
    assert np.isclose(fidelity, direct, atol=1e-10)


# ----------------------------------------------------------- kernel algebra

@pytest.mark.parametrize("kind", ["coset", "ladder"])
def test_kernel_properties(kind, small_group):
    rng = np.random.default_rng(4)
    if kind == "coset":
        feature_map = CosetStateFeatureMap(small_group, window_bits=3)
        inputs = [int(small_group.power(int(x)))
                  for x in rng.integers(0, small_group.order, size=30)]
    else:
        feature_map = LadderFeatureMap(small_group.n_bits, reps=2)
        inputs = list(rng.integers(0, 2, size=(30, small_group.n_bits)))

    K = fidelity_gram(feature_map.statevectors(inputs))
    properties = verify_kernel_properties(K)
    assert properties["symmetric"]
    assert properties["unit_diagonal"]
    assert properties["in_unit_range"]
    assert properties["positive_semidefinite"]
    assert np.allclose(K, K.T)                      # K(a,b) == K(b,a)


def test_rectangular_test_kernel_shape(small_group):
    data = generate_dataset(n_qubits=8, n_train=40, n_test=10, seed=7)
    feature_map = CosetStateFeatureMap(data.group, window_bits=3)
    K_train, K_test, timings = build_quantum_kernels(
        feature_map, data.group_elements("train"), data.group_elements("test"))
    assert K_train.shape == (40, 40)
    assert K_test.shape == (10, 40)                 # sklearn's convention
    assert timings["statevector_dim"] == 2 ** data.group.n_bits


# ------------------------------------------------------------- diagnostics

def test_diagnostics_on_identity_and_ones():
    identity = np.eye(20)
    stats = kernel_diagnostics(identity)
    assert np.isclose(stats["offdiag_mean"], 0.0)
    assert np.isclose(stats["effective_rank"], 20.0)

    ones = np.ones((20, 20))
    stats = kernel_diagnostics(ones)
    assert np.isclose(stats["offdiag_mean"], 1.0)
    assert stats["effective_rank"] < 1.5           # rank-1 matrix


def test_generic_kernel_is_more_concentrated_than_coset():
    """
    The headline diagnostic: the generic map's Gram matrix is far closer to the
    identity than the coset map's, which is the quantum-overfitting signature.
    """
    data = generate_dataset(n_qubits=10, n_train=60, n_test=10, seed=11)
    coset = CosetStateFeatureMap(data.group)
    ladder = LadderFeatureMap(data.group.n_bits, reps=2)

    K_coset = fidelity_gram(coset.statevectors(data.group_elements("train")))
    K_ladder = fidelity_gram(ladder.statevectors(data.features("train").astype(int)))

    coset_stats = kernel_diagnostics(K_coset)
    ladder_stats = kernel_diagnostics(K_ladder)
    assert ladder_stats["offdiag_mean"] < 0.05
    assert coset_stats["offdiag_mean"] > ladder_stats["offdiag_mean"]
    assert coset_stats["effective_rank"] < ladder_stats["effective_rank"]


def test_ladder_angles_avoid_the_degenerate_scale():
    """phi = (pi/2)*b, not pi*b -- otherwise Rz(2 phi) is a global phase."""
    feature_map = LadderFeatureMap(4, reps=1)
    assert np.allclose(feature_map.angles([0, 1, 0, 1]),
                       [0.0, np.pi / 2, 0.0, np.pi / 2])
    distinct = fidelity_gram(feature_map.statevectors(
        [[0, 0, 0, 0], [1, 1, 1, 1]]))
    assert distinct[0, 1] < 0.999      # the two inputs are not identified
