r"""
The structure-aligned coset-state feature map (Liu, Arunachalam & Temme 2021).

    |Phi(h)> = (1/sqrt(M)) * sum_{i=0}^{M-1} |h * g^i mod p>,     M = 2^m

The register stores a group element as a computational basis state, so it needs
n = ceil(log2 p) qubits and the support of |Phi(h)> is the coset-like window
{h, hg, hg^2, ..., hg^(M-1)} -- an arithmetic progression of length M on the
DISCRETE-LOG CIRCLE Z_{p-1}.


Closed form of the induced kernel
---------------------------------
Let a = log_g(h), a' = log_g(h'), N = p - 1, and delta = (a' - a) mod N. The two
supports are length-M arithmetic progressions on Z_N starting at a and a'. Their
intersection therefore has size

    Omega(delta) = max(0, M - delta) + max(0, M - (N - delta))

where the first term counts the forward overlap and the second the wraparound
overlap. Provided 2M <= N the two terms cannot both be positive, so no element
is double counted. Since the states are uniform over their supports with real
positive amplitude 1/sqrt(M),

    <Phi(h)|Phi(h')> = Omega(delta) / M
    K(h,h')          = |<Phi(h)|Phi(h')>|^2 = (Omega(delta) / M)^2.

This is a circular TRIANGULAR kernel in discrete-log coordinates, and it is the
entire mechanism of the result: the quantum state performs the hard change of
coordinates h -> log_g(h) implicitly, so a max-margin classifier in this kernel
is solving one-dimensional interval membership.

`analytic_kernel` implements this formula directly, never touching a simulator,
and tests/test_kernel.py checks it against the simulated Gram matrix.


IMPORTANT -- what the simulation does and does not claim
-------------------------------------------------------
Here |Phi(h)> is constructed as an explicit dense amplitude vector and, for
verification, compiled with Qiskit's `StatePreparation`. Dense preparation costs
Theta(2^n) and is a SIMULATION CONVENIENCE for studying the feature-map
geometry. It is NOT an efficient quantum algorithm and this repository never
claims it is.

The intended quantum implementation prepares the same state in O(n^3) gates on
~2n+3 qubits by Shor-style period finding: build a uniform superposition over
i in [0, M), compute |i>|h * g^i mod p> by modular exponentiation, then
uncompute the index register with an inverse QFT. No step of that routine ever
evaluates log_g(h). See docs/LIMITATIONS.md.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation

from .group import DlogGroup


def default_window_bits(group: DlogGroup) -> int:
    """
    Default coset window M = 2^m with M ~ (p-1)/8.

    Rationale: the kernel is a triangle of half-width M on a circle of
    circumference N = p-1. Too small and almost every pair is orthogonal (no
    information); too large and the triangle flattens toward a constant (no
    discrimination), besides violating the 2M <= N condition of the closed
    form. N/8 sits comfortably between, and the value is exposed as
    `window_bits` so the ablation in run_experiment can sweep it.
    """
    return max(1, int(round(math.log2(max(2.0, group.order / 8.0)))))


class CosetStateFeatureMap:
    """Feature map |Phi(h)>; see module docstring for the mathematics."""

    name = "coset"

    def __init__(self, group: DlogGroup, window_bits: Optional[int] = None):
        self.group = group
        self.n_qubits = group.n_bits
        self.dim = 2 ** self.n_qubits
        self.window_bits = int(window_bits if window_bits is not None
                               else default_window_bits(group))
        self.window = 2 ** self.window_bits

        if self.window < 1:
            raise ValueError("window must be >= 1")
        if 2 * self.window > group.order:
            raise ValueError(
                f"window M = 2^{self.window_bits} = {self.window} is too large: "
                f"the closed-form overlap requires 2M <= p-1 = {group.order}. "
                f"Reduce window_bits."
            )
        self._cache: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------ amplitudes
    def statevector(self, h: int) -> np.ndarray:
        """
        Dense amplitude vector of |Phi(h)>: length 2^n, real, unit norm.

        Built by walking the group forward (h, hg, hg^2, ...), which uses only
        modular multiplication -- the EASY direction. No discrete logarithm is
        computed here, which matters: the feature map must be constructible by
        someone who does not hold the trapdoor.
        """
        h = int(h)
        cached = self._cache.get(h)
        if cached is not None:
            return cached

        vec = np.zeros(self.dim, dtype=complex)
        amplitude = 1.0 / math.sqrt(self.window)
        element = h % self.group.p
        for _ in range(self.window):
            vec[element] += amplitude
            element = (element * self.group.g) % self.group.p

        self._cache[h] = vec
        return vec

    def statevectors(self, h_values) -> np.ndarray:
        """Stack of amplitude vectors, one row per sample."""
        return np.stack([self.statevector(int(h)) for h in h_values])

    # --------------------------------------------------------------- circuit
    def circuit(self, h: int) -> QuantumCircuit:
        """
        Qiskit circuit preparing |Phi(h)> from |0...0>.

        Verification path only -- StatePreparation synthesis is exponential in
        n. Used by tests and by the compute-uncompute audit in kernels.py.
        """
        qc = QuantumCircuit(self.n_qubits, name=f"U_Phi_coset({h})")
        qc.append(StatePreparation(self.statevector(h)), range(self.n_qubits))
        return qc

    # ------------------------------------------------------------ closed form
    def analytic_kernel(self, h_a: int, h_b: int) -> float:
        """
        K(h_a, h_b) from the triangular-overlap formula, with no simulation.

        Uses the ORACLE discrete logarithm, so this is an auditing tool, not
        part of the learner. It exists to prove the simulated Gram matrix has
        the structure the theory predicts.
        """
        N, M = self.group.order, self.window
        delta = (self.group.dlog(int(h_b)) - self.group.dlog(int(h_a))) % N
        overlap = max(0, M - delta) + max(0, M - (N - delta))
        return (overlap / M) ** 2

    def analytic_gram(self, h_a, h_b) -> np.ndarray:
        return np.array([[self.analytic_kernel(int(a), int(b)) for b in h_b]
                         for a in h_a], dtype=float)

    def describe(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "window_bits": self.window_bits,
            "window_M": self.window,
            "group_order": self.group.order,
            "window_fraction_of_group": self.window / self.group.order,
        }
