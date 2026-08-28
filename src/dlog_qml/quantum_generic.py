r"""
Generic hardware-efficient feature map -- the QUANTUM NEGATIVE CONTROL.

Structure, per repetition, over the n bit-features b_j with phi_j = (pi/2) b_j:

    H on every qubit
    Rz(2 phi_j) on qubit j
    CX(j,j+1) . Rz(2 (pi/2 - phi_j)(pi/2 - phi_{j+1})) . CX(j,j+1)     [ZZ term]
    Rx(2 phi_j) on qubit j
    CZ(j,j+1) ladder

This is a Havlicek-style second-order encoding with the Rz/Rx/CZ ladder
structure. It is deliberately a *strong* ansatz, not a straw man: it entangles
across the full register, and kernels of this family are not known to be
efficiently computable classically.


Why it must fail anyway
-----------------------
Expressivity is not the binding constraint in this problem -- the ENCODING is.
This map's kernel is a function of the BITS of h, while the label is a function
of log_g(h). Nothing in the circuit inverts that map, so the kernel carries no
information about the concept and instead concentrates towards the identity as
n grows (the "exponential concentration" / quantum-overfitting failure mode:
K_ij ~ 0 off-diagonal, every training point becomes a support vector, perfect
train accuracy, chance test accuracy).

Including this control is what makes the headline claim falsifiable. If a
generic quantum kernel also separated the classes, the result would be about
quantum circuits in general; because it does not, the result is specifically
about aligning the feature map with the algebraic structure of the concept.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector


class LadderFeatureMap:
    """Hardware-efficient Rz/Rx/CZ ladder map; see module docstring."""

    name = "ladder"

    def __init__(self, n_qubits: int, reps: int = 2):
        if n_qubits < 2:
            raise ValueError("ladder map needs >= 2 qubits")
        if reps < 1:
            raise ValueError("reps must be >= 1")
        self.n_qubits = int(n_qubits)
        self.reps = int(reps)
        self.params = ParameterVector("x", self.n_qubits)
        self.template = self._build_template()

    # --------------------------------------------------------------- circuit
    def _build_template(self) -> QuantumCircuit:
        n = self.n_qubits
        qc = QuantumCircuit(n, name="U_Phi_ladder")
        half_pi = np.pi / 2.0
        for rep in range(self.reps):
            for j in range(n):
                qc.h(j)
            for j in range(n):
                qc.rz(2.0 * self.params[j], j)
            for j in range(n - 1):                      # entangling ZZ ladder
                qc.cx(j, j + 1)
                qc.rz(2.0 * (half_pi - self.params[j])
                      * (half_pi - self.params[j + 1]), j + 1)
                qc.cx(j, j + 1)
            for j in range(n):
                qc.rx(2.0 * self.params[j], j)
            for j in range(n - 1):
                qc.cz(j, j + 1)
            if rep < self.reps - 1:
                qc.barrier()
        return qc

    @staticmethod
    def angles(bits: Sequence[int]) -> np.ndarray:
        """
        Bit -> rotation angle: phi_j = (pi/2) * b_j.

        The pi/2 scale is chosen so that a bit flip is a non-trivial rotation.
        Using pi would make Rz(2 phi) = Rz(2 pi) a global phase, collapsing the
        single-qubit layer to the identity and silently weakening the control.
        """
        return (np.pi / 2.0) * np.asarray(bits, dtype=float)

    def circuit(self, bits: Sequence[int]) -> QuantumCircuit:
        return self.template.assign_parameters(
            dict(zip(self.params, self.angles(bits)))
        )

    # ------------------------------------------------------------ amplitudes
    def statevector(self, bits: Sequence[int]) -> np.ndarray:
        return np.asarray(Statevector(self.circuit(bits)).data, dtype=complex)

    def statevectors(self, bit_rows) -> np.ndarray:
        return np.stack([self.statevector(row) for row in bit_rows])

    def describe(self) -> Dict[str, object]:
        ops = self.template.count_ops()
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "reps": self.reps,
            "depth": int(self.template.depth()),
            "n_gates": int(sum(ops.values())),
            "gate_counts": {k: int(v) for k, v in ops.items()},
        }
