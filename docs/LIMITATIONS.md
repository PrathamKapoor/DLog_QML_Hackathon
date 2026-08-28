# Limitations

This document exists because the fastest way to make this project worthless would be to overstate it. Everything below is a real constraint on what the measurements support. Read it before quoting any number.

---

## 1. The prime is tiny and classically trivial

`p = 1019` is a 10-bit prime. The discrete logarithm here is solvable by brute force in microseconds.

We do not merely concede this — **we exploit it**. `baby_step_giant_step` in `src/dlog_qml/group.py` solves the discrete log in `O(√p)` and is invoked as part of the dataset self-audit on every run, to independently confirm the oracle tables. A cryptographically meaningful instance would need `|p| ≥ 2048` bits (or a ≥256-bit elliptic-curve group).

**Consequence:** nothing in this repository is evidence about cryptographic hardness. The demonstration is about *learning geometry* — what a fixed-similarity learner can and cannot recover from a given encoding — not about breaking anything.

---

## 2. It is a classical simulation

Every quantum quantity here comes from exact statevector simulation on a CPU. There is **no quantum hardware result anywhere in this repository**, and none is claimed.

Because the simulation is exact, results carry no shot noise. That is a *convenience* for isolating the geometry, not a fidelity claim about any physical device.

---

## 3. Dense state preparation is not an efficient quantum algorithm

This is the single most important caveat.

`CosetStateFeatureMap.statevector()` builds `|Φ(h)⟩` as an explicit `2ⁿ`-dimensional amplitude vector, and `circuit()` compiles it with Qiskit's `StatePreparation`. **That synthesis is `Θ(2ⁿ)`.** It is a simulation tool for studying the feature map, and it is emphatically not the algorithm the theory is about.

The intended implementation prepares the same state in `O(n³)` gates on `~2n+3` qubits via Shor-style period finding: a uniform superposition over `i ∈ [0, M)`, modular exponentiation `|i⟩|h·g^i mod p⟩`, then an inverse QFT to uncompute the index register. No step of that routine evaluates `log_g h`.

**Anyone reading "quantum kernel gets 94%" must also read this paragraph.** The efficiency claim rests on a circuit we did not build and could not run. We implemented and verified the *kernel*, not the *fault-tolerant preparation*.

---

## 4. An accuracy separation is not a complexity-theoretic speedup

We measure that certain learners generalise and others do not, on a finite sample, at one problem size. That is an empirical learning result.

It is **not** a proof that no classical polynomial-time algorithm can learn this concept. That statement is a theorem of Liu, Arunachalam & Temme (2021), conditional on the Discrete Logarithm Assumption, and it is theirs, not ours.

A small p-value here rejects exactly one null hypothesis: "this classifier performs at chance on this test set." It says nothing about complexity classes, and we never use it as if it did.

---

## 5. Finite dataset size

The reference run has `n_test = 50`. The binomial standard error at chance is `1/(2√50) = 7.07` percentage points.

Practical consequences we act on throughout:

- A raw score of 0.60 is **1.4σ** from chance and is not evidence of learning. Every reported accuracy carries a Wilson 95% CI and an exact binomial p-value so this cannot be misread.
- The multi-seed sweep (20 seeds) exists because a single 50-item test set is a weak instrument. Trust the `0.989 ± 0.018` figure over any single run.
- Small-`n` rows in the size sweep (e.g. `n = 6`, 12 test points) have very wide intervals. They are reported as measurements, but a single small-`n` point is not evidence of a trend.

---

## 6. Dependence on the chosen baselines

We report eleven competing models (RBF-SVM, tuned RBF-SVM, Linear-SVM, Poly-SVM, RandomForest, ExtraTrees, GradientBoosting, MLP, kNN, the generic quantum kernel, and the Legendre attack). None exceeds chance at `p < 0.01`.

That is evidence, not proof. **We cannot rule out that some classical method we did not try does better.** In particular:

- We did not attempt a learner given side information about the group structure.
- We did not attempt algebraic attacks beyond quadratic residuosity.
- We did not run a large neural architecture search. Given that the target is a one-way permutation composed with an interval indicator, we do not expect scale to help — but we did not test it, and we do not claim it.

The honest framing: *these* learners fail, and the theory says all polynomial-time classical learners should.

---

## 7. The asymptotic claim rests on an unproven assumption

The Discrete Logarithm Assumption is a **conjecture**. It is widely believed and underpins deployed cryptography, but it is not a theorem. If DLP turns out to be classically easy, the asymptotic separation evaporates — though the *finite-size* measurements in this repository would remain exactly as they are, since they never depended on the assumption.

---

## 8. Shot noise and measurement overhead on real hardware

On a physical device the Gram matrix would be estimated entry-by-entry from compute–uncompute sampling rather than computed exactly. Additive error `ε` on a single entry requires `O(ε⁻²)` shots.

Two qualifications we insist on:

- That is a **measurement-sampling statement about one entry**, not a guarantee about total algorithmic runtime. An `N × N` kernel needs `O(N²)` entries, and the constant factors, circuit compilation, and error correction are all outside it.
- The coset kernel's useful signal lives in off-diagonal values of order `10⁻¹`, which is comparatively forgiving. The generic map's off-diagonals average `0.0032`, which would be swamped by realistic shot noise — a separate reason it is unusable here, beyond the ones we measured.

---

## 9. Fault-tolerant state preparation is a serious engineering problem

Modular exponentiation in superposition over a cryptographically sized group is among the most expensive primitives in quantum computing. Resource estimates for Shor-class arithmetic at 2048-bit scale run to millions of physical qubits under realistic error rates. Nothing in this repository lightens that.

---

## 10. The kernel is local, so the method is concept-specific

Demonstrated in §4.4 of the README: on the `parity` concept the coset kernel drops to **0.460**, no better than the classical baselines.

This is a limitation and a strength at once. The coset kernel is not a general-purpose "quantum advantage machine"; it is a triangular kernel on the dlog circle, and it helps exactly when the target concept is smooth in that coordinate. Aligning a feature map with structure requires *knowing the structure*.

---

## 11. Window choice is a hyperparameter

The coset window `M = 2^m` is set by a fixed heuristic (`M ≈ (p-1)/8`), not tuned on the test set. The ablation shows `m = 2…8` all beat every classical baseline, so the result is not knife-edge — but `m = 1` degrades to 0.740, so the parameter is not irrelevant either. At cryptographic scale the right choice of `M` is a design question we have not solved.

---

## 12. Scope of the extra baselines and sweeps

- The multi-seed sweep varies the secret and the sample draw at fixed `p`. It does **not** vary `p` — that is the (single-seed) size sweep, whose per-point uncertainty is correspondingly larger.
- The size sweep runs with `extra_baselines=False` for runtime, so its "best classical" column is over the three core baselines plus the tuned SVM, not all eleven.
- The window ablation is single-seed.

---

## Summary table

| Claim | Status |
|---|---|
| Learning separation on a finite instance, 20 seeds | **measured** |
| Generic quantum kernel also fails | **measured** |
| Parity concept is classically breakable | **measured** |
| Labels are learnable given the hidden coordinate | **measured** |
| Coset kernel fails on a mismatched concept | **measured** |
| Kernel equals the circuit definition to ~1e-13 | **measured** |
| Asymptotic classical hardness | **assumed** (DLog Assumption) |
| Asymptotic quantum separation | **theorem of LAT2021**, not reproduced here |
| `O(n³)` state preparation | **cited**, not implemented |
| Cryptographic-scale advantage | **not claimed** |
| Quantum hardware execution | **not performed** |
