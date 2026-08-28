# Structure-Aligned Quantum Feature Maps on a Discrete-Logarithm Concept Class

### A reproducible, falsifiable finite-size study

**Artifact:** `dlog-qml` v1.0.0 · **Reference run:** seed 20260828, `p = 1019` · **All numbers measured, not projected.**

---

## 1. Executive summary

We construct a binary concept class over `Z_p*` whose label depends on the input only through its discrete logarithm, expose learners solely to the binary encoding of the group element, and measure which learners generalise.

Across 20 independent seeds, a quantum kernel built from Shor-style coset states reaches **98.9% ± 1.8%** held-out accuracy. Eleven competing models — including a tuned RBF-SVM, gradient boosting, an MLP, a polynomial-kernel SVM, and a strong *generic* quantum kernel — cluster at **49.0%–54.1%**, none significantly above chance. Mean separation: **+38.3 ± 6.2 percentage points**, worst case +24.0.

The result is accompanied by four falsification experiments, each designed to kill the claim:

| Control | Question it attacks | Result |
|---|---|---|
| Generic quantum kernel | "Is it just quantum?" | 0.503 ± 0.078 — **no** |
| Legendre/Euler attack | "Is the trapdoor real?" | breaks the naive variant (1.000), fails on ours (0.460) — **yes, real** |
| Hidden-x oracle | "Is the data just noise?" | 0.992 ± 0.015 — **not noise** |
| Coset kernel on parity | "Is the quantum kernel a universal cheat?" | 0.460 — **no** |

The scientific content is *not* "quantum beats classical". It is that **the encoding, not model capacity, is the binding constraint**, and that a feature map aligned with the algebraic structure that produced the encoding can recover geometry that generic methods — quantum or classical — cannot.

---

## 2. Motivation

Claims of quantum advantage in machine learning frequently compare a quantum kernel against classical baselines on a dataset where nobody established that the classical baselines *ought* to fail. When the quantum model wins, the result is unfalsifiable: it may reflect a genuine representational advantage, an accidentally favourable inductive bias, or an under-tuned baseline.

We invert the methodology. We start from a concept class with a *conditional hardness guarantee* from the literature, then spend most of the engineering effort trying to break our own result before reporting it.

---

## 3. Problem statement

**Can a quantum feature map whose inductive bias is aligned with the algebraic structure of a discrete-logarithm concept class produce a substantial learning separation over learners that see only the computational representation?**

Sub-questions, each answered by a control:

1. Would *any* sufficiently expressive quantum feature map do? (§11.2)
2. Is the concept genuinely hidden, or just awkwardly encoded? (§11.3)
3. Is the concept learnable at all? (§11.4)
4. Does the aligned kernel work because it matches *this* structure, or is it generically powerful? (§11.5)

---

## 4. Mathematical construction

### 4.1 Group

`p = 2q + 1` safe prime; `g` a generator of the full group `Z_p*`, order `N = p - 1 = 2q`.

Since the divisors of `2q` are `{1, 2, q, 2q}`, testing `g² ≢ 1` and `g^q ≢ 1` proves order `2q`. (Valid only for safe primes; `find_generator` raises otherwise.)

For `n = 10`: `p = 1019 = 2·509 + 1`, `g = 2`. Note the largest prime below `2¹⁰` is 1021, which is **not** safe — `510 = 2·3·5·17`.

Primality uses deterministic Miller–Rabin with the twelve-base witness set, an exact proof for all `n < 3.317 × 10²⁴`. Verified against trial division for all `n < 2000` and against three Carmichael numbers (561, 1105, 1729) in `tests/test_group.py`.

### 4.2 Secret

```
s = int.from_bytes(SHA-256("dlog-concept-class|p=<p>|g=<g>|seed=<seed>"), "big") mod (p-1)
```

Big-endian, full 32-byte digest, domain-separated by group parameters so a size sweep never reuses a secret across primes. Reference run: `s = 666`.

### 4.3 Concept and exact balance

```
y(x) = +1  if (x - s) mod N ∈ [0, (p-3)/2];   -1 otherwise
```

`(p-3)/2 = (2q-2)/2 = q-1`, so the positive set is `{0,…,q-1}` (exactly `q` integers) and its complement `{q,…,2q-1}` (also exactly `q`). **The concept is exactly balanced over the entire group.** Boundaries, wraparound, and equivalence of the `< q` fast path with the closed-interval definition are tested at all 1018 exponents.

### 4.4 Observation model

The learner receives the LSB-first fixed-width bit decomposition of `h = g^x mod p`, width `p.bit_length()`. Nothing else.

### 4.5 The coset feature map

$$|\Phi(h)\rangle = \frac{1}{\sqrt{M}}\sum_{i=0}^{M-1} |h\,g^{i} \bmod p\rangle,\qquad M = 2^{m}$$

Constructed by forward modular multiplication only — the easy direction. **No discrete logarithm is computed during feature-map construction**, which matters: the map must be buildable by someone who does not hold the trapdoor.

### 4.6 Induced kernel — derivation

Let `a = log_g h`, `a' = log_g h'`, `δ = (a' - a) mod N`. The supports are length-`M` arithmetic progressions on `Z_N` starting at `a` and `a'`. Their intersection has size

$$\Omega(\delta) = \max(0, M-\delta) + \max(0, M-(N-\delta))$$

— forward overlap plus wraparound overlap. When `2M ≤ N` the two terms cannot both be positive, so nothing is double-counted. The states are uniform with real amplitude `1/√M`, hence

$$\langle\Phi(h)|\Phi(h')\rangle = \frac{\Omega(\delta)}{M}, \qquad K(h,h') = \left(\frac{\Omega(\delta)}{M}\right)^{2}.$$

**A circular triangular kernel in discrete-log coordinates.** The state performs the hard change of coordinates implicitly; the SVM then solves one-dimensional interval membership.

Verified three ways: against the compute–uncompute circuit (3.0e-13), against a directly counted set intersection (`tests/test_kernel.py`), and against the closed form over a 60×60 block (4.4e-16).

---

## 5. Dataset generation

Exponents drawn **without replacement** from `Z_N` via a uniform permutation filled into per-class buckets. Exact 50/50 balance; splits share no exponent, therefore no group element.

`DlogDataset.verify()` asserts on every run: `h = g^x mod p` recomputed from scratch; labels recomputed from the rule; bit decomposition round-trips; exact per-split balance; split disjointness; no duplicate exponents; and three independent baby-step giant-step probes that bypass the oracle table entirely (so a corrupt table cannot pass silently).

**Public/oracle boundary.** `features()` returns only bits. The exponent is reachable only via `oracle_exponents()`, with three documented legitimate call sites (self-audit, hidden-x control, plotting). `assert_no_leakage()` runs before every primary learner: feature width, strict binarity, no column equal to `x` or its low bits, and the matrix *is* the bit decomposition of `h`. A test deliberately poisons a column and confirms the guard fires.

---

## 6. Classical baselines

| Model | Configuration |
|---|---|
| RBF-SVM | `C=1.0, gamma='scale'`, standardised |
| RandomForest | 100 trees |
| MLP | `(64, 32)`, ReLU, Adam, `max_iter=3000`, standardised |
| Linear-SVM | `C=1.0`, standardised |
| Poly-SVM | degree 3, `C=1.0`, standardised |
| ExtraTrees | 100 trees |
| GradientBoosting | sklearn defaults |
| kNN | `k=5`, standardised |
| RBF-SVM (tuned) | 5-fold CV over `C ∈ {0.1,1,10,100,1000} × gamma ∈ {scale,0.01,0.1,1,10}` |

**No test-set tuning anywhere.** The tuned SVM selected `C=0.1, gamma=0.01` at CV accuracy 0.555 — itself barely above chance, which is the informative part.

---

## 7. Quantum feature maps

### 7.1 Generic control (`LadderFeatureMap`)

Per repetition, with `φ_j = (π/2)·b_j`: `H` on all qubits; `Rz(2φ_j)`; a `CX(j,j+1) · Rz(2(π/2-φ_j)(π/2-φ_{j+1})) · CX(j,j+1)` `ZZ` ladder; `Rx(2φ_j)`; a `CZ` ladder. Two reps. Measured: 10 qubits, depth 64, 133 gates (`rz`×38, `cx`×36, `h`×20, `rx`×20, `cz`×18).

The `π/2` angle scale is deliberate. With `φ = π·b`, `Rz(2φ) = Rz(2π)` is a global phase and the single-qubit layer silently collapses to the identity — a way to accidentally weaken the control into a straw man. Tested explicitly.

### 7.2 Structure-aligned map (`CosetStateFeatureMap`)

As §4.5. Default `M ≈ (p-1)/8` → `m = 7`, `M = 128` (12.6% of the group), satisfying `2M ≤ N`.

### 7.3 Like-for-like guarantee

Both maps use identical splits, identical Gram machinery, and identical `SVC(kernel='precomputed')` selection over the same `C` grid. **They differ only in the feature map.**

---

## 8. Kernel construction and verification

`K_ij = |⟨0ⁿ|U†(x_j)U(x_i)|0ⁿ⟩|²` via exact statevector simulation: simulate each state once, then one BLAS product — `O(N²·2ⁿ)` instead of `O(N²)` circuit executions.

| Audit | Max abs. error |
|---|---|
| Compute–uncompute, generic map | 1.7e-18 |
| Compute–uncompute, coset map | 3.0e-13 |
| Closed-form triangular overlap | 4.4e-16 |

Algebraic properties checked every run: symmetric, unit diagonal, range `[0,1]`, positive semi-definite (min eigenvalue 8.96e-03 for the coset kernel).

---

## 9. Methodology

Two-sided **exact binomial** test (`H0`: accuracy = 0.5); **Wilson** 95% score intervals; **exact McNemar** for paired comparison (`H0`: equal error rates on the same items); **paired percentile bootstrap** (10 000 resamples) for the accuracy difference. Binomial and McNemar are verified against `scipy.stats.binomtest`.

Why paired tests: two separate one-sample tests do not answer "is A better than B". McNemar does.

Determinism: one seed fixes the secret, the sampling, and every `random_state`.

---

## 10. Results

### 10.1 Reference run

| Model | Test acc. | 95% CI | p | Verdict |
|---|---|---|---|---|
| *Hidden-x oracle* ‡ | *0.960* | *[0.865, 0.989]* | *2.3e-12* | *control* |
| **Quantum coset kernel** | **0.940** | **[0.838, 0.979]** | **3.7e-11** | **learns** |
| Linear-SVM | 0.620 | [0.482, 0.741] | 0.119 | n.s. |
| RBF-SVM (tuned) | 0.600 | [0.462, 0.724] | 0.203 | n.s. |
| Quantum generic kernel | 0.600 | [0.462, 0.724] | 0.203 | n.s. |
| Poly-SVM (d=3) | 0.580 | [0.442, 0.706] | 0.322 | n.s. |
| MLP | 0.520 | [0.385, 0.652] | 0.888 | n.s. |
| RBF-SVM | 0.500 | [0.366, 0.634] | 1.000 | n.s. |
| ExtraTrees | 0.500 | [0.366, 0.634] | 1.000 | n.s. |
| kNN (k=5) | 0.480 | [0.348, 0.615] | 0.888 | n.s. |
| GradientBoosting | 0.460 | [0.330, 0.596] | 0.672 | n.s. |
| Legendre attack | 0.460 | [0.330, 0.596] | 0.672 | n.s. |
| RandomForest | 0.420 | [0.294, 0.558] | 0.322 | n.s. |

Coset kernel: train 1.000, test AUC 0.9968, `C=1.0`, CV 0.990, 195/200 support vectors.

**No competing model clears `p < 0.01`.**

### 10.2 Paired comparison

Coset vs. Linear-SVM: **McNemar exact p = 8.6e-4** (19 items coset-right/other-wrong, 3 reverse); bootstrap 95% CI on the difference **[+0.160, +0.480]**.

### 10.3 Multi-seed (20 seeds)

Coset **0.989 ± 0.018** (median 1.000, min 0.940). Every competitor between 0.490 and 0.541. Generic quantum kernel **0.503 ± 0.078**. **Separation +0.383 ± 0.062, min +0.240, max +0.520, 20/20 completed.**

Two definitions are possible and they differ, so the quoted one is stated explicitly. *Per-seed*: on each seed subtract the best competitor **on that seed**, then average — this is the +0.383 we quote, and it is conservative because the comparison target is a per-seed maximum. *Pooled*: coset mean minus the mean of the single best-performing competitor across all seeds (Poly-SVM), which gives +0.448. The coset kernel is ahead of the best competitor on 20 of 20 seeds. `dlog_qml.audit` recomputes both from the raw per-seed accuracies and the notebook displays the comparison.

### 10.4 Kernel structure

| Diagnostic | Coset | Generic |
|---|---|---|
| Mean off-diagonal `K` | 0.0832 | **0.0032** |
| Off-diagonal s.d. | 0.2064 | 0.0155 |
| Effective rank (of 200) | 30.1 | **195.3** |
| Top-10 eigenvalue mass | 0.680 | **0.079** |
| Condition number | 2361 | 3.44 |
| Kernel-target alignment (centred) | **0.3818** | 0.0704 |
| `K` same-class (mean) | 0.1614 | 0.0033 |
| `K` different-class (mean) | 0.0058 | 0.0031 |
| **Class separation gap** | **+0.1556** | **+0.0002** |
| Train accuracy | 1.000 | 1.000 |
| Test accuracy | 0.940 | 0.600 |

**Kernel-target alignment** is $A(K,Y)=\langle K,Y
angle_F/(\|K\|_F\|Y\|_F)$ with
$Y=yy^{\mathsf T}$ the ideal kernel, computed on the centred kernel $HKH$ so a constant
offset cannot inflate it. It measures how much of the kernel geometry points along the
labels at all, independent of any classifier.

**The mechanism, stated precisely:** the generic feature map exhibits near-orthogonal
concentration, producing a high-effective-rank, spectrally flat kernel with negligible
class structure — same-class and different-class pairs sit at essentially the same
similarity (gap $+0.0002$). There is nothing for a classifier to exploit, so it memorises
perfectly and generalises at chance. This is a mechanistic explanation, not merely an
observation that the score was low.

The generic map spreads 200 points into ~195 near-orthogonal directions: exponential concentration, perfect memorisation, chance generalisation.

### 10.5 Measured size sweep

| n | p | Train/Test | Coset | Generic | Best classical | Sim. time | Memory |
|---|---|---|---|---|---|---|---|
| 6 | 59 | 46/12 | 1.000 | 0.600 | 0.500 | 0.001 s | 0.06 MB |
| 7 | 107 | 84/22 | 0.950 | 0.450 | 0.545 | 0.001 s | 0.21 MB |
| 8 | 227 | 180/46 | 0.977 | 0.386 | 0.543 | 0.006 s | 0.88 MB |
| 9 | 503 | 200/50 | 1.000 | 0.460 | 0.560 | 0.015 s | 1.95 MB |
| 10 | 1019 | 200/50 | 0.940 | 0.600 | 0.620 | 0.031 s | 3.91 MB |

Small-`n` rows have wide intervals (12 test points at `n = 6`); they are measurements, not a demonstrated trend.

---

## 11. Ablation and falsification

### 11.1 Coset window ablation

| m | M | Coset acc. | Off-diag K |
|---|---|---|---|
| 1 | 2 | 0.740 | 0.0005 |
| 2 | 4 | 0.880 | 0.0017 |
| 3 | 8 | 0.980 | 0.0042 |
| 4 | 16 | 0.960 | 0.0092 |
| 5 | 32 | 0.940 | 0.0198 |
| 6 | 64 | 0.940 | 0.0411 |
| **7** | **128** | **0.940** | **0.0832** ← default |
| 8 | 256 | 0.940 | 0.1672 |

`m = 2…8` all beat every classical baseline. The default is a fixed heuristic, is not the maximum, and was not tuned. Only `m = 1` degrades materially, matching the prediction that a 2-element window makes nearly all pairs orthogonal.

### 11.2 Generic quantum control

**0.503 ± 0.078.** Chance. Expressivity is not the binding constraint; the encoding is. The generic map's kernel is a function of the bits of `h`, the label is a function of `log_g h`, and nothing in the circuit inverts that map. **This is what makes the headline falsifiable.**

### 11.3 Number-theoretic attack

The naive labelling `y = (-1)^(log_g h mod 2)` has **no trapdoor**. The LSB of a discrete log in `Z_p*` is the quadratic-residuosity bit, recoverable in `O(log p)` by Euler's criterion:

```
h^((p-1)/2) mod p == 1  ⟺  h is a QR  ⟺  log_g h is even
```

`tests/test_labels.py` verifies this equivalence for **every** element of `Z_1019*`.

| Labelling | Attack accuracy |
|---|---|
| `parity` | **1.000** — broken |
| `half_interval` | 0.460 — chance |

The hardcore bits of a discrete log are the high-order ones (Blum–Micali); the low-order bit is free. Hence the half-interval concept.

### 11.4 Hidden-x oracle

RandomForest on `x` directly: **0.992 ± 0.015**. Not a valid model — it holds the trapdoor. It establishes that the labels are perfectly learnable *given the coordinate*, so the ~50% results are about the **encoding**, not label noise or a broken generator.

### 11.5 Coset kernel on a mismatched concept

On `parity`, the coset kernel drops to **0.460** — indistinguishable from the classical baselines. Its triangular kernel is *local* on the dlog circle; parity alternates every step; a local kernel is maximally uninformative about it.

**The quantum kernel is not a universal cheat.** It wins when its inductive bias matches the concept's structure. This is the strongest single piece of evidence that the pipeline measures something real.

---

## 12. Scaling analysis

Measured behaviour in §10.5. To scale up, three changes:

1. **Drop the `O(p)` oracle tables.** Infeasible past ~2²⁴. Generate forward-only: sample `x`, compute `h = pow(g,x,p)`, label from `x`.
2. **Replace `StatePreparation` with Shor-style period finding.** The essential change: `Θ(2ⁿ)` → `O(n³)` gates on `~2n+3` qubits, via superposition over `i`, modular exponentiation, and an inverse QFT. No step evaluates `log_g h`.
3. **Estimate kernel entries from shots.** `O(ε⁻²)` shots per entry for additive error `ε` — a *per-entry sampling* statement, not a runtime guarantee, and `O(N²)` entries are needed.

The number theory itself requires no change: it is arbitrary-precision already.

**Theoretical roadmap (projection, not measured):** dense-simulation ceiling ~24 qubits (~268 MB per statevector); 64+ qubits requires fault tolerance; cryptographic relevance at 2048-bit `p`, where GNFS costs `exp((1.92+o(1))(ln p)^{1/3}(ln ln p)^{2/3})`.

---

## 13. Hardware considerations

**No hardware was used and no hardware result is reported.**

A physical implementation would estimate each `K_ij` by compute–uncompute: prepare `U(x_i)`, apply `U†(x_j)`, measure, estimate `P(|0ⁿ⟩)`. Requirements:

- `O(ε⁻²)` shots per entry for additive error `ε` — again, a per-entry sampling statement.
- The coset kernel's useful signal sits at off-diagonals of order `10⁻¹`, comparatively forgiving. The generic map's `0.0032` off-diagonals would be swamped by realistic shot noise — an additional, independent reason it is unusable here.
- Modular exponentiation in superposition at cryptographic size is among the most expensive known quantum primitives; Shor-class resource estimates at 2048 bits run to millions of physical qubits.

---

## 14. Limitations

Full treatment in [`LIMITATIONS.md`](LIMITATIONS.md). The load-bearing ones: `p = 1019` is classically trivial (we run BSGS on it ourselves); this is a classical simulation with no hardware; dense state preparation is `Θ(2ⁿ)` and is *not* the efficient algorithm; an accuracy separation is not a complexity-theoretic speedup; `n_test = 50` gives a 7.07-point standard error; eleven baselines failing is evidence, not proof; and the asymptotic claim rests on the unproven Discrete Logarithm Assumption.

---

## 15. Reproducibility

```bash
pip install -r requirements.txt
python -m pytest                      # 187 tests
python scripts/demo.py                # ~30 s
python scripts/run_experiment.py      # ~5 min, full study
```

Config in `configs/default.yaml`; every run writes `results/raw/config_used.json` alongside its outputs. Environment recorded in every result file. See [`EXPERIMENT_PROTOCOL.md`](EXPERIMENT_PROTOCOL.md).

Verified on Windows 11, Python 3.13.14, numpy 2.5.2, pandas 2.3.3, scikit-learn 1.9.0, qiskit 2.5.1, qiskit-machine-learning 0.9.0, scipy 1.17.1, matplotlib 3.10.3.

---

## 16. Conclusion

We demonstrate that a quantum feature map explicitly aligned with the algebraic geometry of a discrete-logarithm concept class produces a substantial empirical learning separation — **+38.3 ± 6.2 percentage points over 20 seeds** — from eleven conventional classical learners *and* from a strong generic quantum feature map. The result is supported by number-theoretic, leakage, and quantum negative controls, by paired significance testing, and by an ablation showing it does not hinge on a tuned hyperparameter.

The small simulated instance is **not** a cryptographic-scale demonstration. It is an experimentally auditable realisation of the learning-theoretic construction. The asymptotic computational claim depends on the Discrete Logarithm Assumption and on an efficient fault-tolerant implementation, neither of which this work establishes.

The transferable lesson is the one the generic-quantum control makes unavoidable:

> **Quantum advantage is not automatic. What matters is whether the quantum representation matches the mathematical structure of the learning problem — and a feature map that matches nothing performs exactly like a classical model that matches nothing.**

---

## 17. References

1. Y. Liu, S. Arunachalam, K. Temme, *A rigorous and robust quantum speed-up in supervised machine learning*, Nature Physics **17**, 1013–1017 (2021). arXiv:2010.02174.
2. V. Havlíček et al., *Supervised learning with quantum-enhanced feature spaces*, Nature **567**, 209–212 (2019). arXiv:1804.11326.
3. P. W. Shor, *Polynomial-time algorithms for prime factorization and discrete logarithms on a quantum computer*, SIAM J. Comput. **26**(5), 1484–1509 (1997). arXiv:quant-ph/9508027.
4. M. Blum, S. Micali, *How to Generate Cryptographically Strong Sequences of Pseudorandom Bits*, SIAM J. Comput. **13**(4), 850–864 (1984). doi:10.1137/0213053.
5. D. Shanks, *Class number, a theory of factorization, and genera*, Proc. Symp. Pure Math. **20**, 415–440 (1971).
6. E. B. Wilson, *Probable inference, the law of succession, and statistical inference*, J. Am. Stat. Assoc. **22**, 209–212 (1927).
7. Q. McNemar, *Note on the sampling error of the difference between correlated proportions or percentages*, Psychometrika **12**(2), 153–157 (1947).
8. Qiskit contributors, *Qiskit: An Open-source Framework for Quantum Computing*. <https://www.ibm.com/quantum/qiskit>
