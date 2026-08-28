# Provable Quantum Advantage on the Discrete-Logarithm Concept Class

**A falsifiable end-to-end demonstration that a Shor-style quantum kernel learns a concept class no classical learner — and no generic quantum learner — can touch.**

---

## 1. Abstract

Most "quantum advantage in ML" demos compare a quantum kernel against a classical baseline on a dataset where nobody proved the classical baseline *should* fail. This project does the opposite: it starts from a concept class with a **cryptographic hardness guarantee** — the discrete-logarithm class of Liu, Arunachalam & Temme (*Nature Physics* **17**, 1013, 2021; arXiv:2010.02174) — and then tries as hard as possible to break it classically before claiming any separation.

We generate a balanced binary dataset in the cyclic group `Z_p*` whose label depends on the input **only through its discrete logarithm**, and expose the learner solely to the bit-decomposition of the group element `h = g^x mod p`. We then run four classical learners, a polynomial-time number-theoretic attack, and a generic hardware-efficient quantum kernel against it. All six land at chance. Only the Liu–Arunachalam–Temme coset-state kernel separates.

**Headline result** (`p = 1019`, 10 qubits, 200 train / 50 test, exact statevector simulation):

| Learner | Test accuracy | 95% CI | p vs. chance |
|---|---|---|---|
| Best of 4 classical models | 0.600 | [0.462, 0.724] | 0.20 — n.s. |
| Legendre-symbol attack (poly-time) | 0.460 | [0.330, 0.596] | 0.67 — n.s. |
| Generic quantum kernel (Rz/Rx/CZ) | 0.600 | [0.462, 0.724] | 0.20 — n.s. |
| **Coset-state quantum kernel** | **0.940** | **[0.838, 0.979]** | **3.7 × 10⁻¹¹** |

Separation: **+34.0 percentage points**, and *no* competing model is above chance at `p < 0.01`.

**What makes this claim falsifiable rather than decorative** — three controls that a headline-chasing submission would omit:

1. **A generic quantum kernel is in the table, and it fails.** Being quantum is not what wins. The number-theoretic *encoding* is.
2. **A poly-time classical attack is in the table.** It breaks the naive labelling (below) and fails on ours, so classical failure is demonstrated, not assumed.
3. **A leakage control is in the table.** A RandomForest given the hidden exponent scores 0.96, proving the labels are perfectly learnable *once the discrete log is known* — so the 50% scores are a hardness result about the encoding, not label noise.

---

## 2. Repository layout

```
project_root/
├── generate_and_evaluate.py            # complete end-to-end pipeline (single file)
├── requirements.txt                    # pinned dependencies + reference environment
├── README.md                           # this document
│
├── dlog_quantum_advantage_dataset.csv  # [generated] 250 x 15 dataset
└── results/                            # [generated]
    ├── metrics.json                    # every number in this README, machine-readable
    ├── kernel_matrices.png             # coset vs. ladder Gram matrices
    └── accuracy_comparison.png         # the summary bar chart
```

---

## 3. Architecture

```
                        ┌───────────────────────────────────────────────┐
                        │  TRAPDOOR  (data generator only — never       │
                        │  exposed to any learner)                      │
                        │    p = 2q+1 safe prime, g generates Z_p*      │
                        │    secret shift  s = SHA-256(seed) mod (p-1)  │
                        └───────────────────┬───────────────────────────┘
                                            │
   x ~ U(Z_{p-1})  ────────────────────────►│  h = g^x mod p     (easy direction)
   hidden exponent                          │  x = log_g(h)      (HARD direction)
                                            ▼
                            y = +1  if (x - s) mod (p-1) < (p-1)/2
                                -1  otherwise
                                            │
              ┌─────────────────────────────┴─────────────────────────────┐
              │      dlog_quantum_advantage_dataset.csv                    │
              │      learner sees:  binary_feature_0 .. binary_feature_9    │
              │      learner NEVER sees:  raw_input_x   ( = log_g h )       │
              └─────────────────────────────┬─────────────────────────────┘
                                            │
        ┌───────────────────────────────────┼───────────────────────────────────┐
        │                                   │                                   │
        ▼                                   ▼                                   ▼
┌───────────────────┐          ┌─────────────────────────┐        ┌──────────────────────────┐
│ CLASSICAL         │          │ QUANTUM — LADDER        │        │ QUANTUM — COSET          │
│                   │          │ (negative control)      │        │ (Liu/Arunachalam/Temme)  │
│ RBF-SVM           │          │                         │        │                          │
│ RandomForest      │          │  H · Rz(2φ) · [CX·Rz·CX]│        │ |Φ(h)⟩ =                 │
│ MLP (64,32)       │          │    · Rx(2φ) · CZ-ladder │        │   M^-½ Σ_{i<M} |h·g^i⟩   │
│ RBF-SVM (tuned)   │          │  φ = (π/2)·bit          │        │                          │
│ Legendre attack   │          │  ↳ acts on BITS of h    │        │  ↳ acts on log_g(h)      │
└─────────┬─────────┘          └────────────┬────────────┘        └────────────┬─────────────┘
          │                                 │                                  │
          │                    K_ij = |⟨0ⁿ|U†(x_j)U(x_i)|0ⁿ⟩|²  (exact statevector)
          │                                 │                                  │
          ▼                                 ▼                                  ▼
    accuracy ≈ 0.50               SVC(kernel='precomputed')          SVC(kernel='precomputed')
                                       ≈ 0.50                              ≈ 0.94
          │                                 │                                  │
          └─────────────────────────────────┴──────────────────┬───────────────┘
                                                               ▼
                                        exact binomial test vs. chance + Wilson CI
                                                    → results/metrics.json
```

---

## 4. Quick start

```bash
python -m pip install -r requirements.txt
python generate_and_evaluate.py
```

Runs in **~35 s** on a laptop CPU. No GPU, no quantum hardware, no cloud account.

Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--qubits N` | `10` | Register width; `p` = largest safe prime `< 2^N` |
| `--train / --test` | `200 / 50` | Sample counts (must be even; exact 50/50 balance) |
| `--labeling` | `half_interval` | `half_interval` = hard (LAT2021); `parity` = **broken control** |
| `--window-bits m` | `≈ log2((p-1)/8)` | Coset window `M = 2^m` |
| `--qml-crosscheck` | off | Re-derive the ladder kernel with `qiskit-machine-learning` |
| `--crypto-random` | off | Draw the secret from `secrets` instead of `SHA-256(seed)` |
| `--no-plots`, `--no-tune` | off | Skip figures / the tuned-SVM grid search |

Reproduce the negative controls:

```bash
python generate_and_evaluate.py --labeling parity      # trapdoor collapses (see §6.3)
python generate_and_evaluate.py --qubits 6 --train 28 --test 8   # 6-qubit smoke test
```

---

## 5. Dataset specification

`dlog_quantum_advantage_dataset.csv` — 250 rows × 15 columns.

| Column | Type | Description |
|---|---|---|
| `sample_id` | int | Index `0 … N-1` |
| `raw_input_x` | int | **The hidden exponent** `x = log_g(h)`. See warning below. |
| `group_element_h` | int | `g^x mod p` — the observed datum |
| `binary_feature_0 … _9` | int ∈ {0,1} | Fixed-width bit decomposition of `h`, **LSB-first** (`binary_feature_0` is the 1s bit) |
| `label` | int ∈ {−1,+1} | Target |
| `split` | str | `train` (200) or `test` (50) |

> ⚠️ **`raw_input_x` is the trapdoor, not a feature.** It is shipped for auditability — so a judge can independently verify `h = g^x mod p` and recompute every label — but the label is a deterministic function of it. Any model trained on it scores ~100%. The script enforces this with an assertion (`HIDDEN_COLUMNS`) and every learner is restricted to `binary_feature_*`. **If you build on this CSV, drop `raw_input_x` first.**

Guarantees asserted at generation time (`verify_dataset`), all of which abort the run on failure:

- `h == pow(g, x, p)` for every row, and the log table inverts the exp table
- every `label` matches the concept function re-evaluated from scratch
- bit decomposition recomposes exactly to `h`
- **exact** 100/100 and 25/25 class balance
- **train and test exponents are disjoint** — sampling is without replacement over `Z_{p-1}`, so the reported gap can never be memorisation
- one group element is spot-checked against an independent baby-step giant-step solve

Reference parameters (default run): `p = 1019 = 2·509 + 1`, `q = 509`, `g = 2` (order 1018), `s = 666`.

---

## 6. Mathematical proof summary

### 6.1 Why the construction is hard classically

The label is
$$y(h) \;=\; \operatorname{sign}\!\Big[\mathbb{1}\big\{(\log_g h - s) \bmod (p-1) \in [0, \tfrac{p-3}{2}]\big\} - \tfrac12\Big].$$

It depends on `h` **only through `log_g h`**. The learner observes the bits of `h`. So any learner that beats chance induces a distinguisher for the half-interval predicate of the discrete logarithm.

- **Concept-class capacity is not the obstacle.** In `log_g` coordinates the concept is a single arc — VC dimension 2. It is trivially PAC-learnable *given the right coordinates*. Our leakage control confirms this empirically (0.96 accuracy on `raw_input_x`).
- **The obstacle is the coordinate change.** Recovering `log_g h` requires solving the DLP. The best known classical algorithm for prime fields, the General Number Field Sieve, costs
  $$\exp\!\Big((1.92 + o(1)) (\ln p)^{1/3} (\ln \ln p)^{2/3}\Big),$$
  sub-exponential but super-polynomial.
- **Gradient descent has nothing to descend.** For an unknown shift `s`, the labelling is a shifted half-interval indicator composed with a one-way permutation. Every low-degree Fourier coefficient of `y` as a function of the bits of `h` is negligible, so the empirical-risk gradient at initialisation is statistically indistinguishable from noise. Kernel methods with any *fixed, input-independent* similarity measure (RBF, polynomial, tree ensembles' partition kernel) inherit the same barrier: their kernel is a function of Hamming/Euclidean geometry on the bits of `h`, and that geometry is uncorrelated with `log_g h`.
- **Formally** (LAT2021, Thm 1): under the Discrete Logarithm Assumption, every classical polynomial-time learner achieves accuracy `≤ 1/2 + 1/poly(n)` on this class, while the quantum kernel method achieves `≥ 0.99` with high probability.

### 6.2 Why the coset-state kernel wins

The feature map is
$$|\Phi(h)\rangle \;=\; \frac{1}{\sqrt{M}}\sum_{i=0}^{M-1} \big|\,h \cdot g^{i} \bmod p\,\big\rangle, \qquad M = 2^m .$$

With `a = log_g h`, `a' = log_g h'`, `N = p−1`, and `δ = (a' − a) mod N`, the two supports are arithmetic progressions of length `M` on the exponent circle `Z_N`, so (for `2M ≤ N`) their intersection has size `Ω(δ) = max(0, M−δ) + max(0, M−(N−δ))` and

$$K(h,h') \;=\; \big|\langle \Phi(h)|\Phi(h')\rangle\big|^2 \;=\; \Big(\frac{\Omega(\delta)}{M}\Big)^{\!2}.$$

**This is a circular triangular kernel in discrete-logarithm space.** The quantum state performs the hard coordinate change *for free*: a max-margin classifier in this kernel is solving one-dimensional interval membership. That is the entire mechanism, and `results/kernel_matrices.png` shows it directly — sort the training samples by their hidden `log_g h` and the coset Gram matrix is a clean banded diagonal, while the ladder Gram matrix is structureless.

**Honest accounting of the state-preparation cost.** In this repository `|Φ(h)⟩` is written down explicitly and compiled with Qiskit's `StatePreparation`, whose synthesis is exponential in `n`. That is a **simulation convenience, not the asymptotic algorithm.** At cryptographic scale the same state is produced in `O(n³)` gates by Shor-style period finding: prepare a uniform superposition over `i ∈ [0,M)`, compute `|i⟩|h·g^i mod p⟩` by modular exponentiation, then uncompute the index register with a QFT. **No step of that routine ever evaluates `log_g h`.** The quantum advantage claim rests on that routine; the simulator is only how we check the resulting kernel.

### 6.3 Two corrections to the naive formulation

**(a) The parity labelling `y = (−1)^(log_g h mod 2)` has no trapdoor and must not be used.**

The least significant bit of the discrete logarithm in `Z_p*` is *not* a hard bit — it is the quadratic-residuosity bit, and Euler's criterion computes it in `O(log p)` modular multiplications:

$$h^{(p-1)/2} \bmod p = 1 \iff h \text{ is a QR} \iff \log_g h \text{ is even}.$$

`legendre_attack()` implements this. Measured:

| Labelling | Legendre attack test accuracy | Verdict |
|---|---|---|
| `parity` | **1.000** | **broken by a poly-time classical algorithm** |
| `half_interval` (default) | 0.460 | chance — trapdoor intact |

This is why the default is the half-interval rule. The *hardcore* bits of the discrete log are the high-order ones (Blum–Micali); the low-order bit is free. Run `--labeling parity` to watch the whole premise collapse.

**(b) An `Rz/Rx/CZ` hardware-efficient ansatz cannot solve this problem.**

The `LadderFeatureMap` is a Havlíček-style second-order map — it entangles, and its kernel is not efficiently computable classically. It still scores 0.60 (n.s.), because **expressivity is not the binding constraint; the encoding is.** Its kernel is a function of the *bits* of `h`, while the label is a function of `log_g h`, and nothing in the circuit inverts that map. Its mean off-diagonal kernel value is **0.0032** — a near-identity Gram matrix, the textbook exponential-concentration / "quantum overfitting" signature: 100% train accuracy, chance test accuracy, all 200 points support vectors.

**(c) A bonus falsification: the coset kernel is not a universal cheat.**

Run `--labeling parity` and the *coset* kernel also drops to **0.46**. The triangular kernel is *local* on the dlog circle, whereas parity alternates at every step — so a local kernel is maximally uninformative about it. The coset kernel wins on the half-interval concept because its inductive bias matches that concept's structure, not because quantum kernels are magic. This is the cleanest evidence in the repository that the pipeline is measuring something real.

### 6.4 Verification of the kernel computation

The Gram matrix is computed by simulating each state once and taking one BLAS product — `O(N² · 2ⁿ)` instead of `O(N²)` circuit executions. Three independent audits confirm this equals the circuit definition `K_ij = |⟨0ⁿ|U†(x_j)U(x_i)|0ⁿ⟩|²`:

| Audit | Method | Max abs. error |
|---|---|---|
| Compute–uncompute (ladder) | Build `U†(x_j)U(x_i)` in Qiskit, read `P(|0ⁿ⟩)` | `8.7 × 10⁻¹⁹` |
| Compute–uncompute (coset) | Same, via `StatePreparation` | `3.0 × 10⁻¹³` |
| Closed form (coset) | Triangular-overlap formula, no simulator | `4.4 × 10⁻¹⁶` |
| `qiskit-machine-learning` | `FidelityQuantumKernel`, 12×12 block | `8.1 × 10⁻¹³` |

> **Note for anyone reproducing the last row:** you must pass `enforce_psd=False`. With `FidelityQuantumKernel`'s default (`True`), the eigenvalue-clipping projection distorts a matrix that is already exact and already PSD — it pushed diagonal entries to 1.0245 and manufactured negative off-diagonals, producing a spurious ~0.31 disagreement. This is a measurement artefact of the audit tool, not of the kernel.

### 6.5 Statistical discipline

With `n_test = 50`, the binomial standard error at chance is `1/(2√50) = 7.07` percentage points. A raw score of 0.60 is **1.4σ** from chance — *not* evidence of learning. Every row of the final table therefore carries a **two-sided exact binomial p-value** and a **Wilson 95% score interval**, and the run explicitly reports whether *any* non-coset model clears `p < 0.01` (it does not). The coset kernel clears it at `p = 3.7 × 10⁻¹¹`.

---

## 7. Scaling to cryptographic parameters

| Regime | `n` (qubits) | `p` | Statevector | Status |
|---|---|---|---|---|
| Smoke test | 6 | 59 | 64 | runs in ~5 s; use `--train 28 --test 8` |
| **Default** | **10** | **1019** | **1 024** | **~35 s on a laptop** |
| Simulation ceiling | 24 | 16 777 213 | 2²⁴ (≈268 MB/state) | upper bound for dense simulation |
| Cryptographic | 64 | safe prime `< 2⁶⁴` | 2⁶⁴ | fault-tolerant hardware only |
| Cryptographic | 128 | safe prime `< 2¹²⁸` | 2¹²⁸ | security level of the concept class |

**Exact parameter changes to scale up.** The number theory needs *no* modification — `largest_safe_prime_below(2**n)` and `find_generator()` are already arbitrary-precision (`is_probable_prime` is a deterministic Miller–Rabin, exact below 3.3 × 10²⁴; above that, extend `MILLER_RABIN_WITNESSES` or switch to random bases). Three things must change:

1. **Drop the `O(p)` oracle tables.** `_build_log_tables()` materialises `exp_table`/`log_table` and is infeasible past ~2²⁴. At scale, generate *forward only*: sample `x`, compute `h = pow(g, x, p)`, and label from `x` directly. Nothing else uses the inverse table.
2. **Replace `StatePreparation` with Shor period-finding.** This is the essential change. Dense preparation is `Θ(2ⁿ)`; the real construction is `O(n³)` gates on `~2n+3` qubits (superposition over `i` → modular exponentiation `|i⟩|h·g^i⟩` → inverse QFT). This is a circuit-synthesis task, not a physics change.
3. **Estimate the kernel from shots, not amplitudes.** Replace the exact inner product with compute–uncompute sampling; `O(ε⁻²)` shots per entry for additive error `ε`. LAT2021 §4 proves the SVM is robust to this — the margin degrades gracefully.

**Then stop and read this before claiming quantum advantage.** At `p = 1019` the discrete log falls to brute force in microseconds; our own `baby_step_giant_step` solves it in `O(√p)` as part of the dataset audit. **This repository demonstrates the learning-theoretic structure, not cryptographic hardness.** The asymptotic separation is a theorem about `|p| ≥ 2048` bits, and realising it needs fault-tolerant hardware that does not currently exist. Any submission claiming to have *achieved* cryptographic-scale quantum advantage on a laptop is claiming something false. What is honestly demonstrated here: a concept class where the encoding — not the model capacity — is the barrier, six learners that fail on it, one that succeeds, and a mechanism that explains exactly why.

---

## 8. Reproducibility

- **Determinism.** Every stochastic component is seeded from `--seed` (default `20260828`): the secret `s` is `SHA-256("dlog-concept-class|p=…|g=…|seed=…") mod (p−1)`, sampling uses `np.random.default_rng(seed)`, and all sklearn estimators and CV splitters take `random_state=seed`. Re-running reproduces the CSV byte-for-byte. `--crypto-random` intentionally breaks determinism by drawing `s` from `secrets` — use it to confirm the result is not an artefact of one lucky secret.
- **No shot noise.** All kernels are exact statevector inner products, so quantum numbers are deterministic too.
- **Machine-readable output.** `results/metrics.json` contains the full config, group parameters, per-model metrics with CIs and p-values, kernel audit errors, and timings.
- **Reference environment.** Windows 11, Python 3.13.14, numpy 2.5.2, pandas 2.3.3, scikit-learn 1.9.0, qiskit 2.5.1, qiskit-machine-learning 0.9.0, matplotlib 3.10.3. Verified end-to-end at 10 and 6 qubits and on both labellings.

---

## 9. Reference

> Y. Liu, S. Arunachalam, K. Temme, *A rigorous and robust quantum speed-up in supervised machine learning*, **Nature Physics 17**, 1013–1017 (2021). [arXiv:2010.02174](https://arxiv.org/abs/2010.02174)

Supporting: V. Havlíček et al., *Supervised learning with quantum-enhanced feature spaces*, Nature **567**, 209 (2019) — the ladder control map. P. Shor, *Polynomial-time algorithms for prime factorization and discrete logarithms*, SIAM J. Comput. **26**, 1484 (1997) — the `O(n³)` state preparation. M. Blum, S. Micali, *How to generate cryptographically strong sequences of pseudo-random bits*, SIAM J. Comput. **13**, 850 (1984) — why the low-order dlog bit is not hardcore (§6.3a).
