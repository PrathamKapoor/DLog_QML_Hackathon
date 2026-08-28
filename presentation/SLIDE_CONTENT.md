# Slide Content — DLOG-QML

Seven slides. Every number is measured; none is projected. Figures referenced live in `figures/`.

---

## Slide 1 — The Hook

# Can quantum learning see what classical learning cannot?

### `h = g^x mod p`

|  |  |
|---|---|
| **x → h** | easy (modular exponentiation) |
| **h → x** | the discrete logarithm problem |

> ### The label depends only on `x`.
> ### The learner only ever sees `h`.

*Speaker note: do not say "quantum" yet. Get them to feel the asymmetry first.*

---

## Slide 2 — The Problem

### The concept is a single interval — in a coordinate system the learner cannot compute

```
y = +1   if  (log_g(h) - s) mod (p-1)  ∈  [0, (p-3)/2]
y = -1   otherwise
```

**Figure:** `fig3_concept_geometry.png` — full width

- **Left:** what the learner sees. Colours fully interleaved.
- **Middle:** identical points, identical strip, plotted against the hidden coordinate. One clean split.
- **Right:** the exponent circle — two label arcs, and the window the quantum map spans.

> In `log_g` coordinates this concept has **VC dimension 2**.
> Capacity was never the problem. **The coordinate system was hidden.**

Setup: `p = 1019 = 2·509+1`, `g = 2`, 10 qubits, 200 train / 50 test, exact 50/50, disjoint exponents.

---

## Slide 3 — The Idea

### Encode the group's own orbit, not the bits

$$|\Phi(h)\rangle = \frac{1}{\sqrt{M}}\sum_{i=0}^{M-1}\big|\,h\cdot g^{i} \bmod p\,\big\rangle$$

Built by forward multiplication only. **No discrete logarithm is ever computed.**

But the support is a contiguous run on the exponent circle, so:

$$K(h,h') = \left(\frac{\max(0,M-\delta)+\max(0,M-(N-\delta))}{M}\right)^{2},\qquad \delta = (\log_g h' - \log_g h)\bmod N$$

> ### A triangular kernel in discrete-log coordinates.
> The state does the hard coordinate change for free. The SVM just solves interval membership.

**Figure:** `fig2_kernel_matrices.png`

| Sorted by hidden `log_g h` | |
|---|---|
| **Coset kernel** | clean diagonal band |
| **Generic quantum kernel** | the identity matrix |
| **Classical RBF kernel** | noise |

| | Coset | Generic |
|---|---|---|
| Mean off-diagonal `K` | 0.0832 | **0.0032** |
| Effective rank (of 200) | 30.1 | **195.3** |
| Kernel-target alignment | **0.382** | 0.070 |
| Class separation gap | **+0.156** | **+0.000** |

---

## Slide 4 — Results

# 98.9% vs 54.1%

### +38.3 ± 6.2 points, over 20 independent seeds

*(measured per seed against that seed's best competitor — the stricter count; pooled means give +44.8. Ahead on 20/20 seeds.)*

**Figure:** `fig1_accuracy_comparison.png`

| Model (20 seeds) | Mean | Std |
|---|---|---|
| *Hidden-x oracle (control)* | *0.992* | *0.015* |
| **Quantum coset kernel** | **0.989** | **0.018** |
| Poly-SVM (d=3) | 0.541 | 0.070 |
| ExtraTrees | 0.541 | 0.082 |
| RBF-SVM (tuned) | 0.537 | 0.095 |
| RandomForest | 0.526 | 0.098 |
| Linear-SVM | 0.524 | 0.084 |
| kNN, RBF-SVM | 0.515 | — |
| **Quantum generic kernel** | **0.503** | **0.078** |
| MLP, GradientBoosting | ~0.50 | — |

Reference run: coset **0.940**, CI [0.838, 0.979], **p = 3.7e-11**, AUC 0.9968.

**Paired on the same items:** McNemar exact **p = 8.6e-4**; bootstrap 95% CI on the difference **[+0.16, +0.48]**.

> **No competing model is above chance at p < 0.01.**
> Worst separation across all 20 seeds: **+24 points.**

---

## Slide 5 — Try To Break It

# Four experiments designed to kill our own claim

### 1. Is it just "quantum"? — **No.**
Generic entangling `Rz/Rx/CZ` map, 133 gates, depth 64, identical pipeline.
### → 0.503 ± 0.078. Chance.

### 2. Is the trapdoor real? — **Yes, and we prove it by breaking the naive version.**
Labelling by dlog *parity* has **no trapdoor**: the low bit is the Legendre symbol, `O(log p)` via Euler's criterion.

| Labelling | Our attack |
|---|---|
| `parity` | **1.000 — broken** |
| `half_interval` (ours) | 0.460 — chance |

### 3. Is the data just noise? — **No.**
RandomForest on the hidden `x`: **0.992**. Perfectly learnable *given the coordinate*.

### 4. Is our quantum kernel a universal cheat? — **No.**
Run it on the parity concept: **0.460.** Its kernel is *local* on the dlog circle; parity flips every step.
### → It wins when its geometry matches the concept. It fails when it doesn't.

**Plus:** window ablation — `m = 2…8` all beat every classical baseline; the default is a fixed heuristic and isn't even the best value.

---

## Slide 5b — Why does the generic kernel fail?

### Not "it was bad" — an actual mechanism

**Figure:** `fig6_kernel_diagnostics.png`

| Diagnostic | Coset | Generic | Reads as |
|---|---|---|---|
| Mean off-diagonal `K` | 0.0832 | **0.0032** | near-orthogonal |
| Effective rank (of 200) | 30.1 | **195.3** | 200 points, ~195 directions |
| Top-10 eigenvalue mass | 0.680 | **0.079** | spectrally flat |
| Kernel-target alignment | **0.382** | 0.070 | geometry ignores the labels |
| `K` same-class | **0.1614** | 0.0033 | |
| `K` different-class | 0.0058 | 0.0031 | |
| **Separation gap** | **+0.1556** | **+0.0002** | **no class structure at all** |

> The generic feature map exhibits **near-orthogonal concentration**, producing a
> high-effective-rank kernel with **negligible class structure**. Same-class and
> different-class pairs sit at essentially the same similarity — so there is nothing
> for any classifier to exploit. 100% train accuracy, every point a support vector,
> chance test accuracy.

*Speaker note: this is the slide that turns a bar chart into science.*

---

## Slide 6 — Scaling and Limitations

### Measured (real runs, real timings)

| n | p | Coset | Generic | Memory |
|---|---|---|---|---|
| 6 | 59 | 1.000 | 0.600 | 0.06 MB |
| 8 | 227 | 0.977 | 0.386 | 0.88 MB |
| 10 | 1019 | 0.940 | 0.600 | 3.91 MB |

**Figure:** `fig4_scaling_measured.png`

### Theoretically motivated (projection — NOT measured)

~24 qubits is the dense-simulation ceiling (~268 MB/statevector). Cryptographic relevance needs 2048-bit `p` and fault-tolerant hardware.

### What we do NOT claim

- ❌ `p = 1019` is **not** cryptographically hard — our own code solves it with baby-step giant-step as a data audit
- ❌ Classical simulation only. **No quantum hardware. No hardware results anywhere.**
- ❌ Dense state preparation is `Θ(2ⁿ)` — **not** the efficient algorithm. The `O(n³)` version is Shor's; we cite it, we did not build it
- ❌ An accuracy separation is **not** a complexity-theoretic speedup
- ❌ The asymptotic theorem is **Liu–Arunachalam–Temme's**, conditional on the discrete-log assumption

---

## Slide 7 — Conclusion

# Quantum advantage is not automatic.

### Our own generic quantum kernel scores 50.3% — in the same table as our 98.9%.

> ## The ingredient that mattered was matching the quantum representation to the algebraic structure of the learning problem.

**What we built:** a deterministic, audited implementation with a hard public/oracle boundary · **187 tests** · a four-experiment falsification suite · exact binomial, Wilson, McNemar and bootstrap statistics · a 20-seed robustness sweep · kernel-structure diagnostics · a measured size sweep and window ablation · explicit claim discipline.

```bash
python scripts/demo.py            # ~30 s
python scripts/run_experiment.py  # full study
python -m pytest                  # 187 tests
```

---

## Appendix slides (hold in reserve)

### A1 — Kernel verification

| Audit | Max abs. error |
|---|---|
| Compute–uncompute circuits (generic) | 1.7e-18 |
| Compute–uncompute circuits (coset) | 3.0e-13 |
| Closed-form triangular overlap | 4.4e-16 |
| `qiskit-machine-learning` `FidelityQuantumKernel` | 8.2e-13 |

*(The last requires `enforce_psd=False`; the default clipping distorts an already-exact PSD matrix.)*

### A2 — Why 1019 and not 1021

1021 is prime but **not safe**: `510 = 2·3·5·17`. The generator shortcut (`g² ≢ 1`, `g^q ≢ 1`) requires a safe prime. Largest safe prime below `2¹⁰` is **1019**.

Also: `ceil(log2(2·125)) = 8` suggests 8 qubits suffice for 125 samples/class — but `p = 227` gives `q = 113 < 125`. The code searches the safe-prime table instead of trusting the log.

### A3 — Statistical discipline

`n_test = 50` → binomial s.e. at chance = **7.07 points**. So 0.60 is **1.4σ** — noise.
Every row carries a Wilson CI and an exact binomial p-value so this cannot be misread.
`H0` for the binomial test: "this classifier is at chance." It says **nothing** about complexity classes.

### A4 — Leakage prevention

`features()` returns bits only. The exponent is reachable solely via `oracle_exponents()`, with three documented legitimate callers. `assert_no_leakage()` runs before every learner and checks width, binarity, no column equal to `x` or its low bits, and that the matrix *is* the bit decomposition of `h`. A test poisons a column on purpose and confirms the guard fires. Train/test exponents are disjoint by construction.
