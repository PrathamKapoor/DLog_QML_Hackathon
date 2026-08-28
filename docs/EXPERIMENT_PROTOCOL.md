# Experiment Protocol

Exact procedure to reproduce every number in `README.md` and `docs/TECHNICAL_REPORT.md`.

---

## 0. Environment

Reference environment (all reported numbers produced here):

| Component | Version |
|---|---|
| OS | Windows 11 (10.0.26200) |
| Python | 3.13.14 |
| numpy | 2.5.2 |
| pandas | 2.3.3 |
| scikit-learn | 1.9.0 |
| qiskit | 2.5.1 |
| qiskit-machine-learning | 0.9.0 (optional) |
| scipy | 1.17.1 (tests only) |
| matplotlib | 3.10.3 |

Setup:

```bash
python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

The environment is recorded automatically into every result file under the `environment` key, so a result can always be attributed to a concrete stack.

---

## 1. Test suite (run first)

```bash
python -m pytest
```

Expected: **187 passed**. Runtime ≈ 100 s (the group-theory tests brute-force whole groups on purpose).

Coverage:

| File | Checks |
|---|---|
| `test_group.py` | Miller–Rabin vs. trial division for `n<2000`; three Carmichael numbers; safe primes at every width; generator full-orbit brute force; LSB-first bit ordering; BSGS vs. oracle table |
| `test_labels.py` | SHA-256 secret formula and domain separation; interval endpoints `0`, `q-1`, `q`, `N-1`; wraparound; exact balance for several secrets; closed-interval vs. `< q` at all 1018 exponents; Euler's criterion equals dlog parity for **every** element of `Z_1019*` |
| `test_dataset.py` | `h = g^x mod p`; exact balance; split disjointness; determinism; **leakage guard accepts clean features and rejects an injected exponent bit**; CSV round-trip |
| `test_diagnostics.py` | Kernel-target alignment vs closed form; class-conditional similarity on hand-built block kernels; eigenspectrum on identity/rank-one; artifact provenance (cached never presentable as fresh); ablation table and controls checklist; headline-number audit catches an injected wrong value |
| `test_kernel.py` | Coset support/normalisation; closed form vs. simulation; triangular formula vs. hand-counted set intersection; kernel depends only on dlog difference; compute–uncompute agreement; PSD/symmetry/unit-diagonal; concentration comparison; the `π/2` angle-scale guard |
| `test_statistics.py` | Binomial vs. closed form and vs. `scipy.stats.binomtest`; "0.60 on n=50 is not significant"; Wilson known values and boundary behaviour; McNemar vs. closed form and scipy; bootstrap determinism |

---

## 2. Demo (~30 s)

```bash
python scripts/demo.py
```

Runs one complete experiment and prints the judge-facing narrative. Every printed value is computed in-process.

| Flag | Effect |
|---|---|
| `--fast` | 8 qubits, 120/30, no extra baselines (~16 s) |
| `--labeling parity` | the BROKEN concept — Legendre attack should hit 1.000 |
| `--qubits N` `--train N` `--test N` `--seed N` | overrides |
| `--no-verify` | skip the compute–uncompute audit |

---

## 3. Full study (~5 min)

```bash
python scripts/run_experiment.py
```

Six stages:

| Stage | Output | Approx. time |
|---|---|---|
| 1. Reference run (with kernel audits) | `raw/single_run.json`, `dlog_dataset.csv` | 30 s |
| 2. Parity falsification | `raw/parity_run.json` | 25 s |
| 3. Multi-seed robustness (20 seeds) | `raw/multi_seed.json`, `tables/multi_seed.*` | ~2.5 min |
| 4. Measured size sweep | `raw/size_sweep.json`, `tables/size_sweep.*` | 25 s |
| 5. Coset window ablation | `raw/window_ablation.json`, `tables/window_ablation.*` | 45 s |
| 6. Figures | `figures/fig1..fig7.png` | 15 s |

Total measured wall-clock on the reference machine: **~200 s**.

Options:

```bash
python scripts/run_experiment.py --quick              # stage 1 + figures only
python scripts/run_experiment.py --seeds 5            # shorter robustness sweep
python scripts/run_experiment.py --skip sweep ablation
python scripts/run_experiment.py --outdir results_alt
```

---

## 4. Figures only

```bash
python scripts/generate_figures.py
```

Re-runs the reference experiment (deterministic, so figures match stored numbers) and reuses saved multi-seed / sweep / ablation JSON when present. The kernel matrices and concept-geometry plot need in-memory Gram matrices and oracle exponents, which are deliberately not serialised — hence the re-run.

---

## 5. Reproduction checklist

Run the full study, then verify:

| # | Check | Expected | Where |
|---|---|---|---|
| 1 | Group parameters | `p=1019, q=509, g=2, s=666` | stage 1 header |
| 2 | Dataset audit | balance 100/100 and 25/25; splits disjoint | stage 1 |
| 3 | Coset accuracy | 0.940, CI [0.838, 0.979], p = 3.7e-11 | `tables/main_results.md` |
| 4 | No competitor significant | `any_competitor_significant_at_1pct: false` | `raw/single_run.json` |
| 5 | Best competitor | Linear-SVM, 0.620 | stage 1 |
| 6 | McNemar | p = 8.6e-4 | `raw/single_run.json` |
| 7 | Kernel audits | ≤ 3.0e-13 (circuits), 4.4e-16 (closed form) | `raw/single_run.json` |
| 8 | Generic kernel concentration | off-diag mean 0.0032, effective rank 195.3 | `raw/single_run.json` |
| 9 | Parity falsification | Legendre 1.000; coset drops to 0.460 | stage 2 |
| 10 | Multi-seed | coset 0.989 ± 0.018; separation +0.383 ± 0.062 | `tables/multi_seed.md` |
| 11 | Window ablation | `m = 2…8` all ≥ 0.880 | `tables/window_ablation.md` |
| 12 | Figures | 7 PNGs written | `figures/` |
| 13 | Ablation table | `tables/ablation_table.md` written | `results/tables/` |
| 14 | Negative controls | 11/11 passed | `tables/negative_controls.md` |
| 15 | Headline-number audit | every row `agrees = True` | notebook, *Where the headline numbers come from* |

---

## 6. Determinism

One seed controls the secret shift, the sample draw, and every `random_state`. Re-running with the same config reproduces the dataset byte-for-byte:

```bash
python scripts/run_experiment.py --quick
python -c "import hashlib;print(hashlib.sha256(open('results/dlog_dataset.csv','rb').read()).hexdigest())"
# run again -- the hash must be identical
```

Quantum results are deterministic too: kernels are exact statevector inner products, so there is no shot noise.

**Deliberate non-determinism:** `ExperimentConfig(crypto_random=...)` is not exposed at this level, but the multi-seed sweep varies the secret across seeds precisely to show the result is not an artefact of one lucky secret.

---

## 7. Independent verification of the dataset

`results/dlog_dataset.csv` ships `raw_input_x` (the trapdoor) specifically so a third party can check the construction without trusting our code:

```python
import pandas as pd
df = pd.read_csv("results/dlog_dataset.csv")
p, g, s, q = 1019, 2, 666, 509

assert (df["raw_input_x"].map(lambda x: pow(g, int(x), p)) == df["group_element_h"]).all()

expected = df["raw_input_x"].map(lambda x: 1 if (x - s) % (p - 1) < q else -1)
assert (expected == df["label"]).all()

bits = [f"binary_feature_{j}" for j in range(10)]
recomposed = sum(df[c] * (1 << j) for j, c in enumerate(bits))   # LSB-first
assert (recomposed == df["group_element_h"]).all()

tr = set(df.loc[df.split == "train", "raw_input_x"])
te = set(df.loc[df.split == "test", "raw_input_x"])
assert not (tr & te)
print("dataset independently verified")
```

`results/dlog_dataset_no_oracle.csv` is the same data with `raw_input_x` removed — the leak-proof version for anyone training on it.

---

## 8. Changing the experiment

Edit `configs/default.yaml` or override on the command line. Constraints the code enforces with actionable errors:

- `n_train`, `n_test` must be **even** (exact 50/50 balance).
- `(n_train + n_test)/2 ≤ q = (p-1)/2`. If violated, the error names the minimum `n_qubits` — computed by searching the safe-prime table, since `ceil(log2(2·per_class))` is wrong (`n=8` gives `p=227`, `q=113`, not 127).
- Coset window must satisfy `2M ≤ p-1` for the closed form to hold.
- `n_qubits ≥ 5`.

---

## 9. Optional: `qiskit-machine-learning` cross-check

An independent audit of the generic kernel against `FidelityQuantumKernel`:

```python
import sys; sys.path.insert(0, "src")
import numpy as np
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from dlog_qml.data import generate_dataset
from dlog_qml.kernels import fidelity_gram
from dlog_qml.quantum_generic import LadderFeatureMap

data = generate_dataset(n_qubits=10, n_train=12, n_test=2, seed=20260828)
fm = LadderFeatureMap(data.group.n_bits, reps=2)
ours = fidelity_gram(fm.statevectors(data.features("train").astype(int)))

# enforce_psd=False is REQUIRED for a like-for-like comparison.
qk = FidelityQuantumKernel(feature_map=fm.template, enforce_psd=False)
ref = np.asarray(qk.evaluate(x_vec=fm.angles(data.features("train"))))
print("max |K_qml - K_ours| =", np.abs(ref - ours).max())    # ~1e-13
```

**`enforce_psd=False` is not optional.** With the default `True`, the eigenvalue-clipping projection distorts a matrix that is already exact and already PSD — in testing it pushed diagonal entries to 1.0245 and manufactured negative off-diagonals, producing a spurious ~0.31 disagreement. That is an artefact of the audit tool, not of the kernel.
