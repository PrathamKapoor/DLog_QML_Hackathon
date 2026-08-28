#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 PROVABLE QUANTUM ADVANTAGE ON THE DISCRETE-LOGARITHM CONCEPT CLASS
 End-to-end dataset generator, classical-failure benchmark, and quantum-kernel
 (QSVM) evaluation pipeline.

 Reference concept class:
   Y. Liu, S. Arunachalam, K. Temme, "A rigorous and robust quantum speed-up in
   supervised machine learning", Nature Physics 17, 1013-1017 (2021).
   arXiv:2010.02174

--------------------------------------------------------------------------------
 WHAT THIS SCRIPT DOES
--------------------------------------------------------------------------------
   1. Builds a cryptographically structured binary classification problem in the
      cyclic group Z_p^* whose labelling rule is a hidden discrete logarithm.
   2. Exports it to `dlog_quantum_advantage_dataset.csv`.
   3. Trains three classical learners (RBF-SVM, RandomForest, MLP) plus a tuned
      RBF-SVM and shows they cannot beat chance.
   4. Runs a *number-theoretic* attack (Legendre symbol) to prove the trapdoor is
      genuine for the shipped labelling and broken for the naive parity labelling.
   5. Builds TWO Qiskit quantum feature maps and their fidelity kernels:
        (a) COSET  -- the Liu/Arunachalam/Temme Shor-style coset state. Separates.
        (b) LADDER -- a generic hardware-efficient Rz/Rx/CZ map. Control: fails.
   6. Trains SVC(kernel='precomputed') on each and prints a comparison table.

--------------------------------------------------------------------------------
 TWO MATHEMATICAL CAVEATS, STATED UP FRONT (see README.md Section 6)
--------------------------------------------------------------------------------
 CAVEAT 1 -- The parity labelling y = (-1)^(dlog mod 2) is NOT classically hard.
   The least significant bit of log_g(h) in Z_p^* equals the quadratic-residuosity
   bit of h, which Euler's criterion computes in O(log p) time:
        h^((p-1)/2) mod p  ==  1  <=>  log_g(h) is even.
   Parity is therefore a *hardcore-bit-free* labelling with no trapdoor. It is
   available via --labeling parity ONLY as a negative control; `legendre_attack`
   below breaks it to ~100% accuracy. The default labelling is the half-interval
   rule of Liu-Arunachalam-Temme, for which no polynomial-time classical
   distinguisher is known under the Discrete Logarithm Assumption.

 CAVEAT 2 -- Simulation scale is a demonstration, not a cryptographic proof.
   At p ~ 10^3 the discrete logarithm is solvable by brute force in microseconds.
   The asymptotic hardness statement applies to |p| >= 2048 bits (or >= 256-bit
   elliptic-curve groups). What this script demonstrates at small scale is that
   the *learning-theoretic structure* is real: models restricted to the group
   encoding cannot generalise, while the coset-state quantum kernel can. See
   THEORETICAL_SCALE below and README.md Section 7 for exact scale-up parameters.
================================================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import sys
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

try:  # keep stdout safe on legacy Windows code pages
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover
    pass


# ==============================================================================
# CONSTANTS
# ==============================================================================

DEFAULT_CSV = "dlog_quantum_advantage_dataset.csv"
DEFAULT_OUTDIR = "results"

#: Columns that encode the SECRET and must never be handed to a learner.
HIDDEN_COLUMNS: Tuple[str, ...] = ("raw_input_x",)

#: Deterministic Miller-Rabin witness set: exact for all n < 3.317e24.
MILLER_RABIN_WITNESSES: Tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

#: Documented scale-up points. `n_qubits` is ceil(log2 p) == register width.
THEORETICAL_SCALE: Dict[str, Dict[str, object]] = {
    "hackathon_6q": {
        "n_qubits": 6,
        "p_approx": 59,
        "statevector_dim": 2 ** 6,
        "max_samples": 58,
        "note": "Smoke-test scale. Group too small for N=250; use --train 28 --test 8.",
    },
    "hackathon_10q": {
        "n_qubits": 10,
        "p_approx": 1019,
        "statevector_dim": 2 ** 10,
        "max_samples": 1018,
        "note": "DEFAULT. Exact statevector simulation in seconds on a laptop.",
    },
    "simulation_ceiling_24q": {
        "n_qubits": 24,
        "p_approx": 16_777_213,
        "statevector_dim": 2 ** 24,
        "max_samples": 16_777_212,
        "note": "~268 MB per complex128 statevector. Upper bound for dense simulation.",
    },
    "cryptographic_64bit": {
        "n_qubits": 64,
        "p_approx": "largest safe prime < 2**64",
        "statevector_dim": "2**64 (NOT simulable)",
        "max_samples": "2**64",
        "note": (
            "Requires fault-tolerant hardware. Coset state prepared by Shor "
            "period-finding: ~2n+3 qubits, O(n^3) gates. Classical GNFS cost "
            "~exp(1.92 (ln p)^(1/3) (ln ln p)^(2/3))."
        ),
    },
    "cryptographic_128bit": {
        "n_qubits": 128,
        "p_approx": "largest safe prime < 2**128",
        "statevector_dim": "2**128 (NOT simulable)",
        "max_samples": "2**128",
        "note": (
            "Security level of the concept class; classical learner provably "
            "restricted to chance under the DLog Assumption (LAT2021 Thm 1)."
        ),
    },
}


# ==============================================================================
# NUMBER-THEORETIC PRIMITIVES
# ==============================================================================

def is_probable_prime(n: int) -> bool:
    """Deterministic Miller-Rabin primality test for every n < 3.317e24."""
    if n < 2:
        return False
    for small in MILLER_RABIN_WITNESSES:
        if n == small:
            return True
        if n % small == 0:
            return False

    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for a in MILLER_RABIN_WITNESSES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def largest_safe_prime_below(bound: int) -> int:
    """Largest safe prime p = 2q + 1 (q prime) strictly below `bound`."""
    candidate = bound - 1 if bound % 2 == 0 else bound - 2
    while candidate >= 5:
        if is_probable_prime(candidate) and is_probable_prime((candidate - 1) // 2):
            return candidate
        candidate -= 2
    raise ValueError(f"No safe prime below {bound}; increase --qubits.")


def exact_binomial_p(accuracy: float, n: int) -> float:
    """
    Two-sided exact binomial p-value for `accuracy` against the chance rate 0.5.

    With only n = 50 held-out samples the standard error of an accuracy estimate
    is 1/(2 sqrt(n)) = 7.1 percentage points, so a raw score of 0.60 sits 1.4
    sigma from chance and is NOT evidence of learning. Reporting this alongside
    every accuracy is what stops a lucky classical baseline -- or a lucky quantum
    one -- from being mistaken for signal.
    """
    if n <= 0:
        return float("nan")
    k = int(round(accuracy * n))
    total = float(2 ** n)
    observed = math.comb(n, k) / total
    tail = sum(math.comb(n, i) / total
               for i in range(n + 1)
               if math.comb(n, i) / total <= observed * (1 + 1e-12))
    return float(min(1.0, tail))


def wilson_interval(accuracy: float, n: int, z: float = 1.959963985) -> Tuple[float, float]:
    """95% Wilson score interval -- well behaved near 0 and 1, unlike Wald."""
    if n <= 0:
        return (float("nan"), float("nan"))
    centre = (accuracy + z * z / (2 * n)) / (1 + z * z / n)
    half = (z / (1 + z * z / n)) * math.sqrt(
        accuracy * (1 - accuracy) / n + z * z / (4 * n * n)
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def baby_step_giant_step(g: int, h: int, p: int) -> int:
    """
    Reference discrete-log solver, O(sqrt(p)) time and space.

    Used ONLY to audit the generated dataset -- never by any learner. Its
    sqrt(p) cost is precisely why the construction is a *demonstration* at
    p ~ 10^3 and a *hardness assumption* at |p| >= 2048 bits.
    """
    order = p - 1
    m = math.isqrt(order) + 1
    table: Dict[int, int] = {}
    e = 1
    for j in range(m):
        table.setdefault(e, j)
        e = (e * g) % p
    factor = pow(g, (order - m) % order, p)
    gamma = h % p
    for i in range(m):
        if gamma in table:
            return (i * m + table[gamma]) % order
        gamma = (gamma * factor) % p
    raise ValueError("Discrete logarithm not found (h outside <g>).")


# ==============================================================================
# 1. DATA GENERATION
# ==============================================================================

@dataclass
class DiscreteLogConceptClass:
    r"""
    The Liu-Arunachalam-Temme discrete-logarithm concept class over Z_p^*.

    Construction
    ------------
    1. p = 2q + 1 is a safe prime, g a generator of the full group Z_p^* of
       order p - 1 = 2q.
    2. A secret shift s is drawn from Z_{p-1} (SHA-256 derived for
       reproducibility, or from `secrets` under --crypto-random).
    3. A hidden exponent x is drawn uniformly from Z_{p-1}; the OBSERVED datum is
       the group element h = g^x mod p, exposed to the learner only as its
       fixed-width binary decomposition.
    4. Labelling (half-interval rule, LAT2021 Definition 1):

           y(h) = +1  if  (log_g(h) - s) mod (p-1)  in  [0, (p-3)/2]
                  -1  otherwise

       The interval holds exactly q = (p-1)/2 exponents, so the concept is
       perfectly balanced over the whole group by construction.

    Why a classical learner is stuck
    --------------------------------
    y depends on h *only* through log_g(h). Recovering log_g(h) from h is the
    Discrete Logarithm Problem. Every bit the learner is shown is a bit of h,
    and no polynomial-time function of those bits is known to correlate with
    the half-interval indicator. See README.md Section 6.
    """

    n_qubits: int = 10
    n_train: int = 200
    n_test: int = 50
    seed: int = 20260828
    labeling: str = "half_interval"          # {"half_interval", "parity"}
    crypto_random: bool = False

    # --- populated by __post_init__ -------------------------------------------
    p: int = field(init=False)
    q: int = field(init=False)
    g: int = field(init=False)
    order: int = field(init=False)
    secret_s: int = field(init=False)
    n_bits: int = field(init=False)
    exp_table: np.ndarray = field(init=False, repr=False)
    log_table: np.ndarray = field(init=False, repr=False)
    dataframe: Optional[pd.DataFrame] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------ setup
    def __post_init__(self) -> None:
        if self.labeling not in ("half_interval", "parity"):
            raise ValueError("labeling must be 'half_interval' or 'parity'")
        if self.n_qubits < 5:
            raise ValueError("n_qubits must be >= 5 (smallest usable safe prime is 11).")
        if self.n_train % 2 or self.n_test % 2:
            raise ValueError("n_train and n_test must be even (exact 50/50 balance).")
        if self.n_train <= 0 or self.n_test <= 0:
            raise ValueError("n_train and n_test must be positive.")

        self.p = largest_safe_prime_below(2 ** self.n_qubits)
        self.q = (self.p - 1) // 2
        self.order = self.p - 1
        self.n_bits = self.p.bit_length()          # == n_qubits for our choices
        self.g = self.find_generator()
        self.secret_s = self._derive_secret()
        self._build_log_tables()
        self._check_capacity()

    def _check_capacity(self) -> None:
        total = self.n_train + self.n_test
        per_class = total // 2
        if per_class > self.q:
            # Search for the true minimum width: ceil(log2(2*per_class)) is NOT
            # sufficient, because the largest safe prime below 2^n can sit well
            # under 2^n (e.g. n = 8 -> p = 227 -> q = 113, not 127).
            need = None
            for width in range(self.n_qubits + 1, 65):
                if (largest_safe_prime_below(2 ** width) - 1) // 2 >= per_class:
                    need = width
                    break
            hint = f"raise --qubits to >= {need}" if need else "raise --qubits"
            raise ValueError(
                f"Group too small: need {per_class} samples per class but only "
                f"q = {self.q} exponents carry each label (p = {self.p}).\n"
                f"  Fix A: {hint}.\n"
                f"  Fix B: lower --train/--test so (train+test)/2 <= {self.q}."
            )

    def find_generator(self) -> int:
        """
        Smallest generator of the FULL group Z_p^*, i.e. an element of order 2q.

        For a safe prime the only divisors of |Z_p^*| = 2q are {1, 2, q, 2q}, so
        it suffices to reject candidates with g^2 == 1 or g^q == 1.
        """
        for candidate in range(2, self.p):
            if pow(candidate, 2, self.p) != 1 and pow(candidate, self.q, self.p) != 1:
                return candidate
        raise RuntimeError(f"No generator found for p = {self.p} (p is not prime?).")

    def _derive_secret(self) -> int:
        """Secret shift s in Z_{p-1}: SHA-256 (reproducible) or CSPRNG."""
        if self.crypto_random:
            return secrets.randbelow(self.order)
        material = f"dlog-concept-class|p={self.p}|g={self.g}|seed={self.seed}"
        digest = hashlib.sha256(material.encode("utf-8")).digest()
        return int.from_bytes(digest, "big") % self.order

    def _build_log_tables(self) -> None:
        """
        Full exponent/logarithm tables for the ORACLE.

        exp_table[i] = g^i mod p ;  log_table[h] = log_g(h).
        This is the trapdoor: only the data generator holds it. It costs O(p)
        memory and is feasible solely because this is a simulation-scale group.
        """
        exp_table = np.zeros(self.order, dtype=np.int64)
        log_table = np.full(self.p, -1, dtype=np.int64)
        value = 1
        for i in range(self.order):
            exp_table[i] = value
            log_table[value] = i
            value = (value * self.g) % self.p
        if value != 1:
            raise RuntimeError("Generator failed order check; group table invalid.")
        self.exp_table = exp_table
        self.log_table = log_table

    # ------------------------------------------------------------- concept
    def label_of_exponent(self, x: int) -> int:
        """Ground-truth label as a function of the HIDDEN exponent x."""
        if self.labeling == "half_interval":
            shifted = (x - self.secret_s) % self.order
            return 1 if shifted < self.q else -1
        return 1 if x % 2 == 0 else -1        # parity (negative control)

    def dlog(self, h: int) -> int:
        """Oracle discrete logarithm via the precomputed table."""
        value = int(self.log_table[h])
        if value < 0:
            raise ValueError(f"{h} is not in Z_{self.p}^*.")
        return value

    # ------------------------------------------------------------ sampling
    def generate_samples(self) -> pd.DataFrame:
        """
        Draw an exactly balanced, split-disjoint dataset.

        Exponents are sampled WITHOUT replacement from Z_{p-1}, so no group
        element appears in both the training and test split -- the accuracy gap
        we report can never be an artefact of memorisation.
        """
        rng = np.random.default_rng(self.seed)
        per_class_total = (self.n_train + self.n_test) // 2
        n_train_pc = self.n_train // 2

        permuted = rng.permutation(self.order)
        pos: List[int] = []
        neg: List[int] = []
        for raw in permuted:
            x = int(raw)
            bucket = pos if self.label_of_exponent(x) == 1 else neg
            if len(bucket) < per_class_total:
                bucket.append(x)
            if len(pos) == per_class_total and len(neg) == per_class_total:
                break
        if len(pos) < per_class_total or len(neg) < per_class_total:
            raise RuntimeError("Balanced sampling failed; group exhausted.")

        train_x = pos[:n_train_pc] + neg[:n_train_pc]
        test_x = pos[n_train_pc:] + neg[n_train_pc:]
        rng.shuffle(train_x)
        rng.shuffle(test_x)

        rows = []
        ordered = [(x, "train") for x in train_x] + [(x, "test") for x in test_x]
        for sample_id, (x, split) in enumerate(ordered):
            h = int(self.exp_table[x])
            rows.append({
                "sample_id": sample_id,
                "raw_input_x": int(x),
                "group_element_h": h,
                **{f"binary_feature_{b}": (h >> b) & 1 for b in range(self.n_bits)},
                "label": self.label_of_exponent(x),
                "split": split,
            })

        self.dataframe = pd.DataFrame(rows)
        self.verify_dataset()
        return self.dataframe

    @property
    def feature_columns(self) -> List[str]:
        return [f"binary_feature_{b}" for b in range(self.n_bits)]

    def verify_dataset(self) -> None:
        """Hard assertions: modular arithmetic, labels, balance, split disjointness."""
        df = self.dataframe
        assert df is not None, "generate_samples() has not been called."

        for _, row in df.iterrows():
            x, h = int(row["raw_input_x"]), int(row["group_element_h"])
            assert pow(self.g, x, self.p) == h, f"h != g^x for sample {row['sample_id']}"
            assert self.dlog(h) == x, "log table inconsistent with exp table"
            assert int(row["label"]) == self.label_of_exponent(x), "label mismatch"
            bits = [int(row[c]) for c in self.feature_columns]
            assert sum(b << i for i, b in enumerate(bits)) == h, "bit decomposition wrong"

        for split in ("train", "test"):
            counts = df.loc[df["split"] == split, "label"].value_counts().to_dict()
            assert counts.get(1, 0) == counts.get(-1, 0), f"{split} split unbalanced"

        train_x = set(df.loc[df["split"] == "train", "raw_input_x"])
        test_x = set(df.loc[df["split"] == "test", "raw_input_x"])
        assert not (train_x & test_x), "train/test exponent leakage"

        # Spot-check the oracle table against an independent BSGS solve.
        probe = int(df["group_element_h"].iloc[0])
        assert baby_step_giant_step(self.g, probe, self.p) == self.dlog(probe)

    def save_csv(self, path: str = DEFAULT_CSV) -> str:
        if self.dataframe is None:
            self.generate_samples()
        self.dataframe.to_csv(path, index=False)
        return os.path.abspath(path)

    # -------------------------------------------------------------- report
    def describe(self) -> str:
        source = "CSPRNG" if self.crypto_random else "SHA-256(seed)"
        return (
            f"  safe prime p           : {self.p}   (= 2*{self.q} + 1, "
            f"{self.p.bit_length()} bits)\n"
            f"  Sophie Germain prime q : {self.q}\n"
            f"  generator g            : {self.g}   (order {self.order} = |Z_p^*|)\n"
            f"  secret shift s         : {self.secret_s}   ({source})\n"
            f"  labelling rule         : {self.labeling}\n"
            f"  feature width          : {self.n_bits} bits -> {self.n_bits} qubits\n"
            f"  samples                : {self.n_train} train / {self.n_test} test "
            f"(exact 50/50)\n"
        )


# ==============================================================================
# 2. NUMBER-THEORETIC ATTACK (proves the trapdoor is real, not just empirical)
# ==============================================================================

def legendre_attack(concept: DiscreteLogConceptClass, df: pd.DataFrame) -> Dict[str, float]:
    """
    Polynomial-time classical attack using Euler's criterion.

        h^((p-1)/2) mod p == 1  <=>  h is a quadratic residue  <=>  log_g(h) even

    This costs O(log p) modular multiplications -- it does NOT solve the discrete
    logarithm, it extracts only its least significant bit.

    Expected outcome
    ----------------
      --labeling parity        -> ~100% accuracy. The parity concept is BROKEN;
                                  it is not a valid quantum-advantage instance.
      --labeling half_interval -> ~50% accuracy. One bit of log_g(h) says nothing
                                  about half-interval membership under an unknown
                                  shift s. Trapdoor intact.
    """
    out: Dict[str, float] = {}
    for split in ("train", "test"):
        sub = df[df["split"] == split]
        preds = []
        for h in sub["group_element_h"].to_numpy():
            is_qr = pow(int(h), (concept.p - 1) // 2, concept.p) == 1
            preds.append(1 if is_qr else -1)          # QR <=> even dlog
        out[f"{split}_accuracy"] = float(
            accuracy_score(sub["label"].to_numpy(), np.asarray(preds))
        )
    return out


# ==============================================================================
# 3. CLASSICAL BASELINES
# ==============================================================================

class ClassicalBaselineEvaluator:
    """
    Trains classical learners on the binary encoding of the group element and
    demonstrates that each memorises the training split while generalising at
    chance level.

    A `leakage_control` model is additionally trained on the HIDDEN exponent
    `raw_input_x`. It reaches near-perfect accuracy. That is the crucial control:
    it proves the label is a clean deterministic function of the discrete log, so
    the ~50% test accuracy of the other models is a *hardness* result about the
    encoding h = g^x mod p, not label noise or a broken generator.
    """

    def __init__(
        self,
        concept: DiscreteLogConceptClass,
        df: pd.DataFrame,
        seed: int = 20260828,
        tune: bool = True,
    ) -> None:
        self.concept = concept
        self.df = df
        self.seed = seed
        self.tune = tune
        self.feature_cols = concept.feature_columns

        for col in HIDDEN_COLUMNS:
            assert col not in self.feature_cols, f"SECRET COLUMN {col} leaked into features"

        train = df[df["split"] == "train"]
        test = df[df["split"] == "test"]
        self.X_train = train[self.feature_cols].to_numpy(dtype=float)
        self.y_train = train["label"].to_numpy(dtype=int)
        self.X_test = test[self.feature_cols].to_numpy(dtype=float)
        self.y_test = test["label"].to_numpy(dtype=int)
        self.x_raw_train = train[["raw_input_x"]].to_numpy(dtype=float)
        self.x_raw_test = test[["raw_input_x"]].to_numpy(dtype=float)
        self.results: Dict[str, Dict[str, object]] = {}

    # ---------------------------------------------------------------- models
    def _model_zoo(self) -> Dict[str, object]:
        return {
            "RBF-SVM": Pipeline([
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=self.seed)),
            ]),
            "RandomForest": RandomForestClassifier(
                n_estimators=100, random_state=self.seed, n_jobs=-1
            ),
            "MLP (64,32)": Pipeline([
                ("scale", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(64, 32), max_iter=3000,
                    random_state=self.seed, early_stopping=False,
                )),
            ]),
        }

    @staticmethod
    def _scores(model, X: np.ndarray) -> np.ndarray:
        """Continuous scores for AUC, whichever API the estimator exposes."""
        if hasattr(model, "decision_function"):
            try:
                return np.asarray(model.decision_function(X)).ravel()
            except Exception:
                pass
        proba = model.predict_proba(X)
        idx = list(model.classes_).index(1)
        return np.asarray(proba)[:, idx]

    def _evaluate(self, name, model, Xtr, ytr, Xte, yte, verbose) -> Dict[str, object]:
        t0 = time.perf_counter()
        model.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0

        pred_tr, pred_te = model.predict(Xtr), model.predict(Xte)
        try:
            auc = float(roc_auc_score(yte, self._scores(model, Xte)))
        except Exception:
            auc = float("nan")

        res = {
            "train_accuracy": float(accuracy_score(ytr, pred_tr)),
            "test_accuracy": float(accuracy_score(yte, pred_te)),
            "test_auc": auc,
            "fit_seconds": round(fit_s, 4),
            "confusion_matrix": confusion_matrix(yte, pred_te, labels=[-1, 1]).tolist(),
        }
        self.results[name] = res

        if verbose:
            print(f"\n  --- {name} " + "-" * max(0, 62 - len(name)))
            print(f"  train accuracy : {res['train_accuracy']:.4f}   "
                  f"(memorisation capacity)")
            print(f"  TEST  accuracy : {res['test_accuracy']:.4f}   "
                  f"TEST AUC : {auc:.4f}   fit {fit_s:.2f}s")
            print("  classification report (test split):")
            report = classification_report(
                yte, pred_te, labels=[-1, 1],
                target_names=["class -1", "class +1"], zero_division=0, digits=4,
            )
            print("".join("    " + line + "\n" for line in report.strip().splitlines()))
        return res

    # ------------------------------------------------------------------ main
    def train_and_evaluate(self, verbose: bool = True) -> Dict[str, Dict[str, object]]:
        if verbose:
            print("\n" + "=" * 78)
            print(" [2] CLASSICAL BASELINES  --  features = binary_feature_0..k only")
            print("=" * 78)

        for name, model in self._model_zoo().items():
            self._evaluate(name, model, self.X_train, self.y_train,
                           self.X_test, self.y_test, verbose)

        if self.tune:
            grid = GridSearchCV(
                Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="rbf"))]),
                param_grid={
                    "clf__C": [0.1, 1.0, 10.0, 100.0, 1000.0],
                    "clf__gamma": ["scale", 0.01, 0.1, 1.0, 10.0],
                },
                cv=StratifiedKFold(5, shuffle=True, random_state=self.seed),
                scoring="accuracy", n_jobs=-1,
            )
            res = self._evaluate("RBF-SVM (tuned)", grid, self.X_train, self.y_train,
                                 self.X_test, self.y_test, verbose)
            res["best_params"] = {k: str(v) for k, v in grid.best_params_.items()}
            res["cv_best_accuracy"] = float(grid.best_score_)
            if verbose:
                print(f"    best hyperparameters : {res['best_params']}")
                print(f"    5-fold CV accuracy   : {grid.best_score_:.4f} "
                      f"(chance = 0.5000)")

        # Leakage control on the HIDDEN exponent -- not a competing baseline.
        control = RandomForestClassifier(n_estimators=300, random_state=self.seed, n_jobs=-1)
        if verbose:
            print("\n  --- CONTROL: RandomForest on the HIDDEN exponent raw_input_x " + "-" * 6)
            print("      (not a baseline -- proves the concept is perfectly learnable")
            print("       ONCE the discrete logarithm is known)")
        self._evaluate("CONTROL: RF on hidden dlog x", control,
                       self.x_raw_train, self.y_train,
                       self.x_raw_test, self.y_test, verbose)
        return self.results


# ==============================================================================
# 4. QUANTUM FEATURE MAPS AND KERNELS
# ==============================================================================

class CosetStateFeatureMap:
    r"""
    The Liu-Arunachalam-Temme feature map (arXiv:2010.02174, Section 2).

        |Phi(h)> = (1 / sqrt(M)) * sum_{i=0}^{M-1} |h * g^i mod p>,     M = 2^m

    The register holds a group element as a computational basis state, so it
    needs n = ceil(log2 p) qubits.

    Closed-form kernel
    ------------------
    Write a = log_g(h), a' = log_g(h'), N = p - 1, delta = (a' - a) mod N. The
    two supports are arithmetic progressions of length M on the exponent circle
    Z_N, so their intersection has size

        Omega(delta) = max(0, M - delta) + max(0, M - (N - delta))      (2M <= N)

    and the fidelity kernel is

        K(h, h') = |<Phi(h)|Phi(h')>|^2 = (Omega(delta) / M)^2.

    This is a circular TRIANGULAR kernel in DISCRETE-LOG SPACE. That is the whole
    trick: the quantum state performs the change of coordinates h -> log_g(h) for
    free, turning a cryptographically scrambled problem into a one-dimensional
    interval-membership problem that a max-margin classifier solves trivially.

    Preparation cost -- the honest statement
    ----------------------------------------
    Here |Phi(h)> is written down explicitly and, for verification, compiled with
    Qiskit's `StatePreparation`. That synthesis is exponential in n and is a
    SIMULATION CONVENIENCE, not the asymptotic algorithm. At cryptographic scale
    the same state is produced in O(n^3) gates by Shor-style period finding:
    prepare a uniform superposition over i in [0, M), compute |i>|h * g^i mod p>
    by modular exponentiation, then uncompute the index register with a QFT. No
    step of that routine ever evaluates log_g(h).
    """

    def __init__(self, concept: DiscreteLogConceptClass, window_bits: Optional[int] = None):
        self.concept = concept
        self.n_qubits = concept.n_bits
        self.dim = 2 ** self.n_qubits

        if window_bits is None:
            window_bits = max(1, int(round(math.log2(max(2.0, concept.order / 8.0)))))
        self.window_bits = int(window_bits)
        self.window = 2 ** self.window_bits
        if 2 * self.window > concept.order:
            raise ValueError(
                f"window 2^{self.window_bits} = {self.window} too large: the closed-form "
                f"overlap requires 2*M <= p-1 = {concept.order}. Lower --window-bits."
            )
        self._cache: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------- amplitudes
    def statevector(self, h: int) -> np.ndarray:
        """Dense amplitude vector of |Phi(h)> (length 2^n, real, normalised)."""
        h = int(h)
        cached = self._cache.get(h)
        if cached is not None:
            return cached
        vec = np.zeros(self.dim, dtype=complex)
        amplitude = 1.0 / math.sqrt(self.window)
        element = h % self.concept.p
        for _ in range(self.window):
            vec[element] += amplitude
            element = (element * self.concept.g) % self.concept.p
        self._cache[h] = vec
        return vec

    def circuit(self, h: int, label: str = "U_Phi") -> QuantumCircuit:
        """Qiskit circuit preparing |Phi(h)> from |0...0> (verification path)."""
        qc = QuantumCircuit(self.n_qubits, name=f"{label}({h})")
        qc.append(StatePreparation(self.statevector(h)), range(self.n_qubits))
        return qc

    # ----------------------------------------------------------- closed form
    def analytic_kernel(self, h_a: int, h_b: int) -> float:
        N, M = self.concept.order, self.window
        delta = (self.concept.dlog(int(h_b)) - self.concept.dlog(int(h_a))) % N
        overlap = max(0, M - delta) + max(0, M - (N - delta))
        return (overlap / M) ** 2


class LadderFeatureMap:
    r"""
    NEGATIVE CONTROL -- a generic hardware-efficient encoding of the raw bits.

    Per repetition, on the n bit-features b_j with phi_j = (pi/2) * b_j:

        H on every qubit
        Rz(2 phi_j) on qubit j
        CX(j, j+1) . Rz(2 (pi/2 - phi_j)(pi/2 - phi_{j+1})) . CX(j, j+1)   [ZZ term]
        Rx(2 phi_j) on qubit j
        CZ(j, j+1) ladder

    This is the Havlicek-style second-order map the brief specifies, and it is a
    *strong* ansatz: it entangles, it is classically hard to simulate in general,
    and its kernel is not efficiently computable classically. It still cannot
    classify this dataset, because expressivity is not the binding constraint --
    the encoding is. K(h, h') here is a function of the BITS of h, while the
    label is a function of log_g(h). Nothing in the circuit inverts that map, so
    the kernel is blind to the concept and concentrates towards the identity,
    which is exactly the "quantum overfitting" failure mode.

    Including this row is what makes the headline claim falsifiable: quantum
    hardware alone does not buy the separation; the number-theoretic feature map
    does.
    """

    def __init__(self, n_qubits: int, reps: int = 2):
        self.n_qubits = int(n_qubits)
        self.reps = int(reps)
        self.params = ParameterVector("x", self.n_qubits)
        self.template = self._build_template()

    def _build_template(self) -> QuantumCircuit:
        n = self.n_qubits
        qc = QuantumCircuit(n, name="U_Phi_ladder")
        half_pi = np.pi / 2.0
        for rep in range(self.reps):
            for j in range(n):
                qc.h(j)
            for j in range(n):
                qc.rz(2.0 * self.params[j], j)
            for j in range(n - 1):
                qc.cx(j, j + 1)
                qc.rz(2.0 * (half_pi - self.params[j]) * (half_pi - self.params[j + 1]), j + 1)
                qc.cx(j, j + 1)
            for j in range(n):
                qc.rx(2.0 * self.params[j], j)
            for j in range(n - 1):
                qc.cz(j, j + 1)
            if rep < self.reps - 1:
                qc.barrier()
        return qc

    def circuit(self, bits: Sequence[int]) -> QuantumCircuit:
        angles = [(np.pi / 2.0) * float(b) for b in bits]
        return self.template.assign_parameters(dict(zip(self.params, angles)))

    def statevector(self, bits: Sequence[int]) -> np.ndarray:
        return np.asarray(Statevector(self.circuit(bits)).data, dtype=complex)


class QuantumKernelEvaluator:
    """
    Builds a quantum feature map, computes the fidelity Gram matrix

        K_ij = |<0^n| U_Phi^dag(x_j) U_Phi(x_i) |0^n>|^2

    and trains SVC(kernel='precomputed') on it.

    Both kernels are evaluated by EXACT statevector simulation, so the reported
    numbers carry no shot noise. The compute-uncompute circuit route (literally
    building U^dag(x_j) U(x_i) in Qiskit and reading the |0^n> amplitude) is
    cross-checked on a random subset in `verify_kernel`, confirming the fast
    inner-product path is identical to the circuit definition to ~1e-12.
    """

    def __init__(
        self,
        concept: DiscreteLogConceptClass,
        df: pd.DataFrame,
        kind: str = "coset",
        window_bits: Optional[int] = None,
        ladder_reps: int = 2,
        seed: int = 20260828,
    ) -> None:
        if kind not in ("coset", "ladder"):
            raise ValueError("kind must be 'coset' or 'ladder'")
        self.concept = concept
        self.df = df
        self.kind = kind
        self.seed = seed
        self.window_bits = window_bits
        self.ladder_reps = ladder_reps
        self.feature_map = None

        train, test = df[df["split"] == "train"], df[df["split"] == "test"]
        self.h_train = train["group_element_h"].to_numpy(dtype=np.int64)
        self.h_test = test["group_element_h"].to_numpy(dtype=np.int64)
        self.bits_train = train[concept.feature_columns].to_numpy(dtype=int)
        self.bits_test = test[concept.feature_columns].to_numpy(dtype=int)
        self.y_train = train["label"].to_numpy(dtype=int)
        self.y_test = test["label"].to_numpy(dtype=int)
        self.results: Dict[str, object] = {}

    # ------------------------------------------------------------ feature map
    def build_quantum_feature_map(self):
        """Instantiate U_Phi for the selected encoding and return it."""
        if self.kind == "coset":
            self.feature_map = CosetStateFeatureMap(self.concept, self.window_bits)
        else:
            self.feature_map = LadderFeatureMap(self.concept.n_bits, self.ladder_reps)
        return self.feature_map

    def _statevectors(self, split: str) -> np.ndarray:
        if self.feature_map is None:
            self.build_quantum_feature_map()
        if self.kind == "coset":
            source = self.h_train if split == "train" else self.h_test
            return np.stack([self.feature_map.statevector(int(h)) for h in source])
        rows = self.bits_train if split == "train" else self.bits_test
        return np.stack([self.feature_map.statevector(row) for row in rows])

    # ----------------------------------------------------------------- kernel
    def compute_kernel_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (K_train_train, K_test_train), both fidelity matrices.

        Amplitudes are simulated once per sample and the Gram matrix follows from
        a single BLAS call: O(N^2 * 2^n) instead of O(N^2) circuit executions.
        Mathematically identical to compute-uncompute (verified below).
        """
        t0 = time.perf_counter()
        psi_train = self._statevectors("train")
        psi_test = self._statevectors("test")
        self.results["statevector_seconds"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        K_train = np.abs(psi_train.conj() @ psi_train.T) ** 2
        K_test = np.abs(psi_test.conj() @ psi_train.T) ** 2
        self.results["gram_seconds"] = round(time.perf_counter() - t0, 3)

        K_train = np.clip(np.real(K_train), 0.0, 1.0)
        K_test = np.clip(np.real(K_test), 0.0, 1.0)
        np.fill_diagonal(K_train, 1.0)

        self.K_train, self.K_test = K_train, K_test
        n = K_train.shape[0]
        eigenvalues = np.linalg.eigvalsh(K_train)
        self.results.update({
            "kernel_min_eigenvalue": float(eigenvalues.min()),
            "kernel_offdiag_mean": float((K_train.sum() - np.trace(K_train)) / (n * n - n)),
        })
        return K_train, K_test

    def verify_kernel(self, n_pairs: int = 6, verbose: bool = True) -> Dict[str, float]:
        """
        Two independent audits of the Gram matrix:
          (a) compute-uncompute -- build U^dag(x_j) U(x_i) in Qiskit and read the
              probability of |0^n>; compare to K_ij.
          (b) closed form (coset map only) -- compare to the triangular-overlap
              formula, which never touches a simulator.
        """
        rng = np.random.default_rng(self.seed)
        n = self.K_train.shape[0]
        pairs = [(int(rng.integers(n)), int(rng.integers(n))) for _ in range(n_pairs)]

        circuit_err = 0.0
        for i, j in pairs:
            if self.kind == "coset":
                left = self.feature_map.circuit(int(self.h_train[i]))
                right = self.feature_map.circuit(int(self.h_train[j]))
            else:
                left = self.feature_map.circuit(self.bits_train[i])
                right = self.feature_map.circuit(self.bits_train[j])
            overlap_circuit = left.compose(right.inverse())
            fidelity = float(abs(Statevector(overlap_circuit).data[0]) ** 2)
            circuit_err = max(circuit_err, abs(fidelity - float(self.K_train[i, j])))

        out: Dict[str, float] = {"max_compute_uncompute_error": circuit_err}

        if self.kind == "coset":
            analytic_err = 0.0
            limit = min(n, 60)
            for i in range(limit):
                for j in range(limit):
                    predicted = self.feature_map.analytic_kernel(
                        int(self.h_train[i]), int(self.h_train[j])
                    )
                    analytic_err = max(analytic_err, abs(predicted - float(self.K_train[i, j])))
            out["max_analytic_error"] = analytic_err

        self.results.update(out)
        if verbose:
            print(f"  kernel audit : max |circuit fidelity - K_ij| = "
                  f"{circuit_err:.3e}  (compute-uncompute, {n_pairs} random pairs)")
            if "max_analytic_error" in out:
                print(f"               : max |closed form  - K_ij| = "
                      f"{out['max_analytic_error']:.3e}  (triangular-overlap formula)")
        return out

    # ------------------------------------------------------------------ QSVM
    def train_qsvm(self, verbose: bool = True) -> Dict[str, object]:
        if not hasattr(self, "K_train"):
            self.compute_kernel_matrix()

        grid = GridSearchCV(
            SVC(kernel="precomputed"),
            param_grid={"C": [0.1, 1.0, 10.0, 100.0, 1000.0]},
            cv=StratifiedKFold(5, shuffle=True, random_state=self.seed),
            scoring="accuracy", n_jobs=-1,
        )
        t0 = time.perf_counter()
        grid.fit(self.K_train, self.y_train)
        fit_s = time.perf_counter() - t0

        model = grid.best_estimator_
        pred_tr = model.predict(self.K_train)
        pred_te = model.predict(self.K_test)
        try:
            auc = float(roc_auc_score(self.y_test, model.decision_function(self.K_test)))
        except Exception:
            auc = float("nan")

        self.results.update({
            "train_accuracy": float(accuracy_score(self.y_train, pred_tr)),
            "test_accuracy": float(accuracy_score(self.y_test, pred_te)),
            "test_auc": auc,
            "fit_seconds": round(fit_s, 4),
            "best_C": float(grid.best_params_["C"]),
            "cv_best_accuracy": float(grid.best_score_),
            "n_support_vectors": int(model.n_support_.sum()),
            "confusion_matrix": confusion_matrix(self.y_test, pred_te, labels=[-1, 1]).tolist(),
        })
        self.model, self.y_pred_test = model, pred_te

        if verbose:
            print(f"  train accuracy : {self.results['train_accuracy']:.4f}")
            print(f"  TEST  accuracy : {self.results['test_accuracy']:.4f}   "
                  f"TEST AUC : {auc:.4f}")
            print(f"  best C = {grid.best_params_['C']}, "
                  f"5-fold CV = {grid.best_score_:.4f}, "
                  f"support vectors = {self.results['n_support_vectors']}")
            print("  classification report (test split):")
            report = classification_report(
                self.y_test, pred_te, labels=[-1, 1],
                target_names=["class -1", "class +1"], zero_division=0, digits=4,
            )
            print("".join("    " + line + "\n" for line in report.strip().splitlines()))
        return self.results


# ==============================================================================
# 5. OPTIONAL CROSS-CHECK AGAINST qiskit-machine-learning
# ==============================================================================

def qml_crosscheck(evaluator: QuantumKernelEvaluator, n_sub: int = 12) -> Dict[str, object]:
    """
    Recompute a small sub-block of the LADDER Gram matrix with
    `qiskit_machine_learning.kernels.FidelityQuantumKernel` and compare.

    Kept optional and small: FidelityQuantumKernel runs one circuit per pair, so
    a full 200x200 block is ~20k executions. Skipped cleanly if the package or
    its fidelity backend is unavailable.
    """
    try:
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
    except Exception as exc:  # pragma: no cover
        return {"status": "skipped", "reason": f"import failed: {exc}"}
    if evaluator.kind != "ladder":
        return {"status": "skipped", "reason": "cross-check applies to the ladder map only"}

    try:
        fmap = evaluator.feature_map or evaluator.build_quantum_feature_map()
        angles = (np.pi / 2.0) * evaluator.bits_train[:n_sub].astype(float)

        # enforce_psd=False is REQUIRED for a like-for-like comparison. With the
        # default (True), FidelityQuantumKernel projects out negative eigenvalues
        # by eigendecomposing and clipping. On this kernel -- already PSD and
        # already exact -- that projection is pure distortion: it pushes diagonal
        # entries off 1.0 (observed: 1.0245) and manufactures negative
        # off-diagonals, producing a spurious ~0.3 disagreement. Disabling it
        # recovers agreement at the 1e-13 level.
        qk = FidelityQuantumKernel(feature_map=fmap.template, enforce_psd=False)
        t0 = time.perf_counter()
        K_ref = np.asarray(qk.evaluate(x_vec=angles))
        elapsed = time.perf_counter() - t0
        delta = float(np.max(np.abs(K_ref - evaluator.K_train[:n_sub, :n_sub])))
        return {"status": "ok", "n_sub": n_sub, "enforce_psd": False,
                "max_abs_difference": delta, "seconds": round(elapsed, 2)}
    except Exception as exc:  # pragma: no cover
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}


# ==============================================================================
# 6. PLOTS
# ==============================================================================

def render_plots(concept, df, coset, ladder, summary_rows, outdir) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  [plots skipped: {exc}]")
        return []

    os.makedirs(outdir, exist_ok=True)
    written: List[str] = []
    train = df[df["split"] == "train"]
    order = np.argsort(train["raw_input_x"].to_numpy())   # sort by HIDDEN dlog

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4))
    panels = (
        (axes[0], coset, "COSET map (Shor-style)\nrows sorted by hidden dlog"),
        (axes[1], ladder, "LADDER map (Rz/Rx/CZ control)\nsame ordering"),
    )
    for ax, ev, title in panels:
        im = ax.imshow(ev.K_train[np.ix_(order, order)], cmap="magma", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("training sample (sorted by $\\log_g h$)")
        ax.set_ylabel("training sample (sorted by $\\log_g h$)")
        fig.colorbar(im, ax=ax, fraction=0.046,
                     label="$K_{ij}=|\\langle\\Phi_i|\\Phi_j\\rangle|^2$")
    fig.suptitle(
        "Quantum kernel Gram matrices: structure exists only in discrete-log coordinates",
        fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "kernel_matrices.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    names = [r[0] for r in summary_rows]
    accs = [r[2] for r in summary_rows]
    colors = ["#2a9d8f" if "COSET" in n else "#8d99ae" if "CONTROL" in n else "#e76f51"
              for n in names]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    bars = ax.barh(names[::-1], accs[::-1], color=colors[::-1])
    ax.axvline(0.5, ls="--", lw=1.6, color="#264653", label="chance (0.50)")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("test accuracy")
    ax.set_title(f"Generalisation on the DLog concept class  (p = {concept.p}, "
                 f"{concept.n_bits} qubits, {concept.n_test} held-out samples)")
    for bar, acc in zip(bars, accs[::-1]):
        ax.text(bar.get_width() + 0.012, bar.get_y() + bar.get_height() / 2,
                f"{acc:.3f}", va="center", fontsize=9)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = os.path.join(outdir, "accuracy_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


# ==============================================================================
# 7. MAIN
# ==============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Discrete-log quantum-advantage dataset generator and benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--qubits", type=int, default=10,
                    help="register width n; p = largest safe prime < 2^n")
    ap.add_argument("--train", type=int, default=200, help="training samples (even)")
    ap.add_argument("--test", type=int, default=50, help="test samples (even)")
    ap.add_argument("--seed", type=int, default=20260828, help="master RNG seed")
    ap.add_argument("--labeling", choices=["half_interval", "parity"],
                    default="half_interval",
                    help="half_interval = LAT2021 (hard); parity = BROKEN control")
    ap.add_argument("--window-bits", type=int, default=None,
                    help="coset window m (M = 2^m); default ~ log2((p-1)/8)")
    ap.add_argument("--ladder-reps", type=int, default=2,
                    help="repetitions of the control feature map")
    ap.add_argument("--crypto-random", action="store_true",
                    help="draw the secret from `secrets` instead of SHA-256(seed)")
    ap.add_argument("--csv", default=DEFAULT_CSV, help="dataset output path")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="metrics/figure directory")
    ap.add_argument("--no-plots", action="store_true", help="skip matplotlib figures")
    ap.add_argument("--no-tune", action="store_true",
                    help="skip the tuned RBF-SVM grid search")
    ap.add_argument("--qml-crosscheck", action="store_true",
                    help="verify the ladder kernel against qiskit-machine-learning")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    t_start = time.perf_counter()

    print("=" * 78)
    print(" PROVABLE QUANTUM ADVANTAGE ON THE DISCRETE-LOGARITHM CONCEPT CLASS")
    print(" Liu, Arunachalam & Temme, Nature Physics 17, 1013 (2021) -- arXiv:2010.02174")
    print("=" * 78)

    # ---------------------------------------------------------------- [1] data
    print("\n" + "=" * 78)
    print(" [1] CRYPTOGRAPHIC PARAMETERS AND DATASET GENERATION")
    print("=" * 78)
    concept = DiscreteLogConceptClass(
        n_qubits=args.qubits, n_train=args.train, n_test=args.test,
        seed=args.seed, labeling=args.labeling, crypto_random=args.crypto_random,
    )
    print(concept.describe())
    if args.labeling == "parity":
        print("  !! WARNING: parity labelling has NO trapdoor. The LSB of log_g(h) is")
        print("  !! the Legendre symbol of h, computable in O(log p). Control only.\n")

    df = concept.generate_samples()
    csv_path = concept.save_csv(args.csv)
    print("  dataset verified (h = g^x, labels, balance, split disjointness, BSGS check)")
    print(f"  rows x cols            : {df.shape[0]} x {df.shape[1]}")
    print(f"  written to             : {csv_path}")
    print(f"  learner-visible columns: {concept.feature_columns[0]} .. "
          f"{concept.feature_columns[-1]}")
    print(f"  withheld from learners : {', '.join(HIDDEN_COLUMNS)}  (the hidden dlog)")

    # ------------------------------------------------ [1b] number-theory attack
    print("\n" + "-" * 78)
    print(" [1b] NUMBER-THEORETIC ATTACK -- Legendre symbol / Euler's criterion")
    print("-" * 78)
    legendre = legendre_attack(concept, df)
    print(f"  test accuracy of the O(log p) quadratic-residuosity attack : "
          f"{legendre['test_accuracy']:.4f}")
    if args.labeling == "parity":
        print("  => parity labelling BROKEN by a polynomial-time classical algorithm.")
    else:
        print("  => chance level. One bit of log_g(h) says nothing about half-interval")
        print("     membership under an unknown shift s. The trapdoor holds.")

    # ------------------------------------------------------------ [2] classical
    classical = ClassicalBaselineEvaluator(concept, df, seed=args.seed, tune=not args.no_tune)
    classical_results = classical.train_and_evaluate(verbose=True)

    # ------------------------------------------------------------- [3] quantum
    print("\n" + "=" * 78)
    print(" [3] QUANTUM KERNEL EVALUATION  (exact statevector simulation, no shot noise)")
    print("=" * 78)

    print("\n  --- (a) LADDER feature map -- hardware-efficient Rz/Rx/CZ CONTROL " + "-" * 11)
    ladder = QuantumKernelEvaluator(concept, df, kind="ladder",
                                    ladder_reps=args.ladder_reps, seed=args.seed)
    ladder_map = ladder.build_quantum_feature_map()
    print(f"  qubits {ladder_map.n_qubits} | reps {ladder_map.reps} | "
          f"depth {ladder_map.template.depth()} | "
          f"gates {sum(ladder_map.template.count_ops().values())}")
    ladder.compute_kernel_matrix()
    ladder.verify_kernel(verbose=True)
    print(f"  mean off-diagonal kernel value = {ladder.results['kernel_offdiag_mean']:.4f} "
          f"(-> 0 means a near-identity, over-expressive kernel)")
    ladder_results = ladder.train_qsvm(verbose=True)

    print("  --- (b) COSET feature map -- Liu/Arunachalam/Temme Shor-style " + "-" * 15)
    coset = QuantumKernelEvaluator(concept, df, kind="coset",
                                   window_bits=args.window_bits, seed=args.seed)
    coset_map = coset.build_quantum_feature_map()
    print(f"  qubits {coset_map.n_qubits} | coset window M = 2^{coset_map.window_bits} = "
          f"{coset_map.window} of |Z_p^*| = {concept.order}")
    print("  kernel = squared triangular overlap on the discrete-log circle")
    coset.compute_kernel_matrix()
    coset.verify_kernel(verbose=True)
    print(f"  mean off-diagonal kernel value = {coset.results['kernel_offdiag_mean']:.4f}   "
          f"min eigenvalue = {coset.results['kernel_min_eigenvalue']:.2e} (PSD)")
    coset_results = coset.train_qsvm(verbose=True)

    crosscheck: Dict[str, object] = {"status": "not requested"}
    if args.qml_crosscheck:
        print("  --- qiskit-machine-learning cross-check (FidelityQuantumKernel) " + "-" * 13)
        crosscheck = qml_crosscheck(ladder)
        if crosscheck.get("status") == "ok":
            print(f"  max |K_qml - K_ours| on a {crosscheck['n_sub']}x{crosscheck['n_sub']} "
                  f"block = {crosscheck['max_abs_difference']:.3e} "
                  f"({crosscheck['seconds']}s)")
        else:
            print(f"  skipped: {crosscheck.get('reason')}")

    # -------------------------------------------------------------- [4] summary
    rows: List[Tuple[str, float, float, float]] = []
    for name in ("RBF-SVM", "RandomForest", "MLP (64,32)", "RBF-SVM (tuned)"):
        if name in classical_results:
            r = classical_results[name]
            rows.append((f"CLASSICAL  {name}", r["train_accuracy"],
                         r["test_accuracy"], r["test_auc"]))
    rows.append(("CLASSICAL  Legendre-symbol attack", legendre["train_accuracy"],
                 legendre["test_accuracy"], float("nan")))
    rows.append(("QUANTUM    LADDER kernel (control)", ladder_results["train_accuracy"],
                 ladder_results["test_accuracy"], ladder_results["test_auc"]))
    rows.append(("QUANTUM    COSET kernel (LAT2021)", coset_results["train_accuracy"],
                 coset_results["test_accuracy"], coset_results["test_auc"]))
    ctrl = classical_results["CONTROL: RF on hidden dlog x"]
    rows.append(("CONTROL    RF on hidden dlog x", ctrl["train_accuracy"],
                 ctrl["test_accuracy"], ctrl["test_auc"]))

    print("\n" + "=" * 78)
    print(" [4] FINAL COMPARISON")
    print("=" * 78)
    n_test = int(args.test)
    print(f"  Held-out n = {n_test}. Chance = 0.5000, binomial s.e. = "
          f"{0.5 / math.sqrt(n_test):.4f}. 'p' is a two-sided exact binomial")
    print("  test against chance; a score is only evidence of learning if p < 0.01.")
    header = (f"  {'MODEL':<36}{'TRAIN':>8}{'TEST':>8}{'AUC':>8}"
              f"{'95% CI':>16}{'p':>10}   VERDICT")
    print(header)
    print("  " + "-" * (len(header) - 2))

    stats: Dict[str, Dict[str, float]] = {}
    for name, tr, te, auc in rows:
        lo, hi = wilson_interval(te, n_test)
        pval = exact_binomial_p(te, n_test)
        stats[name] = {"test_accuracy": te, "ci_low": lo, "ci_high": hi, "p_value": pval}

        auc_s = "n/a" if math.isnan(auc) else f"{auc:.4f}"
        ci_s = f"[{lo:.3f},{hi:.3f}]"
        p_s = "<1e-6" if pval < 1e-6 else f"{pval:.4f}"
        if name.startswith("CONTROL"):
            verdict = "leak check"
        elif pval < 0.01 and te > 0.5:
            verdict = "LEARNS"
        else:
            verdict = "chance (n.s.)"
        print(f"  {name:<36}{tr:>8.4f}{te:>8.4f}{auc_s:>8}{ci_s:>16}{p_s:>10}   {verdict}")

    classical_names = [n for n, _, _, _ in rows if not n.startswith("CONTROL")
                       and "COSET" not in n]
    best_classical_name = max(classical_names, key=lambda n: stats[n]["test_accuracy"])
    best_classical = stats[best_classical_name]["test_accuracy"]
    quantum = coset_results["test_accuracy"]
    gap = quantum - best_classical
    any_significant = any(
        stats[n]["p_value"] < 0.01 and stats[n]["test_accuracy"] > 0.5
        for n in classical_names
    )

    print("\n  " + "-" * 74)
    print(f"  best non-coset test accuracy      : {best_classical:.4f}  "
          f"({best_classical_name.strip()})")
    print(f"  COSET quantum kernel test accuracy: {quantum:.4f}  "
          f"(p = {stats['QUANTUM    COSET kernel (LAT2021)']['p_value']:.3e})")
    print(f"  SEPARATION                        : {gap:+.4f} "
          f"({gap * 100:+.1f} percentage points)")
    print(f"  any non-coset model above chance at p < 0.01 ? "
          f"{'YES -- INVESTIGATE' if any_significant else 'NO'}")
    print("  " + "-" * 74)
    print("  Two caveats to quote alongside the headline:")
    print("   1. The LADDER row shows a generic quantum feature map fails here too.")
    print("      The separation comes from the Shor-style coset encoding, not from")
    print("      being quantum.")
    print("   2. At p = %d the discrete log is brute-forceable. This demonstrates the"
          % concept.p)
    print("      learning-theoretic structure; hardness is asymptotic. README Sec. 6-7.")

    # ---------------------------------------------------------------- artefacts
    os.makedirs(args.outdir, exist_ok=True)
    metrics = {
        "config": vars(args),
        "group": {
            "p": concept.p, "q": concept.q, "g": concept.g,
            "order": concept.order, "secret_s": concept.secret_s,
            "n_bits": concept.n_bits, "labeling": concept.labeling,
        },
        "classical": classical_results,
        "legendre_attack": legendre,
        "quantum_ladder": ladder.results,
        "quantum_coset": coset.results,
        "qml_crosscheck": crosscheck,
        "summary": [
            {
                "model": n, "train_accuracy": tr, "test_accuracy": te, "test_auc": auc,
                "wilson_ci_95": [stats[n]["ci_low"], stats[n]["ci_high"]],
                "binomial_p_vs_chance": stats[n]["p_value"],
            }
            for n, tr, te, auc in rows
        ],
        "best_classical_test_accuracy": best_classical,
        "best_classical_model": best_classical_name.strip(),
        "any_classical_significant_at_1pct": bool(any_significant),
        "quantum_test_accuracy": quantum,
        "separation": gap,
        "theoretical_scale": THEORETICAL_SCALE,
    }
    metrics_path = os.path.join(args.outdir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, default=float)

    print("\n" + "=" * 78)
    print(" [5] ARTEFACTS")
    print("=" * 78)
    print(f"  dataset : {csv_path}")
    print(f"  metrics : {os.path.abspath(metrics_path)}")
    if not args.no_plots:
        for path in render_plots(concept, df, coset, ladder, rows, args.outdir):
            print(f"  figure  : {os.path.abspath(path)}")
    print(f"\n  total wall-clock: {time.perf_counter() - t_start:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
