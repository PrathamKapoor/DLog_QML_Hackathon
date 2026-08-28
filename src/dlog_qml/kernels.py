"""
Fidelity kernel construction, diagnostics, verification, and the kernel SVM.

The quantity every quantum kernel here computes is

    K_ij = |<0^n| U_Phi^dag(x_j) U_Phi(x_i) |0^n>|^2
         = |<Phi(x_j)|Phi(x_i)>|^2

evaluated by EXACT statevector simulation, so reported numbers carry no shot
noise. `verify_against_circuits` independently rebuilds the literal
compute-uncompute circuit U^dag(x_j) U(x_i) in Qiskit and reads the |0^n>
amplitude, confirming the fast inner-product route agrees with the circuit
definition to ~1e-13.

Efficiency note: simulating each state once and taking one BLAS product costs
O(N^2 * 2^n) rather than O(N^2) circuit executions. On physical hardware the
matrix would instead be estimated entry-by-entry from compute-uncompute
sampling; see docs/LIMITATIONS.md for the shot-cost discussion.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from qiskit.quantum_info import Statevector
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


# ---------------------------------------------------------------------------
# Gram matrices
# ---------------------------------------------------------------------------

def fidelity_gram(psi_a: np.ndarray, psi_b: Optional[np.ndarray] = None,
                  symmetric: Optional[bool] = None) -> np.ndarray:
    """
    Fidelity Gram matrix |<a_i|b_j>|^2 from stacked amplitude vectors.

    Rows of `psi_a` / `psi_b` are statevectors. When `psi_b` is None the matrix
    is the symmetric self-Gram and its diagonal is pinned to exactly 1.0 (the
    states are normalised by construction; pinning removes float drift that
    would otherwise make the matrix look non-PSD to sklearn).
    """
    if psi_b is None:
        psi_b, symmetric = psi_a, True
    if symmetric is None:
        symmetric = psi_a is psi_b

    gram = np.abs(psi_a.conj() @ psi_b.T) ** 2
    gram = np.clip(np.real(gram), 0.0, 1.0)
    if symmetric:
        gram = 0.5 * (gram + gram.T)
        np.fill_diagonal(gram, 1.0)
    return gram


def kernel_target_alignment(K: np.ndarray, y: Sequence[int],
                            centered: bool = True) -> float:
    r"""
    Kernel-target alignment (Cristianini et al., 2001).

        A(K, Y) = <K, Y>_F / (||K||_F ||Y||_F),      Y = y y^T

    `Y` is the *ideal* kernel for a binary task: +1 for same-class pairs, -1
    for different-class pairs. Alignment therefore measures how much of the
    kernel's geometry actually points along the label structure, independent
    of any classifier. It answers a question accuracy cannot: *why* one kernel
    generalises and another does not.

    With `centered=True` the kernel is first centred in feature space,
    `K_c = H K H` with `H = I - 11^T/n`. Centring matters here: an uncentred
    alignment is inflated by any constant offset in `K`, which would flatter a
    concentrated kernel for the wrong reason.

    Range is [-1, 1]; higher means better aligned with the labels.
    """
    y = np.asarray(y, dtype=float).ravel()
    K = np.asarray(K, dtype=float)
    if K.shape[0] != y.size:
        raise ValueError("K and y have incompatible shapes")

    if centered:
        n = K.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K = H @ K @ H

    Y = np.outer(y, y)
    denominator = np.linalg.norm(K, "fro") * np.linalg.norm(Y, "fro")
    if denominator <= 1e-15:                              # pragma: no cover
        return float("nan")
    return float((K * Y).sum() / denominator)


def class_conditional_similarity(K: np.ndarray, y: Sequence[int]) -> Dict[str, float]:
    """
    Mean kernel value for same-class vs different-class pairs.

    The geometric explanation behind the accuracy numbers. A kernel useful for
    classification must place same-class points closer than different-class
    points:

        within  >> between   ->  usable class structure
        within  ~= between   ->  the kernel is blind to the labels

    The diagonal is excluded (it is 1 by construction and would bias `within`).
    `separation_ratio` is reported as within/between and is only meaningful
    when `between` is not numerically zero.
    """
    K = np.asarray(K, dtype=float)
    y = np.asarray(y).ravel()
    n = K.shape[0]

    same = np.equal.outer(y, y)
    off_diagonal = ~np.eye(n, dtype=bool)
    within_mask = same & off_diagonal
    between_mask = (~same) & off_diagonal

    within = float(K[within_mask].mean()) if within_mask.any() else float("nan")
    between = float(K[between_mask].mean()) if between_mask.any() else float("nan")
    ratio = float(within / between) if between > 1e-12 else float("inf")

    return {
        "within_class_mean": within,
        "between_class_mean": between,
        "separation_gap": float(within - between),
        "separation_ratio": ratio,
        "within_class_std": float(K[within_mask].std()) if within_mask.any() else float("nan"),
        "between_class_std": float(K[between_mask].std()) if between_mask.any() else float("nan"),
    }


def eigenvalue_spectrum(K: np.ndarray, top: int = 20) -> Dict[str, object]:
    """
    Eigenspectrum summary of a Gram matrix.

    A kernel that concentrates towards the identity has a flat spectrum (all
    eigenvalues near 1, effective rank near n). A kernel with real structure
    concentrates its mass in a few leading directions.
    """
    eigenvalues = np.linalg.eigvalsh(0.5 * (K + K.T))[::-1]   # descending
    total = float(eigenvalues.sum())
    cumulative = np.cumsum(eigenvalues) / total if total > 0 else eigenvalues

    def components_for(fraction: float) -> int:
        return int(np.searchsorted(cumulative, fraction) + 1)

    return {
        "eigenvalues": eigenvalues.tolist(),
        "top_eigenvalues": eigenvalues[:top].tolist(),
        "trace": total,
        "top1_fraction": float(eigenvalues[0] / total) if total > 0 else float("nan"),
        "top10_fraction": float(eigenvalues[:10].sum() / total) if total > 0 else float("nan"),
        "n_components_90pct": components_for(0.90),
        "n_components_99pct": components_for(0.99),
    }


def kernel_diagnostics(K: np.ndarray) -> Dict[str, float]:
    """
    Structural summary of a Gram matrix.

    `offdiag_mean` is the headline number for kernel concentration: a value
    approaching 0 means the kernel is effectively the identity, every point is
    its own island, and the SVM can only memorise. `effective_rank` is the
    exponential of the von Neumann entropy of the normalised eigenspectrum --
    a smooth, numerically stable stand-in for "how many directions the kernel
    actually uses".
    """
    n = K.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    off = K[off_mask]

    eigenvalues = np.linalg.eigvalsh(0.5 * (K + K.T))
    positive = eigenvalues[eigenvalues > 1e-12]
    if positive.size:
        weights = positive / positive.sum()
        entropy = float(-np.sum(weights * np.log(weights)))
        effective_rank = float(np.exp(entropy))
    else:                                                # pragma: no cover
        effective_rank = 0.0

    lam_max, lam_min = float(eigenvalues.max()), float(eigenvalues.min())
    condition = float(lam_max / lam_min) if lam_min > 1e-12 else float("inf")

    return {
        "diagonal_mean": float(np.mean(np.diag(K))),
        "offdiag_mean": float(off.mean()),
        "offdiag_std": float(off.std()),
        "offdiag_max": float(off.max()),
        "min_eigenvalue": lam_min,
        "max_eigenvalue": lam_max,
        "condition_number": condition,
        "effective_rank": effective_rank,
        "n": int(n),
    }


def full_kernel_report(K: np.ndarray, y: Sequence[int]) -> Dict[str, object]:
    """
    Everything we can say about a Gram matrix, in one object.

    Combines the structural diagnostics (concentration, effective rank) with
    the two label-aware measures (alignment, class-conditional separation) and
    the eigenspectrum. This is the mechanistic explanation layer: it turns
    "the generic kernel was bad" into "the generic kernel is near-orthogonal,
    spectrally flat, and has no class structure to exploit".
    """
    report: Dict[str, object] = dict(kernel_diagnostics(K))
    report["target_alignment_centered"] = kernel_target_alignment(K, y, centered=True)
    report["target_alignment_raw"] = kernel_target_alignment(K, y, centered=False)
    report["class_conditional"] = class_conditional_similarity(K, y)
    report["spectrum"] = eigenvalue_spectrum(K)
    return report


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_against_circuits(feature_map, inputs: Sequence, K: np.ndarray,
                            n_pairs: int = 6, seed: int = 0) -> Dict[str, float]:
    """
    Audit K against the literal compute-uncompute circuit.

    For random pairs (i, j) this builds U(x_i) followed by U(x_j)^dag, simulates
    it, and compares P(|0^n>) with K[i, j]. Agreement at ~1e-13 confirms the
    BLAS shortcut computes exactly the circuit-defined kernel and not something
    subtly different.
    """
    rng = np.random.default_rng(seed)
    n = len(inputs)
    max_error = 0.0
    checked = []
    for _ in range(n_pairs):
        i, j = int(rng.integers(n)), int(rng.integers(n))
        left = feature_map.circuit(inputs[i])
        right = feature_map.circuit(inputs[j])
        overlap = left.compose(right.inverse())
        fidelity = float(abs(Statevector(overlap).data[0]) ** 2)
        max_error = max(max_error, abs(fidelity - float(K[i, j])))
        checked.append((i, j))
    return {"max_compute_uncompute_error": float(max_error),
            "n_pairs_checked": int(n_pairs)}


def verify_kernel_properties(K: np.ndarray, tol: float = 1e-9) -> Dict[str, bool]:
    """Symmetry, unit diagonal, range, and positive semi-definiteness."""
    n = K.shape[0]
    eigenvalues = np.linalg.eigvalsh(0.5 * (K + K.T))
    return {
        "symmetric": bool(np.allclose(K, K.T, atol=tol)),
        "unit_diagonal": bool(np.allclose(np.diag(K), 1.0, atol=tol)),
        "in_unit_range": bool(K.min() >= -tol and K.max() <= 1.0 + tol),
        "positive_semidefinite": bool(eigenvalues.min() >= -1e-8),
        "square": bool(K.shape == (n, n)),
    }


# ---------------------------------------------------------------------------
# Kernel SVM
# ---------------------------------------------------------------------------

def train_kernel_svm(K_train: np.ndarray, y_train: np.ndarray,
                     K_test: np.ndarray, y_test: np.ndarray,
                     C_grid: Sequence[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
                     cv_folds: int = 5, seed: int = 0) -> Dict[str, object]:
    """
    Fit SVC(kernel='precomputed') with C selected by cross-validation on the
    TRAINING kernel only. The test kernel is never touched during selection.

    `K_test` must be shaped (n_test, n_train) -- sklearn's convention for a
    precomputed kernel at prediction time.
    """
    grid = GridSearchCV(
        SVC(kernel="precomputed"),
        param_grid={"C": list(C_grid)},
        cv=StratifiedKFold(cv_folds, shuffle=True, random_state=seed),
        scoring="accuracy",
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    grid.fit(K_train, y_train)
    fit_seconds = time.perf_counter() - t0

    model = grid.best_estimator_
    pred_train = model.predict(K_train)
    pred_test = model.predict(K_test)
    scores = model.decision_function(K_test)
    try:
        auc = float(roc_auc_score(y_test, scores))
    except ValueError:                                   # pragma: no cover
        auc = float("nan")

    return {
        "train_accuracy": float(accuracy_score(y_train, pred_train)),
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "test_auc": auc,
        "best_C": float(grid.best_params_["C"]),
        "cv_best_accuracy": float(grid.best_score_),
        "n_support_vectors": int(model.n_support_.sum()),
        "fit_seconds": round(fit_seconds, 4),
        "confusion_matrix": confusion_matrix(y_test, pred_test, labels=[-1, 1]).tolist(),
        "predictions": pred_test.tolist(),
        "scores": scores.tolist(),
    }


def build_quantum_kernels(feature_map, inputs_train, inputs_test
                          ) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Simulate all states once, then form both Gram matrices.

    Returns (K_train (n_tr, n_tr), K_test (n_te, n_tr), timings).
    """
    t0 = time.perf_counter()
    psi_train = feature_map.statevectors(inputs_train)
    psi_test = feature_map.statevectors(inputs_test)
    simulate_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    K_train = fidelity_gram(psi_train)
    K_test = fidelity_gram(psi_test, psi_train, symmetric=False)
    gram_seconds = time.perf_counter() - t0

    timings = {
        "simulate_seconds": round(simulate_seconds, 4),
        "gram_seconds": round(gram_seconds, 4),
        "statevector_dim": int(psi_train.shape[1]),
        "statevector_megabytes": float(
            (psi_train.nbytes + psi_test.nbytes) / 1024 ** 2
        ),
    }
    return K_train, K_test, timings
