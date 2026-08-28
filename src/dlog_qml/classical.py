"""
Classical baselines and the two classical control experiments.

All primary baselines see EXACTLY ONE thing: the LSB-first bit decomposition of
h = g^x mod p. They never receive the exponent x, the secret shift, or the group
order. `run_classical_baselines` asserts this at entry.

Two experiments in this module deliberately break that rule and are labelled
accordingly:

  hidden_x_oracle_control  -- trains on x itself. NOT a baseline. It is the
                              positive control establishing that the concept is
                              perfectly learnable once the right coordinate is
                              exposed, so that ~50% elsewhere is a statement
                              about the encoding rather than about label noise
                              or a broken generator.

  legendre_attack          -- a polynomial-time number-theoretic attack on h.
                              Breaks the parity concept, fails on the
                              half-interval concept.
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import ExperimentConfig
from .data import DlogDataset
from .labels import legendre_predict

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def _scores(model, X: np.ndarray) -> np.ndarray:
    """Continuous scores for AUC, whichever API the estimator exposes."""
    if hasattr(model, "decision_function"):
        try:
            return np.asarray(model.decision_function(X)).ravel()
        except (AttributeError, ValueError):
            pass
    proba = model.predict_proba(X)
    positive_index = list(model.classes_).index(1)
    return np.asarray(proba)[:, positive_index]


def evaluate_model(model, X_train, y_train, X_test, y_test) -> Dict[str, object]:
    """Fit and score one estimator. Records everything needed to reproduce it."""
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    try:
        auc = float(roc_auc_score(y_test, _scores(model, X_test)))
    except (ValueError, AttributeError):                 # pragma: no cover
        auc = float("nan")

    return {
        "train_accuracy": float(accuracy_score(y_train, pred_train)),
        "test_accuracy": float(accuracy_score(y_test, pred_test)),
        "test_auc": auc,
        "fit_seconds": round(fit_seconds, 4),
        "confusion_matrix": confusion_matrix(y_test, pred_test, labels=[-1, 1]).tolist(),
        "predictions": pred_test.tolist(),
        "classification_report": classification_report(
            y_test, pred_test, labels=[-1, 1],
            target_names=["class -1", "class +1"], zero_division=0, digits=4,
        ),
    }


def build_model_zoo(cfg: ExperimentConfig) -> Dict[str, object]:
    """
    The classical baselines, all with recorded hyperparameters.

    Tree models take raw bits; distance/gradient models are wrapped in a
    StandardScaler. Nothing here was tuned against the test split.
    """
    zoo: Dict[str, object] = {
        "RBF-SVM": Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=cfg.seed)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=cfg.rf_n_estimators, random_state=cfg.seed, n_jobs=-1,
        ),
        "MLP": Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=tuple(cfg.mlp_hidden),
                activation="relu", solver="adam", max_iter=cfg.mlp_max_iter,
                random_state=cfg.seed, early_stopping=False,
            )),
        ]),
    }

    if cfg.extra_baselines:
        # Not a huge suite -- just enough to show the result is not an artefact
        # of one weak model family. Linear and polynomial kernels, a different
        # tree ensemble, boosting, and a distance-based learner.
        zoo.update({
            "Linear-SVM": Pipeline([
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="linear", C=1.0, random_state=cfg.seed)),
            ]),
            "Poly-SVM (d=3)": Pipeline([
                ("scale", StandardScaler()),
                ("clf", SVC(kernel="poly", degree=3, C=1.0,
                            gamma="scale", random_state=cfg.seed)),
            ]),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=cfg.rf_n_estimators, random_state=cfg.seed, n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingClassifier(random_state=cfg.seed),
            "kNN (k=5)": Pipeline([
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=5)),
            ]),
        })
    return zoo


def tuned_rbf_svm(cfg: ExperimentConfig, X_train, y_train, X_test, y_test
                  ) -> Dict[str, object]:
    """
    Transparent CV grid over (C, gamma), selected on TRAIN ONLY.

    This exists so nobody can attribute classical failure to an untuned model.
    The grid, the CV scheme, and the winning hyperparameters are all recorded.
    """
    grid = GridSearchCV(
        Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="rbf"))]),
        param_grid={"clf__C": list(cfg.svm_C_grid),
                    "clf__gamma": list(cfg.svm_gamma_grid)},
        cv=StratifiedKFold(cfg.cv_folds, shuffle=True, random_state=cfg.seed),
        scoring="accuracy", n_jobs=-1,
    )
    result = evaluate_model(grid, X_train, y_train, X_test, y_test)
    result["best_params"] = {k: str(v) for k, v in grid.best_params_.items()}
    result["cv_best_accuracy"] = float(grid.best_score_)
    result["search_space"] = {"C": list(cfg.svm_C_grid),
                              "gamma": [str(v) for v in cfg.svm_gamma_grid],
                              "cv_folds": cfg.cv_folds}
    return result


def assert_no_leakage(dataset: DlogDataset, X: np.ndarray, split: str) -> None:
    """
    Hard guard run before every primary learner sees data.

    Verifies that the feature matrix is (a) exactly n_bits wide, (b) strictly
    binary, and (c) carries no column that reproduces the hidden exponent --
    checked both directly and against its low bits, since `x mod 2^j` would be
    a partial leak that a binary-only check would miss.
    """
    n_bits = dataset.group.n_bits
    if X.shape[1] != n_bits:
        raise AssertionError(f"feature width {X.shape[1]} != n_bits {n_bits}")
    if not set(np.unique(X)).issubset({0.0, 1.0}):
        raise AssertionError("learner features are not strictly binary")

    x_hidden = dataset.oracle_exponents(split).astype(np.int64)
    for column in range(X.shape[1]):
        col = X[:, column].astype(np.int64)
        if np.array_equal(col, x_hidden):
            raise AssertionError(f"column {column} leaks the hidden exponent")
        # A bit of x is a genuine partial leak; a bit of h is legitimate.
        if np.array_equal(col, x_hidden & 1):
            raise AssertionError(f"column {column} leaks a bit of the hidden exponent")

    expected = np.stack([dataset.group.to_bits(int(h))
                         for h in dataset.group_elements(split)]).astype(float)
    if not np.array_equal(X, expected):
        raise AssertionError("features are not the bit decomposition of h")


def run_classical_baselines(dataset: DlogDataset, cfg: ExperimentConfig
                            ) -> Dict[str, Dict[str, object]]:
    """Every primary classical learner, on the public bit features only."""
    X_train = dataset.features("train")
    X_test = dataset.features("test")
    y_train = dataset.labels("train")
    y_test = dataset.labels("test")

    assert_no_leakage(dataset, X_train, "train")
    assert_no_leakage(dataset, X_test, "test")

    results: Dict[str, Dict[str, object]] = {}
    for name, model in build_model_zoo(cfg).items():
        results[name] = evaluate_model(model, X_train, y_train, X_test, y_test)
    results["RBF-SVM (tuned)"] = tuned_rbf_svm(cfg, X_train, y_train, X_test, y_test)
    return results


def hidden_x_oracle_control(dataset: DlogDataset, cfg: ExperimentConfig
                            ) -> Dict[str, object]:
    """
    ORACLE / LEAKAGE CONTROL -- NOT a valid model under the task constraints.

    Trains a RandomForest directly on the hidden exponent x. Near-perfect
    accuracy here proves the labels are a clean deterministic function of the
    discrete-log coordinate. Without this control, ~50% everywhere else would
    be equally consistent with "the dataset is pure noise".
    """
    model = RandomForestClassifier(n_estimators=300, random_state=cfg.seed, n_jobs=-1)
    result = evaluate_model(
        model,
        dataset.oracle_exponents("train").reshape(-1, 1).astype(float),
        dataset.labels("train"),
        dataset.oracle_exponents("test").reshape(-1, 1).astype(float),
        dataset.labels("test"),
    )
    result["role"] = "oracle_control_not_a_baseline"
    result["input"] = "hidden exponent x = log_g(h)"
    return result


def legendre_attack(dataset: DlogDataset) -> Dict[str, object]:
    """
    Polynomial-time number-theoretic attack via Euler's criterion.

    Costs O(log p) modular multiplications per sample and recovers exactly one
    bit of log_g(h): its parity. Expected behaviour --
      parity concept        -> ~100% (the concept is BROKEN, no trapdoor)
      half-interval concept -> ~50%  (one bit is useless against a shifted
                                      half-interval; trapdoor intact)
    """
    out: Dict[str, object] = {"role": "number_theoretic_control",
                              "cost": "O(log p) modular multiplications"}
    for split in ("train", "test"):
        predictions = legendre_predict(dataset.group, dataset.group_elements(split))
        truth = dataset.labels(split)
        out[f"{split}_accuracy"] = float(accuracy_score(truth, predictions))
        if split == "test":
            out["predictions"] = predictions.tolist()
            out["confusion_matrix"] = confusion_matrix(
                truth, predictions, labels=[-1, 1]).tolist()
    return out


def classical_rbf_gram(dataset: DlogDataset, gamma: Optional[float] = None) -> np.ndarray:
    """
    Classical RBF Gram matrix on the training bits, for the kernel-comparison
    figure. Uses sklearn's 'scale' heuristic when gamma is None.
    """
    from sklearn.metrics.pairwise import rbf_kernel
    X = dataset.features("train")
    if gamma is None:
        gamma = 1.0 / (X.shape[1] * X.var()) if X.var() > 0 else 1.0
    return rbf_kernel(X, X, gamma=gamma)
