"""
Tests for the kernel-geometry diagnostics, the artifact provenance layer, the
formal ablation table, and the negative-controls checklist.

These back the notebook's mechanistic explanation of *why* the generic kernel
fails, so they are correctness tests against hand-computable cases, not smoke
tests.
"""

import json

import numpy as np
import pytest

from dlog_qml.artifacts import Provenance, load_artifact, require, run_or_load
from dlog_qml.data import generate_dataset
from dlog_qml.kernels import (
    class_conditional_similarity,
    eigenvalue_spectrum,
    fidelity_gram,
    full_kernel_report,
    kernel_target_alignment,
)
from dlog_qml.quantum_coset import CosetStateFeatureMap
from dlog_qml.quantum_generic import LadderFeatureMap


# ---------------------------------------------------------------- alignment

def test_alignment_of_the_ideal_kernel_is_one():
    """K = yy^T is perfectly aligned with itself."""
    y = np.array([1, 1, -1, -1, 1, -1])
    K = np.outer(y, y).astype(float)
    assert kernel_target_alignment(K, y, centered=False) == pytest.approx(1.0)


def test_alignment_of_anti_ideal_kernel_is_minus_one():
    y = np.array([1, 1, -1, -1])
    K = -np.outer(y, y).astype(float)
    assert kernel_target_alignment(K, y, centered=False) == pytest.approx(-1.0)


def test_alignment_matches_explicit_frobenius_formula():
    rng = np.random.default_rng(0)
    y = rng.choice([-1, 1], size=12)
    A = rng.normal(size=(12, 12))
    K = A @ A.T
    Y = np.outer(y, y).astype(float)
    expected = (K * Y).sum() / (np.linalg.norm(K, "fro") * np.linalg.norm(Y, "fro"))
    assert kernel_target_alignment(K, y, centered=False) == pytest.approx(expected)


def test_alignment_is_scale_invariant():
    rng = np.random.default_rng(1)
    y = rng.choice([-1, 1], size=10)
    A = rng.normal(size=(10, 10))
    K = A @ A.T
    a = kernel_target_alignment(K, y, centered=True)
    b = kernel_target_alignment(7.5 * K, y, centered=True)
    assert a == pytest.approx(b)


def test_centering_makes_alignment_invariant_to_a_constant_offset():
    """
    Adding a constant to every kernel entry adds no class information. The
    centred alignment must therefore be unchanged, while the uncentred one
    drifts. This is exactly why the notebook reports the centred number.
    """
    rng = np.random.default_rng(4)
    y = np.array([1] * 8 + [-1] * 8)
    A = rng.normal(size=(16, 16))
    K = A @ A.T

    centred_before = kernel_target_alignment(K, y, centered=True)
    centred_after = kernel_target_alignment(K + 5.0, y, centered=True)
    assert centred_before == pytest.approx(centred_after, abs=1e-10)

    raw_before = kernel_target_alignment(K, y, centered=False)
    raw_after = kernel_target_alignment(K + 5.0, y, centered=False)
    assert raw_before != pytest.approx(raw_after, abs=1e-6)


def test_alignment_of_a_degenerate_kernel_is_nan_not_zero():
    """
    Centring a constant kernel yields the zero matrix, so alignment is a
    genuine 0/0. We return NaN rather than silently reporting 0.0, which would
    be indistinguishable from a real measurement of 'no alignment'.
    """
    y = np.array([1, 1, 1, -1, -1, -1])
    assert np.isnan(kernel_target_alignment(np.ones((6, 6)), y, centered=True))


def test_identity_kernel_has_low_alignment():
    y = np.array([1] * 10 + [-1] * 10)
    assert abs(kernel_target_alignment(np.eye(20), y, centered=True)) < 0.3


def test_alignment_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        kernel_target_alignment(np.eye(4), np.array([1, -1]))


# ------------------------------------------------- class-conditional similarity

def test_class_conditional_on_a_block_kernel():
    """Two perfectly separated blocks: within = 1, between = 0."""
    y = np.array([1, 1, 1, -1, -1, -1])
    K = np.zeros((6, 6))
    K[:3, :3] = 1.0
    K[3:, 3:] = 1.0
    stats = class_conditional_similarity(K, y)
    assert stats["within_class_mean"] == pytest.approx(1.0)
    assert stats["between_class_mean"] == pytest.approx(0.0)
    assert stats["separation_gap"] == pytest.approx(1.0)
    assert stats["separation_ratio"] == float("inf")


def test_class_conditional_excludes_the_diagonal():
    """A pure identity has zero within-class similarity once the diagonal goes."""
    y = np.array([1, 1, -1, -1])
    stats = class_conditional_similarity(np.eye(4), y)
    assert stats["within_class_mean"] == pytest.approx(0.0)
    assert stats["between_class_mean"] == pytest.approx(0.0)


def test_class_conditional_on_a_structureless_kernel():
    y = np.array([1] * 15 + [-1] * 15)
    K = np.full((30, 30), 0.3)
    np.fill_diagonal(K, 1.0)
    stats = class_conditional_similarity(K, y)
    assert stats["separation_gap"] == pytest.approx(0.0, abs=1e-12)
    assert stats["separation_ratio"] == pytest.approx(1.0)


# ------------------------------------------------------------------ spectrum

def test_spectrum_of_identity_is_flat():
    spectrum = eigenvalue_spectrum(np.eye(20))
    assert spectrum["trace"] == pytest.approx(20.0)
    assert spectrum["top1_fraction"] == pytest.approx(0.05)
    assert spectrum["n_components_90pct"] == 18


def test_spectrum_of_rank_one_is_concentrated():
    ones = np.ones((20, 20))
    spectrum = eigenvalue_spectrum(ones)
    assert spectrum["top1_fraction"] == pytest.approx(1.0)
    assert spectrum["n_components_90pct"] == 1


def test_spectrum_eigenvalues_are_descending():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(15, 15))
    eigenvalues = eigenvalue_spectrum(A @ A.T)["eigenvalues"]
    assert all(a >= b - 1e-12 for a, b in zip(eigenvalues, eigenvalues[1:]))


# ------------------------------------- the headline mechanistic comparison

@pytest.fixture(scope="module")
def kernel_reports():
    data = generate_dataset(n_qubits=10, n_train=80, n_test=20, seed=20260828)
    y = data.labels("train")
    coset = CosetStateFeatureMap(data.group)
    ladder = LadderFeatureMap(data.group.n_bits, reps=2)
    K_coset = fidelity_gram(coset.statevectors(data.group_elements("train")))
    K_ladder = fidelity_gram(ladder.statevectors(data.features("train").astype(int)))
    return full_kernel_report(K_coset, y), full_kernel_report(K_ladder, y)


def test_coset_kernel_is_better_aligned_than_generic(kernel_reports):
    coset, generic = kernel_reports
    assert coset["target_alignment_centered"] > generic["target_alignment_centered"]
    assert coset["target_alignment_centered"] > 0.1


def test_coset_kernel_separates_classes_generic_does_not(kernel_reports):
    """The geometric explanation for the accuracy gap."""
    coset, generic = kernel_reports
    assert coset["class_conditional"]["separation_gap"] > 0.02
    assert abs(generic["class_conditional"]["separation_gap"]) < 0.01
    assert (coset["class_conditional"]["separation_gap"]
            > generic["class_conditional"]["separation_gap"])


def test_generic_kernel_spectrum_is_flatter(kernel_reports):
    coset, generic = kernel_reports
    assert coset["spectrum"]["top10_fraction"] > generic["spectrum"]["top10_fraction"]
    assert generic["effective_rank"] > coset["effective_rank"]


def test_full_report_has_every_field(kernel_reports):
    coset, _ = kernel_reports
    for key in ("offdiag_mean", "effective_rank", "target_alignment_centered",
                "target_alignment_raw", "class_conditional", "spectrum"):
        assert key in coset


# ------------------------------------------------------------- provenance

def test_provenance_banner_distinguishes_fresh_from_cached():
    fresh = Provenance(source="fresh", name="demo", runtime_seconds=1.25)
    cached = Provenance(source="cached", name="demo", path="/x.json",
                        written_at="2026-08-28 12:00:00")
    assert fresh.is_fresh and "FRESHLY COMPUTED" in fresh.banner()
    assert not cached.is_fresh and "CACHED ARTIFACT" in cached.banner()
    # A cached load must never be presentable as fresh.
    assert "FRESH" not in cached.banner().upper().replace("FRESHLY", "")


def test_load_artifact_round_trip(tmp_path):
    path = tmp_path / "thing.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")
    payload, provenance = load_artifact(str(path))
    assert payload == {"a": 1}
    assert provenance.source == "cached"
    assert provenance.written_at is not None


def test_load_artifact_missing_returns_none(tmp_path):
    payload, provenance = load_artifact(str(tmp_path / "nope.json"))
    assert payload is None
    assert provenance.source == "missing"


def test_run_or_load_prefers_cache(tmp_path):
    path = tmp_path / "cached.json"
    path.write_text(json.dumps({"cached": True}), encoding="utf-8")
    calls = []

    def compute():
        calls.append(1)
        return {"cached": False}

    payload, provenance = run_or_load("x", str(path), compute, prefer_cached=True)
    assert payload == {"cached": True}
    assert provenance.source == "cached"
    assert not calls                       # compute must not have run


def test_run_or_load_forces_fresh(tmp_path):
    path = tmp_path / "cached.json"
    path.write_text(json.dumps({"cached": True}), encoding="utf-8")
    payload, provenance = run_or_load("x", str(path), lambda: {"cached": False},
                                      prefer_cached=False)
    assert payload == {"cached": False}
    assert provenance.source == "fresh"
    assert provenance.runtime_seconds is not None


def test_require_raises_with_hint():
    _, provenance = load_artifact("definitely_missing.json")
    with pytest.raises(FileNotFoundError, match="run_experiment"):
        require(None, provenance, "Run: python scripts/run_experiment.py")


# ------------------------------------ ablation table and controls checklist

@pytest.fixture(scope="module")
def small_run():
    from dlog_qml.config import ExperimentConfig
    from dlog_qml.experiments import run_single_experiment
    cfg = ExperimentConfig(n_qubits=8, n_train=60, n_test=20,
                           extra_baselines=False, cv_folds=3)
    return run_single_experiment(cfg, verify=True)


def test_ablation_table_structure(small_run):
    from dlog_qml.experiments import build_ablation_table
    rows = build_ablation_table(small_run)
    assert rows[0]["component"] == "Full system"
    assert rows[0]["delta_vs_full"] == 0.0
    for row in rows:
        assert set(row) == {"component", "variant", "accuracy", "delta_vs_full",
                            "question", "interpretation"}
        assert row["question"], "every ablation row must state its question"
        assert row["delta_vs_full"] == pytest.approx(
            row["accuracy"] - small_run["coset_accuracy"])


def test_ablation_table_includes_representation_swap(small_run):
    from dlog_qml.experiments import build_ablation_table
    variants = [r["variant"] for r in build_ablation_table(small_run)]
    assert any("generic" in v for v in variants)


def test_controls_checklist_evaluates_pass_conditions(small_run):
    from dlog_qml.experiments import negative_controls_checklist
    checks = negative_controls_checklist(small_run)
    assert checks, "checklist must not be empty"
    for check in checks:
        assert set(check) == {"control", "expectation", "observed", "passed",
                              "would_invalidate_if"}
        assert isinstance(check["passed"], bool)
        assert check["would_invalidate_if"], "each control must state its failure mode"

    names = [c["control"] for c in checks]
    assert "Hidden-x oracle succeeds" in names
    assert "Generic quantum kernel fails" in names
    assert "Train/test exponents disjoint" in names


def test_core_controls_pass_on_a_real_run(small_run):
    from dlog_qml.experiments import negative_controls_checklist
    by_name = {c["control"]: c for c in negative_controls_checklist(small_run)}
    assert by_name["Hidden-x oracle succeeds"]["passed"]
    assert by_name["Train/test exponents disjoint"]["passed"]
    assert by_name["Exact class balance"]["passed"]
    assert by_name["Kernel equals circuit definition"]["passed"]


# ------------------------------------------------- headline-number auditing

def test_headline_numbers_recomputes_separation(small_run):
    from dlog_qml.audit import headline_numbers
    numbers = headline_numbers(small_run)
    ref = numbers["reference"]
    # separation must be derived, not copied from the run dict
    assert ref["separation"] == pytest.approx(
        ref["coset_accuracy"] - ref["best_competitor_accuracy"])


def test_headline_numbers_survive_a_corrupted_summary_field(small_run):
    """
    If someone hand-edits a stored separation, the recomputed one must disagree.
    This is the failure mode the audit exists to catch.
    """
    import copy
    from dlog_qml.audit import headline_numbers
    tampered = copy.deepcopy(small_run)
    tampered["separation"] = 0.99                      # a lie
    numbers = headline_numbers(tampered)
    assert numbers["reference"]["separation"] != pytest.approx(0.99)


def test_best_competitor_excludes_coset_and_oracle():
    from dlog_qml.audit import best_competitor_over_seeds
    from dlog_qml.experiments import COSET_KEY, LADDER_KEY, ORACLE_KEY
    multi_seed = {"per_model": {
        COSET_KEY: {"mean": 0.99}, ORACLE_KEY: {"mean": 1.00},
        LADDER_KEY: {"mean": 0.51}, "RandomForest": {"mean": 0.55},
    }}
    # The oracle is higher than everything but must never be chosen.
    assert best_competitor_over_seeds(multi_seed) == "RandomForest"


def test_consistency_check_passes_on_a_real_run(small_run):
    from dlog_qml.audit import check_consistency, headline_numbers
    checks = check_consistency(headline_numbers(small_run), small_run)
    assert checks, "audit produced no checks"
    for row in checks:
        assert set(row) == {"quantity", "quoted", "recomputed", "agrees"}
        assert row["agrees"], f"{row['quantity']} disagrees: {row}"


def test_consistency_check_detects_a_wrong_number(small_run):
    from dlog_qml.audit import check_consistency, headline_numbers
    numbers = headline_numbers(small_run)
    numbers["reference"]["separation"] += 0.25         # inject an error
    checks = check_consistency(numbers, small_run)
    failed = [c for c in checks if not c["agrees"]]
    assert failed and failed[0]["quantity"] == "reference separation"
