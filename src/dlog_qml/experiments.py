"""
Experiment orchestration: single run, multi-seed robustness, measured size
sweep, and the window-size ablation.

Everything returns plain dictionaries so results serialise straight to JSON and
every number in the README/report is traceable back to a stored artefact.
"""

from __future__ import annotations

import platform
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from .classical import (
    classical_rbf_gram,
    hidden_x_oracle_control,
    legendre_attack,
    run_classical_baselines,
)
from .config import ExperimentConfig
from .data import DlogDataset, generate_dataset, max_balanced_split
from .group import DlogGroup
from .kernels import (
    build_quantum_kernels,
    full_kernel_report,
    kernel_diagnostics,
    train_kernel_svm,
    verify_against_circuits,
    verify_kernel_properties,
)
from .quantum_coset import CosetStateFeatureMap, default_window_bits
from .quantum_generic import LadderFeatureMap
from .statistics import (
    AccuracySummary,
    bootstrap_diff_ci,
    describe_runs,
    mcnemar_exact,
)

#: Labels used consistently across results, tables, figures, and the demo.
COSET_KEY = "Quantum coset kernel"
LADDER_KEY = "Quantum generic kernel"
ORACLE_KEY = "Hidden-x oracle (control)"
LEGENDRE_KEY = "Legendre attack (control)"


def environment_info() -> Dict[str, str]:
    """Record the environment so results are attributable to a concrete stack."""
    versions: Dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for module in ("numpy", "pandas", "sklearn", "qiskit",
                   "qiskit_machine_learning", "scipy", "matplotlib"):
        try:
            versions[module] = __import__(module).__version__
        except Exception:
            versions[module] = "not installed"
    return versions


# ---------------------------------------------------------------------------
# Quantum blocks
# ---------------------------------------------------------------------------

def run_quantum_kernel(dataset: DlogDataset, cfg: ExperimentConfig, kind: str,
                       verify: bool = True) -> Dict[str, object]:
    """
    Build a quantum feature map, form its Gram matrices, audit them, and fit the
    kernel SVM.

    `kind` is 'coset' (structure-aligned) or 'ladder' (generic control). The two
    differ ONLY in the feature map -- identical splits, identical kernel
    machinery, identical model selection -- so the comparison is like for like.
    """
    if kind == "coset":
        feature_map = CosetStateFeatureMap(dataset.group, cfg.window_bits)
        inputs_train = dataset.group_elements("train")
        inputs_test = dataset.group_elements("test")
    elif kind == "ladder":
        feature_map = LadderFeatureMap(dataset.group.n_bits, cfg.ladder_reps)
        inputs_train = dataset.features("train").astype(int)
        inputs_test = dataset.features("test").astype(int)
    else:
        raise ValueError(f"unknown feature map kind {kind!r}")

    K_train, K_test, timings = build_quantum_kernels(
        feature_map, inputs_train, inputs_test)

    result: Dict[str, object] = {
        "feature_map": feature_map.describe(),
        "timings": timings,
        "kernel_diagnostics": kernel_diagnostics(K_train),
        "kernel_properties": verify_kernel_properties(K_train),
        # Label-aware geometry: alignment, class-conditional separation and the
        # eigenspectrum. This is what explains the accuracy mechanistically.
        "kernel_report": full_kernel_report(K_train, dataset.labels("train")),
    }

    if verify:
        result["verification"] = verify_against_circuits(
            feature_map, list(inputs_train), K_train, n_pairs=6, seed=cfg.seed)
        if kind == "coset":
            # Independent check against the closed-form triangular overlap,
            # which never touches a simulator.
            limit = min(60, len(inputs_train))
            analytic = feature_map.analytic_gram(
                inputs_train[:limit], inputs_train[:limit])
            result["verification"]["max_analytic_error"] = float(
                np.abs(analytic - K_train[:limit, :limit]).max())

    result.update(train_kernel_svm(
        K_train, dataset.labels("train"), K_test, dataset.labels("test"),
        C_grid=cfg.qsvm_C_grid, cv_folds=cfg.cv_folds, seed=cfg.seed,
    ))
    result["_K_train"] = K_train          # retained in-memory for plotting only
    result["_K_test"] = K_test
    return result


# ---------------------------------------------------------------------------
# Single full experiment
# ---------------------------------------------------------------------------

def run_single_experiment(cfg: ExperimentConfig, verify: bool = True,
                          keep_kernels: bool = False) -> Dict[str, object]:
    """
    One complete experiment at one seed: data, classical, controls, quantum,
    statistics, paired comparisons.
    """
    t_start = time.perf_counter()
    cfg.validate()

    dataset = generate_dataset(
        n_qubits=cfg.n_qubits, n_train=cfg.n_train, n_test=cfg.n_test,
        seed=cfg.seed, labeling=cfg.labeling,
    )
    audit = dataset.verify()
    y_test = dataset.labels("test")
    n_test = len(y_test)

    classical = run_classical_baselines(dataset, cfg)
    oracle = hidden_x_oracle_control(dataset, cfg)
    legendre = legendre_attack(dataset)
    ladder = run_quantum_kernel(dataset, cfg, "ladder", verify=verify)
    coset = run_quantum_kernel(dataset, cfg, "coset", verify=verify)

    # ---- unified accuracy table with CIs and p-values ----------------------
    summaries: Dict[str, Dict[str, float]] = {}
    predictions: Dict[str, List[int]] = {}

    for name, res in classical.items():
        summaries[name] = AccuracySummary.build(res["test_accuracy"], n_test).to_dict()
        predictions[name] = res["predictions"]
    for key, res in ((LADDER_KEY, ladder), (COSET_KEY, coset)):
        summaries[key] = AccuracySummary.build(res["test_accuracy"], n_test).to_dict()
        predictions[key] = res["predictions"]
    summaries[LEGENDRE_KEY] = AccuracySummary.build(
        legendre["test_accuracy"], n_test).to_dict()
    predictions[LEGENDRE_KEY] = legendre["predictions"]
    summaries[ORACLE_KEY] = AccuracySummary.build(
        oracle["test_accuracy"], n_test).to_dict()
    predictions[ORACLE_KEY] = oracle["predictions"]

    # ---- competitor analysis ----------------------------------------------
    # Everything that is a legitimate model under the task constraints, i.e.
    # excluding the coset kernel itself and the oracle control.
    competitor_names = [n for n in summaries
                        if n not in (COSET_KEY, ORACLE_KEY)]
    best_competitor = max(competitor_names,
                          key=lambda n: summaries[n]["accuracy"])
    any_competitor_significant = any(
        summaries[n]["significant_at_1pct"] for n in competitor_names)

    paired = {
        "comparison": f"{COSET_KEY} vs {best_competitor}",
        "mcnemar": mcnemar_exact(y_test, predictions[COSET_KEY],
                                 predictions[best_competitor]),
        "bootstrap_accuracy_difference": bootstrap_diff_ci(
            y_test, predictions[COSET_KEY], predictions[best_competitor],
            n_boot=10000, seed=cfg.seed),
    }

    result: Dict[str, object] = {
        "config": cfg.to_dict(),
        "environment": environment_info(),
        "group": dataset.group.describe(),
        "secret_shift": getattr(dataset.label_rule, "secret_s", None),
        "label_rule": dataset.label_rule.name,
        "dataset_audit": audit,
        "classical": classical,
        "oracle_control": oracle,
        "legendre_control": legendre,
        "quantum_generic": {k: v for k, v in ladder.items() if not k.startswith("_")},
        "quantum_coset": {k: v for k, v in coset.items() if not k.startswith("_")},
        "summaries": summaries,
        "best_competitor": best_competitor,
        "best_competitor_accuracy": summaries[best_competitor]["accuracy"],
        "any_competitor_significant_at_1pct": bool(any_competitor_significant),
        "coset_accuracy": summaries[COSET_KEY]["accuracy"],
        "separation": float(summaries[COSET_KEY]["accuracy"]
                            - summaries[best_competitor]["accuracy"]),
        "paired_test": paired,
        "wall_clock_seconds": round(time.perf_counter() - t_start, 2),
    }

    if keep_kernels:
        result["_kernels"] = {
            "coset_train": coset["_K_train"],
            "ladder_train": ladder["_K_train"],
            "classical_rbf_train": classical_rbf_gram(dataset),
        }
        result["_dataset"] = dataset
    return result


# ---------------------------------------------------------------------------
# Multi-seed robustness
# ---------------------------------------------------------------------------

def run_multi_seed(cfg: ExperimentConfig, n_seeds: Optional[int] = None,
                   progress: bool = True) -> Dict[str, object]:
    """
    Repeat the full experiment across independent seeds.

    Each seed redraws the secret shift AND the sample set, so this measures
    robustness to both, not just to sampling noise. Reported per model:
    mean, std, median, min, max over the actual completed runs.
    """
    n_seeds = int(n_seeds if n_seeds is not None else cfg.n_seeds)
    per_model: Dict[str, List[float]] = {}
    separations: List[float] = []
    seeds_used: List[int] = []
    failures: List[Dict[str, str]] = []

    for index in range(n_seeds):
        seed = cfg.seed + 1000 * index
        try:
            run = run_single_experiment(cfg.replace(seed=seed), verify=False)
        except Exception as exc:                          # pragma: no cover
            failures.append({"seed": str(seed), "error": f"{type(exc).__name__}: {exc}"})
            continue

        for name, summary in run["summaries"].items():
            per_model.setdefault(name, []).append(summary["accuracy"])
        separations.append(run["separation"])
        seeds_used.append(seed)

        if progress:
            print(f"    seed {seed}  ({index + 1}/{n_seeds})  "
                  f"coset={run['coset_accuracy']:.3f}  "
                  f"best-competitor={run['best_competitor_accuracy']:.3f}  "
                  f"sep={run['separation']:+.3f}", flush=True)

    return {
        "n_seeds_requested": n_seeds,
        "n_seeds_completed": len(seeds_used),
        "seeds": seeds_used,
        "failures": failures,
        "per_model": {name: describe_runs(values) for name, values in per_model.items()},
        "raw_accuracies": per_model,
        "separation": describe_runs(separations),
        "raw_separations": separations,
    }


# ---------------------------------------------------------------------------
# Measured size sweep
# ---------------------------------------------------------------------------

def run_size_sweep(cfg: ExperimentConfig, qubit_sizes: Optional[Sequence[int]] = None,
                   progress: bool = True) -> Dict[str, object]:
    """
    ACTUALLY MEASURED sweep across register widths.

    For each width the largest feasible balanced dataset is used (small groups
    cannot supply 200/50), and we record real accuracies, real simulation
    wall-clock, and real statevector memory. These are measurements, not
    projections -- the theoretical roadmap lives separately in
    docs/TECHNICAL_REPORT.md and is labelled as such.
    """
    qubit_sizes = list(qubit_sizes if qubit_sizes is not None else cfg.sweep_qubits)
    rows: List[Dict[str, object]] = []

    for n_qubits in qubit_sizes:
        n_train, n_test = max_balanced_split(n_qubits, cfg.n_train, cfg.n_test)
        local = cfg.replace(n_qubits=n_qubits, n_train=n_train, n_test=n_test,
                            extra_baselines=False)
        try:
            run = run_single_experiment(local, verify=False)
        except Exception as exc:                          # pragma: no cover
            rows.append({"n_qubits": n_qubits, "error": f"{type(exc).__name__}: {exc}"})
            continue

        coset_timing = run["quantum_coset"]["timings"]
        ladder_timing = run["quantum_generic"]["timings"]
        rows.append({
            "n_qubits": n_qubits,
            "p": run["group"]["p"],
            "n_train": n_train,
            "n_test": n_test,
            "coset_accuracy": run["coset_accuracy"],
            "coset_ci": [run["summaries"][COSET_KEY]["ci_low"],
                         run["summaries"][COSET_KEY]["ci_high"]],
            "generic_accuracy": run["summaries"][LADDER_KEY]["accuracy"],
            "best_competitor": run["best_competitor"],
            "best_competitor_accuracy": run["best_competitor_accuracy"],
            "separation": run["separation"],
            "coset_simulate_seconds": coset_timing["simulate_seconds"],
            "ladder_simulate_seconds": ladder_timing["simulate_seconds"],
            "statevector_dim": coset_timing["statevector_dim"],
            "statevector_megabytes": coset_timing["statevector_megabytes"],
            "wall_clock_seconds": run["wall_clock_seconds"],
        })
        if progress:
            print(f"    n={n_qubits:2d}  p={rows[-1]['p']:>7}  "
                  f"coset={rows[-1]['coset_accuracy']:.3f}  "
                  f"generic={rows[-1]['generic_accuracy']:.3f}  "
                  f"sim={coset_timing['simulate_seconds']:.3f}s  "
                  f"{coset_timing['statevector_megabytes']:.2f} MB", flush=True)

    return {"measured": True, "rows": rows, "qubit_sizes": qubit_sizes}


# ---------------------------------------------------------------------------
# Ablation: coset window size
# ---------------------------------------------------------------------------

def run_window_ablation(cfg: ExperimentConfig, window_bits: Optional[Sequence[int]] = None,
                        progress: bool = True) -> Dict[str, object]:
    """
    Sweep the coset window M = 2^m.

    Predicted behaviour, from the triangular-kernel geometry: very small M makes
    almost all pairs orthogonal (kernel -> identity, no generalisation), while M
    approaching (p-1)/2 flattens the triangle toward a constant and violates the
    2M <= N condition. The useful regime is in between -- this ablation checks
    that the result is not a knife-edge artefact of one lucky window.
    """
    group = DlogGroup.from_qubits(cfg.n_qubits)
    default_m = default_window_bits(group)
    max_m = int(np.floor(np.log2(group.order / 2)))
    if window_bits is None:
        window_bits = [m for m in range(1, max_m + 1)]

    rows: List[Dict[str, object]] = []
    for m in window_bits:
        if 2 * (2 ** m) > group.order:
            continue
        try:
            run = run_single_experiment(
                cfg.replace(window_bits=m, extra_baselines=False), verify=False)
        except Exception as exc:                          # pragma: no cover
            rows.append({"window_bits": m, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rows.append({
            "window_bits": m,
            "window_M": 2 ** m,
            "fraction_of_group": 2 ** m / group.order,
            "coset_accuracy": run["coset_accuracy"],
            "coset_auc": run["quantum_coset"]["test_auc"],
            "offdiag_mean": run["quantum_coset"]["kernel_diagnostics"]["offdiag_mean"],
            "effective_rank": run["quantum_coset"]["kernel_diagnostics"]["effective_rank"],
            "is_default": m == default_m,
        })
        if progress:
            print(f"    m={m:2d}  M={2 ** m:>5}  "
                  f"acc={rows[-1]['coset_accuracy']:.3f}  "
                  f"offdiag={rows[-1]['offdiag_mean']:.4f}"
                  f"{'   <- default' if m == default_m else ''}", flush=True)

    return {"default_window_bits": default_m, "group_order": group.order, "rows": rows}


# ---------------------------------------------------------------------------
# Formal ablation table and controls checklist
# ---------------------------------------------------------------------------

def build_ablation_table(run: Dict[str, object],
                         ablation: Optional[Dict[str, object]] = None,
                         parity: Optional[Dict[str, object]] = None
                         ) -> List[Dict[str, object]]:
    """
    Assemble every ablation into one table, each row answering a stated question.

    `delta_vs_full` is measured against the full structure-aware system (the
    coset kernel at its default window on the half-interval concept), so a
    negative delta is the cost of removing or perturbing that component.
    """
    full_accuracy = float(run["coset_accuracy"])
    rows: List[Dict[str, object]] = [{
        "component": "Full system",
        "variant": "coset kernel, default window",
        "accuracy": full_accuracy,
        "delta_vs_full": 0.0,
        "question": "reference configuration",
        "interpretation": "structure-aware representation",
    }]

    # -- representation ablation: swap the feature map, keep everything else --
    generic_accuracy = float(run["summaries"][LADDER_KEY]["accuracy"])
    rows.append({
        "component": "Representation",
        "variant": "generic Rz/Rx/CZ feature map",
        "accuracy": generic_accuracy,
        "delta_vs_full": generic_accuracy - full_accuracy,
        "question": "Would any expressive quantum feature map do?",
        "interpretation": "geometry mismatch -- being quantum is not sufficient",
    })

    # -- representation ablation: drop quantum entirely -----------------------
    best_classical = float(run["best_competitor_accuracy"])
    rows.append({
        "component": "Representation",
        "variant": f"best classical ({run['best_competitor']})",
        "accuracy": best_classical,
        "delta_vs_full": best_classical - full_accuracy,
        "question": "Is a classical kernel on the same bits enough?",
        "interpretation": "no access to the dlog coordinate",
    })

    # -- window ablation ------------------------------------------------------
    if ablation:
        for row in ablation.get("rows", []):
            if "error" in row:
                continue
            accuracy = float(row["coset_accuracy"])
            rows.append({
                "component": "Coset window",
                "variant": f"m={row['window_bits']} (M={row['window_M']})",
                "accuracy": accuracy,
                "delta_vs_full": accuracy - full_accuracy,
                "question": "Does the result hinge on a tuned window size?",
                "interpretation": ("default" if row.get("is_default") else
                                   "too narrow: near-orthogonal kernel"
                                   if row["window_bits"] <= 1 else "robust"),
            })

    # -- concept ablation: keep the kernel, change the target -----------------
    if parity:
        parity_accuracy = float(parity["coset_accuracy"])
        rows.append({
            "component": "Target concept",
            "variant": "coset kernel on parity concept",
            "accuracy": parity_accuracy,
            "delta_vs_full": parity_accuracy - full_accuracy,
            "question": "Is the coset kernel a universal cheat?",
            "interpretation": "local kernel vs alternating concept -- fails",
        })
        legendre_parity = float(parity["legendre_control"]["test_accuracy"])
        rows.append({
            "component": "Target concept",
            "variant": "Legendre attack on parity concept",
            "accuracy": legendre_parity,
            "delta_vs_full": legendre_parity - full_accuracy,
            "question": "Is every dlog-derived label actually hard?",
            "interpretation": "parity has no trapdoor -- broken in O(log p)",
        })

    return rows


def negative_controls_checklist(run: Dict[str, object],
                                parity: Optional[Dict[str, object]] = None,
                                multi_seed: Optional[Dict[str, object]] = None,
                                ablation: Optional[Dict[str, object]] = None
                                ) -> List[Dict[str, object]]:
    """
    Every sanity and falsification control, with its pass condition evaluated.

    Each entry states what would have INVALIDATED the result, so a reader can
    see the claim was genuinely exposed to failure rather than merely asserted.
    """
    summaries = run["summaries"]
    checks: List[Dict[str, object]] = []

    def add(name: str, expectation: str, observed: str, passed: bool,
            invalidates: str) -> None:
        checks.append({"control": name, "expectation": expectation,
                       "observed": observed, "passed": bool(passed),
                       "would_invalidate_if": invalidates})

    oracle = float(summaries[ORACLE_KEY]["accuracy"])
    add("Hidden-x oracle succeeds", "high accuracy given the hidden coordinate",
        f"{oracle:.3f}", oracle > 0.85,
        "labels are noise or the generator is broken")

    generic = float(summaries[LADDER_KEY]["accuracy"])
    add("Generic quantum kernel fails", "chance level",
        f"{generic:.3f} (p={summaries[LADDER_KEY]['p_value']:.3f})",
        not summaries[LADDER_KEY]["significant_at_1pct"],
        "the effect is 'quantum', not 'structure-aware'")

    legendre = float(summaries[LEGENDRE_KEY]["accuracy"])
    add("Legendre attack fails on target", "chance level on half-interval",
        f"{legendre:.3f}", not summaries[LEGENDRE_KEY]["significant_at_1pct"],
        "a poly-time classical shortcut explains the label")

    add("No competitor beats chance", "no legitimate model significant at 1%",
        "none" if not run["any_competitor_significant_at_1pct"] else "AT LEAST ONE",
        not run["any_competitor_significant_at_1pct"],
        "the concept is classically learnable by a tested method")

    add("Train/test exponents disjoint", "no shared exponent between splits",
        str(run["dataset_audit"]["splits_disjoint"]),
        bool(run["dataset_audit"]["splits_disjoint"]),
        "the gap could be memorisation")

    balance = run["dataset_audit"]["balance"]
    balanced = (balance["train"]["positive"] == balance["train"]["negative"]
                and balance["test"]["positive"] == balance["test"]["negative"])
    add("Exact class balance", "50/50 in both splits",
        f"train {balance['train']}, test {balance['test']}", balanced,
        "chance level would not be 0.5")

    verification = run["quantum_coset"].get("verification", {})
    if verification:
        error = float(verification.get("max_compute_uncompute_error", float("nan")))
        analytic = float(verification.get("max_analytic_error", float("nan")))
        add("Kernel equals circuit definition", "agreement < 1e-9",
            f"{error:.1e} (circuits), {analytic:.1e} (closed form)",
            error < 1e-9, "the simulated kernel is not the fidelity kernel")

    if parity:
        parity_coset = float(parity["coset_accuracy"])
        add("Coset kernel fails on parity", "chance on a mismatched concept",
            f"{parity_coset:.3f}", parity_coset < 0.65,
            "the quantum kernel is a universal cheat, not structure-aware")
        parity_legendre = float(parity["legendre_control"]["test_accuracy"])
        add("Legendre attack breaks parity", "near-perfect on the naive concept",
            f"{parity_legendre:.3f}", parity_legendre > 0.9,
            "our falsification tool does not actually work")

    if multi_seed and multi_seed.get("n_seeds_completed", 0) > 1:
        separation = multi_seed["separation"]
        add("Separation holds across seeds", "minimum separation > 0.10",
            f"min {separation['min']:+.3f} over "
            f"{multi_seed['n_seeds_completed']} seeds",
            separation["min"] > 0.10, "the result is one lucky seed")

    if ablation:
        good = [r for r in ablation.get("rows", []) if "error" not in r]
        strong = [r for r in good if r["coset_accuracy"] > 0.85]
        add("Window ablation remains strong", ">= 3 windows above 0.85",
            f"{len(strong)}/{len(good)} windows above 0.85", len(strong) >= 3,
            "the result depends on one tuned hyperparameter")

    return checks
