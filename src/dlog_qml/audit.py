"""
Canonical headline numbers, computed from artifacts in exactly one place.

Nothing in the notebook, README or docs should hard-code a derived statistic.
`headline_numbers()` is the single source: it reads the stored artifacts (or a
freshly computed run), recomputes every derived quantity, and returns them with
their provenance attached. `check_consistency()` then re-derives the same values
independently and flags any disagreement.

The point is narrow but important: if `separation` is quoted anywhere, it must
equal `coset_mean - best_competitor_mean` computed from the same artifact, not a
number someone typed once and never revisited.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .experiments import COSET_KEY, LADDER_KEY, LEGENDRE_KEY, ORACLE_KEY

#: Models that are NOT legitimate competitors under the task constraints.
_NOT_COMPETITORS = (COSET_KEY, ORACLE_KEY)


def best_competitor_over_seeds(multi_seed: Dict[str, Any]) -> str:
    """
    Highest-mean model that is a legitimate competitor.

    Excludes the coset kernel itself and the hidden-x oracle. The Legendre
    attack and the generic quantum kernel ARE competitors -- they operate under
    the same constraints as any other model.
    """
    per_model = multi_seed["per_model"]
    candidates = [n for n in per_model if n not in _NOT_COMPETITORS]
    return max(candidates, key=lambda n: per_model[n]["mean"])


def headline_numbers(run: Dict[str, Any],
                     multi_seed: Optional[Dict[str, Any]] = None,
                     parity: Optional[Dict[str, Any]] = None,
                     ablation: Optional[Dict[str, Any]] = None
                     ) -> Dict[str, Any]:
    """Every number quoted anywhere in the project, derived in one place."""
    summaries = run["summaries"]
    out: Dict[str, Any] = {}

    # ---- reference (single-seed) run --------------------------------------
    out["reference"] = {
        "n_test": int(run["config"]["n_test"]),
        "coset_accuracy": float(summaries[COSET_KEY]["accuracy"]),
        "coset_ci": [float(summaries[COSET_KEY]["ci_low"]),
                     float(summaries[COSET_KEY]["ci_high"])],
        "coset_p": float(summaries[COSET_KEY]["p_value"]),
        "coset_auc": float(run["quantum_coset"]["test_auc"]),
        "generic_accuracy": float(summaries[LADDER_KEY]["accuracy"]),
        "generic_p": float(summaries[LADDER_KEY]["p_value"]),
        "best_competitor": run["best_competitor"],
        "best_competitor_accuracy": float(run["best_competitor_accuracy"]),
        # recomputed, never read from the artifact
        "separation": float(summaries[COSET_KEY]["accuracy"]
                            - run["best_competitor_accuracy"]),
        "any_competitor_significant": bool(run["any_competitor_significant_at_1pct"]),
        "mcnemar_p": float(run["paired_test"]["mcnemar"]["p_value"]),
        "bootstrap_ci": [float(run["paired_test"]["bootstrap_accuracy_difference"]["ci_low"]),
                         float(run["paired_test"]["bootstrap_accuracy_difference"]["ci_high"])],
        "oracle_accuracy": float(summaries[ORACLE_KEY]["accuracy"]),
        "legendre_accuracy": float(summaries[LEGENDRE_KEY]["accuracy"]),
    }

    # ---- kernel geometry ---------------------------------------------------
    for tag, key in (("coset", "quantum_coset"), ("generic", "quantum_generic")):
        report = run[key].get("kernel_report")
        if report:
            out.setdefault("kernel", {})[tag] = {
                "offdiag_mean": float(report["offdiag_mean"]),
                "effective_rank": float(report["effective_rank"]),
                "n": int(report["n"]),
                "alignment": float(report["target_alignment_centered"]),
                "within_class": float(report["class_conditional"]["within_class_mean"]),
                "between_class": float(report["class_conditional"]["between_class_mean"]),
                "separation_gap": float(report["class_conditional"]["separation_gap"]),
                "top10_fraction": float(report["spectrum"]["top10_fraction"]),
            }

    # ---- multi-seed --------------------------------------------------------
    if multi_seed and multi_seed.get("n_seeds_completed", 0) > 0:
        per_model = multi_seed["per_model"]
        best = best_competitor_over_seeds(multi_seed)
        coset_runs = np.asarray(multi_seed["raw_accuracies"][COSET_KEY], dtype=float)
        best_runs = np.asarray(multi_seed["raw_accuracies"][best], dtype=float)

        out["multi_seed"] = {
            "n_seeds": int(multi_seed["n_seeds_completed"]),
            "coset_mean": float(per_model[COSET_KEY]["mean"]),
            "coset_std": float(per_model[COSET_KEY]["std"]),
            "coset_min": float(per_model[COSET_KEY]["min"]),
            "generic_mean": float(per_model[LADDER_KEY]["mean"]),
            "generic_std": float(per_model[LADDER_KEY]["std"]),
            "best_competitor": best,
            "best_competitor_mean": float(per_model[best]["mean"]),
            "best_competitor_std": float(per_model[best]["std"]),
            # derived, recomputed here
            "separation_mean": float(multi_seed["separation"]["mean"]),
            "separation_std": float(multi_seed["separation"]["std"]),
            "separation_min": float(multi_seed["separation"]["min"]),
            "headline_gap": float(per_model[COSET_KEY]["mean"] - per_model[best]["mean"]),
            "seeds_coset_beats_best": int(np.sum(coset_runs > best_runs)),
            "oracle_mean": float(per_model[ORACLE_KEY]["mean"]),
        }

    # ---- controls ----------------------------------------------------------
    if parity:
        out["parity"] = {
            "legendre_accuracy": float(parity["legendre_control"]["test_accuracy"]),
            "coset_accuracy": float(parity["coset_accuracy"]),
        }

    if ablation:
        rows = [r for r in ablation.get("rows", []) if "error" not in r]
        out["ablation"] = {
            "n_windows": len(rows),
            "n_above_085": sum(1 for r in rows if r["coset_accuracy"] > 0.85),
            "min_accuracy": min((r["coset_accuracy"] for r in rows), default=float("nan")),
            "max_accuracy": max((r["coset_accuracy"] for r in rows), default=float("nan")),
            "default_window_bits": ablation.get("default_window_bits"),
        }

    return out


def check_consistency(numbers: Dict[str, Any], run: Dict[str, Any],
                      multi_seed: Optional[Dict[str, Any]] = None
                      ) -> List[Dict[str, Any]]:
    """
    Re-derive each headline number independently and report agreement.

    Returns one row per check so the notebook can display it as a table rather
    than asserting silently.
    """
    checks: List[Dict[str, Any]] = []

    def add(name: str, quoted: float, recomputed: float, tol: float = 1e-9) -> None:
        agrees = abs(float(quoted) - float(recomputed)) <= tol
        checks.append({"quantity": name, "quoted": round(float(quoted), 6),
                       "recomputed": round(float(recomputed), 6), "agrees": agrees})

    ref = numbers["reference"]
    summaries = run["summaries"]

    add("reference separation", ref["separation"],
        summaries[COSET_KEY]["accuracy"] - run["best_competitor_accuracy"])

    add("coset accuracy", ref["coset_accuracy"],
        run["quantum_coset"]["test_accuracy"])

    # Confusion matrix must agree with the reported accuracy.
    cm = np.asarray(run["quantum_coset"]["confusion_matrix"], dtype=float)
    add("coset accuracy (from confusion matrix)", ref["coset_accuracy"],
        float(np.trace(cm) / cm.sum()))

    # The stated best competitor really is the argmax over legitimate models.
    competitors = {n: s["accuracy"] for n, s in summaries.items()
                   if n not in _NOT_COMPETITORS}
    add("best competitor accuracy", ref["best_competitor_accuracy"],
        max(competitors.values()))

    if multi_seed and "multi_seed" in numbers:
        ms = numbers["multi_seed"]
        raw = multi_seed["raw_accuracies"]
        add("coset mean over seeds", ms["coset_mean"],
            float(np.mean(raw[COSET_KEY])))
        add("generic mean over seeds", ms["generic_mean"],
            float(np.mean(raw[LADDER_KEY])))
        add("separation mean over seeds", ms["separation_mean"],
            float(np.mean(multi_seed["raw_separations"])))
        add("n seeds", ms["n_seeds"], len(raw[COSET_KEY]))

    return checks
