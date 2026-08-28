#!/usr/bin/env python3
"""
DLOG-QML full experimental study.

Runs, in order:
  1. the reference single-seed experiment (with kernel audits),
  2. the parity falsification run (the BROKEN concept),
  3. the multi-seed robustness sweep,
  4. the measured size sweep across register widths,
  5. the coset-window ablation,
  6. all figures and result tables.

Everything is written to `results/` as JSON, CSV, Markdown, and PNG.

    python scripts/run_experiment.py                # full study (~10-20 min)
    python scripts/run_experiment.py --quick        # single run + figures only
    python scripts/run_experiment.py --seeds 5      # shorter robustness sweep
    python scripts/run_experiment.py --skip sweep ablation
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from dlog_qml.config import ExperimentConfig                       # noqa: E402
from dlog_qml.experiments import (                                 # noqa: E402
    COSET_KEY, LADDER_KEY, ORACLE_KEY, build_ablation_table, environment_info,
    negative_controls_checklist, run_multi_seed, run_single_experiment,
    run_size_sweep, run_window_ablation,
)
from dlog_qml.plotting import generate_all                         # noqa: E402
from dlog_qml.utils import (                                       # noqa: E402
    accuracy_row, format_p, header, rule, save_csv_table, save_json,
    save_markdown_table, section,
)

STAGES = ("parity", "seeds", "sweep", "ablation", "figures")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="DLOG-QML full experimental study.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--qubits", type=int, default=None)
    ap.add_argument("--train", type=int, default=None)
    ap.add_argument("--test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None,
                    help="number of seeds for the robustness sweep")
    ap.add_argument("--outdir", default=None, help="results directory")
    ap.add_argument("--quick", action="store_true",
                    help="single run + figures only (skip seeds/sweep/ablation/parity)")
    ap.add_argument("--skip", nargs="*", default=[], choices=list(STAGES),
                    help="stages to skip")
    return ap.parse_args(argv)


def build_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig.load(args.config)
    changes = {}
    for key, value in (("n_qubits", args.qubits), ("n_train", args.train),
                       ("n_test", args.test), ("seed", args.seed),
                       ("n_seeds", args.seeds), ("results_dir", args.outdir)):
        if value is not None:
            changes[key] = value
    return cfg.replace(**changes) if changes else cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    skip = set(args.skip) | (set(STAGES) - {"figures"} if args.quick else set())

    root = os.path.abspath(cfg.results_dir)
    raw_dir = os.path.join(root, "raw")
    tables_dir = os.path.join(root, "tables")
    figures_dir = os.path.abspath(cfg.figures_dir)
    for path in (raw_dir, tables_dir, figures_dir):
        os.makedirs(path, exist_ok=True)

    t_start = time.perf_counter()
    print(header("DLOG-QML", "FULL EXPERIMENTAL STUDY"))
    print("\nEnvironment:")
    for key, value in environment_info().items():
        print(f"  {key:26} {value}")
    print(f"\nResults directory: {root}")
    cfg.save(os.path.join(raw_dir, "config_used.json"))

    # ---------------------------------------------------- 1. reference run
    print(section("[1/6] REFERENCE EXPERIMENT (single seed, with kernel audits)"))
    run = run_single_experiment(cfg, verify=True, keep_kernels=True)
    save_json(run, os.path.join(raw_dir, "single_run.json"))

    dataset = run["_dataset"]
    csv_path = dataset.save_csv(
        os.path.join(root, "dlog_dataset.csv"), include_oracle=True)
    dataset.save_csv(os.path.join(root, "dlog_dataset_no_oracle.csv"),
                     include_oracle=False)

    summaries = run["summaries"]
    print(f"\n  {'model':<30}{'accuracy':>8}  {'95% CI':>16}  {'p':>9}   verdict")
    for name in summaries:
        print(accuracy_row(name, summaries[name]))
    print(f"\n  best competitor : {run['best_competitor']} "
          f"({run['best_competitor_accuracy']:.3f})")
    print(f"  coset kernel    : {run['coset_accuracy']:.3f}")
    print(f"  separation      : {run['separation']:+.3f}")
    print(f"  McNemar (paired): p = "
          f"{format_p(run['paired_test']['mcnemar']['p_value'])}")
    print(f"  dataset         : {csv_path}")

    # ------------------------------------------------ 2. parity falsification
    parity = None
    if "parity" not in skip:
        print(section("[2/6] PARITY FALSIFICATION (the BROKEN concept)"))
        parity = run_single_experiment(cfg.replace(labeling="parity"), verify=False)
        save_json(parity, os.path.join(raw_dir, "parity_run.json"))
        print(f"  Legendre/Euler attack accuracy : "
              f"{parity['legendre_control']['test_accuracy']:.3f}   "
              f"(half-interval run: {run['legendre_control']['test_accuracy']:.3f})")
        print(f"  coset kernel on parity concept : {parity['coset_accuracy']:.3f}")
        print("  => a poly-time classical attack solves parity outright, and the")
        print("     coset kernel -- being LOCAL on the dlog circle -- fails on it.")
        print("     The coset kernel is not a universal cheat.")
    else:
        print(section("[2/6] PARITY FALSIFICATION -- skipped"))

    # --------------------------------------------------- 3. multi-seed sweep
    multi_seed = None
    if "seeds" not in skip:
        print(section(f"[3/6] MULTI-SEED ROBUSTNESS ({cfg.n_seeds} seeds)"))
        multi_seed = run_multi_seed(cfg, progress=True)
        save_json(multi_seed, os.path.join(raw_dir, "multi_seed.json"))
        print(f"\n  completed {multi_seed['n_seeds_completed']}"
              f"/{multi_seed['n_seeds_requested']} seeds")
        print(f"\n  {'model':<30}{'mean':>8}{'std':>8}{'median':>8}"
              f"{'min':>8}{'max':>8}")
        rows = []
        for name, stats in sorted(multi_seed["per_model"].items(),
                                  key=lambda kv: -kv[1]["mean"]):
            print(f"  {name:<30}{stats['mean']:>8.3f}{stats['std']:>8.3f}"
                  f"{stats['median']:>8.3f}{stats['min']:>8.3f}{stats['max']:>8.3f}")
            rows.append({"model": name, **{k: stats[k] for k in
                                           ("mean", "std", "median", "min", "max",
                                            "n_runs")}})
        save_markdown_table(rows, os.path.join(tables_dir, "multi_seed.md"),
                            title=f"Multi-seed accuracy "
                                  f"({multi_seed['n_seeds_completed']} seeds)")
        save_csv_table(rows, os.path.join(tables_dir, "multi_seed.csv"))
        sep = multi_seed["separation"]
        print(f"\n  separation over seeds: mean {sep['mean']:+.3f} "
              f"+/- {sep['std']:.3f}  (min {sep['min']:+.3f}, max {sep['max']:+.3f})")
    else:
        print(section("[3/6] MULTI-SEED ROBUSTNESS -- skipped"))

    # ---------------------------------------------------- 4. measured sweep
    sweep = None
    if "sweep" not in skip:
        print(section("[4/6] MEASURED SIZE SWEEP (real runs, real timings)"))
        sweep = run_size_sweep(cfg, progress=True)
        save_json(sweep, os.path.join(raw_dir, "size_sweep.json"))
        good = [r for r in sweep["rows"] if "error" not in r]
        save_markdown_table(
            good, os.path.join(tables_dir, "size_sweep.md"),
            columns=["n_qubits", "p", "n_train", "n_test", "coset_accuracy",
                     "generic_accuracy", "best_competitor_accuracy", "separation",
                     "coset_simulate_seconds", "statevector_megabytes"],
            title="Measured size sweep")
        save_csv_table(good, os.path.join(tables_dir, "size_sweep.csv"))
    else:
        print(section("[4/6] MEASURED SIZE SWEEP -- skipped"))

    # -------------------------------------------------------- 5. ablation
    ablation = None
    if "ablation" not in skip:
        print(section("[5/6] COSET WINDOW ABLATION"))
        ablation = run_window_ablation(cfg, progress=True)
        save_json(ablation, os.path.join(raw_dir, "window_ablation.json"))
        good = [r for r in ablation["rows"] if "error" not in r]
        save_markdown_table(
            good, os.path.join(tables_dir, "window_ablation.md"),
            columns=["window_bits", "window_M", "fraction_of_group",
                     "coset_accuracy", "coset_auc", "offdiag_mean",
                     "effective_rank", "is_default"],
            title="Coset window ablation")
        save_csv_table(good, os.path.join(tables_dir, "window_ablation.csv"))
    else:
        print(section("[5/6] COSET WINDOW ABLATION -- skipped"))

    # --------------------------------------------------------- 6. figures
    figures = []
    if "figures" not in skip and cfg.make_plots:
        print(section("[6/6] FIGURES"))
        figures = generate_all(run, figures_dir, multi_seed=multi_seed,
                               sweep=sweep, ablation=ablation)
        for path in figures:
            print(f"  {path}")
    else:
        print(section("[6/6] FIGURES -- skipped"))

    # ------------------------------------------------------ summary table
    main_rows = [{
        "model": name,
        "accuracy": summaries[name]["accuracy"],
        "wilson_ci_95": [summaries[name]["ci_low"], summaries[name]["ci_high"]],
        "binomial_p": summaries[name]["p_value"],
        "above_chance_at_1pct": summaries[name]["significant_at_1pct"],
        "role": ("structure-aligned quantum" if name == COSET_KEY else
                 "generic quantum control" if name == LADDER_KEY else
                 "oracle control (not a baseline)" if name == ORACLE_KEY else
                 "control" if "control" in name.lower() else "classical baseline"),
    } for name in sorted(summaries, key=lambda n: -summaries[n]["accuracy"])]
    save_markdown_table(main_rows, os.path.join(tables_dir, "main_results.md"),
                        title="Reference run: held-out accuracy")
    save_csv_table(main_rows, os.path.join(tables_dir, "main_results.csv"))

    # ---- formal ablation table and negative-controls checklist -------------
    ablation_table = build_ablation_table(run, ablation=ablation, parity=parity)
    save_markdown_table(
        ablation_table, os.path.join(tables_dir, "ablation_table.md"),
        columns=["component", "variant", "accuracy", "delta_vs_full",
                 "question", "interpretation"],
        title="Formal ablation table")
    save_csv_table(ablation_table, os.path.join(tables_dir, "ablation_table.csv"))

    controls = negative_controls_checklist(run, parity=parity,
                                           multi_seed=multi_seed, ablation=ablation)
    save_markdown_table(
        controls, os.path.join(tables_dir, "negative_controls.md"),
        columns=["control", "expectation", "observed", "passed",
                 "would_invalidate_if"],
        title="Negative and sanity controls")
    save_csv_table(controls, os.path.join(tables_dir, "negative_controls.csv"))

    print(section("NEGATIVE / SANITY CONTROLS"))
    for check in controls:
        print(f"  [{'PASS' if check['passed'] else 'FAIL'}]  "
              f"{check['control']:<34} {check['observed']}")
    n_passed = sum(1 for c in controls if c["passed"])
    print(f"\n  {n_passed}/{len(controls)} controls passed")

    save_json({
        "reference_run": {k: v for k, v in run.items() if not k.startswith("_")},
        "ablation_table": ablation_table,
        "negative_controls": controls,
        "parity_run": parity,
        "multi_seed": multi_seed,
        "size_sweep": sweep,
        "window_ablation": ablation,
        "figures": figures,
        "environment": environment_info(),
    }, os.path.join(root, "summary.json"))

    print(section("DONE"))
    print(f"  raw JSON   : {raw_dir}")
    print(f"  tables     : {tables_dir}")
    print(f"  figures    : {figures_dir}")
    print(f"  summary    : {os.path.join(root, 'summary.json')}")
    print(f"\n  total wall-clock: {time.perf_counter() - t_start:.1f}s")
    print(rule("="))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
