#!/usr/bin/env python3
"""
Regenerate every figure.

The kernel matrices and the concept-geometry plot need in-memory objects (the
Gram matrices and the oracle exponents), which are deliberately NOT serialised
into results/raw. This script therefore re-runs the reference experiment -- it
is deterministic, so the figures match the stored numbers exactly -- and reuses
the saved multi-seed / sweep / ablation JSON when present.

    python scripts/generate_figures.py
    python scripts/generate_figures.py --outdir figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from dlog_qml.config import ExperimentConfig                       # noqa: E402
from dlog_qml.experiments import run_single_experiment             # noqa: E402
from dlog_qml.plotting import generate_all                         # noqa: E402


def load_optional(path: str):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate DLOG-QML figures.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--results", default=None, help="results root (default from config)")
    ap.add_argument("--outdir", default=None, help="figure output directory")
    args = ap.parse_args(argv)

    cfg = ExperimentConfig.load(args.config)
    root = os.path.abspath(args.results or cfg.results_dir)
    outdir = os.path.abspath(args.outdir or cfg.figures_dir)
    raw = os.path.join(root, "raw")
    os.makedirs(outdir, exist_ok=True)

    print("Re-running the reference experiment (deterministic) for figure data ...")
    run = run_single_experiment(cfg, verify=False, keep_kernels=True)
    print(f"  coset {run['coset_accuracy']:.3f}   "
          f"best competitor {run['best_competitor_accuracy']:.3f} "
          f"({run['best_competitor']})")

    multi_seed = load_optional(os.path.join(raw, "multi_seed.json"))
    sweep = load_optional(os.path.join(raw, "size_sweep.json"))
    ablation = load_optional(os.path.join(raw, "window_ablation.json"))
    for name, payload in (("multi_seed", multi_seed), ("size_sweep", sweep),
                          ("window_ablation", ablation)):
        print(f"  {name:16} {'loaded' if payload else 'not found (figure skipped)'}")

    written = generate_all(run, outdir, multi_seed=multi_seed,
                           sweep=sweep, ablation=ablation)
    print("\nFigures written:")
    for path in written:
        print(f"  {path}")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
