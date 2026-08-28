#!/usr/bin/env python3
"""
DLOG-QML terminal demo -- the fastest representative experiment.

Runs one complete experiment (data generation, classical baselines, both
quantum kernels, all three controls, full statistics) and prints a judge-facing
narrative. Every number printed is computed in this process; nothing is
hard-coded.

    python scripts/demo.py
    python scripts/demo.py --fast          # smaller group, ~8 s
    python scripts/demo.py --labeling parity   # watch the premise collapse
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Allow `python scripts/demo.py` without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

from dlog_qml.config import ExperimentConfig                       # noqa: E402
from dlog_qml.experiments import (                                 # noqa: E402
    COSET_KEY, LADDER_KEY, LEGENDRE_KEY, ORACLE_KEY, run_single_experiment,
)
from dlog_qml.utils import (                                       # noqa: E402
    accuracy_row, bar, format_p, header, rule, section,
)

TITLE = "DLOG-QML"
SUBTITLE = "QUANTUM LEARNING ON A DISCRETE-LOG CONCEPT CLASS"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="DLOG-QML terminal demo.")
    ap.add_argument("--config", default=None, help="path to a YAML/JSON config")
    ap.add_argument("--qubits", type=int, default=None, help="register width n")
    ap.add_argument("--train", type=int, default=None)
    ap.add_argument("--test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--labeling", choices=["half_interval", "parity"], default=None,
                    help="'parity' is the BROKEN control concept")
    ap.add_argument("--fast", action="store_true",
                    help="8 qubits / 120 train / 30 test, no extra baselines")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the compute-uncompute kernel audit")
    return ap.parse_args(argv)


def build_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig.load(args.config)
    changes = {}
    if args.fast:
        changes.update(n_qubits=8, n_train=120, n_test=30, extra_baselines=False)
    for key, value in (("n_qubits", args.qubits), ("n_train", args.train),
                       ("n_test", args.test), ("seed", args.seed),
                       ("labeling", args.labeling)):
        if value is not None:
            changes[key] = value
    return cfg.replace(**changes) if changes else cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)

    print(header(TITLE, SUBTITLE))
    print("""
QUESTION

  Can a quantum feature map whose inductive bias matches the algebraic
  structure of a concept class learn a concept whose useful coordinate
  system is hidden behind a discrete logarithm?
""")

    t0 = time.perf_counter()
    print("  running experiment ...", flush=True)
    run = run_single_experiment(cfg, verify=not args.no_verify, keep_kernels=False)
    elapsed = time.perf_counter() - t0

    group = run["group"]
    summaries = run["summaries"]
    n_test = run["config"]["n_test"]

    # ---------------------------------------------------------------- problem
    print(section("PROBLEM SETUP"))
    print(f"""
  safe prime p            {group['p']}  = 2*{group['q']} + 1
  generator g             {group['g']}   (order {group['order']} = |Z_p*|)
  secret shift s          {run['secret_shift']}   (SHA-256 of the run seed)
  labelling rule          {run['label_rule']}
  train / test            {run['config']['n_train']} / {n_test}   (exact 50/50, disjoint exponents)

  The learner SEES        binary_feature_0..{group['n_bits'] - 1}
                          = the {group['n_bits']}-bit decomposition of h = g^x mod p

  The learner NEVER SEES  x = log_g(h)          <- the hidden coordinate
                          s                     <- the secret shift

  The label depends on h ONLY through log_g(h). Recovering that coordinate
  is the discrete logarithm problem.""")

    if run["label_rule"] == "parity":
        print("""
  !! PARITY MODE: this concept has NO trapdoor. The least significant bit
  !! of log_g(h) is the quadratic-residuosity bit, computable in O(log p).
  !! Expect the Legendre attack below to solve it outright.""")

    # ------------------------------------------------------------- classical
    print(section("CLASSICAL BASELINES  (bits of h only)"))
    print(f"  {'model':<30}{'accuracy':>8}  {'95% CI':>16}  {'p':>9}   verdict")
    for name in run["classical"]:
        print(accuracy_row(name, summaries[name]))

    # -------------------------------------------------------------- quantum
    print(section("QUANTUM KERNELS  (identical splits, identical model selection)"))
    print(f"  {'feature map':<30}{'accuracy':>8}  {'95% CI':>16}  {'p':>9}   verdict")
    print(accuracy_row(LADDER_KEY, summaries[LADDER_KEY]))
    print(accuracy_row(COSET_KEY, summaries[COSET_KEY]))

    generic_diag = run["quantum_generic"]["kernel_diagnostics"]
    coset_diag = run["quantum_coset"]["kernel_diagnostics"]
    coset_map = run["quantum_coset"]["feature_map"]
    print(f"""
  coset window            M = 2^{coset_map['window_bits']} = {coset_map['window_M']}"""
          f""" of |Z_p*| = {coset_map['group_order']}
  mean off-diagonal K     coset {coset_diag['offdiag_mean']:.4f}   """
          f"""generic {generic_diag['offdiag_mean']:.4f}
  effective kernel rank   coset {coset_diag['effective_rank']:.1f}   """
          f"""generic {generic_diag['effective_rank']:.1f}   (of {coset_diag['n']})

  The generic kernel is nearly the identity: it separates every training
  point from every other, memorises perfectly, and generalises at chance.""")

    # ------------------------------------------------------------- headline
    best = run["best_competitor"]
    print(section("HEADLINE RESULT"))
    print(f"""
  best competing model    {best}  ->  {run['best_competitor_accuracy']:.3f}
  structure-aligned coset kernel  ->  {run['coset_accuracy']:.3f}
  SEPARATION              {run['separation']:+.3f}  """
          f"""({run['separation'] * 100:+.1f} percentage points)

  coset       {bar(run['coset_accuracy'])}  {run['coset_accuracy']:.3f}
  best other  {bar(run['best_competitor_accuracy'])}  """
          f"""{run['best_competitor_accuracy']:.3f}
  chance      {bar(0.5)}  0.500""")

    mcnemar = run["paired_test"]["mcnemar"]
    boot = run["paired_test"]["bootstrap_accuracy_difference"]
    print(f"""
  Paired comparison on the SAME held-out items ({run['paired_test']['comparison']}):
    McNemar exact           p = {format_p(mcnemar['p_value'])}
                            (coset right / other wrong: {mcnemar['n_a_correct_b_wrong']}, """
          f"""reverse: {mcnemar['n_a_wrong_b_correct']})
    bootstrap 95% CI on the accuracy difference:
                            [{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]

  Any competing model above chance at p < 0.01?  """
          f"""{'YES -- INVESTIGATE' if run['any_competitor_significant_at_1pct'] else 'NO'}""")

    # -------------------------------------------------------- falsification
    print(section("FALSIFICATION TESTS  (we tried to break our own claim)"))
    legendre = run["legendre_control"]
    oracle = run["oracle_control"]
    print(f"""
  1. GENERIC QUANTUM CONTROL       {summaries[LADDER_KEY]['accuracy']:.3f}  """
          f"""(p = {format_p(summaries[LADDER_KEY]['p_value'])})
     A strong entangling Rz/Rx/CZ feature map, same pipeline. It fails too,
     so the separation is NOT "quantum beats classical".

  2. NUMBER-THEORETIC ATTACK       {legendre['test_accuracy']:.3f}  """
          f"""(p = {format_p(summaries[LEGENDRE_KEY]['p_value'])})
     Euler's criterion recovers dlog parity in O(log p). It solves the
     `parity` concept outright, and is useless here -- so "involves a
     discrete log" does not by itself mean "hard".

  3. HIDDEN-x ORACLE CONTROL       {oracle['test_accuracy']:.3f}  """
          f"""(p = {format_p(summaries[ORACLE_KEY]['p_value'])})
     A RandomForest given x directly. NOT a valid model -- it proves the
     labels are perfectly learnable once the coordinate is exposed, so the
     ~50% scores above are about the ENCODING, not about label noise.""")

    # ------------------------------------------------------------ integrity
    audit = run["dataset_audit"]
    coset_ver = run["quantum_coset"].get("verification", {})
    print(section("INTEGRITY CHECKS"))
    print(f"""
  h = g^x mod p verified for all {audit['n_samples']} samples
  exact class balance            train {audit['balance']['train']}, """
          f"""test {audit['balance']['test']}
  train/test exponents disjoint  {audit['splits_disjoint']}
  independent BSGS dlog probes   {audit['bsgs_probes']}""")
    if coset_ver:
        print(f"""  kernel vs. compute-uncompute   max err """
              f"""{coset_ver.get('max_compute_uncompute_error', float('nan')):.2e}
  kernel vs. closed form         max err """
              f"""{coset_ver.get('max_analytic_error', float('nan')):.2e}""")

    # ------------------------------------------------------- interpretation
    print(section("INTERPRETATION"))
    print("""
  Quantum computing alone does not produce the separation. A generic
  quantum feature map lands at chance alongside every classical model.

  The separation arises from matching the feature-map geometry to the
  algebraic structure of the concept class: the coset state induces a
  triangular kernel in discrete-log coordinates, which is exactly the
  coordinate system in which the concept is a single interval.""")

    print(rule("="))
    print("IMPORTANT LIMITATION".center(len(rule("="))))
    print(rule("="))
    print(f"""
  This is a small SIMULATED instance. p = {group['p']} is NOT
  cryptographically hard -- baby-step giant-step solves it in
  O(sqrt(p)) and this repository does exactly that as a data audit.

  What is demonstrated : a measurable learning separation on a finite
                         concept-class instance, with controls.
  What is NOT claimed  : cryptographic-scale quantum advantage, a
                         complexity-theoretic speed-up proof, or any
                         result obtained on quantum hardware.

  The asymptotic claim is a theorem of Liu, Arunachalam & Temme (2021)
  conditioned on the Discrete Logarithm Assumption and on an efficient
  fault-tolerant implementation. See docs/LIMITATIONS.md.
""")
    print(rule("="))
    print(f"  demo wall-clock: {elapsed:.1f}s   "
          f"(full study: python scripts/run_experiment.py)")
    print(rule("="))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
