"""
Publication-quality static figures.

Design rule followed throughout: never manipulate a visualisation to exaggerate
a difference. Kernel heatmaps share a common vmin/vmax of [0, 1], accuracy axes
start at 0, the chance line is always drawn, and anything that is a theoretical
projection rather than a measurement is labelled as such inside the figure.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .experiments import COSET_KEY, LADDER_KEY, LEGENDRE_KEY, ORACLE_KEY

# Colour roles, used consistently across every figure.
C_COSET = "#2a9d8f"      # structure-aligned quantum
C_GENERIC = "#e76f51"    # generic quantum control
C_CLASSICAL = "#4a5568"  # classical baselines
C_ORACLE = "#8d99ae"     # oracle / leakage control
C_CHANCE = "#264653"     # chance reference line


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return plt


def _colour_for(name: str) -> str:
    if name == COSET_KEY:
        return C_COSET
    if name == LADDER_KEY:
        return C_GENERIC
    if name == ORACLE_KEY:
        return C_ORACLE
    return C_CLASSICAL


# ---------------------------------------------------------------------------
# Figure 1 -- accuracy comparison
# ---------------------------------------------------------------------------

def plot_accuracy_comparison(run: Dict, outdir: str,
                             multi_seed: Optional[Dict] = None) -> str:
    """
    Horizontal bar chart of held-out accuracy for every model.

    When multi-seed results exist the bars show the mean across seeds with
    +/- 1 s.d. error bars and the single-run value overlaid as a marker;
    otherwise single-run accuracies are shown with their Wilson 95% intervals.
    """
    plt = _pyplot()
    summaries: Dict[str, Dict] = run["summaries"]
    use_seeds = bool(multi_seed and multi_seed.get("n_seeds_completed", 0) > 1)
    per_model = (multi_seed or {}).get("per_model", {})

    def plotted_value(name: str) -> float:
        if use_seeds and name in per_model:
            return per_model[name]["mean"]
        return summaries[name]["accuracy"]

    # Sort by the quantity actually drawn, not by the single-run accuracy --
    # otherwise the bar order contradicts the printed labels.
    names = sorted(summaries, key=plotted_value)

    values, errors, single = [], [], []
    for name in names:
        single.append(summaries[name]["accuracy"])
        values.append(plotted_value(name))
        errors.append(per_model[name]["std"]
                      if use_seeds and name in per_model else 0.0)

    fig, ax = plt.subplots(figsize=(10.5, 0.46 * len(names) + 2.6))
    ax.barh(names, values, xerr=errors if use_seeds else None,
            color=[_colour_for(n) for n in names],
            error_kw={"ecolor": "#22223b", "capsize": 3, "lw": 1.1})

    if use_seeds:
        ax.scatter(single, names, marker="|", s=110, color="white",
                   zorder=3, linewidths=1.6,
                   label="single reference run (seed %d)" % run["config"]["seed"])

    ax.axvline(0.5, ls="--", lw=1.6, color=C_CHANCE, label="chance (0.50)")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("held-out accuracy")

    n_seeds = multi_seed.get("n_seeds_completed") if use_seeds else None
    subtitle = (f"mean $\\pm$ 1 s.d. over {n_seeds} seeds"
                if use_seeds else "single run, Wilson 95% CI as text")
    ax.set_title(f"Learning the discrete-log concept class  "
                 f"(p = {run['group']['p']}, {run['group']['n_bits']} qubits, "
                 f"n_test = {run['config']['n_test']})\n{subtitle}", fontsize=11)

    for index, name in enumerate(names):
        # Offset past the error bar so the label never sits on top of it.
        ax.text(values[index] + errors[index] + 0.016, index,
                f"{values[index]:.3f}", va="center", fontsize=9)

    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, "fig1_accuracy_comparison.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2 -- kernel matrices
# ---------------------------------------------------------------------------

def plot_kernel_matrices(run: Dict, outdir: str) -> Optional[str]:
    """
    Side-by-side Gram matrices, rows sorted by the HIDDEN discrete log.

    Sorting by log_g(h) is the whole point: it is the coordinate the learner
    cannot compute. In that ordering the coset kernel is a clean banded
    diagonal (plus wraparound corners, since the exponents live on a circle),
    while the generic quantum and classical RBF kernels show no structure.
    All three panels share the same [0, 1] colour scale.
    """
    kernels = run.get("_kernels")
    dataset = run.get("_dataset")
    if kernels is None or dataset is None:
        return None

    plt = _pyplot()
    order = np.argsort(dataset.oracle_exponents("train"))

    panels = [
        ("coset_train", "Structure-aligned coset kernel\n(quantum)", C_COSET),
        ("ladder_train", "Generic hardware-efficient kernel\n(quantum control)", C_GENERIC),
        ("classical_rbf_train", "Classical RBF kernel\n(on bits of h)", C_CLASSICAL),
    ]
    available = [(k, t, c) for k, t, c in panels if kernels.get(k) is not None]

    fig, axes = plt.subplots(1, len(available), figsize=(5.1 * len(available), 5.0))
    if len(available) == 1:
        axes = [axes]

    for ax, (key, title, colour) in zip(axes, available):
        K = np.asarray(kernels[key])[np.ix_(order, order)]
        image = ax.imshow(K, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=10, color=colour)
        ax.set_xlabel("training sample (sorted by hidden $\\log_g h$)")
        ax.set_ylabel("training sample (sorted by hidden $\\log_g h$)")
        ax.grid(False)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03,
                     label="$K_{ij}$")

    fig.suptitle("Kernel structure exists only in discrete-log coordinates "
                 "(identical ordering, identical colour scale)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "fig2_kernel_matrices.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3 -- concept geometry
# ---------------------------------------------------------------------------

def plot_concept_geometry(run: Dict, outdir: str) -> Optional[str]:
    """
    The core intuition in one picture.

    Left  : label vs. the OBSERVED group element h. Structureless -- this is
            what every classical baseline sees.
    Middle: label vs. the HIDDEN shifted exponent (x - s) mod (p-1). Two clean
            contiguous blocks: the concept is an interval.
    Right : the same on the exponent circle, showing the two arcs and the
            triangular kernel window M that the coset feature map spans.
    """
    dataset = run.get("_dataset")
    if dataset is None:
        return None

    plt = _pyplot()
    group = dataset.group
    secret = getattr(dataset.label_rule, "secret_s", 0)
    order = group.order

    h = dataset.group_elements("train")
    x = dataset.oracle_exponents("train")
    y = dataset.labels("train")
    shifted = (x - secret) % order
    positive, negative = y == 1, y == -1

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))

    # Both 1-D panels are drawn as the SAME kind of colour strip: every point
    # sits at the same height and only its colour carries the label. Splitting
    # the classes onto separate rows would manufacture visual separation in the
    # left panel, where the whole point is that no structure exists.
    rng = np.random.default_rng(0)
    jitter_pos = rng.uniform(-0.35, 0.35, int(positive.sum()))
    jitter_neg = rng.uniform(-0.35, 0.35, int(negative.sum()))

    # -- left: what the learner sees -------------------------------------
    ax = axes[0]
    ax.scatter(h[positive], jitter_pos, s=14, alpha=0.85,
               color=C_COSET, label="$y = +1$")
    ax.scatter(h[negative], jitter_neg, s=14, alpha=0.85,
               color=C_GENERIC, label="$y = -1$")
    ax.set_yticks([])
    ax.set_ylim(-1.4, 1.4)
    # NB: matplotlib mathtext has no \bmod -- use \mathrm{mod} instead.
    ax.set_xlabel("observed group element $h = g^x\\ \\mathrm{mod}\\ p$")
    ax.set_title("What the learner sees\n"
                 "colours fully interleaved: no structure", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # -- middle: the hidden coordinate ------------------------------------
    ax = axes[1]
    ax.scatter(shifted[positive], jitter_pos, s=14, alpha=0.85, color=C_COSET)
    ax.scatter(shifted[negative], jitter_neg, s=14, alpha=0.85, color=C_GENERIC)
    ax.axvline(group.q, ls="--", lw=1.6, color=C_CHANCE)
    ax.text(group.q, 1.05, " boundary at $q=(p-1)/2$", fontsize=8, color=C_CHANCE)
    ax.set_yticks([])
    ax.set_ylim(-1.4, 1.4)
    ax.set_xlabel("hidden coordinate $(\\log_g h - s)\\ \\mathrm{mod}\\ (p-1)$")
    ax.set_title("The same points, same strip, dlog coordinates\n"
                 "one clean split: the concept is an interval", fontsize=10)

    # -- right: the exponent circle ---------------------------------------
    ax = axes[2]
    theta = 2 * np.pi * shifted / order
    ax.scatter(np.cos(theta[positive]), np.sin(theta[positive]),
               s=16, color=C_COSET, label="$y=+1$")
    ax.scatter(np.cos(theta[negative]), np.sin(theta[negative]),
               s=16, color=C_GENERIC, label="$y=-1$")

    window_bits = run["quantum_coset"]["feature_map"]["window_bits"]
    window = 2 ** window_bits
    arc = np.linspace(0, 2 * np.pi * window / order, 200)
    ax.plot(1.22 * np.cos(arc), 1.22 * np.sin(arc), lw=3.2, color="#1d3557",
            solid_capstyle="round",
            label=f"coset window $M = 2^{{{window_bits}}} = {window}$")
    ax.set_aspect("equal")
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(-1.45, 1.45)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_title("Exponent circle $\\mathbb{Z}_{p-1}$\n"
                 "two arcs; kernel is triangular in arc distance", fontsize=10)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.16))

    fig.suptitle("Why the encoding, not the model, is the barrier", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "fig3_concept_geometry.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4 -- measured scaling
# ---------------------------------------------------------------------------

def plot_scaling(sweep: Dict, outdir: str) -> Optional[str]:
    """
    MEASURED scaling across register widths -- accuracy, simulation wall-clock,
    and statevector memory. Every point is an experiment that actually ran; the
    axis titles say so explicitly to prevent it being read as a projection.
    """
    rows = [r for r in sweep.get("rows", []) if "error" not in r]
    if len(rows) < 2:
        return None

    plt = _pyplot()
    n = [r["n_qubits"] for r in rows]
    coset = [r["coset_accuracy"] for r in rows]
    generic = [r["generic_accuracy"] for r in rows]
    classical = [r["best_competitor_accuracy"] for r in rows]
    seconds = [r["coset_simulate_seconds"] + r["ladder_simulate_seconds"] for r in rows]
    megabytes = [r["statevector_megabytes"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))

    ax = axes[0]
    ax.plot(n, coset, "o-", color=C_COSET, lw=2, label="coset kernel")
    ax.plot(n, generic, "s--", color=C_GENERIC, lw=1.6, label="generic quantum")
    ax.plot(n, classical, "^--", color=C_CLASSICAL, lw=1.6, label="best classical")
    ax.axhline(0.5, ls=":", color=C_CHANCE, label="chance")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("register width $n$ (qubits)")
    ax.set_ylabel("held-out accuracy")
    ax.set_title("MEASURED accuracy vs. problem size", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.semilogy(n, seconds, "o-", color="#1d3557", lw=2)
    ax.set_xlabel("register width $n$ (qubits)")
    ax.set_ylabel("state simulation wall-clock (s)")
    ax.set_title("MEASURED simulation cost", fontsize=10)

    ax = axes[2]
    ax.semilogy(n, megabytes, "o-", color="#7d4f9a", lw=2)
    ax.set_xlabel("register width $n$ (qubits)")
    ax.set_ylabel("statevector memory (MB)")
    ax.set_title("MEASURED dense-statevector memory\n"
                 "(exponential in $n$: the simulation wall)", fontsize=10)

    fig.suptitle("Measured scaling of the simulated demonstration "
                 "(NOT a projection to cryptographic size)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "fig4_scaling_measured.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_window_ablation(ablation: Dict, outdir: str) -> Optional[str]:
    """Coset accuracy and kernel concentration as a function of window M."""
    rows = [r for r in ablation.get("rows", []) if "error" not in r]
    if len(rows) < 2:
        return None

    plt = _pyplot()
    m = [r["window_bits"] for r in rows]
    accuracy = [r["coset_accuracy"] for r in rows]
    offdiag = [r["offdiag_mean"] for r in rows]
    default_m = ablation.get("default_window_bits")

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(m, accuracy, "o-", color=C_COSET, lw=2, label="coset kernel accuracy")
    ax.axhline(0.5, ls="--", color=C_CHANCE, lw=1.4, label="chance")
    if default_m is not None:
        ax.axvline(default_m, ls=":", color="#1d3557", lw=1.6,
                   label=f"default $m={default_m}$")
    ax.set_xlabel("coset window exponent $m$   ($M = 2^m$)")
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(0, 1.05)

    twin = ax.twinx()
    twin.plot(m, offdiag, "s--", color=C_GENERIC, lw=1.5, alpha=0.85,
              label="mean off-diagonal $K_{ij}$")
    twin.set_ylabel("mean off-diagonal kernel value")
    twin.grid(False)

    lines, labels = ax.get_legend_handles_labels()
    tlines, tlabels = twin.get_legend_handles_labels()
    ax.legend(lines + tlines, labels + tlabels, fontsize=8, loc="lower right")
    ax.set_title("Ablation: coset window size\n"
                 "too small -> orthogonal kernel; too large -> flat kernel",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(outdir, "fig5_window_ablation.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------

def plot_kernel_diagnostics(run: Dict, outdir: str) -> Optional[str]:
    """
    Why the generic kernel fails -- the mechanistic panel.

    Left   : eigenvalue spectra. A concentrated (near-identity) kernel has a
             flat spectrum; a structured kernel front-loads its mass.
    Middle : class-conditional similarity. A usable kernel puts same-class
             pairs closer than different-class pairs.
    Right  : kernel-target alignment. How much of the geometry points along
             the labels at all.
    """
    coset = run["quantum_coset"].get("kernel_report")
    generic = run["quantum_generic"].get("kernel_report")
    if not coset or not generic:
        return None

    kernels = run.get("_kernels")
    dataset = run.get("_dataset")

    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.4))

    # -- (a) eigenspectrum -------------------------------------------------
    ax = axes[0, 0]
    for report, label, colour in ((coset, "coset", C_COSET),
                                  (generic, "generic", C_GENERIC)):
        eigenvalues = np.asarray(report["spectrum"]["eigenvalues"], dtype=float)
        ax.semilogy(np.arange(1, len(eigenvalues) + 1),
                    np.maximum(eigenvalues, 1e-6), lw=2, color=colour, label=label)
    ax.set_xlabel("eigenvalue index (descending)")
    ax.set_ylabel("eigenvalue (log)")
    ax.set_title("(a) Eigenspectrum", fontsize=10)
    ax.legend(fontsize=9)

    # -- (b,c) class-conditional similarity DISTRIBUTIONS -------------------
    # Means hide the shape. The coset kernel's same-class mass is a long right
    # tail (nearby points on the dlog circle); the generic kernel's two
    # distributions sit on top of each other.
    for panel, (key, report, title, colour) in enumerate((
            ("coset_train", coset, "(b) coset kernel", C_COSET),
            ("ladder_train", generic, "(c) generic kernel", C_GENERIC))):
        ax = axes[0, 1] if panel == 0 else axes[1, 0]
        if kernels is not None and dataset is not None and kernels.get(key) is not None:
            K = np.asarray(kernels[key])
            y = dataset.labels("train")
            n = K.shape[0]
            same = np.equal.outer(y, y) & ~np.eye(n, dtype=bool)
            diff = (~np.equal.outer(y, y)) & ~np.eye(n, dtype=bool)
            bins = np.linspace(0, max(1e-6, float(K[same | diff].max())), 45)
            ax.hist(K[same], bins=bins, alpha=0.65, color=colour,
                    label="same class", log=True)
            ax.hist(K[diff], bins=bins, alpha=0.55, color="#6c757d",
                    label="different class", log=True)
            ax.set_xlabel("$K_{ij}$ (off-diagonal)")
            ax.set_ylabel("pair count (log)")
            ax.legend(fontsize=8)
        cc = report["class_conditional"]
        ax.set_title(f"{title}\nmean same {cc['within_class_mean']:.4f} vs "
                     f"diff {cc['between_class_mean']:.4f}", fontsize=10)

    # -- (d) alignment and effective rank ----------------------------------
    ax = axes[1, 1]
    x = np.arange(2)
    align = [coset["target_alignment_centered"], generic["target_alignment_centered"]]
    rank_frac = [coset["effective_rank"] / coset["n"],
                 generic["effective_rank"] / generic["n"]]
    ax.bar(x - 0.19, align, width=0.36, color=[C_COSET, C_GENERIC],
           label="target alignment")
    ax.bar(x + 0.19, rank_frac, width=0.36, color=[C_COSET, C_GENERIC],
           alpha=0.45, hatch="//", label="effective rank / n")
    for xi, (a, r) in enumerate(zip(align, rank_frac)):
        ax.text(xi - 0.19, a + 0.02, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(xi + 0.19, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["coset", "generic"])
    ax.set_ylim(0, 1.12)
    ax.set_title("(d) Alignment vs kernel concentration", fontsize=10)
    ax.legend(fontsize=8, loc="upper center")

    fig.suptitle("Kernel geometry: what differs between the two feature maps",
                 fontsize=12)
    fig.tight_layout()
    path = os.path.join(outdir, "fig6_kernel_diagnostics.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_seed_distribution(multi_seed: Dict, outdir: str) -> Optional[str]:
    """
    Per-seed accuracy for the three headline models.

    A mean and a standard deviation can hide a bimodal or heavy-tailed result.
    Plotting every seed shows whether the separation is uniform or driven by a
    few runs.
    """
    raw = multi_seed.get("raw_accuracies") or {}
    if not raw:
        return None

    from .experiments import COSET_KEY, LADDER_KEY, LEGENDRE_KEY, ORACLE_KEY

    classical = [n for n in raw if n not in (COSET_KEY, LADDER_KEY,
                                             ORACLE_KEY, LEGENDRE_KEY)]
    if not classical:
        return None
    best_classical = max(classical,
                         key=lambda n: float(np.mean(raw[n])))

    series = [(COSET_KEY, "coset quantum", C_COSET),
              (f"best classical\n({best_classical})", best_classical, C_CLASSICAL),
              (LADDER_KEY, "generic quantum", C_GENERIC)]

    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    # -- left: accuracy per seed ------------------------------------------
    ax = axes[0]
    for key, label, colour in [(COSET_KEY, "coset quantum", C_COSET),
                               (best_classical, f"best classical ({best_classical})",
                                C_CLASSICAL),
                               (LADDER_KEY, "generic quantum", C_GENERIC)]:
        values = np.asarray(raw[key], dtype=float)
        ax.plot(np.arange(1, len(values) + 1), values, "o-", ms=4, lw=1.3,
                color=colour, label=label)
    ax.axhline(0.5, ls="--", lw=1.4, color=C_CHANCE, label="chance")
    ax.set_xlabel("seed index")
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Accuracy on each of the {len(raw[COSET_KEY])} seeds", fontsize=10)
    ax.legend(fontsize=8, loc="center right")

    # -- right: separation distribution ------------------------------------
    ax = axes[1]
    separations = np.asarray(multi_seed.get("raw_separations", []), dtype=float)
    if separations.size:
        ax.hist(separations, bins=10, color=C_COSET, alpha=0.85,
                edgecolor="white")
        ax.axvline(0.0, ls="--", lw=1.4, color=C_CHANCE)
        ax.axvline(float(separations.mean()), lw=1.8, color="#1d3557",
                   label=f"mean {separations.mean():+.3f}")
        ax.set_xlabel("separation (coset - best competitor)")
        ax.set_ylabel("seeds")
        ax.set_title(f"Separation per seed\nworst {separations.min():+.3f}",
                     fontsize=10)
        ax.legend(fontsize=8)

    fig.tight_layout()
    path = os.path.join(outdir, "fig7_seed_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_all(run: Dict, outdir: str, multi_seed: Optional[Dict] = None,
                 sweep: Optional[Dict] = None,
                 ablation: Optional[Dict] = None) -> List[str]:
    """Render every figure whose inputs are available. Never raises."""
    os.makedirs(outdir, exist_ok=True)
    written: List[str] = []
    jobs = [
        ("accuracy comparison", lambda: plot_accuracy_comparison(run, outdir, multi_seed)),
        ("kernel matrices", lambda: plot_kernel_matrices(run, outdir)),
        ("concept geometry", lambda: plot_concept_geometry(run, outdir)),
        ("kernel diagnostics", lambda: plot_kernel_diagnostics(run, outdir)),
    ]
    if sweep:
        jobs.append(("scaling", lambda: plot_scaling(sweep, outdir)))
    if ablation:
        jobs.append(("window ablation", lambda: plot_window_ablation(ablation, outdir)))
    if multi_seed:
        jobs.append(("seed distribution",
                     lambda: plot_seed_distribution(multi_seed, outdir)))

    for label, job in jobs:
        try:
            path = job()
            if path:
                written.append(path)
        except Exception as exc:                          # pragma: no cover
            print(f"  [figure '{label}' skipped: {type(exc).__name__}: {exc}]")
    return written
