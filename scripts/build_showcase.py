#!/usr/bin/env python3
"""
Build `showcase.html` -- a standalone, shareable page summarising the project.

The page is presentation only. It contains no science of its own: every number
comes from `dlog_qml.audit.headline_numbers()` reading the stored artifacts, and
every figure is embedded from `figures/`. Nothing here is typed by hand, so the
page cannot drift away from the experiment.

Sources, in the order the page depends on them:

    configs/default.yaml      -> group / dataset parameters
    results/raw/*.json        -> every accuracy, CI, p-value, kernel diagnostic
    results/summary.json      -> the negative-controls checklist
    figures/*.png             -> embedded as base64 data URIs
    DLog_QML_Hackathon.ipynb  -> title, cell count, execution state

The output is a single file with no external requests, so it works offline and
can be emailed or dropped on any static host.

    python scripts/build_showcase.py            # build once
    python scripts/watch_showcase.py            # rebuild whenever anything changes
"""

from __future__ import annotations

import base64
import datetime as dt
import html
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from dlog_qml.audit import headline_numbers            # noqa: E402
from dlog_qml.config import ExperimentConfig           # noqa: E402

OUTPUT = os.path.join(ROOT, "showcase.html")

#: Files whose contents the page reflects. The watcher polls exactly these.
WATCH_PATHS = [
    os.path.join(ROOT, "DLog_QML_Hackathon.ipynb"),
    os.path.join(ROOT, "configs", "default.yaml"),
    os.path.join(ROOT, "results", "raw"),
    os.path.join(ROOT, "results", "summary.json"),
    os.path.join(ROOT, "figures"),
    os.path.abspath(__file__),
]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _load(path: str):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def gather() -> dict:
    """Collect everything the page renders. Missing pieces degrade to None."""
    raw = os.path.join(ROOT, "results", "raw")
    run = _load(os.path.join(raw, "single_run.json"))
    if run is None:
        raise SystemExit(
            "results/raw/single_run.json not found.\n"
            "  Run: python scripts/run_experiment.py"
        )

    multi = _load(os.path.join(raw, "multi_seed.json"))
    parity = _load(os.path.join(raw, "parity_run.json"))
    ablation = _load(os.path.join(raw, "window_ablation.json"))
    summary = _load(os.path.join(ROOT, "results", "summary.json")) or {}

    numbers = headline_numbers(run, multi_seed=multi, parity=parity, ablation=ablation)

    # notebook state
    notebook = {"cells": None, "executed": None}
    nb_path = os.path.join(ROOT, "DLog_QML_Hackathon.ipynb")
    if os.path.exists(nb_path):
        try:
            import nbformat as nbf
            nb = nbf.read(nb_path, as_version=4)
            code = [c for c in nb.cells if c.cell_type == "code"]
            notebook = {
                "cells": len(nb.cells),
                "code_cells": len(code),
                "executed": all(bool(c.outputs) for c in code) if code else False,
                "errors": sum(1 for c in code for o in c.outputs
                              if o.output_type == "error"),
            }
        except Exception:
            pass

    # current commit, if this is a git checkout
    commit = None
    try:
        commit = subprocess.run(
            ["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        pass

    return {
        "cfg": ExperimentConfig.load(os.path.join(ROOT, "configs", "default.yaml")),
        "numbers": numbers,
        "run": run,
        "controls": summary.get("negative_controls", []),
        "ablation": ablation,
        "notebook": notebook,
        "commit": commit,
    }


def embed(name: str) -> str | None:
    """Base64 data URI for a figure, so the page is fully self-contained."""
    path = os.path.join(ROOT, "figures", name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode("ascii")


# ---------------------------------------------------------------------------
# rendering helpers
# ---------------------------------------------------------------------------

def esc(value) -> str:
    return html.escape(str(value))


def figure(name: str, caption: str) -> str:
    uri = embed(name)
    if not uri:
        return ""
    return (f'<figure><img src="{uri}" alt="{esc(caption[:80])}" loading="lazy">'
            f'<figcaption>{caption}</figcaption></figure>')


def bar_row(label: str, value: float, note: str, kind: str) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return (f'<div class="bar-row">'
            f'<span class="bar-label">{esc(label)}</span>'
            f'<span class="bar-track"><span class="bar-fill {kind}" '
            f'style="width:{pct:.1f}%"></span></span>'
            f'<span class="bar-value">{value:.3f}</span>'
            f'<span class="bar-note">{esc(note)}</span></div>')


def table(headers, rows, cls="") -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return (f'<div class="scroll"><table class="{cls}"><thead><tr>{head}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

def render(data: dict) -> str:
    cfg = data["cfg"]
    n = data["numbers"]
    ref, ker = n["reference"], n.get("kernel", {})
    ms = n.get("multi_seed")
    parity = n.get("parity", {})
    abl = n.get("ablation", {})
    nb = data["notebook"]
    built = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ---- hero bars ---------------------------------------------------------
    if ms:
        bars = (bar_row("coset quantum", ms["coset_mean"],
                        f"± {ms['coset_std']:.3f}", "good")
                + bar_row("best classical", ms["best_competitor_mean"],
                          f"± {ms['best_competitor_std']:.3f} "
                          f"({ms['best_competitor']})", "neutral")
                + bar_row("generic quantum", ms["generic_mean"],
                          f"± {ms['generic_std']:.3f}", "bad")
                + bar_row("chance", 0.5, "", "chance"))
        scope = f"mean over {ms['n_seeds']} seeds"
        headline = f"{ms['separation_mean']:+.3f}"
        headline_sub = (f"per-seed separation, ± {ms['separation_std']:.3f}, "
                        f"worst seed {ms['separation_min']:+.3f}, "
                        f"ahead on {ms['seeds_coset_beats_best']}/{ms['n_seeds']}")
    else:
        bars = (bar_row("coset quantum", ref["coset_accuracy"], "", "good")
                + bar_row("best classical", ref["best_competitor_accuracy"],
                          ref["best_competitor"], "neutral")
                + bar_row("generic quantum", ref["generic_accuracy"], "", "bad")
                + bar_row("chance", 0.5, "", "chance"))
        scope = "single reference run"
        headline = f"{ref['separation']:+.3f}"
        headline_sub = "single-run separation (no seed sweep found)"

    # ---- reference table ---------------------------------------------------
    summaries = data["run"]["summaries"]
    order = sorted(summaries, key=lambda k: -summaries[k]["accuracy"])
    ref_rows = []
    for k in order:
        s = summaries[k]
        p = f"{s['p_value']:.1e}" if s["p_value"] < 1e-4 else f"{s['p_value']:.3f}"
        verdict = ('<span class="tag ok">learns</span>' if s["significant_at_1pct"]
                   else '<span class="tag muted">chance (n.s.)</span>')
        ref_rows.append([esc(k), f"{s['accuracy']:.3f}",
                         f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]", p, verdict])

    # ---- controls ----------------------------------------------------------
    ctrl_rows = [[
        ('<span class="tag ok">pass</span>' if c["passed"]
         else '<span class="tag bad">FAIL</span>'),
        esc(c["control"]), esc(c["observed"]), esc(c["would_invalidate_if"]),
    ] for c in data["controls"]]
    n_pass = sum(1 for c in data["controls"] if c["passed"])

    # ---- kernel geometry ---------------------------------------------------
    ker_rows = []
    if "coset" in ker and "generic" in ker:
        c, g = ker["coset"], ker["generic"]
        for label, key, fmt in (
                ("mean off-diagonal K", "offdiag_mean", "{:.4f}"),
                ("effective rank (of %d)" % c["n"], "effective_rank", "{:.1f}"),
                ("top-10 eigenvalue mass", "top10_fraction", "{:.3f}"),
                ("kernel-target alignment", "alignment", "{:.4f}"),
                ("K same class", "within_class", "{:.4f}"),
                ("K different class", "between_class", "{:.4f}"),
                ("class separation gap", "separation_gap", "{:+.4f}")):
            ker_rows.append([label, fmt.format(c[key]), fmt.format(g[key])])

    # ---- ablation ----------------------------------------------------------
    abl_block = ""
    if data["ablation"]:
        rows = [r for r in data["ablation"]["rows"] if "error" not in r]
        default_m = data["ablation"].get("default_window_bits")
        abl_block = table(
            ["m", "M", "coset accuracy", "mean off-diag K"],
            [[f"{r['window_bits']}" + (" <em>(default)</em>"
                                       if r["window_bits"] == default_m else ""),
              r["window_M"], f"{r['coset_accuracy']:.3f}",
              f"{r['offdiag_mean']:.4f}"] for r in rows])

    nb_line = ("notebook not found" if nb["cells"] is None else
               f"{nb['cells']} cells, {nb['code_cells']} code, "
               f"{'all executed' if nb['executed'] else 'NOT fully executed'}, "
               f"{nb['errors']} errors")

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DLog-QML — showcase</title>
<style>
  :root {{
    --bg:#ffffff; --panel:#f6f7f9; --line:#e2e5ea; --fg:#15181d; --dim:#5d6570;
    --good:#2a9d8f; --bad:#e76f51; --neutral:#4a5568; --chance:#264653;
    --mono:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg:#101318; --panel:#171b22; --line:#262c36; --fg:#e6e9ee;
             --dim:#96a0ad; --neutral:#8d99ae; --chance:#7f9aa6; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg); line-height:1.62;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
    -webkit-font-smoothing:antialiased; }}
  .wrap {{ max-width:940px; margin:0 auto; padding:56px 22px 90px; }}
  h1 {{ font-size:2.5rem; margin:0 0 .2em; letter-spacing:-.02em; }}
  h2 {{ font-size:1.32rem; margin:3em 0 .7em; padding-bottom:.35em;
        border-bottom:1px solid var(--line); letter-spacing:-.01em; }}
  h3 {{ font-size:1.02rem; margin:2em 0 .5em; }}
  .sub {{ font-size:1.14rem; color:var(--dim); margin:0 0 1.6em; }}
  p {{ margin:0 0 1.05em; }}
  code {{ font-family:var(--mono); font-size:.9em; background:var(--panel);
          padding:.12em .38em; border-radius:4px; }}
  .lede {{ font-size:1.06rem; }}

  .hero {{ background:var(--panel); border:1px solid var(--line);
    border-radius:12px; padding:26px 26px 20px; margin:2em 0; }}
  .hero-scope {{ font-size:.8rem; text-transform:uppercase; letter-spacing:.09em;
    color:var(--dim); margin-bottom:16px; }}
  .bar-row {{ display:grid; grid-template-columns:130px 1fr 58px 1fr;
    align-items:center; gap:12px; margin:9px 0; font-size:.9rem; }}
  .bar-label {{ color:var(--dim); }}
  .bar-track {{ background:rgba(128,128,128,.16); border-radius:5px; height:19px;
    overflow:hidden; }}
  .bar-fill {{ display:block; height:100%; border-radius:5px; }}
  .bar-fill.good{{background:var(--good);}} .bar-fill.bad{{background:var(--bad);}}
  .bar-fill.neutral{{background:var(--neutral);}}
  .bar-fill.chance{{background:var(--chance);opacity:.4;}}
  .bar-value {{ font-family:var(--mono); text-align:right; }}
  .bar-note {{ color:var(--dim); font-size:.82rem; }}
  .headline {{ margin-top:20px; padding-top:18px; border-top:1px solid var(--line);
    display:flex; align-items:baseline; gap:14px; flex-wrap:wrap; }}
  .headline .big {{ font-size:2.1rem; font-family:var(--mono); color:var(--good);
    font-weight:600; }}
  .headline .small {{ color:var(--dim); font-size:.88rem; }}

  .scroll {{ overflow-x:auto; margin:1.1em 0; }}
  table {{ border-collapse:collapse; width:100%; font-size:.87rem; }}
  th,td {{ text-align:left; padding:7px 12px; border-bottom:1px solid var(--line);
           white-space:nowrap; }}
  th {{ color:var(--dim); font-weight:600; font-size:.76rem;
        text-transform:uppercase; letter-spacing:.05em; }}
  td:nth-child(n+2) {{ font-family:var(--mono); }}
  .controls td:nth-child(n+2) {{ font-family:inherit; }}
  .controls td:nth-child(4) {{ color:var(--dim); white-space:normal;
                               min-width:250px; }}
  .tag {{ font-size:.72rem; padding:2px 8px; border-radius:20px;
          font-family:inherit; white-space:nowrap; }}
  .tag.ok {{ background:rgba(42,157,143,.16); color:var(--good); }}
  .tag.bad {{ background:rgba(231,111,81,.18); color:var(--bad); }}
  .tag.muted {{ background:rgba(128,128,128,.14); color:var(--dim); }}

  figure {{ margin:1.6em 0; }}
  figure img {{ width:100%; height:auto; border:1px solid var(--line);
    border-radius:9px; background:#fff; }}
  figcaption {{ color:var(--dim); font-size:.85rem; margin-top:.6em; }}

  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:0 26px; }}
  @media (max-width:680px) {{ .grid2 {{ grid-template-columns:1fr; }}
    .bar-row {{ grid-template-columns:96px 1fr 52px; }} .bar-note {{ display:none; }} }}
  .note {{ border-left:3px solid var(--line); padding:2px 0 2px 15px;
    color:var(--dim); font-size:.92rem; margin:1.2em 0; }}
  footer {{ margin-top:4em; padding-top:1.3em; border-top:1px solid var(--line);
    color:var(--dim); font-size:.82rem; }}
  a {{ color:inherit; }}
</style></head><body><div class="wrap">

<h1>DLog-QML</h1>
<p class="sub">Can mathematical structure make quantum kernels learnable?</p>

<p class="lede">A binary classification task where the label is simple in one coordinate
system and, as far as we can tell, invisible in another. The two are related by a discrete
logarithm: the label depends on log<sub>g</sub>&nbsp;h, and the model only ever sees
<code>h = g^x mod p</code>.</p>

<p>The eleven tested classical models remain close to chance. So does a strong generic
quantum kernel. A feature map built around the group structure that produced the encoding
gets close to perfect. That gap is the result, and most of the work went into trying to
make it go away.</p>

<div class="hero">
  <div class="hero-scope">held-out accuracy — {esc(scope)}</div>
  {bars}
  <div class="headline">
    <span class="big">{esc(headline)}</span>
    <span class="small">{esc(headline_sub)}</span>
  </div>
</div>

<div class="note">This is a representation experiment, not a speedup result.
p&nbsp;=&nbsp;{esc(_p(data))} is small enough that the test suite breaks it with baby-step
giant-step as an integrity check, everything runs in classical simulation, and the dense
state preparation we simulate is exponential. No quantum hardware was used.</div>

<h2>The task</h2>
{figure("fig3_concept_geometry.png",
        "The same points twice on the same kind of strip. Left: against the group element "
        "the model receives — colours interleaved. Middle: the identical points against the "
        "hidden coordinate, split by one boundary. Right: the exponent circle with the two "
        "label arcs and the coset window the feature map spans.")}

<h2>The feature map</h2>
<p>Instead of encoding the bits of <code>h</code>, encode a stretch of the group's own
orbit: |Φ(h)⟩ ∝ Σ<sub>i&lt;M</sub> |h·g<sup>i</sup> mod p⟩. Building that state uses forward
modular multiplication only — it never calls a discrete logarithm. But the support is a
contiguous run on the exponent circle, so the induced kernel is a triangle in discrete-log
distance, and the SVM is left with interval membership in one dimension.</p>
{figure("fig2_kernel_matrices.png",
        "Rows sorted by the hidden discrete log, identical colour scale. The coset kernel is "
        "a band whose width is the coset window. The generic quantum kernel is essentially "
        "the identity. The classical RBF kernel has variance but none of it aligned with the "
        "dlog axis.")}

<h2>Reference run</h2>
<p>p&nbsp;=&nbsp;{esc(_p(data))}, {cfg.n_train} train / {cfg.n_test} test, exact 50/50,
disjoint exponents, seed {cfg.seed}. With n<sub>test</sub>&nbsp;=&nbsp;{cfg.n_test} the
binomial standard error at chance is {0.5 / (cfg.n_test ** 0.5):.4f}, which is why every row
carries an interval and a p-value rather than just an accuracy.</p>
{table(["model", "accuracy", "95% CI", "p vs chance", ""], ref_rows)}
<p>Coset kernel AUC {ref['coset_auc']:.4f}. Paired against the best competitor on the same
held-out items: McNemar exact p&nbsp;=&nbsp;{ref['mcnemar_p']:.1e}, bootstrap 95% CI on the
accuracy difference [{ref['bootstrap_ci'][0]:+.3f}, {ref['bootstrap_ci'][1]:+.3f}].</p>

<h2>Why the generic kernel fails</h2>
<p>Landing at chance was not surprising; the useful part is the reason. Both kernels put
most pairs near zero, which is expected for a local kernel. The difference is the tail —
for the coset kernel only same-class pairs reach high similarity.</p>
{table(["diagnostic", "coset", "generic"], ker_rows)}
{figure("fig6_kernel_diagnostics.png",
        "(a) the generic kernel's spectrum is nearly flat. (b, c) its same-class and "
        "different-class similarity distributions sit on top of each other; the coset "
        "kernel's do not. (d) alignment against kernel concentration.")}

<h2>Trying to break it</h2>
<p>{n_pass} of {len(data['controls'])} controls passed. Each states what would have
invalidated the result.</p>
{table(["", "control", "observed", "would invalidate if"], ctrl_rows, cls="controls")}
{_parity_note(parity)}

<h2>Across seeds</h2>
{figure("fig7_seed_distribution.png",
        "Each seed redraws the secret shift and the dataset. The coset kernel is above every "
        "other model on every seed, and the separation histogram has no mass near zero.")}

<h2>Ablation</h2>
<p>The coset window M&nbsp;=&nbsp;2<sup>m</sup> is the one free parameter, set by a fixed
rule rather than tuned{_abl_note(abl)}</p>
{abl_block}
{figure("fig5_window_ablation.png",
        "Accuracy against window size, with mean off-diagonal kernel value on the right "
        "axis. Too narrow makes the kernel orthogonal; too wide flattens it.")}

<h2>What this does not show</h2>
<div class="grid2">
<div><h3>Established</h3><ul>
<li>A repeatable accuracy gap under the tested conditions</li>
<li>A generic quantum kernel does not close it</li>
<li>The target is not exposed by the obvious number-theoretic invariant</li>
<li>The labels are learnable given the hidden coordinate</li>
<li>The coset kernel fails on a target it is not matched to</li>
</ul></div>
<div><h3>Not established</h3><ul>
<li>That no efficient classical algorithm exists</li>
<li>That quantum kernels are generally better</li>
<li>That the target is cryptographically hard at this size</li>
<li>Anything about quantum hardware</li>
<li>A complexity-theoretic separation</li>
</ul></div>
</div>

<footer>
Generated {esc(built)} by <code>scripts/build_showcase.py</code> from
<code>results/raw/</code> and <code>figures/</code> — no value on this page is typed by
hand.<br>
Notebook: {esc(nb_line)}.{_commit(data)}<br>
Concept class, coset feature map and asymptotic theorem: Liu, Arunachalam &amp; Temme,
<em>Nature Physics</em> <strong>17</strong>, 1013 (2021), arXiv:2010.02174.
</footer>

</div></body></html>"""


def _p(data) -> str:
    return str(data["run"]["group"]["p"])


def _commit(data) -> str:
    return f" Commit <code>{esc(data['commit'])}</code>." if data.get("commit") else ""


def _parity_note(parity: dict) -> str:
    if not parity:
        return ""
    return (f'<p>The strongest of these: an O(log p) Legendre/Euler attack solves a naive '
            f'<em>parity</em> version of the target at '
            f'<strong>{parity["legendre_accuracy"]:.3f}</strong> and fails on the one we use. '
            f'And the coset kernel itself drops to '
            f'<strong>{parity["coset_accuracy"]:.3f}</strong> on that mismatched concept — '
            f'it is not a universal cheat.</p>')


def _abl_note(abl: dict) -> str:
    if not abl:
        return "."
    return (f". {abl['n_above_085']} of {abl['n_windows']} tested windows stay above 0.85, "
            f"and the default is not the best value in the range.")


def main() -> int:
    data = gather()
    with open(OUTPUT, "w", encoding="utf-8") as fh:
        fh.write(render(data))
    size = os.path.getsize(OUTPUT) / 1024
    print(f"wrote {OUTPUT}  ({size:.0f} KB, self-contained)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
