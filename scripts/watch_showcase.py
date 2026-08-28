#!/usr/bin/env python3
"""
Watch the project and rebuild `showcase.html` whenever anything it displays changes.

    python scripts/watch_showcase.py                 # watch, rebuild on change
    python scripts/watch_showcase.py --interval 1    # poll faster
    python scripts/watch_showcase.py --once          # build once and exit
    python scripts/watch_showcase.py --open          # open the page after the first build

Watched:

    DLog_QML_Hackathon.ipynb   the notebook -- edit or re-run it and the page follows
    configs/default.yaml       group / dataset parameters
    results/raw/*.json         every accuracy, CI, p-value, kernel diagnostic
    results/summary.json       the negative-controls checklist
    figures/*.png              embedded into the page
    scripts/build_showcase.py  the generator itself

Polling by (mtime, size) rather than using an OS watch API keeps this
dependency-free -- no watchdog, no extra install. A poll over ~30 files costs
well under a millisecond, so a 2 s interval is free in practice.

Note on what "changing a value in the notebook" does: editing a *markdown* or
*code* cell changes the notebook file, and the page rebuilds immediately with
the new notebook metadata. But the accuracies on the page come from
`results/raw/`, which only `scripts/run_experiment.py` writes. So to move a
*number*, re-run the experiment (or the notebook cells that regenerate figures)
-- the watcher picks up either automatically. This is deliberate: the page
should never show a number that no run produced.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_showcase as builder                       # noqa: E402


def snapshot(paths) -> dict:
    """(mtime, size) for every watched file. Directories are walked one level."""
    state: dict = {}
    for path in paths:
        if os.path.isdir(path):
            for entry in sorted(os.listdir(path)):
                full = os.path.join(path, entry)
                if os.path.isfile(full):
                    try:
                        st = os.stat(full)
                        state[full] = (st.st_mtime, st.st_size)
                    except OSError:
                        pass
        elif os.path.isfile(path):
            try:
                st = os.stat(path)
                state[path] = (st.st_mtime, st.st_size)
            except OSError:
                pass
    return state


def describe(before: dict, after: dict) -> str:
    """Human-readable summary of what moved, for the log line."""
    changed = [p for p in after if p in before and after[p] != before[p]]
    added = [p for p in after if p not in before]
    removed = [p for p in before if p not in after]
    parts = []
    for label, items in (("changed", changed), ("added", added), ("removed", removed)):
        if items:
            names = ", ".join(os.path.relpath(p, ROOT).replace("\\", "/")
                              for p in sorted(items)[:3])
            more = f" (+{len(items) - 3} more)" if len(items) > 3 else ""
            parts.append(f"{label}: {names}{more}")
    return " | ".join(parts) or "no visible change"


def rebuild(reason: str) -> bool:
    stamp = time.strftime("%H:%M:%S")
    try:
        data = builder.gather()
        with open(builder.OUTPUT, "w", encoding="utf-8") as fh:
            fh.write(builder.render(data))
        size = os.path.getsize(builder.OUTPUT) / 1024
        ms = data["numbers"].get("multi_seed")
        headline = (f"coset {ms['coset_mean']:.3f} vs {ms['best_competitor_mean']:.3f}"
                    if ms else
                    f"coset {data['numbers']['reference']['coset_accuracy']:.3f}")
        print(f"[{stamp}] rebuilt showcase.html ({size:.0f} KB) — {headline}")
        print(f"          {reason}")
        return True
    except Exception as exc:                            # keep the loop alive
        print(f"[{stamp}] BUILD FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=2)
        print(f"          {reason}")
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild showcase.html on change.")
    ap.add_argument("--interval", type=float, default=2.0,
                    help="seconds between polls (default 2)")
    ap.add_argument("--once", action="store_true", help="build once and exit")
    ap.add_argument("--open", action="store_true",
                    help="open the page in a browser after the first build")
    args = ap.parse_args(argv)

    paths = builder.WATCH_PATHS
    print("watching:")
    for p in paths:
        rel = os.path.relpath(p, ROOT).replace("\\", "/")
        kind = "dir " if os.path.isdir(p) else "file"
        exists = "" if os.path.exists(p) else "   [missing]"
        print(f"  {kind}  {rel}{exists}")
    print()

    ok = rebuild("initial build")
    if args.open and ok:
        import webbrowser
        webbrowser.open("file://" + builder.OUTPUT.replace("\\", "/"))
    if args.once:
        return 0 if ok else 1

    state = snapshot(paths)
    print(f"\npolling every {args.interval:g}s — edit the notebook or re-run the "
          f"experiment and this rebuilds. Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(args.interval)
            new = snapshot(paths)
            if new != state:
                # Small settle delay: editors and Jupyter write in several steps,
                # and rebuilding mid-write reads a truncated file.
                time.sleep(0.4)
                new = snapshot(paths)
                rebuild(describe(state, new))
                state = new
    except KeyboardInterrupt:
        print("\nstopped.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
