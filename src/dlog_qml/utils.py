"""Small shared helpers: JSON serialisation, tables, and terminal formatting."""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars, arrays, and NaN/inf."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            value = float(o)
            return value if math.isfinite(value) else None
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return super().default(o)


def strip_private(obj: Any) -> Any:
    """Recursively drop keys beginning with '_' (in-memory kernels, datasets)."""
    if isinstance(obj, dict):
        return {k: strip_private(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def save_json(payload: Any, path: str) -> str:
    """Write JSON, creating parent directories and stripping private keys."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(strip_private(payload), fh, indent=2, cls=NumpyJSONEncoder)
    return os.path.abspath(path)


def save_markdown_table(rows: Sequence[Dict[str, Any]], path: str,
                        columns: Sequence[str] | None = None,
                        title: str | None = None) -> str:
    """Write a list of dicts as a GitHub-flavoured Markdown table."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    if not rows:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write((f"# {title}\n\n" if title else "") + "_No rows._\n")
        return os.path.abspath(path)

    columns = list(columns or rows[0].keys())
    lines: List[str] = []
    if title:
        lines += [f"# {title}", ""]
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * len(columns)) + "|")
    for row in rows:
        cells = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                cells.append("n/a" if not math.isfinite(value) else f"{value:.4f}")
            elif isinstance(value, (list, tuple)):
                cells.append("[" + ", ".join(
                    f"{v:.3f}" if isinstance(v, float) else str(v) for v in value) + "]")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return os.path.abspath(path)


def save_csv_table(rows: Sequence[Dict[str, Any]], path: str) -> str:
    """Write a list of dicts as CSV via pandas."""
    import pandas as pd
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Terminal formatting (ASCII only -- no colour dependency, no Unicode boxes,
# so the demo renders identically in every console including legacy Windows)
# ---------------------------------------------------------------------------

WIDTH = 74


def rule(char: str = "=") -> str:
    return char * WIDTH


def header(title: str, subtitle: str = "") -> str:
    lines = [rule("="), title.center(WIDTH)]
    if subtitle:
        lines.append(subtitle.center(WIDTH))
    lines.append(rule("="))
    return "\n".join(lines)


def section(title: str) -> str:
    return f"\n{title}\n{rule('-')}"


def format_p(p_value: float) -> str:
    """Compact p-value rendering that never loses the exponent."""
    if p_value is None or (isinstance(p_value, float) and math.isnan(p_value)):
        return "n/a"
    if p_value < 1e-4:
        return f"{p_value:.1e}"
    return f"{p_value:.4f}"


def accuracy_row(name: str, summary: Dict[str, Any], width: int = 30) -> str:
    """One formatted line: name, accuracy, 95% CI, p-value, verdict."""
    accuracy = summary["accuracy"]
    ci = f"[{summary['ci_low']:.3f}, {summary['ci_high']:.3f}]"
    verdict = "LEARNS" if summary.get("significant_at_1pct") else "chance (n.s.)"
    return (f"  {name:<{width}}{accuracy:>8.3f}  {ci:>16}  "
            f"{format_p(summary['p_value']):>9}   {verdict}")


def bar(value: float, width: int = 28, vmax: float = 1.0) -> str:
    """ASCII bar for the terminal demo."""
    filled = int(round(width * min(max(value, 0.0), vmax) / vmax))
    return "#" * filled + "." * (width - filled)
