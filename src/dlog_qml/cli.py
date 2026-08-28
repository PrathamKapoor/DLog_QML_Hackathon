"""
Console-script entry points declared in pyproject.toml.

    dlog-qml-demo        -> scripts/demo.py
    dlog-qml-experiment  -> scripts/run_experiment.py

These import the script modules by path so there is exactly one implementation
of each command rather than a copy that can drift.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from typing import Optional, Sequence

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "scripts",
)


def _load(module_name: str, filename: str):
    path = os.path.join(_SCRIPTS, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cannot find {path}. The console scripts run from a source "
            f"checkout; use `python scripts/{filename}` instead."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def demo_main(argv: Optional[Sequence[str]] = None) -> int:
    return _load("_dlog_qml_demo", "demo.py").main(argv)


def experiment_main(argv: Optional[Sequence[str]] = None) -> int:
    return _load("_dlog_qml_experiment", "run_experiment.py").main(argv)
