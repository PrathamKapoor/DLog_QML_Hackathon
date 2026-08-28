"""
Experiment configuration: a single dataclass, loadable from and dumpable to
YAML/JSON so that every reported number is traceable to an exact config.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs", "default.yaml",
)


@dataclass
class ExperimentConfig:
    """All knobs for one experiment. Defaults reproduce the headline result."""

    # --- group / concept ----------------------------------------------------
    n_qubits: int = 10
    labeling: str = "half_interval"      # {"half_interval", "parity"}
    seed: int = 20260828

    # --- dataset ------------------------------------------------------------
    n_train: int = 200
    n_test: int = 50

    # --- quantum feature maps ----------------------------------------------
    window_bits: Optional[int] = None    # coset window M = 2^m; None -> auto
    ladder_reps: int = 2

    # --- classical baselines ------------------------------------------------
    rf_n_estimators: int = 100
    mlp_hidden: List[int] = field(default_factory=lambda: [64, 32])
    mlp_max_iter: int = 3000
    svm_C_grid: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0, 1000.0])
    svm_gamma_grid: List[Any] = field(default_factory=lambda: ["scale", 0.01, 0.1, 1.0, 10.0])
    qsvm_C_grid: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0, 100.0, 1000.0])
    cv_folds: int = 5
    extra_baselines: bool = True         # ExtraTrees / linear+poly SVM / kNN / GB

    # --- robustness ---------------------------------------------------------
    n_seeds: int = 20                    # multi-seed sweep in run_experiment
    sweep_qubits: List[int] = field(default_factory=lambda: [6, 7, 8, 9, 10])

    # --- output -------------------------------------------------------------
    results_dir: str = "results"
    figures_dir: str = "figures"      # top-level; referenced by README/notebook
    make_plots: bool = True

    # ------------------------------------------------------------------ util
    def validate(self) -> "ExperimentConfig":
        if self.labeling not in ("half_interval", "parity"):
            raise ValueError("labeling must be 'half_interval' or 'parity'")
        if self.n_train % 2 or self.n_test % 2:
            raise ValueError("n_train and n_test must be even (exact 50/50 balance)")
        if self.n_train <= 0 or self.n_test <= 0:
            raise ValueError("n_train and n_test must be positive")
        if self.n_qubits < 5:
            raise ValueError("n_qubits must be >= 5")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be >= 2")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def replace(self, **changes: Any) -> "ExperimentConfig":
        data = self.to_dict()
        data.update(changes)
        return ExperimentConfig(**data).validate()

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return os.path.abspath(path)

    # ---------------------------------------------------------------- loaders
    @classmethod
    def load(cls, path: Optional[str] = None) -> "ExperimentConfig":
        """Load from YAML or JSON. Falls back to defaults when path is absent."""
        path = path or DEFAULT_CONFIG_PATH
        if not os.path.exists(path):
            return cls().validate()
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
                data = yaml.safe_load(text) or {}
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "PyYAML is required to read a .yaml config. "
                    "Install it, or pass a .json config instead."
                ) from exc
        else:
            data = json.loads(text)

        known = {f for f in cls().to_dict()}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data).validate()
