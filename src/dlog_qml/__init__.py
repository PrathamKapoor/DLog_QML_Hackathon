"""
dlog_qml -- Quantum learning on a cryptographically hidden coordinate system.

A reproducible, falsifiable study of the discrete-logarithm concept class of
Liu, Arunachalam & Temme (Nature Physics 17, 1013, 2021).

Central claim, stated at the altitude the evidence supports:

    A classification concept that is trivial in discrete-logarithm coordinates
    becomes unlearnable by conventional methods when the learner receives only
    the corresponding group element. A quantum feature map whose inductive bias
    matches the algebraic structure recovers the concept; a generic quantum
    feature map does not.

Quantum advantage is NOT automatic. See docs/LIMITATIONS.md before quoting any
number from this package.
"""

from .config import ExperimentConfig
from .data import DlogDataset, generate_dataset
from .group import DlogGroup, baby_step_giant_step, find_generator, is_safe_prime
from .labels import HalfIntervalLabel, ParityLabel, derive_secret, legendre_predict
from .quantum_coset import CosetStateFeatureMap
from .quantum_generic import LadderFeatureMap
from .statistics import AccuracySummary, exact_binomial_p, mcnemar_exact, wilson_interval

__version__ = "1.0.0"

__all__ = [
    "ExperimentConfig",
    "DlogDataset",
    "generate_dataset",
    "DlogGroup",
    "baby_step_giant_step",
    "find_generator",
    "is_safe_prime",
    "HalfIntervalLabel",
    "ParityLabel",
    "derive_secret",
    "legendre_predict",
    "CosetStateFeatureMap",
    "LadderFeatureMap",
    "AccuracySummary",
    "exact_binomial_p",
    "mcnemar_exact",
    "wilson_interval",
    "__version__",
]
