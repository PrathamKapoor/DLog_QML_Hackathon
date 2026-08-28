"""
Dataset generation with a hard separation between what a learner may see and
what only the oracle may see.

The central invariant of this repository:

    PUBLIC   -> the LSB-first bit decomposition of h = g^x mod p
    ORACLE   -> the exponent x = log_g(h), the secret shift s, the label rule

`DlogDataset.features()` returns only the public view. The oracle exponents are
reachable exclusively through `oracle_exponents()`, whose name is deliberately
awkward and whose every call site is audited in tests/test_dataset.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .group import DlogGroup, baby_step_giant_step, minimum_qubits_for
from .labels import build_label_rule


@dataclass
class DlogDataset:
    """
    A generated instance of the concept class.

    Arrays are aligned row-wise; `split_mask` selects train vs test.
    """

    group: DlogGroup
    label_rule: object
    seed: int

    h: np.ndarray = field(repr=False)             # group elements (public)
    bits: np.ndarray = field(repr=False)          # (N, n_bits) LSB-first (public)
    y: np.ndarray = field(repr=False)             # labels in {-1,+1} (public)
    split: np.ndarray = field(repr=False)         # 'train' / 'test' (public)
    _x: np.ndarray = field(repr=False)            # hidden exponents (ORACLE ONLY)

    # ---------------------------------------------------------------- public
    @property
    def feature_names(self) -> List[str]:
        return [f"binary_feature_{j}" for j in range(self.group.n_bits)]

    @property
    def train_mask(self) -> np.ndarray:
        return self.split == "train"

    @property
    def test_mask(self) -> np.ndarray:
        return self.split == "test"

    def features(self, split: Optional[str] = None) -> np.ndarray:
        """
        THE ONLY legitimate model input: the bit decomposition of h.

        Returns float64 because every sklearn estimator here expects it.
        """
        mask = slice(None) if split is None else (self.split == split)
        return self.bits[mask].astype(np.float64)

    def labels(self, split: Optional[str] = None) -> np.ndarray:
        mask = slice(None) if split is None else (self.split == split)
        return self.y[mask].astype(np.int64)

    def group_elements(self, split: Optional[str] = None) -> np.ndarray:
        """Raw h values. Public -- the bits are just a re-encoding of these."""
        mask = slice(None) if split is None else (self.split == split)
        return self.h[mask]

    # ---------------------------------------------------------------- oracle
    def oracle_exponents(self, split: Optional[str] = None) -> np.ndarray:
        """
        THE HIDDEN COORDINATE x = log_g(h). NOT a feature.

        Legitimate callers, and only these:
          * `verify()` in this module (dataset self-audit)
          * the hidden-x leakage control in classical.py
          * plotting.py, for the concept-geometry figure
        Any other use is a leak and will fail tests/test_dataset.py.
        """
        mask = slice(None) if split is None else (self.split == split)
        return self._x[mask]

    # ----------------------------------------------------------------- audit
    def verify(self, bsgs_probes: int = 3) -> Dict[str, object]:
        """
        Full self-audit. Raises AssertionError on any violation.

        Checks the modular arithmetic, the label rule, the bit encoding, exact
        class balance per split, and -- most importantly -- that no exponent
        appears in both splits.
        """
        g, p = self.group.g, self.group.p

        # 1. h really is g^x mod p, recomputed from scratch.
        recomputed = np.array([pow(g, int(x), p) for x in self._x], dtype=np.int64)
        assert np.array_equal(recomputed, self.h), "h != g^x mod p"

        # 2. Labels really are the concept function of the hidden exponent.
        expected = self.label_rule.vectorised(self._x)
        assert np.array_equal(expected.astype(np.int64), self.y.astype(np.int64)), \
            "labels do not match the label rule"

        # 3. Bit encoding round-trips (LSB-first).
        for idx in range(len(self.h)):
            assert self.group.from_bits(self.bits[idx]) == int(self.h[idx]), \
                f"bit decomposition wrong at row {idx}"
        assert self.bits.shape[1] == self.group.n_bits, "wrong feature width"
        assert set(np.unique(self.bits)).issubset({0, 1}), "features are not binary"

        # 4. Exact class balance per split.
        balance = {}
        for name in ("train", "test"):
            ys = self.y[self.split == name]
            pos, neg = int((ys == 1).sum()), int((ys == -1).sum())
            assert pos == neg, f"{name} split unbalanced: {pos}/+1 vs {neg}/-1"
            balance[name] = {"positive": pos, "negative": neg}

        # 5. No exponent shared between splits -- the anti-memorisation guard.
        train_x = set(self._x[self.train_mask].tolist())
        test_x = set(self._x[self.test_mask].tolist())
        assert not (train_x & test_x), "train/test exponent overlap (leakage)"
        assert len(train_x) == int(self.train_mask.sum()), "duplicate train exponents"
        assert len(test_x) == int(self.test_mask.sum()), "duplicate test exponents"

        # 6. Independent discrete-log solve on a few probes, via BSGS rather
        #    than the oracle table, so a corrupt table cannot pass silently.
        probes = []
        for idx in range(min(bsgs_probes, len(self.h))):
            solved = baby_step_giant_step(g, int(self.h[idx]), p)
            assert solved == int(self._x[idx]), f"BSGS disagrees at row {idx}"
            probes.append(solved)

        return {
            "n_samples": int(len(self.h)),
            "balance": balance,
            "splits_disjoint": True,
            "bsgs_probes": probes,
            "feature_width": int(self.bits.shape[1]),
        }

    # ------------------------------------------------------------------ i/o
    def to_dataframe(self, include_oracle: bool = True) -> pd.DataFrame:
        """
        Tabular view.

        `include_oracle=True` adds `raw_input_x`, which is the TRAPDOOR, not a
        feature. It is shipped so a judge can independently recompute every
        label and verify h = g^x mod p. Any model trained on it scores ~100%.
        Use `include_oracle=False` for a leak-proof CSV.
        """
        data: Dict[str, np.ndarray] = {"sample_id": np.arange(len(self.h))}
        if include_oracle:
            data["raw_input_x"] = self._x
        data["group_element_h"] = self.h
        for j, name in enumerate(self.feature_names):
            data[name] = self.bits[:, j]
        data["label"] = self.y
        data["split"] = self.split
        return pd.DataFrame(data)

    def save_csv(self, path: str, include_oracle: bool = True) -> str:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.to_dataframe(include_oracle=include_oracle).to_csv(path, index=False)
        return os.path.abspath(path)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def max_balanced_samples(group: DlogGroup, label_rule) -> int:
    """
    Largest total sample count admitting an exact 50/50 split without replacement.

    For the half-interval rule each class holds exactly q exponents, so the cap
    is 2q = p - 1. Computed empirically so it stays correct for any label rule.
    """
    labels = label_rule.vectorised(np.arange(group.order))
    return 2 * int(min((labels == 1).sum(), (labels == -1).sum()))


def generate_dataset(
    n_qubits: int = 10,
    n_train: int = 200,
    n_test: int = 50,
    seed: int = 20260828,
    labeling: str = "half_interval",
    group: Optional[DlogGroup] = None,
) -> DlogDataset:
    """
    Build an exactly balanced, split-disjoint dataset.

    Sampling is WITHOUT REPLACEMENT over Z_{p-1}: a uniform permutation of all
    exponents is drawn, then filled into per-class buckets until each is full.
    Because no exponent is reused, no group element can appear in both splits,
    so a reported train/test gap can never be memorisation.
    """
    if n_train % 2 or n_test % 2:
        raise ValueError("n_train and n_test must be even (exact 50/50 balance)")
    if n_train <= 0 or n_test <= 0:
        raise ValueError("n_train and n_test must be positive")

    group = group or DlogGroup.from_qubits(n_qubits)
    group.build_tables()
    label_rule = build_label_rule(labeling, group, seed)

    per_class_total = (n_train + n_test) // 2
    if 2 * per_class_total > max_balanced_samples(group, label_rule):
        need = minimum_qubits_for(per_class_total)
        hint = f"raise n_qubits to >= {need}" if need else "raise n_qubits"
        raise ValueError(
            f"Group too small: need {per_class_total} samples per class but "
            f"p = {group.p} admits only {max_balanced_samples(group, label_rule) // 2}.\n"
            f"  Fix A: {hint}.\n"
            f"  Fix B: lower n_train/n_test so (n_train+n_test)/2 <= "
            f"{max_balanced_samples(group, label_rule) // 2}."
        )

    rng = np.random.default_rng(seed)
    permuted = rng.permutation(group.order)
    all_labels = label_rule.vectorised(permuted)

    pos = permuted[all_labels == 1][:per_class_total]
    neg = permuted[all_labels == -1][:per_class_total]
    if len(pos) < per_class_total or len(neg) < per_class_total:
        raise RuntimeError("Balanced sampling failed; group exhausted.")

    n_train_pc = n_train // 2
    train_x = np.concatenate([pos[:n_train_pc], neg[:n_train_pc]])
    test_x = np.concatenate([pos[n_train_pc:], neg[n_train_pc:]])
    rng.shuffle(train_x)
    rng.shuffle(test_x)

    x = np.concatenate([train_x, test_x]).astype(np.int64)
    split = np.array(["train"] * len(train_x) + ["test"] * len(test_x))
    h = np.array([group.power(int(v)) for v in x], dtype=np.int64)
    bits = np.stack([group.to_bits(int(v)) for v in h])
    y = label_rule.vectorised(x)

    dataset = DlogDataset(group=group, label_rule=label_rule, seed=seed,
                          h=h, bits=bits, y=y, split=split, _x=x)
    dataset.verify()
    return dataset


def max_balanced_split(n_qubits: int, cap_train: int = 200,
                       cap_test: int = 50) -> Tuple[int, int]:
    """
    Largest even (n_train, n_test) feasible at a given register width.

    Used by the measured size sweep so small groups get the biggest honest
    dataset they can support instead of failing or silently resampling.
    """
    group = DlogGroup.from_qubits(n_qubits)
    total = min(group.order, cap_train + cap_test)
    total -= total % 2
    n_test = min(cap_test, total // 5)
    n_test -= n_test % 2
    n_test = max(n_test, 2)
    n_train = total - n_test
    n_train -= n_train % 2
    return int(n_train), int(n_test)
