"""
Dataset integrity and -- most importantly -- leakage tests.

The single failure mode that would invalidate this entire project is the hidden
exponent reaching a learner. These tests attack that directly.
"""

import numpy as np
import pytest

from dlog_qml.classical import assert_no_leakage
from dlog_qml.config import ExperimentConfig
from dlog_qml.data import generate_dataset, max_balanced_samples, max_balanced_split
from dlog_qml.group import DlogGroup


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(n_qubits=10, n_train=200, n_test=50, seed=20260828)


# ----------------------------------------------------------------- integrity

def test_self_audit_passes(dataset):
    audit = dataset.verify()
    assert audit["n_samples"] == 250
    assert audit["splits_disjoint"] is True
    assert audit["feature_width"] == 10


def test_modular_arithmetic_holds(dataset):
    g, p = dataset.group.g, dataset.group.p
    for x, h in zip(dataset.oracle_exponents(), dataset.group_elements()):
        assert pow(g, int(x), p) == int(h)


def test_exact_class_balance(dataset):
    for split, expected in (("train", 100), ("test", 25)):
        y = dataset.labels(split)
        assert int((y == 1).sum()) == expected
        assert int((y == -1).sum()) == expected


def test_train_test_exponents_disjoint(dataset):
    train = set(dataset.oracle_exponents("train").tolist())
    test = set(dataset.oracle_exponents("test").tolist())
    assert not (train & test)
    assert len(train) == 200 and len(test) == 50


def test_group_elements_also_disjoint(dataset):
    """Disjoint exponents imply disjoint h, since x -> g^x is a bijection."""
    train = set(dataset.group_elements("train").tolist())
    test = set(dataset.group_elements("test").tolist())
    assert not (train & test)


def test_feature_width_and_binarity(dataset):
    X = dataset.features()
    assert X.shape == (250, dataset.group.n_bits)
    assert set(np.unique(X)).issubset({0.0, 1.0})


def test_features_are_the_bits_of_h(dataset):
    expected = np.stack([dataset.group.to_bits(int(h))
                         for h in dataset.group_elements()]).astype(float)
    assert np.array_equal(dataset.features(), expected)


def test_determinism():
    a = generate_dataset(n_qubits=10, n_train=40, n_test=10, seed=99)
    b = generate_dataset(n_qubits=10, n_train=40, n_test=10, seed=99)
    assert np.array_equal(a.h, b.h)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.oracle_exponents(), b.oracle_exponents())


def test_different_seeds_give_different_data():
    a = generate_dataset(n_qubits=10, n_train=40, n_test=10, seed=1)
    b = generate_dataset(n_qubits=10, n_train=40, n_test=10, seed=2)
    assert not np.array_equal(a.h, b.h)


# ------------------------------------------------------------------- leakage

def test_public_features_do_not_contain_the_exponent(dataset):
    """No feature column may reproduce x, or any low bit of x."""
    X = dataset.features("train")
    x = dataset.oracle_exponents("train")
    for column in range(X.shape[1]):
        col = X[:, column].astype(np.int64)
        assert not np.array_equal(col, x)
        assert not np.array_equal(col, x & 1)
        assert not np.array_equal(col, (x >> 1) & 1)


def test_leakage_guard_accepts_clean_features(dataset):
    assert_no_leakage(dataset, dataset.features("train"), "train")
    assert_no_leakage(dataset, dataset.features("test"), "test")


def test_leakage_guard_rejects_injected_exponent_bit(dataset):
    """Deliberately poison a column and confirm the guard fires."""
    X = dataset.features("train").copy()
    X[:, 0] = (dataset.oracle_exponents("train") & 1).astype(float)
    with pytest.raises(AssertionError):
        assert_no_leakage(dataset, X, "train")


def test_leakage_guard_rejects_wrong_width(dataset):
    with pytest.raises(AssertionError):
        assert_no_leakage(dataset, dataset.features("train")[:, :5], "train")


def test_leakage_guard_rejects_non_binary(dataset):
    X = dataset.features("train").copy()
    X[0, 0] = 7.0
    with pytest.raises(AssertionError):
        assert_no_leakage(dataset, X, "train")


def test_csv_without_oracle_has_no_exponent_column(dataset, tmp_path):
    path = dataset.save_csv(str(tmp_path / "clean.csv"), include_oracle=False)
    import pandas as pd
    frame = pd.read_csv(path)
    assert "raw_input_x" not in frame.columns
    assert "group_element_h" in frame.columns
    assert frame.shape[0] == 250


def test_csv_with_oracle_is_verifiable(dataset, tmp_path):
    """The shipped CSV must let a third party recompute h = g^x mod p."""
    import pandas as pd
    path = dataset.save_csv(str(tmp_path / "full.csv"), include_oracle=True)
    frame = pd.read_csv(path)
    g, p = dataset.group.g, dataset.group.p
    recomputed = frame["raw_input_x"].map(lambda x: pow(g, int(x), p))
    assert (recomputed == frame["group_element_h"]).all()


# ------------------------------------------------------------------ capacity

def test_max_balanced_samples_is_two_q():
    from dlog_qml.labels import build_label_rule
    group = DlogGroup.from_qubits(10).build_tables()
    rule = build_label_rule("half_interval", group, 0)
    assert max_balanced_samples(group, rule) == group.order == 2 * group.q


def test_oversized_request_raises_with_actionable_hint():
    with pytest.raises(ValueError, match="Group too small"):
        generate_dataset(n_qubits=6, n_train=200, n_test=50, seed=0)


def test_odd_sample_counts_rejected():
    with pytest.raises(ValueError):
        generate_dataset(n_qubits=10, n_train=201, n_test=50, seed=0)


@pytest.mark.parametrize("n_qubits", [6, 7, 8, 9, 10])
def test_max_balanced_split_is_feasible(n_qubits):
    n_train, n_test = max_balanced_split(n_qubits)
    assert n_train % 2 == 0 and n_test % 2 == 0
    data = generate_dataset(n_qubits=n_qubits, n_train=n_train,
                            n_test=n_test, seed=3)
    assert len(data.labels("train")) == n_train
    assert len(data.labels("test")) == n_test


# -------------------------------------------------------------------- config

def test_config_round_trip(tmp_path):
    cfg = ExperimentConfig(n_qubits=9, n_train=40, n_test=10)
    path = cfg.save(str(tmp_path / "cfg.json"))
    assert ExperimentConfig.load(path).to_dict() == cfg.to_dict()


def test_config_rejects_unknown_keys(tmp_path):
    import json
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"not_a_real_key": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config keys"):
        ExperimentConfig.load(str(path))


def test_config_validation():
    with pytest.raises(ValueError):
        ExperimentConfig(n_train=201).validate()
    with pytest.raises(ValueError):
        ExperimentConfig(labeling="nope").validate()
