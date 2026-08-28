"""Tests for the number-theoretic layer."""

import numpy as np
import pytest

from dlog_qml.group import (
    DlogGroup,
    baby_step_giant_step,
    find_generator,
    is_probable_prime,
    is_safe_prime,
    largest_safe_prime_below,
    minimum_qubits_for,
)


# ---------------------------------------------------------------- primality

@pytest.mark.parametrize("n", [2, 3, 5, 7, 11, 13, 101, 227, 509, 1019, 65537])
def test_known_primes(n):
    assert is_probable_prime(n)


@pytest.mark.parametrize("n", [0, 1, 4, 9, 15, 21, 100, 561, 1024, 1105, 1729])
def test_known_composites(n):
    # 561, 1105 and 1729 are Carmichael numbers: they defeat a naive Fermat
    # test, so this is a real check that Miller-Rabin is implemented correctly.
    assert not is_probable_prime(n)


def test_primality_matches_trial_division():
    for n in range(2, 2000):
        expected = all(n % d for d in range(2, int(n ** 0.5) + 1))
        assert is_probable_prime(n) == expected, n


# -------------------------------------------------------------- safe primes

@pytest.mark.parametrize("p", [11, 23, 47, 59, 107, 227, 503, 1019])
def test_known_safe_primes(p):
    assert is_safe_prime(p)
    assert is_probable_prime((p - 1) // 2)


@pytest.mark.parametrize("p", [13, 17, 31, 61, 127, 251, 1021])
def test_non_safe_primes(p):
    # Each is prime but (p-1)/2 is not, so none is a safe prime. 1021 in
    # particular is a tempting near-miss below 2^10; the correct answer is 1019.
    assert is_probable_prime(p)
    assert not is_safe_prime(p)


@pytest.mark.parametrize("n_qubits,expected", [(6, 59), (7, 107), (8, 227),
                                               (9, 503), (10, 1019)])
def test_largest_safe_prime_below_power_of_two(n_qubits, expected):
    assert largest_safe_prime_below(2 ** n_qubits) == expected


def test_minimum_qubits_accounts_for_prime_gaps():
    # Naive ceil(log2(2 * 125)) = 8, but p = 227 gives q = 113 < 125.
    # The correct answer is 9 (p = 503, q = 251).
    assert minimum_qubits_for(125) == 9
    assert (largest_safe_prime_below(2 ** 8) - 1) // 2 < 125
    assert (largest_safe_prime_below(2 ** 9) - 1) // 2 >= 125


# --------------------------------------------------------------- generators

@pytest.mark.parametrize("n_qubits", [6, 7, 8, 9, 10])
def test_generator_has_full_order(n_qubits):
    p = largest_safe_prime_below(2 ** n_qubits)
    g = find_generator(p)
    q = (p - 1) // 2
    assert pow(g, p - 1, p) == 1           # Fermat
    assert pow(g, 2, p) != 1               # order is not 1 or 2
    assert pow(g, q, p) != 1               # order is not q
    # Brute force: the orbit of g must be all of Z_p^*.
    orbit = set()
    value = 1
    for _ in range(p - 1):
        orbit.add(value)
        value = (value * g) % p
    assert value == 1
    assert orbit == set(range(1, p))


def test_find_generator_rejects_non_safe_prime():
    with pytest.raises(ValueError):
        find_generator(1021)               # prime, but not a safe prime


# ------------------------------------------------------------ group object

def test_group_tables_round_trip():
    group = DlogGroup.from_qubits(8).build_tables()
    for x in range(group.order):
        h = group.power(x)
        assert group.dlog(h) == x
    assert len(set(group.exp_table.tolist())) == group.order


def test_group_rejects_bad_generator():
    with pytest.raises(ValueError):
        DlogGroup(p=1019, g=1018)          # 1018 = -1 mod 1019, order 2


def test_bit_encoding_is_lsb_first():
    group = DlogGroup.from_qubits(10)
    bits = group.to_bits(9)                # 9 = 0b1001
    assert bits[0] == 1 and bits[3] == 1
    assert bits[1] == 0 and bits[2] == 0
    assert len(bits) == group.n_bits == 10


def test_bit_round_trip_over_whole_group():
    group = DlogGroup.from_qubits(9)
    for h in range(1, group.p):
        assert group.from_bits(group.to_bits(h)) == h


def test_bit_width_covers_every_element():
    for n_qubits in (6, 7, 8, 9, 10):
        group = DlogGroup.from_qubits(n_qubits)
        assert group.p - 1 < 2 ** group.n_bits
        assert group.n_bits == n_qubits


# ------------------------------------------------------------------- BSGS

@pytest.mark.parametrize("n_qubits", [6, 8, 10])
def test_bsgs_agrees_with_table(n_qubits):
    group = DlogGroup.from_qubits(n_qubits).build_tables()
    rng = np.random.default_rng(0)
    for x in rng.integers(0, group.order, size=15):
        h = group.power(int(x))
        assert baby_step_giant_step(group.g, h, group.p) == int(x)


def test_bsgs_rejects_out_of_group_element():
    group = DlogGroup.from_qubits(6)
    with pytest.raises(ValueError):
        baby_step_giant_step(group.g, 0, group.p)   # 0 is not in Z_p^*
