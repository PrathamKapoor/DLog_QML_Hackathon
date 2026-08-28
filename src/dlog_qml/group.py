"""
Number-theoretic primitives for the multiplicative group Z_p^*.

Everything here is exact integer arithmetic on Python ints, so the module is
already arbitrary-precision and needs no modification to run at cryptographic
parameter sizes. The only component that does not scale is the optional
`exp_table` / `log_table` pair, which costs O(p) memory and exists purely so the
*data generator* (never a learner) can invert the discrete logarithm at
simulation scale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

#: Deterministic Miller-Rabin witness set. Testing these twelve bases is an
#: exact primality proof for every n < 3,317,044,064,679,887,385,961,981
#: (~3.3e24). Beyond that bound the test degrades to probabilistic, which is
#: why `is_probable_prime` is named as it is.
MILLER_RABIN_WITNESSES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

#: Above this value the deterministic witness set is no longer a proof.
DETERMINISTIC_BOUND = 3_317_044_064_679_887_385_961_981


def is_probable_prime(n: int) -> bool:
    """
    Miller-Rabin primality test.

    Deterministic (an exact proof) for n < DETERMINISTIC_BOUND; probabilistic
    with the same fixed bases above it.
    """
    if n < 2:
        return False
    for small in MILLER_RABIN_WITNESSES:
        if n == small:
            return True
        if n % small == 0:
            return False

    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for a in MILLER_RABIN_WITNESSES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def is_safe_prime(p: int) -> bool:
    """A safe prime is p = 2q + 1 with q ('Sophie Germain prime') also prime."""
    return p > 4 and p % 2 == 1 and is_probable_prime(p) and is_probable_prime((p - 1) // 2)


def largest_safe_prime_below(bound: int) -> int:
    """Largest safe prime strictly below `bound`."""
    candidate = bound - 1 if bound % 2 == 0 else bound - 2
    while candidate >= 5:
        if is_safe_prime(candidate):
            return candidate
        candidate -= 2
    raise ValueError(f"No safe prime below {bound}.")


def minimum_qubits_for(per_class: int, max_width: int = 64) -> Optional[int]:
    """
    Smallest register width n such that the largest safe prime below 2^n admits
    `per_class` distinct exponents in each label class.

    ceil(log2(2 * per_class)) is NOT a valid answer: the largest safe prime
    below 2^n can sit well under 2^n (n = 8 gives p = 227, q = 113, not 127).
    """
    for width in range(5, max_width + 1):
        if (largest_safe_prime_below(2 ** width) - 1) // 2 >= per_class:
            return width
    return None


def find_generator(p: int, q: Optional[int] = None) -> int:
    """
    Smallest generator of the FULL group Z_p^*, i.e. an element of order p-1.

    For a safe prime p = 2q + 1 the order of any element divides |Z_p^*| = 2q,
    whose only divisors are {1, 2, q, 2q}. Rejecting g^2 == 1 (orders 1 and 2)
    and g^q == 1 (order q) therefore leaves exactly order 2q. This shortcut is
    valid ONLY for safe primes.
    """
    if q is None:
        q = (p - 1) // 2
    if not is_safe_prime(p):
        raise ValueError(f"p = {p} is not a safe prime; the order shortcut is invalid.")
    for candidate in range(2, p):
        if pow(candidate, 2, p) != 1 and pow(candidate, q, p) != 1:
            return candidate
    raise RuntimeError(f"No generator found for p = {p}.")


def baby_step_giant_step(g: int, h: int, p: int) -> int:
    """
    Reference discrete-log solver in O(sqrt(p)) time and space.

    Used ONLY to audit generated data and to make the tractability of the
    demonstration instance explicit -- never by any learner. Its sqrt(p) cost is
    exactly why p = 1019 is a *demonstration* and |p| >= 2048 bits is a
    *hardness assumption*.
    """
    order = p - 1
    m = math.isqrt(order) + 1
    table: Dict[int, int] = {}
    e = 1
    for j in range(m):
        table.setdefault(e, j)
        e = (e * g) % p
    factor = pow(g, (order - m) % order, p)
    gamma = h % p
    for i in range(m):
        if gamma in table:
            return (i * m + table[gamma]) % order
        gamma = (gamma * factor) % p
    raise ValueError(f"Discrete logarithm of {h} not found (outside <g>).")


@dataclass
class DlogGroup:
    """
    A concrete cyclic group Z_p^* with a chosen generator.

    Attributes
    ----------
    p, q : safe prime and its Sophie Germain prime, p = 2q + 1.
    g     : generator of the full group, of order `order` = p - 1 = 2q.
    n_bits: fixed feature width, ceil(log2 p) == p.bit_length() for our primes.
            Every group element lies in [1, p-1] and so fits in n_bits bits.
    """

    p: int
    g: int
    n_bits: int = field(init=False)
    q: int = field(init=False)
    order: int = field(init=False)
    exp_table: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    log_table: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not is_safe_prime(self.p):
            raise ValueError(f"p = {self.p} is not a safe prime.")
        self.q = (self.p - 1) // 2
        self.order = self.p - 1
        self.n_bits = self.p.bit_length()
        if pow(self.g, 2, self.p) == 1 or pow(self.g, self.q, self.p) == 1:
            raise ValueError(f"g = {self.g} does not generate Z_{self.p}^*.")

    # ------------------------------------------------------------------ build
    @classmethod
    def from_qubits(cls, n_qubits: int) -> "DlogGroup":
        """Largest safe prime representable in `n_qubits` bits."""
        if n_qubits < 5:
            raise ValueError("n_qubits must be >= 5 (smallest usable safe prime is 11).")
        p = largest_safe_prime_below(2 ** n_qubits)
        return cls(p=p, g=find_generator(p))

    def build_tables(self) -> "DlogGroup":
        """
        Materialise exp/log tables. O(p) time and memory -- SIMULATION ONLY.

        This is the trapdoor held by the data generator. At cryptographic scale
        it must be dropped entirely and data generated forward-only (sample x,
        compute h = g^x mod p, label from x).
        """
        if self.exp_table is not None:
            return self
        exp_table = np.zeros(self.order, dtype=np.int64)
        log_table = np.full(self.p, -1, dtype=np.int64)
        value = 1
        for i in range(self.order):
            exp_table[i] = value
            log_table[value] = i
            value = (value * self.g) % self.p
        if value != 1:
            raise RuntimeError("Generator failed its order check; tables invalid.")
        self.exp_table, self.log_table = exp_table, log_table
        return self

    # ------------------------------------------------------------- operations
    def power(self, x: int) -> int:
        """g^x mod p -- the easy direction."""
        return pow(self.g, int(x) % self.order, self.p)

    def dlog(self, h: int) -> int:
        """
        log_g(h) -- the hard direction.

        Uses the precomputed table when available, else baby-step giant-step.
        Either way this is ORACLE machinery, never exposed to a learner.
        """
        h = int(h)
        if not 1 <= h < self.p:
            raise ValueError(f"{h} is not in Z_{self.p}^*.")
        if self.log_table is not None:
            value = int(self.log_table[h])
            if value < 0:
                raise ValueError(f"{h} is not in <g>.")
            return value
        return baby_step_giant_step(self.g, h, self.p)

    def to_bits(self, h: int) -> np.ndarray:
        """
        Fixed-width binary decomposition of a group element, LSB-FIRST.

        Index j holds the coefficient of 2^j, so bits[0] is the ones bit. This
        ordering is asserted in tests/test_group.py and stated in the README;
        the reverse convention would silently permute every feature column.
        """
        h = int(h)
        return np.array([(h >> j) & 1 for j in range(self.n_bits)], dtype=np.int8)

    def from_bits(self, bits) -> int:
        """Inverse of `to_bits` (LSB-first)."""
        return int(sum(int(b) << j for j, b in enumerate(bits)))

    def describe(self) -> Dict[str, int]:
        return {"p": self.p, "q": self.q, "g": self.g,
                "order": self.order, "n_bits": self.n_bits}
