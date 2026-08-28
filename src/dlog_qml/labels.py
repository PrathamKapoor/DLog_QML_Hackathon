"""
Labelling rules for the discrete-logarithm concept class.

Two rules are implemented and the difference between them is the scientific
crux of this repository:

  HalfIntervalLabel  -- the Liu/Arunachalam/Temme concept. No polynomial-time
                        classical distinguisher is known under the Discrete
                        Logarithm Assumption.

  ParityLabel        -- y = (-1)^(log_g h mod 2). BROKEN, and included only as
                        a falsification control. The least significant bit of a
                        discrete logarithm in Z_p^* is *not* a hard bit: it is
                        the quadratic-residuosity bit, recoverable in O(log p)
                        modular multiplications by Euler's criterion. See
                        `legendre_predict` below and docs/LIMITATIONS.md.

The hardcore bits of the discrete logarithm are the high-order ones
(Blum-Micali); the low-order bit is free. Any construction that labels by dlog
parity therefore has no trapdoor at all.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .group import DlogGroup


def derive_secret(seed: int, p: int, g: int) -> int:
    """
    Deterministic secret shift s in Z_{p-1}, derived from the run seed.

    The byte->integer conversion is made explicit here because it is a silent
    reproducibility hazard: we take the full 32-byte SHA-256 digest, interpret
    it as a BIG-ENDIAN unsigned integer, and reduce mod (p-1).

        s = int.from_bytes(SHA256(domain_string), "big") mod (p - 1)

    The domain string binds the secret to the group as well as the seed, so two
    runs with the same seed but different p do not share a secret.
    """
    material = f"dlog-concept-class|p={p}|g={g}|seed={seed}".encode("utf-8")
    digest = hashlib.sha256(material).digest()          # 32 bytes
    return int.from_bytes(digest, "big") % (p - 1)


@dataclass(frozen=True)
class HalfIntervalLabel:
    r"""
    Balanced half-interval concept (Liu/Arunachalam/Temme).

        y(x) = +1  if  (x - s) mod (p-1)  in  [0, (p-3)/2]
               -1  otherwise

    Off-by-one accounting, stated exactly because p-1 is even
    -------------------------------------------------------
    Write N = p - 1 = 2q. The positive interval is the CLOSED integer range
    [0, (p-3)/2]. Since (p-3)/2 = (2q - 2)/2 = q - 1, that range is
    {0, 1, ..., q-1}: exactly q integers. Its complement in Z_N is
    {q, ..., 2q-1}: also exactly q integers. So the concept is *exactly*
    balanced over the full group, not approximately -- which is why the
    generated datasets can hit a perfect 50/50 split by construction.

    Equivalently and more cheaply: y = +1 iff shifted < q. The implementation
    uses the `< q` form; `positive_interval` exposes the closed interval so
    tests can check the two agree at every boundary.
    """

    secret_s: int
    order: int          # N = p - 1
    q: int              # N // 2

    def __post_init__(self) -> None:
        if self.order != 2 * self.q:
            raise ValueError(f"order {self.order} must equal 2q = {2 * self.q}.")
        if not 0 <= self.secret_s < self.order:
            raise ValueError(f"secret_s must lie in [0, {self.order}).")

    @property
    def name(self) -> str:
        return "half_interval"

    @property
    def positive_interval(self) -> tuple:
        """Closed integer interval [lo, hi] of shifted exponents labelled +1."""
        return (0, (self.order - 2) // 2)          # == (0, q - 1)

    def __call__(self, x: int) -> int:
        return 1 if (int(x) - self.secret_s) % self.order < self.q else -1

    def vectorised(self, xs: Sequence[int]) -> np.ndarray:
        shifted = (np.asarray(xs, dtype=np.int64) - self.secret_s) % self.order
        return np.where(shifted < self.q, 1, -1).astype(np.int8)


@dataclass(frozen=True)
class ParityLabel:
    """
    BROKEN control concept: y = +1 iff log_g(h) is even.

    Kept so the falsification suite can demonstrate that "involves a discrete
    logarithm" does NOT imply "classically hard". Never use as the primary task.
    """

    order: int

    @property
    def name(self) -> str:
        return "parity"

    @property
    def positive_interval(self) -> tuple:
        return (0, 0)   # not an interval concept; present for API symmetry

    def __call__(self, x: int) -> int:
        return 1 if int(x) % 2 == 0 else -1

    def vectorised(self, xs: Sequence[int]) -> np.ndarray:
        return np.where(np.asarray(xs, dtype=np.int64) % 2 == 0, 1, -1).astype(np.int8)


def build_label_rule(name: str, group: DlogGroup, seed: int):
    """Factory: 'half_interval' (default, hard) or 'parity' (broken control)."""
    if name == "half_interval":
        return HalfIntervalLabel(
            secret_s=derive_secret(seed, group.p, group.g),
            order=group.order,
            q=group.q,
        )
    if name == "parity":
        return ParityLabel(order=group.order)
    raise ValueError(f"Unknown labelling rule {name!r}.")


# ---------------------------------------------------------------------------
# The number-theoretic attack
# ---------------------------------------------------------------------------

def legendre_predict(group: DlogGroup, h_values: Sequence[int]) -> np.ndarray:
    """
    Predict labels from `h` alone in O(log p) per sample, via Euler's criterion.

        h^((p-1)/2) mod p == 1   <=>   h is a quadratic residue
                                 <=>   log_g(h) is even

    (The forward implication is Euler's criterion; the equivalence with dlog
    parity holds because g generates Z_p^*, so g^x is a QR exactly when x is
    even.)

    This does NOT solve the discrete logarithm -- it extracts one bit of it.
    That single bit is enough to solve the parity concept completely and is
    worthless against the half-interval concept, which is the entire point of
    including it.
    """
    exponent = (group.p - 1) // 2
    return np.array(
        [1 if pow(int(h), exponent, group.p) == 1 else -1 for h in h_values],
        dtype=np.int8,
    )
