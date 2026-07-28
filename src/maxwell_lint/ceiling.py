"""The screening ceiling, and the error law that follows from it.

Physics
-------
For a conductor pair (i, j) embedded in an array, define

    k_ij = |C_full,ij| / |C_iso,ij|

the ratio of the pair's mutual capacitance with the whole array present to its
value with only the pair present. Every other conductor in the array *screens*
the field between i and j, so

    0 < k_ij <= 1                                      (the ceiling)

Screening can only reduce coupling below the isolated-pair value. A predicted
k > 1 is anti-screening: it says a third grounded conductor placed between two
others *increases* their coupling, which no arrangement of passive conductors
in a linear medium can do.

The error law
-------------
A pairwise-superposition extractor -- the method inside essentially every fast
quasi-static parasitic extractor -- assumes k == 1, i.e. it reports the
isolated-pair value. Writing the screening depth as

    delta = -log10(k)   >= 0

its relative error is a function of depth alone:

    E(delta) = |C_iso - C_full| / C_full = |1/k - 1| = 10**delta - 1

which is zero at delta = 0, non-negative everywhere, and **strictly
increasing**. Three consequences, each stronger than a measured correlation:

  * pairwise superposition is exact if and only if there is no screening;
  * its error is one-sided -- it never under-predicts a screened pair;
  * there is no depth beyond which it stops getting worse.

This module implements the ceiling test and the error law. It does not ship a
coupling extractor -- you bring your own, and this tells you whether it is
predicting possible physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CeilingReport",
    "screening_factor",
    "screening_depth",
    "pairwise_error",
    "check_ceiling",
]


def screening_factor(c_full: np.ndarray, c_iso: np.ndarray) -> np.ndarray:
    """k = |C_full| / |C_iso|, elementwise, off-diagonal entries only.

    Diagonal entries are returned as NaN: the self term is not a screened
    coupling and the ceiling does not apply to it.
    """
    c_full = np.asarray(c_full, dtype=float)
    c_iso = np.asarray(c_iso, dtype=float)
    if c_full.shape != c_iso.shape:
        raise ValueError(f"shape mismatch: {c_full.shape} vs {c_iso.shape}")
    if c_full.ndim != 2 or c_full.shape[0] != c_full.shape[1]:
        raise ValueError(f"expected a square matrix, got {c_full.shape}")

    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.abs(c_full) / np.abs(c_iso)
    np.fill_diagonal(k, np.nan)
    return k


def screening_depth(k: np.ndarray) -> np.ndarray:
    """delta = -log10(k). Deeper screening is a larger delta."""
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log10(k)


def pairwise_error(delta: np.ndarray | float) -> np.ndarray | float:
    """E(delta) = 10**delta - 1: the relative error of assuming k == 1.

    Strictly increasing in delta, zero only at delta == 0, non-negative for
    delta >= 0.
    """
    return np.power(10.0, delta) - 1.0


@dataclass
class CeilingReport:
    n_pairs: int
    n_violations: int
    violation_fraction: float
    max_k: float
    worst_pair: tuple[int, int] | None
    median_depth: float
    median_pairwise_error: float
    passed: bool
    tol: float
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_pairs": self.n_pairs,
            "n_violations": self.n_violations,
            "violation_fraction": self.violation_fraction,
            "max_k": self.max_k,
            "worst_pair": list(self.worst_pair) if self.worst_pair else None,
            "median_screening_depth": self.median_depth,
            "median_pairwise_superposition_error": self.median_pairwise_error,
            "passed": self.passed,
            "tolerance": self.tol,
            **({"detail": self.detail} if self.detail else {}),
        }

    def summary(self) -> str:
        if self.passed:
            return (
                f"{self.n_pairs} pairs, 0 ceiling violations, max k = {self.max_k:.6f}. "
                "Predictions are physically admissible."
            )
        pct = 100.0 * self.violation_fraction
        wp = self.worst_pair
        return (
            f"{self.n_violations} of {self.n_pairs} pairs ({pct:.1f}%) predict k > 1 -- "
            f"anti-screening, which no passive arrangement can produce. "
            f"Worst k = {self.max_k:.4f}"
            + (f" at pair ({wp[0]}, {wp[1]})." if wp else ".")
        )


def check_ceiling(
    c_full: np.ndarray,
    c_iso: np.ndarray,
    tol: float = 1e-9,
) -> CeilingReport:
    """Test an extractor's predictions against the screening ceiling k <= 1.

    Parameters
    ----------
    c_full : (N, N) predicted full-array coupling matrix
    c_iso  : (N, N) isolated-pair baseline for the same geometry
    tol    : floating-point slack; k <= 1 + tol passes
    """
    k = screening_factor(c_full, c_iso)
    off = ~np.eye(k.shape[0], dtype=bool)
    vals = k[off]
    finite = np.isfinite(vals)
    n_nonfinite = int(np.count_nonzero(~finite))
    vals = vals[finite]

    if vals.size == 0:
        return CeilingReport(0, 0, 0.0, float("nan"), None, float("nan"),
                             float("nan"), False, tol,
                             detail={"reason": "no finite off-diagonal pairs"})

    viol = vals > 1.0 + tol
    n_viol = int(np.count_nonzero(viol))
    max_k = float(np.max(vals))

    worst = None
    kk = np.where(np.isfinite(k), k, -np.inf)
    idx = np.unravel_index(int(np.argmax(kk)), kk.shape)
    if np.isfinite(k[idx]):
        worst = (int(idx[0]), int(idx[1]))

    depth = screening_depth(vals)
    med_depth = float(np.median(depth))
    med_err = float(np.median(pairwise_error(depth)))

    detail = {}
    if n_nonfinite:
        detail["nonfinite_pairs_skipped"] = n_nonfinite

    return CeilingReport(
        n_pairs=int(vals.size),
        n_violations=n_viol,
        violation_fraction=float(n_viol / vals.size),
        max_k=max_k,
        worst_pair=worst,
        median_depth=med_depth,
        median_pairwise_error=med_err,
        passed=(n_viol == 0),
        tol=tol,
        detail=detail,
    )
