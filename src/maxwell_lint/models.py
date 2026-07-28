"""Reference coupling models, including the ones that fail.

These exist so the ceiling test has something to be run against out of the box,
and so the failure mode is demonstrable rather than asserted.

`monopole_closure` is a zero-parameter analytic multiple-scattering closure: it
represents each conductor by a monopole source, assembles the induced-charge
system over all conductors, and inverts it. Because the inversion is global it
captures all orders of scattering, and it respects the ceiling by construction.

`born_second_order` is the plausible cheap alternative -- a truncated
perturbation series in the scattering strength. It is the thing a competent
engineer reaches for when a full solve is too slow, and it is exactly what
breaks: truncating an alternating series at second order overshoots, and the
overshoot can drive the predicted screening factor above unity.
"""

from __future__ import annotations

import numpy as np

__all__ = ["Layout", "random_layout", "isolated_pair_matrix",
           "monopole_closure", "born_second_order", "mean_field"]

EPS0 = 8.8541878128e-12


class Layout:
    """A planar array of parallel circular conductors.

    Parameters
    ----------
    xy       : (N, 2) positions in metres
    radius   : (N,) conductor radii in metres
    eps_r    : relative permittivity of the surrounding medium
    """

    def __init__(self, xy: np.ndarray, radius: np.ndarray, eps_r: float = 4.6):
        self.xy = np.asarray(xy, dtype=float)
        self.radius = np.asarray(radius, dtype=float)
        self.eps_r = float(eps_r)
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError("xy must be (N, 2)")
        if self.radius.shape != (self.xy.shape[0],):
            raise ValueError("radius must be (N,)")

    @property
    def n(self) -> int:
        return int(self.xy.shape[0])

    def distances(self) -> np.ndarray:
        d = np.linalg.norm(self.xy[:, None, :] - self.xy[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return d


def random_layout(n: int, seed: int = 0, pitch_um: float = 100.0,
                  diameter_um: float = 40.0, jitter: float = 0.25) -> Layout:
    """A random manufacturable-ish array: min pitch respected, mild jitter."""
    rng = np.random.default_rng(seed)
    side = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(side):
        for j in range(side):
            if len(pts) >= n:
                break
            pts.append((i * pitch_um, j * pitch_um))
    pts = np.array(pts[:n], dtype=float)
    pts += rng.uniform(-jitter, jitter, pts.shape) * pitch_um * 0.5
    return Layout(pts * 1e-6, np.full(n, diameter_um * 1e-6 / 2.0))


def _potential_matrix(lay: Layout) -> np.ndarray:
    """Maxwell potential coefficient matrix P (thin-wire / monopole form)."""
    d = lay.distances()
    scale = 1.0 / (2.0 * np.pi * EPS0 * lay.eps_r)
    with np.errstate(divide="ignore"):
        p = -scale * np.log(d)
    np.fill_diagonal(p, -scale * np.log(lay.radius))
    return p


def isolated_pair_matrix(lay: Layout) -> np.ndarray:
    """Mutual capacitance of each pair computed with only that pair present."""
    n = lay.n
    out = np.zeros((n, n))
    d = lay.distances()
    scale = 1.0 / (2.0 * np.pi * EPS0 * lay.eps_r)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            p = np.array([
                [-scale * np.log(lay.radius[i]), -scale * np.log(d[i, j])],
                [-scale * np.log(d[i, j]), -scale * np.log(lay.radius[j])],
            ])
            c = np.linalg.inv(p)
            out[i, j] = abs(c[0, 1])
    return out


def monopole_closure(lay: Layout) -> np.ndarray:
    """Full-array coupling by global inversion. Respects the ceiling."""
    p = _potential_matrix(lay)
    c = np.linalg.inv(p)
    out = np.abs(c)
    np.fill_diagonal(out, 0.0)
    return out


def born_second_order(lay: Layout) -> np.ndarray:
    """Second-order Born (truncated Neumann) approximation to the inverse.

    P = D(I + D^-1 R), so P^-1 ~ (I - A + A^2) D^-1 with A = D^-1 R.
    Truncating an alternating series overshoots; at tight pitch the overshoot
    pushes the predicted screening factor above unity.
    """
    p = _potential_matrix(lay)
    d = np.diag(np.diag(p))
    r = p - d
    dinv = np.linalg.inv(d)
    a = dinv @ r
    approx = (np.eye(lay.n) - a + a @ a) @ dinv
    out = np.abs(approx)
    np.fill_diagonal(out, 0.0)
    return out


def mean_field(lay: Layout) -> np.ndarray:
    """Screen each pair by an averaged effect of the rest, not resolved positions.

    Included as a comparison: averaging discards exactly the positional
    information screening depends on, so it degrades as the array grows.
    """
    iso = isolated_pair_matrix(lay)
    n = lay.n
    d = lay.distances()
    finite = np.isfinite(d)
    mean_inv = np.mean(1.0 / d[finite]) if finite.any() else 0.0
    shield = 1.0 / (1.0 + (n - 2) * mean_inv * np.median(lay.radius) * 2.0)
    return iso * max(min(shield, 1.0), 0.0)
