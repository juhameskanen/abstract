"""
Correlated spatial direction field for one level, built from the derived
pair-correlation function g(r) (pair_correlation.py) instead of i.i.d. noise.

APPROXIMATION BEING MADE (flagged explicitly): g(r) is a ratio of match
PROBABILITIES, not a linear correlation coefficient of a continuous field.
There is no unique, exact way to turn it into a Gaussian-field covariance
kernel. What's implemented here is a reasonable, explicitly-approximate
construction: each chain position s gets an independent "innovation" draw,
then a position's assigned vector is that innovation plus a
sqrt(excess-correlation)-weighted blend of its neighbors' innovations
within the g(r) support (r=1..w-1), normalized to a unit direction. This
is validated below by checking the EMPIRICAL directional correlation this
produces has the right shape (peaked at small r, flat beyond w) -- not
that it reproduces g(r)'s exact magnitude, which would require a more
careful (e.g. copula-based) construction if the amplitude itself matters
later.
"""
from __future__ import annotations
import numpy as np
from pair_correlation import pair_correlation


def build_direction_field(n: int, w: int, k: float, a: int, seed: int) -> np.ndarray:
    """Return a (n, 3) array: field[s] is the unit direction assigned to
    chain position s, spatially correlated per g(r) within r < w."""
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=(n, 3))

    r_values = np.arange(1, w)
    g = pair_correlation(n, w, int(round(k)), a, r_values)  # length w-1
    weights = np.sqrt(np.clip(g - 1.0, 0.0, None))

    field = innovations.copy()
    for idx, r in enumerate(r_values):
        wgt = weights[idx]
        if wgt <= 0:
            continue
        field += wgt * 0.5 * (np.roll(innovations, r, axis=0) + np.roll(innovations, -r, axis=0))

    norms = np.linalg.norm(field, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return field / norms


def empirical_direction_correlation(field: np.ndarray, r_values) -> np.ndarray:
    """Mean cosine similarity between direction(s) and direction(s+r), averaged over s."""
    n = field.shape[0]
    out = np.zeros(len(r_values))
    for i, r in enumerate(r_values):
        shifted = np.roll(field, -r, axis=0)
        out[i] = np.mean(np.sum(field * shifted, axis=1))
    return out


if __name__ == "__main__":
    n, w, a, k = 184, 6, 2, 92
    field = build_direction_field(n, w, k, a, seed=7)
    r_values = np.arange(1, 20)
    emp = empirical_direction_correlation(field, r_values)
    g = pair_correlation(n, w, k, a, np.clip(r_values, 1, w - 1))
    print(f"{'r':>4} {'empirical cos-sim':>18} {'g(r) (clipped r<w)':>20}")
    for r, e, gv in zip(r_values, emp, g):
        tag = "" if r < w else "  <- beyond overlap support"
        print(f"{r:4d} {e:18.4f} {gv:20.4f}{tag}")
