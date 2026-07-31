"""
Exact two-point (pair) correlation function for the hypergeometric structure
model in multiclock.py's level_state.

Setup: n bits, k of them "flipped" (a uniformly random k-subset of the n
positions -- the same assumption level_state already makes for the single-
window marginal via hypergeom.pmf). Two width-w windows start at positions
i and i+r (r = start-to-start separation, r >= 1). We want the exact joint
probability that BOTH windows have composition exactly a, as a function of
r and k, and compare it to the product of the (already-existing) single-
window marginals to get a correlation ratio g(r) = P_joint / P_single^2.
g(r) = 1 means independence (what "distribute randomly" assumes);
g(r) > 1 means positively correlated (real knots cluster); g(r) < 1 means
anti-correlated.
"""
from __future__ import annotations
from math import comb
import numpy as np
from scipy.stats import hypergeom


def _C(nn: int, kk: int) -> int:
    return comb(nn, kk) if 0 <= kk <= nn else 0


def joint_match_prob(n: int, w: int, k: int, a: int, r: int) -> float:
    """Exact P(window at i has composition a AND window at i+r has composition a).

    r=0 is the degenerate identical-window case (excluded -- not a real pair).
    1 <= r < w: overlapping windows, s = w-r shared bits.
    r >= w: disjoint windows (result is exactly constant in r here -- the
            hypergeometric model is fully exchangeable, so it only "knows"
            about overlap, not literal chain distance beyond that).
    """
    if r <= 0:
        raise ValueError("r must be >= 1 (r=0 is the trivial identical-window case)")
    s = max(w - r, 0)          # bits shared by both windows
    edge = min(r, w)           # size of each window's non-shared ("only") region
    rest = n - w - r if r < w else n - 2 * w
    total = 0.0
    Cnk = _C(n, k)
    if Cnk == 0:
        return 0.0
    j_lo = max(0, a - edge)
    j_hi = min(s, a)
    for j in range(j_lo, j_hi + 1):
        x = a - j
        if not (0 <= x <= edge):
            continue
        rest_ones = k - j - 2 * x
        term = _C(s, j) * _C(edge, x) * _C(edge, x) * _C(rest, rest_ones)
        total += term
    return total / Cnk


def pair_correlation(n: int, w: int, k: int, a: int, r_values) -> np.ndarray:
    """g(r) = P_joint(r) / P_single^2 for an array of separations r (r>=1)."""
    p_single = hypergeom.pmf(a, n, k, w)
    denom = p_single ** 2
    out = np.zeros(len(r_values))
    for idx, r in enumerate(r_values):
        p_joint = joint_match_prob(n, w, k, a, int(r))
        out[idx] = p_joint / denom if denom > 0 else np.nan
    return out


if __name__ == "__main__":
    n, w, a = 184, 6, 2
    r_values = np.arange(1, 16)
    print(f"n={n} w={w} a={a}")
    for k in [40, 92, 140]:
        g = pair_correlation(n, w, k, a, r_values)
        print(f"\nk={k}:")
        for r, gv in zip(r_values, g):
            marker = "  <-- overlap" if r < w else ""
            print(f"  r={r:2d}  g(r)={gv:8.3f}{marker}")
