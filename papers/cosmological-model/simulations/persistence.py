"""
Exact time-persistence (Ehrenfest) correlation for a width-w window under
the SAME dynamics multiclock.py already assumes: each raw tick, one of the
n bit positions is chosen uniformly and toggled (symmetric Ehrenfest process
on the full n-bit string).

Derivation
----------
View one specific width-w window. Each raw tick affects it (i.e. touches one
of its w bits) with probability p = w/n; when it does, the affected bit is
uniform among the w window bits (conditioning a uniform choice over n on
landing in a fixed w-subset keeps it uniform over that subset) and gets
toggled. So, conditional on "this raw tick touched the window," the window's
own composition evolves by exactly one step of a SMALLER Ehrenfest chain on
w bits (state = number of ones in the window, 0..w; from state i, hop to
i-1 w.p. i/w, to i+1 w.p. (w-i)/w).

Therefore, viewed every raw tick (whether or not it touched the window), the
window's composition is a Markov chain with one-tick transition matrix

    T = (1-p) * I  +  p * M

where M is the (w+1)x(w+1) internal Ehrenfest transition matrix and
p = w/n. This is exact -- not a large-n approximation -- because raw ticks
are i.i.d. Bernoulli(touches-window) trials, so composing Delta of them is
just T^Delta.

Persistence(Delta; a) = (T^Delta)[a, a]
    = P(window composition is a again after Delta raw ticks | it was a now)

Sanity check: for w=1 this must reduce to the single-bit persistence
(1 - 2/n)^Delta, since a lone bit's own Ehrenfest sub-chain is the 2x2 swap.
"""
from __future__ import annotations
import numpy as np
from math import comb


def _ehrenfest_matrix(w: int) -> np.ndarray:
    """Exact (w+1)x(w+1) one-step transition matrix for an Ehrenfest chain
    on w bits (state = number of ones)."""
    M = np.zeros((w + 1, w + 1))
    for i in range(w + 1):
        if i > 0:
            M[i, i - 1] = i / w
        if i < w:
            M[i, i + 1] = (w - i) / w
    return M


def persistence_matrix(n: int, w: int, delta: int) -> np.ndarray:
    """Full (w+1)x(w+1) exact Delta-raw-tick transition matrix T^Delta."""
    M = _ehrenfest_matrix(w)
    p = w / n
    T = (1 - p) * np.eye(w + 1) + p * M
    return np.linalg.matrix_power(T, delta)


def persistence(n: int, w: int, a: int, delta: int) -> float:
    """P(window composition == a again after `delta` raw ticks | == a now)."""
    return persistence_matrix(n, w, delta)[a, a]


if __name__ == "__main__":
    n = 184

    # Sanity check: w=1 must reduce to the single-bit formula (1-2/n)^Delta
    for delta in [1, 5, 20, 100]:
        exact_w1 = persistence(n, 1, 0, delta)
        formula = (1 - 2 / n) ** delta
        print(f"w=1 delta={delta:4d}: matrix={exact_w1:.6f}  formula={formula:.6f}  "
              f"match={'OK' if abs(exact_w1-formula) < 1e-9 else 'MISMATCH'}")

    print()
    w, a = 6, 2
    for delta in [1, 5, 10, 20, 50, 100, 300, 1000]:
        print(f"w={w} a={a} delta={delta:5d}: persistence={persistence(n, w, a, delta):.5f}")
