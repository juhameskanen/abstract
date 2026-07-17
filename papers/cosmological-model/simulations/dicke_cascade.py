"""
dicke_cascade.py -- the recursive multi-scale extension of dicke_layer.py.

VERIFIED building block (see test_dicke_cascade.py): conditioning |D_n^k>
on a window of w modes having exactly j excitations EXACTLY factorizes:

    |D_n^k>  -->  |D_w^j>  (x)  |D_{n-w}^{k-j}>      with probability hypergeom.pmf(j; n, k, w)

Checked by exact linear algebra: the conditional remainder state is
IDENTICAL (overlap 1.0, machine precision) for every window basis state
consistent with j, it equals the actual smaller Dicke state exactly, and
the total probability mass matches hypergeom.pmf exactly. This is not an
approximation -- it's why a recursive cascade is possible at all: the
leftover after peeling off one window is itself the same kind of object.

This lets us build a literal, exact, disjoint-mode recursive cascade: peel
off level 0's w_0 modes, keep the "rise" (promoted) branches, peel off
level 1's w_1 modes from what's LEFT (not from the original n_bits again),
etc. -- mirroring the fall/bump/rise structure of multiclock.level_state,
but as an exact quantum-state factorization rather than a scalar-pool
rescaling.

FLAGGED HONESTLY, NOT SWEPT UNDER THE RUG: this is NOT proven identical to
multiclock's classical level_state cascade, and level >= 1 should be
expected to disagree with it (see test_dicke_cascade.py for the actual
numeric divergence). The classical cascade re-examines the SAME total
n_bits population at every level, using a level-specific RETARDED TIME to
rescale an abstract pool (bit-budget) -- it never literally shrinks the
mode count. This module instead peels off literal DISJOINT modes,
shrinking N level by level. Level 0 is identical in both models by
construction (verified below) because there's no prior level to disagree
about yet. Which construction is the physically correct one -- rescale-
by-retarded-time on a fixed population, or literal-mode-shrinking -- is an
open modeling question this code does not resolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import hypergeom


@dataclass
class Branch:
    """One leaf of the recursive cascade: a surviving 'promoted' sub-population.

    Represents the (verified, exact) pure state |D_{n_remaining}^{k_remaining}>
    on whatever modes have not yet been peeled off, reached with cumulative
    Born-rule probability `prob` via the sequence of local counts in `history`.
    """
    n_remaining: int
    k_remaining: int
    prob: float
    history: tuple


@dataclass
class LevelSummary:
    width: int
    fabric_prob: float           # total probability mass that 'fell' (j=0) at this level
    structure_prob: float        # total probability mass that 'bumped' (formed structure)
    promoted_prob: float         # total probability mass still alive, carried to the next level
    entropy_contributed: float   # sum over incoming branches of (branch.prob * that window's own S_vN)
    n_branches_in: int
    n_branches_out: int


def recursive_cascade(n_bits: int, k: int, widths: Sequence[int]) -> tuple[list[LevelSummary], list[Branch]]:
    """Exact, literal disjoint-mode recursive cascade. See module docstring for caveats
    about how this relates (and doesn't) to multiclock.level_state.

    Returns (per-level summaries, final surviving branches).
    """
    branches = [Branch(n_remaining=n_bits, k_remaining=k, prob=1.0, history=())]
    summaries = []

    for w in widths:
        fabric_p = 0.0
        structure_p = 0.0
        promoted_p = 0.0
        entropy_contrib = 0.0
        next_branches = []
        n_in = len(branches)

        for br in branches:
            N, K = br.n_remaining, br.k_remaining
            if N <= 0 or K < 0 or br.prob <= 0:
                continue
            w_eff = min(w, N)
            j_max = min(w_eff, K)
            thresh = int(np.ceil(w_eff / 2.0))

            j_range = np.arange(0, j_max + 1)
            pmf = hypergeom.pmf(j_range, N, K, w_eff)

            p_nonzero = pmf[pmf > 0]
            if len(p_nonzero) > 0:
                S_local = -float(np.sum(p_nonzero * np.log2(p_nonzero)))
                entropy_contrib += br.prob * S_local

            for j in j_range:
                p_j = pmf[j] * br.prob
                if p_j <= 0:
                    continue
                if j == 0:
                    fabric_p += p_j
                elif j < thresh:
                    structure_p += p_j
                else:
                    promoted_p += p_j
                    next_branches.append(Branch(
                        n_remaining=N - w_eff, k_remaining=K - j,
                        prob=p_j, history=br.history + (int(j),),
                    ))

        summaries.append(LevelSummary(
            width=w, fabric_prob=fabric_p, structure_prob=structure_p,
            promoted_prob=promoted_p, entropy_contributed=entropy_contrib,
            n_branches_in=n_in, n_branches_out=len(next_branches),
        ))
        branches = next_branches

    return summaries, branches
