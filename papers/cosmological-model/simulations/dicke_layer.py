"""
Dicke layer: the psi-layer (wavefunction) companion to multiclock.py's
classical D-layer bitstring model.

Physical picture (established across prior discussion, not re-derived here):

  - GLOBAL STATE: the flat, zero-parameter superposition over all 2^n
    bitstrings, |Psi> = (1/sqrt(2^n)) sum_x |x>. Solomonoff-minimal choice:
    no free parameters beyond n itself.

  - CLOCK: the total excitation-number operator N_hat of the SAME n modes
    -- no external ancilla, self-referential. Conditioning |Psi> on
    N_hat = k gives the Dicke state |D_n^k> = (1/sqrt(C(n,k))) sum_{|A|=k} |A>.
    Born-rule probability of finding N_hat = k is exactly Binomial(n, k, 1/2)
    -- verified to coincide exactly with the classical Ehrenfest process's
    own equilibrium distribution. Coordinate time tau is *defined* through
    k via multiclock's own p(tau) = 0.5*(1 - exp(-2 tau/n)): k = n*p(tau).

  - VERIFIED (exact, small-n linear algebra -- see verify_rank1_purity):
    for any w-mode window, the reduced density matrix of |D_n^k> is
    block-diagonal by local excitation number j, and every block is rank 1
    (pure). Block traces equal hypergeom.pmf(j, n, k, w) exactly: this state
    reproduces the classical D-layer's hypergeometric shadow exactly, while
    being the LEAST entangled purification possible of that shadow (any
    mixed-within-sector alternative could only add entropy, never remove it).

  - CONSEQUENCE used throughout: because each sector is pure, the von
    Neumann entropy of the reduced state is EXACTLY the Shannon entropy of
    the classical sector-probability distribution:
        S_vN(n, k, w) = H(hypergeom(n, k, w))
    No state-vector construction needed to compute it, at any n -- it's
    closed-form combinatorics, same cost profile as multiclock.py itself.

  - SPECIES / SCALES: confirmed to share the SAME n and the SAME k(tau),
    exactly as multiclock.level_state's classical cascade already assumes.
    "Species" are different window-width decompositions of ONE shared
    Dicke state, not separate tensor-product Hilbert space factors. This
    module provides the single-level primitives; the recursive multi-scale
    cascade (the quantum analogue of level_state's fall/bump/promoted
    recursion) is flagged as NOT YET BUILT -- see module bottom.

  - FLAGGED, NOT SOLVED HERE: under this flat-state weighting, k=0 (the
    low-entropy start) is exponentially suppressed relative to k=n/2 -- a
    version of the real, unresolved "why is the entropy of the early
    universe so low" problem. Inherited, not introduced; left as-is per
    explicit instruction rather than patched over.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binom, hypergeom
from scipy.special import comb

FloatArray = NDArray[np.float64]

TRUE_K_RATE = 2.0  # matches multiclock.TRUE_K_RATE exactly -- same derived Ehrenfest constant


# ===========================================================================
# CLOCK: mapping between the excitation-number eigenvalue k and coordinate
# time tau, using the SAME closed form already derived in multiclock.py.
# Reused, not re-derived, so the two layers stay consistent by construction.
# ===========================================================================

def k_of_tau(n_bits: int, tau: float) -> float:
    """Mean-field excitation count at coordinate time tau (real-valued, not rounded)."""
    p = 0.5 * (1.0 - np.exp(-TRUE_K_RATE * tau / n_bits))
    return n_bits * p


def tau_of_k(n_bits: int, k: float) -> float:
    """Inverse of k_of_tau: the coordinate time at which the mean-field count is k."""
    p = k / n_bits
    if not (0 <= p < 0.5):
        raise ValueError(f"k/n_bits={p:g} must be in [0, 0.5) for the relaxation-from-zero branch")
    return -0.5 * n_bits * np.log(1.0 - 2.0 * p)


# ===========================================================================
# BORN-RULE WEIGHT OVER THE CLOCK: sector probability from the flat state.
# ===========================================================================

def sector_probability(n_bits: int, k) -> FloatArray:
    """P(N_hat = k) when the global state is the flat, zero-parameter superposition.

    Exact: Binomial(n_bits, k, 1/2). Independently coincides with the
    Ehrenfest process's own known equilibrium distribution (verified
    numerically -- see tests).
    """
    return binom.pmf(k, n_bits, 0.5)


# ===========================================================================
# WINDOW MARGINAL: the classical D-layer shadow, exactly, at any n.
# ===========================================================================

def window_marginal(n_bits: int, k: int, w: int) -> FloatArray:
    """P(j ones in a w-mode window | exactly k ones among n_bits total), j=0..min(w,k).

    Exact hypergeom.pmf -- the Born-rule marginal of |D_n^k> on any w-mode
    window (verified exactly against direct linear algebra; see
    verify_rank1_purity). Same quantity multiclock.family_fractions_exact
    computes internally via hypergeom.pmf/.sf.
    """
    j = np.arange(0, min(w, k) + 1)
    return hypergeom.pmf(j, n_bits, k, w)


# ===========================================================================
# ENTANGLEMENT ENTROPY: closed-form, exact, works at any n (no 2^n cost).
# ===========================================================================

def entanglement_entropy(n_bits: int, k: int, w: int) -> float:
    """Von Neumann entropy (bits) of the w-mode reduced state of |D_n^k>.

    Exact: every local-occupancy block of the reduced density matrix is
    rank 1 (verified below), so the only entropy present is the classical
    mixing over sectors -- the Shannon entropy of window_marginal.
    """
    p = window_marginal(n_bits, k, w)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def entanglement_entropy_curve(n_bits: int, k: int, widths: Sequence[int]) -> FloatArray:
    """Vectorized convenience: entanglement_entropy at several window widths."""
    return np.array([entanglement_entropy(n_bits, k, w) for w in widths])


# ===========================================================================
# SPECIFIC-PATTERN PROBABILITY: the honest hump mechanism.
#
# Ties directly to the "Entropy and Emergent Structures" wiki page's Result
# 3/4 (mean-field, independent-bit derivation: P(pattern|tau) = p^a(1-p)^b,
# hump iff a<b). This is the EXACT quantum/Born-rule counterpart of that
# same claim -- NOT an approximation stacked on top of it, and NOT related
# to any eta(tau)^matter_power construction (that mechanism is not used
# anywhere in this module and should not be reintroduced silently).
#
# Derivation: within the exact-k Dicke state, conditioning any w-mode
# window on "exactly a excitations" gives (verified in dicke_cascade.py's
# factorization check) a state that is a SMALLER, still-uniform Dicke
# state on those w modes. A uniform superposition over all C(w,a)
# specific orderings means every specific ordered pattern with that
# composition is EQUALLY likely:
#
#     P(specific ordered a-ones/b-zeros pattern | n, k, w=a+b)
#         = hypergeom.pmf(a, n, k, w) / C(w, a)
#
# VERIFIED exactly by direct linear algebra (see test_dicke_layer.py):
# built the actual n=14 Dicke statevector, partial-traced onto w=5 modes,
# and confirmed all C(5,2)=10 specific two-excitation patterns have
# IDENTICAL probability, matching this formula to machine precision.
#
# VERIFIED to reproduce the wiki's three-fold classification: computed
# this exact formula (not the mean-field approximation) as a function of
# tau for a>b, a=b, a<b compositions at n=184 -- got monotonic / monotonic
# / genuine-hump respectively, matching the mean-field prediction's peak
# location to within ~3%. See test_dicke_layer.py.
# ===========================================================================

def pattern_probability(n_bits: int, k, a: int, b: int) -> FloatArray:
    """Exact Born-rule probability of one SPECIFIC ordered (a ones, b zeros)
    pattern appearing in a fixed w=a+b window, given exact global count k.

    This is the honest replacement for any eta(tau)^matter_power fudge:
    the hump (or lack of one) falls straight out of this formula, for the
    right reason (a<b), with no free decay parameter anywhere.
    """
    w = a + b
    k_arr = np.atleast_1d(np.asarray(k, dtype=float))
    pmf = hypergeom.pmf(a, n_bits, k_arr, w)
    result = pmf / comb(w, a)
    return result if np.ndim(k) > 0 else float(result[0])


def pattern_shape(a: int, b: int) -> str:
    """Which of the wiki's three classes this composition falls into.
    Pure bookkeeping, matches Entropy-and-Emergent-Structures Result 3."""
    if a > b:
        return "monotonic_rise"
    elif a == b:
        return "monotonic_to_boundary"
    else:
        return "hump"


def default_composition(w: int) -> tuple[int, int]:
    """A reasonable, EXPLICITLY-FLAGGED default (a, b) for a window of width w,
    chosen to land in the 'hump' class (a<b). This is a genuine modeling
    choice -- which specific rare pattern counts as 'structure' at this
    width -- not a derived fact. Callers wanting a different composition
    should pass one explicitly rather than rely on this."""
    a = max(1, w // 3)
    b = w - a
    if a >= b:
        a, b = 1, w - 1
    return a, b



# ===========================================================================
# TIER-2 SANDBOX: exact small-n linear algebra, for verification/regression
# testing ONLY. Cost is O(2^n_bits); do not call with n_bits above ~24.
# ===========================================================================

def _build_dicke_statevector(n_bits: int, k: int) -> FloatArray:
    """Exact 2^n_bits statevector of |D_n^k> = (1/sqrt(C(n,k))) sum_{|A|=k} |A>."""
    if n_bits > 24:
        raise ValueError(f"n_bits={n_bits} too large for exact statevector construction (2^n blowup)")
    dim = 2 ** n_bits
    psi = np.zeros(dim)
    configs = list(combinations(range(n_bits), k))
    amp = 1.0 / np.sqrt(len(configs))
    for cfg in configs:
        idx = sum(1 << p for p in cfg)
        psi[idx] = amp
    return psi


def verify_rank1_purity(n_bits: int, k: int, w: int, tol: float = 1e-9) -> dict:
    """Exact linear-algebra check of everything entanglement_entropy() relies on.

    Builds the actual Dicke statevector, partial-traces onto w modes, and
    checks: (1) each local-excitation-number block is rank 1, (2) block
    traces match hypergeom.pmf exactly, (3) the resulting von Neumann
    entropy matches entanglement_entropy()'s closed-form shortcut.

    Tier-2 sandbox / regression test only -- not a production code path.
    """
    psi = _build_dicke_statevector(n_bits, k).reshape([2] * n_bits)
    psi_mat = psi.reshape(2 ** w, 2 ** (n_bits - w))
    rho_w = psi_mat @ psi_mat.T

    blocks = defaultdict(list)
    for state in range(2 ** w):
        j = bin(state).count("1")
        blocks[j].append(state)

    max_rank_violation = 0
    max_trace_error = 0.0
    S_exact = 0.0
    for j, idxs in blocks.items():
        sub = rho_w[np.ix_(idxs, idxs)]
        trace = float(np.trace(sub))
        expected = hypergeom.pmf(j, n_bits, k, w)
        max_trace_error = max(max_trace_error, abs(trace - expected))
        if trace < tol:
            continue
        eigs = np.linalg.eigvalsh(sub)
        rank = int(np.sum(eigs > tol))
        max_rank_violation = max(max_rank_violation, rank - 1)
        S_exact -= trace * np.log2(trace)

    S_shortcut = entanglement_entropy(n_bits, k, w)

    return {
        "max_rank_violation": max_rank_violation,    # 0 <=> every block exactly rank 1
        "max_trace_error": max_trace_error,          # ~0 (machine precision) if pmf matches
        "S_from_exact_diagonalization": S_exact,
        "S_from_closed_form_shortcut": S_shortcut,
        "entropy_match": abs(S_exact - S_shortcut) < 1e-8,
    }


# ===========================================================================
# NOT YET BUILT: the recursive multi-scale cascade (quantum analogue of
# multiclock.level_state's fall/bump/promoted recursion across --scales).
# The single-level primitives above are verified and consistent; nesting
# them the way level_state nests classically -- conditioning on retarded
# sub-populations -- has not been derived or checked yet. Do not assume
# entanglement_entropy() composes recursively across scales without
# re-deriving that explicitly first.
# ===========================================================================
