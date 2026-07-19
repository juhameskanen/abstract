"""
Dicke layer: the psi-layer (wavefunction) companion to multiclock.py's
classical D-layer bitstring model.
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


def sector_probability(n_bits: int, k) -> FloatArray:
    """P(N_hat = k) when the global state is the flat, zero-parameter superposition."""
    return binom.pmf(k, n_bits, 0.5)


def window_marginal(n_bits: int, k: int, w: int) -> FloatArray:
    """P(j ones in a w-mode window | exactly k ones among n_bits total), j=0..min(w,k)."""
    j = np.arange(0, min(w, k) + 1)
    return hypergeom.pmf(j, n_bits, k, w)


def entanglement_entropy(n_bits: int, k: int, w: int) -> float:
    """Von Neumann entropy (bits) of the w-mode reduced state of |D_n^k>."""
    p = window_marginal(n_bits, k, w)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def entanglement_entropy_curve(n_bits: int, k: int, widths: Sequence[int]) -> FloatArray:
    """Vectorized convenience: entanglement_entropy at several window widths."""
    return np.array([entanglement_entropy(n_bits, k, w) for w in widths])


def pattern_probability(n_bits: int, k, a: int, b: int) -> FloatArray:
    """Exact Born-rule probability of one SPECIFIC ordered (a ones, b zeros)
    pattern appearing in a fixed w=a+b window, given exact global count k.
    """
    w = a + b
    k_arr = np.atleast_1d(np.asarray(k, dtype=float))
    pmf = hypergeom.pmf(a, n_bits, k_arr, w)
    result = pmf / comb(w, a)
    return result if np.ndim(k) > 0 else float(result[0])


def class_probability(n_bits: int, k, a: int, b: int) -> FloatArray:
    """Exact Born-rule probability that a w=a+b window has exactly `a`
    excitations, in ANY of the C(w,a) equivalent orderings -- i.e. the
    whole (a,b) equivalence class, not one specific sequence.

    This is exactly pattern_probability(...) * C(w,a); equivalently, it's
    just hypergeom.pmf(a, n_bits, k, w) on its own, with no combinatorial
    division. Use this (not pattern_probability) when "matter" is meant to
    mean "a window with this composition, any arrangement", which is the
    class-level notion multiclock.family_fractions_exact already uses on
    the classical side (threshold on COUNT, not on exact sequence).

    pattern_probability answers a different, still-valid question --
    "how much entanglement/distinguishing information is in a specific
    microstate" -- but it is NOT the right quantity for "how much matter
    has formed", because it divides out the C(w,a) degeneracy that a
    real particle-formation event shouldn't care about. See the module-
    level discussion for why conflating the two silently suppresses the
    macroscopic matter signal by a factor of C(w,a) per level, compounding
    multiplicatively across a chained cascade.
    """
    w = a + b
    k_arr = np.atleast_1d(np.asarray(k, dtype=float))
    pmf = hypergeom.pmf(a, n_bits, k_arr, w)
    return pmf if np.ndim(k) > 0 else float(pmf[0])


def pattern_shape(a: int, b: int) -> str:
    """Which of the wiki's three classes this composition falls into."""
    if a > b:
        return "monotonic_rise"
    elif a == b:
        return "monotonic_to_boundary"
    else:
        return "hump"


def default_composition(w: int) -> tuple[int, int]:
    """A reasonable, EXPLICITLY-FLAGGED default (a, b) for a window of width w."""
    a = max(1, w // 3)
    b = w - a
    if a >= b:
        a, b = 1, w - 1
    return a, b


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
    """Exact linear-algebra check of everything entanglement_entropy() relies on."""
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
        "max_rank_violation": max_rank_violation,
        "max_trace_error": max_trace_error,
        "S_from_exact_diagonalization": S_exact,
        "S_from_closed_form_shortcut": S_shortcut,
        "entropy_match": abs(S_exact - S_shortcut) < 1e-8,
    }
