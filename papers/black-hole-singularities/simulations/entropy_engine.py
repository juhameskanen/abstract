"""
entropy_engine.py
==================
The substrate-level Ehrenfest relaxation engine, factored out of
emergent_structure_relativistic.py so it has one home and both the
cosmological simulation and the black hole radial profile reuse the exact
same functions instead of two copies drifting apart.

Nothing here is new physics -- this is a straight extraction of
p_of_tau / ground_truth_pool / order_parameter / family_fractions_exact /
the multi-clock scale hierarchy / matter_content, unchanged.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import hypergeom
from scipy.special import gammaln

FloatArray = NDArray[np.float64]


def p_of_tau(tau: FloatArray, k_rate: float) -> FloatArray:
    return 0.5 * (1.0 - np.exp(-k_rate * tau))


def ones_count_from_p(p: FloatArray, n_bits: int) -> FloatArray:
    return np.clip(np.round(n_bits * p), 0, n_bits)


def combinatorial_entropy_bits(n_bits: int, k: FloatArray) -> FloatArray:
    k = np.clip(np.round(k), 0, n_bits)
    log2_comb = (gammaln(n_bits + 1) - gammaln(k + 1) - gammaln(n_bits - k + 1)) / np.log(2.0)
    return log2_comb / n_bits


def ground_truth_pool(tau: FloatArray, n_bits: int, k_rate: float) -> tuple[FloatArray, FloatArray]:
    p = p_of_tau(tau, k_rate)
    k = ones_count_from_p(p, n_bits)
    return n_bits * combinatorial_entropy_bits(n_bits, k), k


def order_parameter(tau: FloatArray, k_rate: float) -> FloatArray:
    return np.exp(-k_rate * tau)


def family_fractions_exact(n_bits: int, k: FloatArray, width: float) -> tuple[FloatArray, FloatArray, FloatArray]:
    w = int(round(width))
    k_int = np.clip(np.round(k), 0, n_bits).astype(np.int64)
    j_thresh = int(np.ceil(w / 2.0))
    f_fall = hypergeom.pmf(0, n_bits, k_int, w)
    f_rise = hypergeom.sf(j_thresh - 1, n_bits, k_int, w)
    f_bump = np.clip(1.0 - f_rise - f_fall, 0.0, 1.0)
    return f_fall, f_bump, f_rise


def retard(tau: FloatArray, width: float, n_bits: int) -> FloatArray:
    step = width / n_bits
    return step * np.floor(tau / step)


@dataclass
class ScaleResult:
    width: float
    lapse: float
    tau_local: FloatArray
    pool_true: FloatArray
    pool_eff: FloatArray
    f_fall: FloatArray
    f_bump: FloatArray
    f_rise: FloatArray
    fabric: FloatArray
    structure: FloatArray
    promoted: FloatArray
    pending: FloatArray
    structure_count: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        self.structure_count = self.structure / self.width


def level_state(tau, level_idx, widths, n_bits, k_rate, cache=None) -> ScaleResult:
    if cache is None:
        cache = {}
    key = (level_idx, np.asarray(tau).tobytes())
    if key in cache:
        return cache[key]

    width = widths[level_idx]
    tau_local = retard(tau, width, n_bits)

    if level_idx == 0:
        pool_true, _ = ground_truth_pool(tau, n_bits, k_rate)
        pool_eff, k_local = ground_truth_pool(tau_local, n_bits, k_rate)
    else:
        prev = level_state(tau, level_idx - 1, widths, n_bits, k_rate, cache)
        prev_local = level_state(tau_local, level_idx - 1, widths, n_bits, k_rate, cache)
        pool_true = prev.promoted
        pool_eff = prev_local.promoted
        _, k_local = ground_truth_pool(tau_local, n_bits, k_rate)

    f_fall, f_bump, f_rise = family_fractions_exact(n_bits, k_local, width)
    fabric = pool_eff * f_fall
    structure = pool_eff * f_bump
    promoted = pool_eff * f_rise
    pending = pool_true - pool_eff

    result = ScaleResult(
        width=width, lapse=1.0 / width, tau_local=tau_local,
        pool_true=pool_true, pool_eff=pool_eff,
        f_fall=f_fall, f_bump=f_bump, f_rise=f_rise,
        fabric=fabric, structure=structure, promoted=promoted, pending=pending,
    )
    cache[key] = result
    return result


def build_scale_hierarchy(tau, scales, n_bits, k_rate) -> list[ScaleResult]:
    cache: dict = {}
    return [level_state(tau, i, scales, n_bits, k_rate, cache) for i in range(len(scales))]


def matter_content(levels, tau, k_rate, matter_power):
    eta = order_parameter(tau, k_rate)
    weight = eta ** matter_power
    per_scale_bits = [lvl.structure * weight for lvl in levels]
    total_bits = sum(per_scale_bits)
    per_scale_count = [mb / lvl.width for mb, lvl in zip(per_scale_bits, levels)]
    total_count = sum(per_scale_count)
    return total_bits, per_scale_bits, total_count, per_scale_count


def default_k_rate(t_bf_max: float, sat_fraction: float = 0.99) -> float:
    """Same convention as run_simulation: calibrate k_rate so p reaches
    sat_fraction of its asymptote by t_bf_max."""
    return -np.log(1.0 - sat_fraction) / t_bf_max


def matter_diffusion_rate(levels: list[ScaleResult], tau: FloatArray, k_rate: float,
                            matter_power: float, n_bits: int) -> FloatArray:
    """
    Variance-accumulation rate (per unit tau) of the local matter/structure
    reading a single particle actually experiences, as opposed to the
    ensemble-mean matter_content() curve.

    A single window's bump/fall/rise classification is a Bernoulli draw
    with probability f_bump (exact -- f_bump itself is not approximated).
    Var[X_i] = (pool_eff_i * eta^p)^2 * f_bump_i * (1 - f_bump_i) is the
    exact variance of that draw's contribution. Each scale's draw renews
    once per its own local tick (lapse = 1/width, same as retard()), so
    the renewal rate is n_bits/width.

    Two explicit approximations, not hidden: scales are combined as
    independent (no cross-scale covariance derived), and the per-tick
    renewal is treated as a locally-white process rather than deriving
    the true correlation time.
    """
    eta = order_parameter(tau, k_rate)
    weight = eta ** matter_power
    D_tau = np.zeros_like(np.asarray(tau, dtype=np.float64))
    for lvl in levels:
        var_i = (lvl.pool_eff * weight) ** 2 * lvl.f_bump * (1.0 - lvl.f_bump)
        renewal_rate_i = n_bits / lvl.width
        D_tau = D_tau + var_i * renewal_rate_i
    return D_tau
