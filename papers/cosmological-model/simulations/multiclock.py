"""
Emergent Structure from a Relaxing Bitstring — Multi-Clock (GR-Compatible) Version
====================================================================================

Model
-----
A length-``n`` bitstring starts all-zero and relaxes under the symmetric
Ehrenfest process: at each raw tick, one of the ``n`` positions is chosen
uniformly and toggled. The ensemble-mean per-bit flip probability obeys the
EXACT, parameter-free closed form

    p(tau) = 0.5 * (1 - exp(-2 * tau)),      tau = raw_tick_count / n

This is not a fitted curve: it is the solution of d<k>/dt = 1 - 2<k>/n for
the symmetric Ehrenfest chain. k_rate = 2 is therefore the correct, derived
relaxation constant in these units, not a free parameter.

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Sequence

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
from scipy.special import gammaln

T_PLANCK_YR: float = 1.71e-51
T_AGE_YR: float = 13.8e9

FloatArray = NDArray[np.float64]

TRUE_K_RATE: float = 2.0  # derived Ehrenfest relaxation constant (NOT a free parameter)


# ===========================================================================
# TIME-AXIS MAPPING (for plotting only)
# ===========================================================================

def years_to_tbf(t_yr: FloatArray | float, t_today: float) -> FloatArray | float:
    """Map physical years onto raw bit-flip coordinate time, log-uniformly."""
    ln_min = np.log(T_PLANCK_YR)
    ln_max = np.log(T_AGE_YR)
    return t_today * (np.log(np.maximum(t_yr, T_PLANCK_YR)) - ln_min) / (ln_max - ln_min)


# ===========================================================================
# EHRENFEST BIT-FLIP PRIMITIVES (ground truth, substrate-level -- no observer)
# All functions here take RAW tick counts (tau_raw) plus n_bits explicitly,
# and normalize internally. This is the fix: no function silently assumes
# its argument is already divided by n.
# ===========================================================================

def p_of_tau(tau_raw: FloatArray, n_bits: int, k_rate: float) -> FloatArray:
    """Per-bit probability of having flipped 0->1 by raw tick count ``tau_raw``.

    Args:
        tau_raw: RAW bit-flip tick count (NOT divided by n).
        n_bits: Total bitstring length n. Used to normalize internally.
        k_rate: Relaxation rate constant in normalized-tau units. Use
            TRUE_K_RATE (2.0) for the physically exact Ehrenfest process.

    Returns:
        Per-bit flip probability at each tau_raw, in [0, 0.5).
    """
    tau_norm = tau_raw / n_bits
    return 0.5 * (1.0 - np.exp(-k_rate * tau_norm))


def ones_count_from_p(p: FloatArray, n_bits: int) -> FloatArray:
    """Map the mean-field flip probability onto an integer ones-count k."""
    return np.clip(np.round(n_bits * p), 0, n_bits)


def combinatorial_entropy_bits(n_bits: int, k: FloatArray) -> FloatArray:
    """Exact combinatorial Shannon entropy per bit, log2 C(n,k) / n."""
    k = np.clip(np.round(k), 0, n_bits)
    log2_comb = (gammaln(n_bits + 1) - gammaln(k + 1) - gammaln(n_bits - k + 1)) / np.log(2.0)
    return log2_comb / n_bits


def ground_truth_pool(
    tau_raw: FloatArray, n_bits: int, k_rate: float
) -> tuple[FloatArray, FloatArray]:
    """The substrate's own entropy-unfolded bit budget -- no observer involved."""
    p = p_of_tau(tau_raw, n_bits, k_rate)
    k = ones_count_from_p(p, n_bits)
    return n_bits * combinatorial_entropy_bits(n_bits, k), k


def order_parameter(tau_raw: FloatArray, n_bits: int, k_rate: float) -> FloatArray:
    """Departure-from-equilibrium order parameter, eta(tau) = 1 - 2*p(tau).

    Equals exp(-k_rate * tau_raw / n_bits) exactly.
    """
    return np.exp(-k_rate * tau_raw / n_bits)


def family_fractions_exact(
    n_bits: int, k: FloatArray, width: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Exact falling/bump/rising probability mass fractions for one window."""
    w = int(round(width))
    k_int = np.clip(np.round(k), 0, n_bits).astype(np.int64)
    j_thresh = int(np.ceil(w / 2.0))
    f_fall = hypergeom.pmf(0, n_bits, k_int, w)
    f_rise = hypergeom.sf(j_thresh - 1, n_bits, k_int, w)
    f_bump = np.clip(1.0 - f_rise - f_fall, 0.0, 1.0)
    return f_fall, f_bump, f_rise


# ===========================================================================
# MULTI-CLOCK CASCADE: each scale gets its own retarded proper time
# ===========================================================================

def retard(tau_raw: FloatArray, width: float) -> FloatArray:
    """Quantize raw tick count down to this scale's own tick resolution.

    FIXED: operates natively in RAW tick units. A width-w structure needs
    w raw flips to advance one unit of its own proper time, so its own
    clock only ticks at multiples of w raw ticks. No n_bits normalization
    is needed here at all — this was the source of the earlier bug (the
    old version quantized in normalized units while receiving raw ticks).

    Args:
        tau_raw: Raw bit-flip coordinate time, scalar or array.
        width: This scale's window width in bits.

    Returns:
        tau_raw floored to the nearest multiple of width.
    """
    return width * np.floor(tau_raw / width)


@dataclass
class ScaleResult:
    """Retarded fall/bump/rise decomposition and ledger for one scale."""

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


def matter_content(
    levels: list[ScaleResult], tau_raw: FloatArray, n_bits: int, k_rate: float, matter_power: float
) -> tuple[FloatArray, list[FloatArray], FloatArray, list[FloatArray]]:
    """Reweight each scale's structure by the order parameter to get real matter."""
    eta = order_parameter(tau_raw, n_bits, k_rate)
    weight = eta ** matter_power
    per_scale_bits = [lvl.structure * weight for lvl in levels]
    total_bits = sum(per_scale_bits)
    per_scale_count = [mb / lvl.width for mb, lvl in zip(per_scale_bits, levels)]
    total_count = sum(per_scale_count)
    return total_bits, per_scale_bits, total_count, per_scale_count


def level_state(
    tau_raw: FloatArray,
    level_idx: int,
    widths: Sequence[float],
    n_bits: int,
    k_rate: float,
    cache: dict | None = None,
) -> ScaleResult:
    """Recursively compute one scale's retarded state at the given raw tau."""
    if cache is None:
        cache = {}
    key = (level_idx, np.asarray(tau_raw).tobytes())
    if key in cache:
        return cache[key]

    width = widths[level_idx]
    tau_local = retard(tau_raw, width)

    if level_idx == 0:
        pool_true, _ = ground_truth_pool(tau_raw, n_bits, k_rate)
        pool_eff, k_local = ground_truth_pool(tau_local, n_bits, k_rate)
    else:
        prev = level_state(tau_raw, level_idx - 1, widths, n_bits, k_rate, cache)
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


def build_scale_hierarchy(
    tau_raw: FloatArray, scales: Sequence[float], n_bits: int, k_rate: float
) -> list[ScaleResult]:
    """Evaluate every scale's retarded state on a shared raw-tau grid."""
    cache: dict = {}
    return [
        level_state(tau_raw, i, scales, n_bits, k_rate, cache)
        for i in range(len(scales))
    ]


# ===========================================================================
# SIMULATION
# ===========================================================================

@dataclass
class SimulationResult:
    t_bf: FloatArray
    t_bf_max: float
    t_today: float
    k_rate: float
    n_bits: int
    unfolded_bits: FloatArray
    entropy: FloatArray
    levels: list[ScaleResult]
    size_measure: FloatArray
    conservation_max_error: float
    matter_power: float
    total_matter: FloatArray
    per_scale_matter: list[FloatArray]
    t_today_auto: bool
    saturation_reached: float  # NEW: diagnostic — actual p-saturation at t_bf_max


def run_simulation(
    n_bits: float,
    scales: Sequence[float],
    steps: int = 3000,
    t_bf_max: float | None = None,
    k_rate: float | None = None,
    t_today: float | None = None,
    matter_power: float = 1.0,
) -> SimulationResult:
    """Run the full multi-clock entropy-unfolding and scale-decomposition.
    """
    resolved_t_bf_max = (
        t_bf_max * n_bits if t_bf_max is not None else n_bits * np.log(n_bits)
    )
    resolved_k_rate = TRUE_K_RATE
    n_int = int(round(n_bits))

    t_bf = np.linspace(1e-9, resolved_t_bf_max, steps)
    unfolded_bits, k_ones = ground_truth_pool(t_bf, n_int, resolved_k_rate)
    entropy = unfolded_bits / n_int

    # diagnostic only: how saturated is p() at the chosen display endpoint?
    p_end = p_of_tau(np.array([resolved_t_bf_max]), n_int, resolved_k_rate)[0]
    saturation_reached = p_end / 0.5

    levels = build_scale_hierarchy(t_bf, scales, n_int, resolved_k_rate)

    pending_total = sum(lvl.pending for lvl in levels)
    fabric_total = sum(lvl.fabric for lvl in levels)
    structure_total = sum(lvl.structure for lvl in levels)
    promoted_last = levels[-1].promoted

    total_accounted = pending_total + fabric_total + structure_total + promoted_last
    conservation_max_error = float(np.max(np.abs(total_accounted - unfolded_bits)))

    total_matter_bits, per_scale_matter_bits, total_matter, per_scale_matter = matter_content(
        levels, t_bf, n_int, resolved_k_rate, matter_power
    )

    size_measure = (unfolded_bits - pending_total - total_matter_bits) / n_bits

    t_today_auto = t_today is None
    resolved_t_today = float(t_bf[int(np.argmax(total_matter))]) if t_today_auto else t_today

    return SimulationResult(
        t_bf=t_bf, t_bf_max=resolved_t_bf_max, t_today=resolved_t_today, k_rate=resolved_k_rate,
        n_bits=n_int, unfolded_bits=unfolded_bits, entropy=entropy, levels=levels,
        size_measure=size_measure, conservation_max_error=conservation_max_error,
        matter_power=matter_power, total_matter=total_matter, per_scale_matter=per_scale_matter,
        t_today_auto=t_today_auto, saturation_reached=saturation_reached,
    )


# ===========================================================================
# WORLDLINES
# ===========================================================================

def build_worldlines(
    structure_count: FloatArray, n_slots: int, seed: int
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Assign a fixed set of comoving slots and activate a fraction over time."""
    rng = np.random.default_rng(seed)
    y_comoving = np.linspace(-0.5, 0.5, n_slots)
    slot_rank = rng.permutation(n_slots)
    frac = structure_count / max(structure_count.max(), 1e-12)
    active_count = np.clip((frac * n_slots).astype(int), 0, n_slots)
    active_mask = slot_rank[None, :] < active_count[:, None]
    return y_comoving, active_mask
