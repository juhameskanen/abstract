"""
Emergent Structure from a Relaxing Bitstring
=============================================

A parameter-free model of structure formation, built from nothing but a
bitstring length ``n``.

Model
-----
A length-``n`` bitstring starts all-zero (the unique, minimal-description
state, zero Shannon entropy) and relaxes under the symmetric Ehrenfest
process: at each raw tick, one of the ``n`` positions is chosen uniformly
and toggled. Each bit is then independently ``1`` with probability

    p(tau) = 0.5 * (1 - exp(-2 * tau)),      tau = raw_tick_count / n

a closed-form, ``n``-independent relaxation curve. There is no external
clock: an emergent internal observer experiences "time" as the minimal
change between two configurations it passes through -- one bit flip.

For a window of ``w`` consecutive bits, the number of 1-bits ``K`` in that
window is distributed as ``Binomial(w, p(tau))``. Classifying ``K``
against ``w/2`` splits every possible window composition into exactly
three shape families, with no fitted parameters:

    falling  (K = 0)        the window is still exactly all-zero
    bump     (0 < K < w/2)  a genuine, transient interior structure
    rising   (K >= w/2)     the window has become majority-flipped

These three probabilities are mutually exclusive and exhaustive outcomes
of a single random variable (the window's own bit count), so they sum to
exactly 1 at every tau -- this is what makes the accounting below
conserve exactly, with no free "matter" parameters beyond the coherence
scales chosen for inspection.

Because "falling" and "rising" are mirror images of each other under the
zero-start/one-start complementation symmetry, only the "bump" family
represents a genuine deviation from either homogeneous extreme -- and,
for the same reason, it is the *typical* family: the population mean of
``K`` sits strictly below ``w/2`` for every finite tau, so it is
overwhelmingly the bump region, not the rising region, that captures
where most windows actually sit.

Coherence scales
-----------------
The same fall/bump/rise decomposition can be evaluated at any window
width ``w``. Small ``w`` gives a gradual, ongoing, spatially-local
"rising" onset; large ``w`` (comparable to ``n`` itself) gives a sharp,
late-switching, effectively global onset, a direct consequence of
concentration of measure (the width of the transition region in ``p``
shrinks as ``w`` grows). This script treats a list of coherence scales
uniformly -- there is no privileged, named tier -- and nests them so that
each scale's condensed (rising) fraction becomes the input budget for the
next, coarser scale.

Conservation
------------
At each scale ``i``, the incoming bit budget ``pool_i`` splits into:

    structure_i  = pool_i * f_bump_i    (transient, visible at this scale)
    pool_{i+1}   = pool_i * f_rise_i    (promoted to the next scale)

and, since ``f_fall_i + f_bump_i + f_rise_i == 1`` exactly, summing
fabric (the scale-0 falling fraction) with every scale's structure and
settled-credit terms reproduces the entropy-unfolded budget ``n * S(tau)``
to floating-point precision -- verified numerically below.
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
from scipy.stats import binom

T_PLANCK_YR: float = 1.71e-51
T_AGE_YR: float = 13.8e9

FloatArray = NDArray[np.float64]


# ===========================================================================
# TIME-AXIS MAPPING
# ===========================================================================

def years_to_tbf(t_yr: FloatArray | float, t_today: float) -> FloatArray | float:
    """Map physical years onto bit-flip time, log-uniformly.

    Maps the interval ``[T_PLANCK_YR, T_AGE_YR]`` onto ``[0, t_today]`` via
    ``t -> ln t``, so that a wide range of physical timescales can share a
    single bit-flip-time axis for plotting.

    Args:
        t_yr: Physical time(s) in years.
        t_today: Bit-flip time corresponding to the present day.

    Returns:
        The corresponding bit-flip time(s).
    """
    ln_min = np.log(T_PLANCK_YR)
    ln_max = np.log(T_AGE_YR)
    return t_today * (np.log(np.maximum(t_yr, T_PLANCK_YR)) - ln_min) / (ln_max - ln_min)


# ===========================================================================
# EHRENFEST BIT-FLIP PRIMITIVES
# ===========================================================================

def p_of_tau(tau: FloatArray, k_rate: float) -> FloatArray:
    """Per-bit probability of having flipped 0->1 by bit-flip time ``tau``.

    Exact Ehrenfest urn relaxation curve, starting from the all-zero state.

    Args:
        tau: Bit-flip time (raw tick count / n), scalar or array.
        k_rate: Overall relaxation rate constant.

    Returns:
        The per-bit flip probability at each ``tau``, in ``[0, 0.5)``.
    """
    return 0.5 * (1.0 - np.exp(-k_rate * tau))


def binary_entropy_bits(p: FloatArray) -> FloatArray:
    """Exact Shannon entropy per bit, in bits.

    Args:
        p: Per-bit probability of being ``1``.

    Returns:
        ``H(p) = -p log2(p) - (1-p) log2(1-p)``, in bits.
    """
    p = np.clip(p, 1e-300, 1.0 - 1e-300)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def family_fractions(width: float, p: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Exact falling/bump/rising probability mass fractions for one window.

    For a window of ``width`` bits, each independently ``1`` with
    probability ``p``, the number of 1-bits ``K`` is ``Binomial(width, p)``.
    This partitions the outcomes of ``K`` into three mutually exclusive,
    exhaustive classes, so the three returned arrays sum to 1 at every
    point (up to floating-point precision).

    Uses ``scipy.stats.binom`` (regularized incomplete beta under the
    hood) rather than summing ``width + 1`` terms directly, so this stays
    exact and fast for any window width, small or large.

    Args:
        width: Window width in bits.
        p: Per-bit flip probability (scalar or array).

    Returns:
        A tuple ``(f_fall, f_bump, f_rise)`` of arrays matching the shape
        of ``p``:
          - ``f_fall``: ``P(K = 0)``.
          - ``f_bump``: ``P(0 < K < width/2)``.
          - ``f_rise``: ``P(K >= width/2)``.
    """
    w = int(round(width))
    k_thresh = int(np.ceil(w / 2.0))       # smallest integer K with K >= w/2
    f_rise = binom.sf(k_thresh - 1, w, p)  # P(K >= k_thresh)
    f_fall = binom.pmf(0, w, p)            # P(K = 0)
    f_bump = np.clip(1.0 - f_rise - f_fall, 0.0, 1.0)
    return f_fall, f_bump, f_rise


# ===========================================================================
# SCALE HIERARCHY
# ===========================================================================

@dataclass
class ScaleResult:
    """Fall/bump/rise decomposition and derived quantities for one scale.

    Attributes:
        width: Window width in bits for this scale.
        pool_bits: Incoming bit budget at this scale, before splitting.
        f_fall: ``P(K=0)`` at this scale, per tau.
        f_bump: ``P(0<K<width/2)`` at this scale, per tau.
        f_rise: ``P(K>=width/2)`` at this scale, per tau.
        structure_bits: Transient (bump) bit-mass visible at this scale.
        promoted_bits: Condensed (rise) bit-mass passed to the next scale.
        structure_count: ``structure_bits / width``, in entity units.
    """

    width: float
    pool_bits: FloatArray
    f_fall: FloatArray
    f_bump: FloatArray
    f_rise: FloatArray
    structure_bits: FloatArray
    promoted_bits: FloatArray
    structure_count: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        self.structure_count = self.structure_bits / self.width


def build_scale_hierarchy(
    scales: Sequence[float], unfolded_bits: FloatArray, p: FloatArray
) -> list[ScaleResult]:
    """Nest an arbitrary list of coherence scales over one shared p(tau).

    Each scale's condensed (rising) bit-mass becomes the next scale's
    incoming budget. No scale is treated as privileged or named; the list
    order only determines the nesting order (finest-grained first).

    Args:
        scales: Window widths in bits, ordered from finest to coarsest.
        unfolded_bits: Entropy-unfolded bit budget ``n * S(tau)`` feeding
            the first scale.
        p: Per-bit flip probability, shared across all scales.

    Returns:
        One :class:`ScaleResult` per entry in ``scales``, in the same
        order.
    """
    results: list[ScaleResult] = []
    pool = unfolded_bits
    for width in scales:
        f_fall, f_bump, f_rise = family_fractions(width, p)
        structure_bits = pool * f_bump
        promoted_bits = pool * f_rise
        results.append(
            ScaleResult(
                width=width,
                pool_bits=pool,
                f_fall=f_fall,
                f_bump=f_bump,
                f_rise=f_rise,
                structure_bits=structure_bits,
                promoted_bits=promoted_bits,
            )
        )
        pool = promoted_bits
    return results


def settled_credit(levels: list[ScaleResult]) -> list[FloatArray]:
    """Per-scale entity credit: condensed structure not yet re-absorbed.

    A scale's condensed (rising) bit-mass keeps its "+1 credit" toward the
    structural size measure only for the portion not itself swept up into
    the next scale's bump/rise formation -- i.e. the fraction of it that
    remains "falling" (untouched) under the *next* scale's classification.
    The coarsest scale has no next scale, so all of its condensed mass
    counts as settled.

    Args:
        levels: Scale hierarchy as returned by :func:`build_scale_hierarchy`.

    Returns:
        One array per scale, in entity-count units (bits / width).
    """
    credits: list[FloatArray] = []
    for i, level in enumerate(levels):
        if i + 1 < len(levels):
            next_level = levels[i + 1]
            settled_bits = next_level.pool_bits * next_level.f_fall
        else:
            settled_bits = level.promoted_bits
        credits.append(settled_bits / level.width)
    return credits


# ===========================================================================
# SIMULATION
# ===========================================================================

@dataclass
class SimulationResult:
    """Full output of one run of :func:`run_simulation`.

    Attributes:
        t_bf: Bit-flip time grid.
        t_bf_max: Maximum bit-flip time simulated.
        t_today: Bit-flip time identified with the present day.
        k_rate: Relaxation rate constant used.
        entropy: Shannon entropy per bit, ``S(tau)``.
        unfolded_bits: Entropy-unfolded bit budget, ``n * S(tau)``.
        fabric_bits: Untouched (scale-0 falling) bit-mass.
        levels: Per-scale decomposition, see :class:`ScaleResult`.
        entity_credit: Per-scale settled entity credit, see
            :func:`settled_credit`.
        size_measure: ``(fabric_bits + sum(entity_credit * width)) / n``,
            a structural size proxy analogous to a scale factor.
        conservation_max_error: Maximum absolute deviation between the
            full bit-mass accounting and ``unfolded_bits``; should be at
            floating-point precision.
    """

    t_bf: FloatArray
    t_bf_max: float
    t_today: float
    k_rate: float
    entropy: FloatArray
    unfolded_bits: FloatArray
    fabric_bits: FloatArray
    levels: list[ScaleResult]
    entity_credit: list[FloatArray]
    size_measure: FloatArray
    conservation_max_error: float


def run_simulation(
    n_bits: float,
    scales: Sequence[float],
    steps: int = 3000,
    t_bf_max: float | None = None,
    sat_fraction: float = 0.99,
    k_rate: float | None = None,
    t_today: float = 74.0,
) -> SimulationResult:
    """Run the full entropy-unfolding and scale-decomposition simulation.

    Args:
        n_bits: Total bitstring length ``n``.
        scales: Coherence scales (window widths, in bits) to evaluate,
            ordered from finest to coarsest.
        steps: Number of bit-flip-time grid points.
        t_bf_max: Maximum bit-flip time to simulate. Defaults to
            ``n * ln(n)``, the Ehrenfest mixing/cutoff timescale.
        sat_fraction: Fraction of equilibrium saturation reached by
            ``t_bf_max``, used to calibrate ``k_rate`` when not given
            explicitly.
        k_rate: Relaxation rate constant. Derived from ``sat_fraction``
            and ``t_bf_max`` if not given.
        t_today: Bit-flip time identified with the present day.

    Returns:
        A populated :class:`SimulationResult`.
    """
    resolved_t_bf_max = t_bf_max if t_bf_max is not None else n_bits * np.log(n_bits)
    resolved_k_rate = (
        k_rate
        if k_rate is not None
        else -np.log(1.0 - sat_fraction) / resolved_t_bf_max
    )

    t_bf = np.linspace(1e-9, resolved_t_bf_max, steps)
    p = p_of_tau(t_bf, resolved_k_rate)
    entropy = binary_entropy_bits(p)
    unfolded_bits = n_bits * entropy  # nothing has "come into being" until entropy unfolds it

    levels = build_scale_hierarchy(scales, unfolded_bits, p)
    fabric_bits = unfolded_bits * levels[0].f_fall
    entity_credit = settled_credit(levels)

    credit_bits_total = sum(c * lvl.width for c, lvl in zip(entity_credit, levels))
    size_measure = (fabric_bits + credit_bits_total) / n_bits

    structure_bits_total = sum(lvl.structure_bits for lvl in levels)
    total_accounted = fabric_bits + structure_bits_total + credit_bits_total
    conservation_max_error = float(np.max(np.abs(total_accounted - unfolded_bits)))

    return SimulationResult(
        t_bf=t_bf,
        t_bf_max=resolved_t_bf_max,
        t_today=t_today,
        k_rate=resolved_k_rate,
        entropy=entropy,
        unfolded_bits=unfolded_bits,
        fabric_bits=fabric_bits,
        levels=levels,
        entity_credit=entity_credit,
        size_measure=size_measure,
        conservation_max_error=conservation_max_error,
    )


# ===========================================================================
# WORLDLINES
# ===========================================================================

def build_worldlines(
    structure_count: FloatArray, n_slots: int, seed: int
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Assign a fixed set of comoving slots and activate a fraction over time.

    Args:
        structure_count: Structure abundance over time for one scale.
        n_slots: Number of comoving worldline slots to allocate.
        seed: RNG seed, so each scale gets an independent, reproducible
            slot assignment.

    Returns:
        A tuple ``(y_comoving, active_mask)`` where ``y_comoving`` are the
        slots' fixed comoving y-positions and ``active_mask[t, i]`` is
        whether slot ``i`` is active at time-step ``t``.
    """
    rng = np.random.default_rng(seed)
    y_comoving = np.linspace(-0.5, 0.5, n_slots)
    slot_rank = rng.permutation(n_slots)
    frac = structure_count / max(structure_count.max(), 1e-12)
    active_count = np.clip((frac * n_slots).astype(int), 0, n_slots)
    active_mask = slot_rank[None, :] < active_count[:, None]
    return y_comoving, active_mask


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_results(sim: SimulationResult, n_bits: float, slots_per_scale: int, output_path: str) -> None:
    """Render the worldline and metric panels and save to disk.

    Args:
        sim: Simulation output from :func:`run_simulation`.
        n_bits: Total bitstring length, used for axis annotation.
        slots_per_scale: Number of worldline slots to draw per scale.
        output_path: File path to save the figure to.
    """
    t_bf = sim.t_bf
    envelope = sim.fabric_bits / n_bits
    cmap = plt.get_cmap("plasma")
    colors = [cmap(0.15 + 0.7 * i / max(len(sim.levels) - 1, 1)) for i in range(len(sim.levels))]

    tick_years = [1e-40, 1e-30, 1e-20, 1e-10, 1e-4, 1e3, 1e9, 13.8e9]
    tick_tbf = [years_to_tbf(t, sim.t_today) for t in tick_years]
    tick_labels = ["10\u207b\u2074\u2070", "10\u207b\u00b3\u2070", "10\u207b\u00b2\u2070",
                   "10\u207b\u00b9\u2070", "10\u207b\u2074", "10\u00b3", "10\u2079", "now"]
    valid = [(tb, lb) for tb, lb in zip(tick_tbf, tick_labels) if 0 <= tb <= sim.t_bf_max]
    tick_tbf_v, tick_labels_v = zip(*valid) if valid else ([], [])

    fig, (ax_st, ax_met) = plt.subplots(1, 2, figsize=(17, 7))
    scale_str = ", ".join(f"{lvl.width:g}" for lvl in sim.levels)
    fig.suptitle(
        "Emergent Structure from a Relaxing Bitstring\n"
        f"n={n_bits:g}  scales=[{scale_str}] bits  |  conservation error={sim.conservation_max_error:.2e}",
        fontsize=10, fontweight="bold",
    )

    # --- Left panel: worldlines ---
    ax_st.set_facecolor("#020205")
    ax_st.set_xlabel("Physical time (years, log scale via t\u2192ln t)")
    ax_st.set_ylabel("Comoving y \u00d7 fabric(t)/n")
    ax_st.set_xticks(tick_tbf_v)
    ax_st.set_xticklabels(tick_labels_v, fontsize=7)

    ax_st.fill_between(t_bf, -envelope / 2, envelope / 2, color="gainsboro", alpha=0.15)
    ax_st.plot(t_bf, envelope / 2, color="white", lw=2.2, alpha=0.9)
    ax_st.plot(t_bf, -envelope / 2, color="white", lw=2.2, alpha=0.9)

    for level, color in zip(sim.levels, colors):
        y_slots, active = build_worldlines(level.structure_count, slots_per_scale, seed=int(level.width))
        for i, y0 in enumerate(y_slots):
            mask = active[:, i]
            if mask.any():
                ax_st.plot(t_bf[mask], y0 * envelope[mask], color=color, lw=0.6, alpha=0.35)

    ax_st.plot([], [], color="white", lw=2, label="fabric (scale-0 falling)")
    for level, color in zip(sim.levels, colors):
        ax_st.plot([], [], color=color, lw=1.2, label=f"structure, scale w={level.width:g} (bump)")
    ax_st.axvline(sim.t_today, color="lime", lw=1.5, ls="--", alpha=0.85, label=f"now (t={sim.t_today:g})")
    ax_st.legend(loc="upper left", facecolor="#111115", edgecolor="gray", labelcolor="white", fontsize=8)
    ax_st.text(0.02, 0.02, f"2^{n_bits:.0f} \u2248 10^{n_bits * np.log10(2.0):.1f} states at saturation",
               transform=ax_st.transAxes, color="gray", fontsize=7)

    # --- Right panel: metrics ---
    ax_met.set_facecolor("#0a0a0a")
    ax_met.set_xlabel("Physical time (years, log scale)")
    ax_met.set_ylabel("Fraction of n (normalized)")
    ax_met.set_xticks(tick_tbf_v)
    ax_met.set_xticklabels(tick_labels_v, fontsize=7)

    ax_met.plot(t_bf, sim.entropy, color="red", lw=1.5, ls="--", label="S(t) = H(p(t))  [entropy/bit]")
    ax_met.plot(t_bf, sim.fabric_bits / n_bits, color="white", lw=3.0, label="fabric(t)/n")
    ax_met.plot(t_bf, sim.size_measure, color="orange", lw=1.5, label="size measure (fabric+condensed)/n")

    for level, color in zip(sim.levels, colors):
        norm = level.structure_count / max(level.structure_count.max(), 1e-12)
        ax_met.plot(t_bf, norm, color=color, lw=1.2, label=f"structure w={level.width:g} [norm, bump]")

    ax_met.set_ylim(-0.05, 1.15)
    ax_met.axvline(sim.t_today, color="lime", lw=1.5, ls="--", alpha=0.85)
    ax_met.legend(loc="upper left", fontsize=7, facecolor="#111115", edgecolor="gray", labelcolor="white")
    ax_met.set_title(
        f"k_rate={sim.k_rate:.5f}  t_bf_max={sim.t_bf_max:.0f}  conservation error={sim.conservation_max_error:.2e}",
        fontsize=7,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="white")
    print(f"Saved \u2192 {output_path}")
    print(f"Conservation check (fabric + all structure + all settled credit vs unfolded), "
          f"max abs error: {sim.conservation_max_error:.3e}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_scales(raw: str) -> list[float]:
    """Parse a comma-separated list of coherence scales from the CLI.

    Args:
        raw: Comma-separated widths, e.g. ``"6,12,20"``.

    Returns:
        The parsed widths as floats, in the given order.
    """
    return [float(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Parameter-free emergent-structure demonstrator: a relaxing "
                    "bitstring decomposed, at any set of coherence scales, into "
                    "exact falling/bump/rising fractions."
    )
    parser.add_argument("--n_bits", type=float, default=184.0)
    parser.add_argument("--t_bf_max", type=float, default=None, help="Default: n*ln(n)")
    parser.add_argument("--sat_fraction", type=float, default=0.99)
    parser.add_argument("--k_rate", type=float, default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--t_today", type=float, default=74.0, help="Bit-flip ticks = present day.")
    parser.add_argument(
        "--scales", type=str, default="6,12,20",
        help="Comma-separated coherence scales (window widths, in bits), "
             "finest to coarsest.",
    )
    parser.add_argument("--slots", type=int, default=50, help="Worldline slots per scale.")
    parser.add_argument("--output", type=str, default="emergent_structure.png")

    args = parser.parse_args()
    scales = parse_scales(args.scales)

    sim = run_simulation(
        n_bits=args.n_bits,
        scales=scales,
        steps=args.steps,
        t_bf_max=args.t_bf_max,
        sat_fraction=args.sat_fraction,
        k_rate=args.k_rate,
        t_today=args.t_today,
    )
    plot_results(sim, n_bits=args.n_bits, slots_per_scale=args.slots, output_path=args.output)


if __name__ == "__main__":
    main()
