"""
Emergent Structure from a Relaxing Bitstring — Multi-Clock (GR-Compatible) Version
====================================================================================

A parameter-free model of structure formation, built from nothing but a
bitstring length ``n``, now with genuine per-scale proper time.

Model
-----
A length-``n`` bitstring starts all-zero (the unique, minimal-description
state, zero Shannon entropy) and relaxes under the symmetric Ehrenfest
process: at each raw tick, one of the ``n`` positions is chosen uniformly
and toggled. Each bit is then independently ``1`` with probability

    p(tau) = 0.5 * (1 - exp(-2 * tau)),      tau = raw_tick_count / n

a closed-form, ``n``-independent relaxation curve. There is no external
clock: an emergent internal observer experiences "time" as the minimal
change between two configurations it passes through -- one bit flip. ``tau``
itself is not anyone's experienced time; it is the bookkeeping coordinate
of the raw substrate, in the same sense that Schwarzschild coordinate time
is not any local observer's proper time.

At bit-flip time ``tau`` the whole ``n``-bit string has a single, definite
number of ones, ``k(tau) = round(n * p(tau))`` -- a fixed finite population.
A window of ``w`` consecutive bits is a sample of ``w`` positions drawn
*without replacement* from that population, so its ones-count is exactly
``Hypergeometric(n, k(tau), w)``. Classifying that count against ``w/2``
splits every window composition into three mutually exclusive, exhaustive
shape families, with no fitted parameters:

    falling  (K = 0)        the window is still exactly all-zero
    bump     (0 < K < w/2)  a genuine, transient interior structure
    rising   (K >= w/2)     the window has become majority-flipped

Multi-clock extension: the counting equation as a lapse function
------------------------------------------------------------------
Every earlier version of this model gave all scales the same instantaneous
``tau`` -- a width-20 structure and a width-6 structure were both evaluated
at the identical instant of raw time, with no penalty for being bigger.
That is inconsistent with the model's own counting-equation assumption: it
takes as many raw bit-flips to drive a width-``w`` structure forward by one
unit of *its own* time as it took bits to encode it in space. A structure
that took ``w`` bits to encode is blind to anything finer than ``w`` raw
flips; its own clock only ticks once per ``w`` raw ticks.

That gives each scale an explicit **lapse function**

    N_w = 1 / w        (local proper ticks per raw tick)

-- structurally the same role a lapse/redshift factor plays in GR: a
"heavier" (larger, more-bit-encoded) structure runs its own clock slower
relative to the background coordinate, exactly the correspondence this
framework's mass candidate already points at (mass as persistence cost /
repair-rate against Ehrenfest background corruption -- a bigger structure
needs more raw flips per unit of its own proper time to keep existing,
which is the same currency as a slower clock).

Concretely, a scale of width ``w`` only re-samples the string (recomputes
its own fall/bump/rise classification) at the raw instants

    tau_w(tau) = (w/n) * floor(tau * n / w)

-- a staircase, not a continuous function of ``tau``. Between its own
ticks it is frozen: literally, not all worldlines saturate at the same
rate. Large-``w`` structures age in long, infrequent jumps; small-``w``
structures age almost continuously. This is genuine differential aging,
not a plotting artifact.

Conservation, restated as a retarded (causal) telescoping identity
---------------------------------------------------------------------
Giving every scale its own clock breaks the *naive* same-instant balance
sheet (fabric + all structure + all settled credit == unfolded_bits at
one shared tau) -- not because bits get lost, but because a slow-clocked
scale is, at any given raw instant, reporting on a retarded snapshot. The
substrate keeps unfolding entropy continuously in between; the slow scale
simply hasn't been "informed" yet. That backlog is real, non-negative
bit-mass, tracked here explicitly as each scale's **pending** term:

    pending_i(tau) = pool_true_i(tau) - pool_eff_i(tau)

where ``pool_true_i`` is the continuous input arriving from the scale
below (or the raw substrate, for the finest scale) and ``pool_eff_i`` is
that same input evaluated at THIS scale's own retarded instant -- i.e.
what it has actually had the resolution to see so far.

With ``pending`` included, each scale's incoming pool splits exactly
three ways plus the backlog:

    pool_true_i  =  pending_i  +  fabric_i  +  structure_i  +  promoted_i

and this telescopes, level by level, all the way back to the ground
truth:

    unfolded_bits(tau)  =  sum_i [pending_i(tau) + fabric_i(tau)
                                  + structure_i(tau)]  +  promoted_last(tau)

an EXACT identity at every continuous raw ``tau`` (verified numerically
below, to floating-point precision) -- conservation restored, now as a
statement about retarded/causal bookkeeping across observers with
different clock rates, rather than a same-instant snapshot. This is the
same move GR makes: covariant local conservation survives even though
different observers disagree about simultaneity.

``S(tau) = log2 C(n, k(tau)) / n`` is the exact combinatorial entropy per
bit (log-count of equally likely arrangements at ones-count k), not the
per-bit binary-entropy approximation ``H(p(tau))``; the two coincide only
in the large-n Stirling limit, and this script always uses the exact form.
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


# ===========================================================================
# TIME-AXIS MAPPING (for plotting only)
# ===========================================================================

def years_to_tbf(t_yr: FloatArray | float, t_today: float) -> FloatArray | float:
    """Map physical years onto raw bit-flip coordinate time, log-uniformly.

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
# EHRENFEST BIT-FLIP PRIMITIVES (ground truth, substrate-level -- no observer)
# ===========================================================================

def p_of_tau(tau: FloatArray, k_rate: float) -> FloatArray:
    """Per-bit probability of having flipped 0->1 by raw bit-flip time ``tau``.

    Args:
        tau: Raw bit-flip coordinate time (raw tick count / n).
        k_rate: Overall relaxation rate constant.

    Returns:
        The per-bit flip probability at each ``tau``, in ``[0, 0.5)``.
    """
    return 0.5 * (1.0 - np.exp(-k_rate * tau))


def ones_count_from_p(p: FloatArray, n_bits: int) -> FloatArray:
    """Map the mean-field flip probability onto an integer ones-count ``k``.

    Args:
        p: Per-bit flip probability (scalar or array), in ``[0, 0.5)``.
        n_bits: Total bitstring length ``n``.

    Returns:
        Integer-valued ones-counts, clipped to ``[0, n_bits]``.
    """
    return np.clip(np.round(n_bits * p), 0, n_bits)


def combinatorial_entropy_bits(n_bits: int, k: FloatArray) -> FloatArray:
    """Exact combinatorial Shannon entropy per bit, ``log2 C(n,k) / n``.

    Args:
        n_bits: Total bitstring length ``n``.
        k: Ones-count(s), in ``[0, n_bits]``.

    Returns:
        Exact combinatorial entropy per bit, in ``[0, 1]``.
    """
    k = np.clip(np.round(k), 0, n_bits)
    log2_comb = (gammaln(n_bits + 1) - gammaln(k + 1) - gammaln(n_bits - k + 1)) / np.log(2.0)
    return log2_comb / n_bits


def ground_truth_pool(
    tau: FloatArray, n_bits: int, k_rate: float
) -> tuple[FloatArray, FloatArray]:
    """The substrate's own entropy-unfolded bit budget -- no observer involved.

    This is the one genuinely continuous, frame-independent quantity in the
    whole model: the actual current macrostate of the single bitstring. Every
    emergent scale below samples a retarded view of THIS.

    Args:
        tau: Raw bit-flip coordinate time, scalar or array.
        n_bits: Total bitstring length ``n``.
        k_rate: Overall relaxation rate constant.

    Returns:
        A tuple ``(unfolded_bits, k)`` -- the continuous bit budget
        ``n * S(tau)`` and the underlying integer ones-count.
    """
    p = p_of_tau(tau, k_rate)
    k = ones_count_from_p(p, n_bits)
    return n_bits * combinatorial_entropy_bits(n_bits, k), k


def order_parameter(tau: FloatArray, k_rate: float) -> FloatArray:
    """Departure-from-equilibrium order parameter, ``eta(tau) = 1 - 2*p(tau)``.

    Equals ``exp(-k_rate * tau)`` exactly (no new free parameter -- it falls
    straight out of the closed-form relaxation curve). ``eta=1`` at ``tau=0``
    (pure vacuum: every bit's marginal is still 0, so any window not
    matching the origin is genuinely distinguishable structure). ``eta->0``
    as ``tau->inf`` (the population's own mean has reached w/2 too, so a
    window landing below vs at/above w/2 is a coin flip around a background
    that is *already* homogeneous -- no longer a meaningful signal).

    Args:
        tau: Raw bit-flip coordinate time, scalar or array.
        k_rate: Overall relaxation rate constant.

    Returns:
        The order parameter, in ``(0, 1]``.
    """
    return np.exp(-k_rate * tau)


def family_fractions_exact(
    n_bits: int, k: FloatArray, width: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Exact falling/bump/rising probability mass fractions for one window.

    Args:
        n_bits: Total bitstring length ``n`` (population size).
        k: Ones-count(s) in the full string, in ``[0, n_bits]``.
        width: Window width in bits (number of draws, without replacement).

    Returns:
        A tuple ``(f_fall, f_bump, f_rise)`` matching the shape of ``k``:
          - ``f_fall``: ``P(J = 0)``.
          - ``f_bump``: ``P(0 < J < width/2)``.
          - ``f_rise``: ``P(J >= width/2)``.
    """
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

def retard(tau: FloatArray, width: float, n_bits: int) -> FloatArray:
    """Quantize ``tau`` down to this scale's own tick resolution.

    The counting-equation / lapse assumption: a structure encoded in
    ``width`` bits needs ``width`` raw flips to advance by one unit of its
    own proper time, so it is blind to anything finer. This is the
    ``N_w = 1/w`` lapse function made concrete as a staircase in raw tau.

    Args:
        tau: Raw bit-flip coordinate time, scalar or array.
        width: This scale's window width in bits.
        n_bits: Total bitstring length ``n``.

    Returns:
        ``tau`` floored to the nearest multiple of ``width / n_bits``.
    """
    step = width / n_bits
    return step * np.floor(tau / step)


@dataclass
class ScaleResult:
    """Retarded fall/bump/rise decomposition and ledger for one scale.

    Attributes:
        width: Window width in bits for this scale.
        lapse: This scale's clock rate relative to raw tau, ``1/width``.
        tau_local: This scale's own retarded (staircase) time.
        pool_true: Continuous incoming bit budget (from the scale below,
            or the substrate for the finest scale) -- what has actually
            arrived by raw tau.
        pool_eff: That same incoming budget as THIS scale has actually
            seen it -- i.e. evaluated at its own last tick.
        f_fall, f_bump, f_rise: Retarded family fractions at this scale.
        fabric: Untouched (falling) bit-mass at this scale's own tick.
        structure: Transient (bump) bit-mass visible at this scale.
        promoted: Condensed (rise) bit-mass passed to the next scale.
        pending: ``pool_true - pool_eff`` -- bit-mass that has arrived at
            the substrate/lower-scale level but that this scale has not
            yet had the temporal resolution to register. Always >= 0.
        structure_count: ``structure / width``, in entity units.
    """

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
    levels: list[ScaleResult], tau: FloatArray, k_rate: float, matter_power: float
) -> tuple[FloatArray, list[FloatArray], FloatArray, list[FloatArray]]:
    """Reweight each scale's structure by the order parameter to get real matter.

    This does NOT touch the exact bit-conservation cascade -- matter is a
    diagnostic subset of the already-conserved ``structure`` bit-mass,
    representing the portion of it that is genuinely distinguishable from a
    homogenized background rather than an artifact of a static K=w/2
    threshold. The complement, ``structure*(1-eta**power)``, isn't lost --
    it has become indistinguishable background/radiation, and is released
    back toward the fabric/size budget (see how ``size_measure`` uses
    ``total_matter_bits``, not just ``pending``).

    Bit-mass units (not entity counts) are the primary output here, because
    ``size_measure`` needs to subtract matter from ``unfolded_bits`` --
    both must be in the same (bit) units for the counting equation to hold.
    Entity counts ("number of emergent structures") are derived from the
    bit-mass version by dividing through by each scale's own width.

    Args:
        levels: Scale hierarchy as returned by :func:`build_scale_hierarchy`.
        tau: Raw bit-flip coordinate time grid matching ``levels``.
        k_rate: Overall relaxation rate constant.
        matter_power: Exponent ``p`` in ``eta(tau)**p``; higher values make
            the matter->radiation handoff sharper.

    Returns:
        A tuple ``(total_matter_bits, per_scale_matter_bits,
        total_matter_count, per_scale_matter_count)``.
    """
    eta = order_parameter(tau, k_rate)
    weight = eta ** matter_power
    per_scale_bits = [lvl.structure * weight for lvl in levels]
    total_bits = sum(per_scale_bits)
    per_scale_count = [mb / lvl.width for mb, lvl in zip(per_scale_bits, levels)]
    total_count = sum(per_scale_count)
    return total_bits, per_scale_bits, total_count, per_scale_count


def level_state(
    tau: FloatArray,
    level_idx: int,
    widths: Sequence[float],
    n_bits: int,
    k_rate: float,
    cache: dict | None = None,
) -> ScaleResult:
    """Recursively compute one scale's retarded state at the given tau.

    Scale 0 reads a retarded view of the substrate's own continuous
    entropy-unfolding; scale i>0 reads a retarded view of scale (i-1)'s
    continuous "promoted" output. Every quantity here is a pure, exactly
    reproducible function of ``tau`` -- there is no lookup/interpolation,
    only closed-form re-evaluation at whatever instant is requested, which
    is what lets each scale be queried at both its own retarded instant
    and (recursively) at a coarser scale's retarded instant.

    Args:
        tau: Raw bit-flip coordinate time, scalar or array, at which to
            evaluate this scale.
        level_idx: Index into ``widths`` for this scale (0 = finest).
        widths: All coherence scales, finest to coarsest.
        n_bits: Total bitstring length ``n``.
        k_rate: Overall relaxation rate constant.
        cache: Optional memoization dict (recursion revisits the same
            (level, tau) pairs); pass ``{}`` from the top-level caller.

    Returns:
        This scale's :class:`ScaleResult` at the requested ``tau``.
    """
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


def build_scale_hierarchy(
    tau: FloatArray, scales: Sequence[float], n_bits: int, k_rate: float
) -> list[ScaleResult]:
    """Evaluate every scale's retarded state on a shared raw-tau grid.

    Args:
        tau: Raw bit-flip coordinate time grid.
        scales: Window widths in bits, finest to coarsest.
        n_bits: Total bitstring length ``n``.
        k_rate: Overall relaxation rate constant.

    Returns:
        One :class:`ScaleResult` per entry in ``scales``, in the same order.
    """
    cache: dict = {}
    return [
        level_state(tau, i, scales, n_bits, k_rate, cache)
        for i in range(len(scales))
    ]


# ===========================================================================
# SIMULATION
# ===========================================================================

@dataclass
class SimulationResult:
    """Full output of one run of :func:`run_simulation`.

    Attributes:
        t_bf: Raw bit-flip coordinate-time grid.
        t_bf_max: Maximum raw bit-flip time simulated.
        t_today: Raw bit-flip time identified with the present day.
        k_rate: Relaxation rate constant used.
        n_bits: Total bitstring length ``n``.
        unfolded_bits: Ground-truth continuous bit budget, ``n * S(tau)``.
        entropy: Exact combinatorial Shannon entropy per bit.
        levels: Per-scale retarded decomposition, see :class:`ScaleResult`.
        size_measure: Actual "size of the universe" -- the counting-equation
            budget minus both the retardation backlog and whatever is still
            tied up as live, distinguishable matter,
            ``(unfolded_bits - sum(pending) - total_matter_bits) / n``. As
            matter decays this converges onto ``entropy`` itself (the pure,
            structure-free relaxed curve) -- no matter, no contraction.
        conservation_max_error: Max abs deviation of the telescoping
            identity ``sum(pending+fabric+structure) + promoted_last``
            from ``unfolded_bits``; should be at floating-point precision.
        matter_power: Exponent used to reweight structure into matter.
        total_matter: Order-parameter-weighted total structure count --
            the physically honest "number of emergent structures", which
            now actually vanishes as the string saturates.
        per_scale_matter: Same, broken out per scale.
        t_today_auto: Whether ``t_today`` was auto-set to the peak of
            ``total_matter`` (True) or taken from an explicit override
            (False).
    """

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


def run_simulation(
    n_bits: float,
    scales: Sequence[float],
    steps: int = 3000,
    t_bf_max: float | None = None,
    sat_fraction: float = 0.99,
    k_rate: float | None = None,
    t_today: float | None = None,
    matter_power: float = 1.0,
) -> SimulationResult:
    """Run the full multi-clock entropy-unfolding and scale-decomposition.

    Args:
        n_bits: Total bitstring length ``n``.
        scales: Coherence scales (window widths, in bits), finest to
            coarsest. Order matters: each scale's own lapse (1/width)
            should be non-increasing as widths grow.
        steps: Number of raw-tau grid points.
        t_bf_max: Maximum raw bit-flip time, in units of ``n_bits``.
            Defaults to ``ln(n_bits)``, the Ehrenfest mixing timescale.
        sat_fraction: Fraction of equilibrium saturation reached by
            ``t_bf_max``, used to calibrate ``k_rate`` when not given.
        k_rate: Relaxation rate constant. Derived from ``sat_fraction``
            and ``t_bf_max`` if not given.
        t_today: Raw bit-flip time identified with the present day. If
            ``None`` (default), auto-set to the raw tau where the total
            order-parameter-weighted matter content peaks -- i.e. "now"
            is defined as the instant where the number of emergent
            structures supporting an observer is highest, not an
            arbitrary fixed tick.
        matter_power: Exponent ``p`` in ``eta(tau)**p`` used to reweight
            structure into physically honest matter (see
            :func:`matter_content`).

    Returns:
        A populated :class:`SimulationResult`.
    """
    resolved_t_bf_max = (
        t_bf_max * n_bits if t_bf_max is not None else n_bits * np.log(n_bits)
    )
    resolved_k_rate = (
        k_rate if k_rate is not None else -np.log(1.0 - sat_fraction) / resolved_t_bf_max
    )
    n_int = int(round(n_bits))

    t_bf = np.linspace(1e-9, resolved_t_bf_max, steps)
    unfolded_bits, k_ones = ground_truth_pool(t_bf, n_int, resolved_k_rate)
    entropy = unfolded_bits / n_int

    levels = build_scale_hierarchy(t_bf, scales, n_int, resolved_k_rate)

    pending_total = sum(lvl.pending for lvl in levels)
    fabric_total = sum(lvl.fabric for lvl in levels)
    structure_total = sum(lvl.structure for lvl in levels)
    promoted_last = levels[-1].promoted

    total_accounted = pending_total + fabric_total + structure_total + promoted_last
    conservation_max_error = float(np.max(np.abs(total_accounted - unfolded_bits)))

    total_matter_bits, per_scale_matter_bits, total_matter, per_scale_matter = matter_content(
        levels, t_bf, resolved_k_rate, matter_power
    )

    # Size is the counting-equation budget MINUS whatever is still tied up as
    # live, distinguishable matter (bit units, commensurate with unfolded_bits)
    # -- not just minus the retardation backlog. As matter decays (eta -> 0),
    # this converges onto the pure relaxed-entropy curve: no structures, no
    # contraction, exactly the "no matter at all" limit.
    size_measure = (unfolded_bits - pending_total - total_matter_bits) / n_bits

    t_today_auto = t_today is None
    resolved_t_today = float(t_bf[int(np.argmax(total_matter))]) if t_today_auto else t_today

    return SimulationResult(
        t_bf=t_bf, t_bf_max=resolved_t_bf_max, t_today=resolved_t_today, k_rate=resolved_k_rate,
        n_bits=n_int, unfolded_bits=unfolded_bits, entropy=entropy, levels=levels,
        size_measure=size_measure, conservation_max_error=conservation_max_error,
        matter_power=matter_power, total_matter=total_matter, per_scale_matter=per_scale_matter,
        t_today_auto=t_today_auto,
    )


# ===========================================================================
# WORLDLINES
# ===========================================================================

def build_worldlines(
    structure_count: FloatArray, n_slots: int, seed: int
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Assign a fixed set of comoving slots and activate a fraction over time.

    Because ``structure_count`` is now a retarded staircase (see
    :class:`ScaleResult`), the resulting worldlines step and hold rather
    than track continuously -- this is where differential aging becomes
    visible: coarse scales' worldlines update in long, infrequent jumps.

    Args:
        structure_count: Structure abundance over time for one scale.
        n_slots: Number of comoving worldline slots to allocate.
        seed: RNG seed, so each scale gets an independent, reproducible
            slot assignment.

    Returns:
        A tuple ``(y_comoving, active_mask)``.
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

def plot_results(sim: SimulationResult, slots_per_scale: int, output_path: str) -> None:
    """Render worldline, metric, and proper-time-ledger panels; save to disk.

    Args:
        sim: Simulation output from :func:`run_simulation`.
        slots_per_scale: Number of worldline slots to draw per scale.
        output_path: File path to save the figure to.
    """
    t_bf = sim.t_bf
    n_bits = sim.n_bits
    fabric_env = sum(lvl.fabric for lvl in sim.levels) / n_bits
    cmap = plt.get_cmap("plasma")
    colors = [cmap(0.15 + 0.7 * i / max(len(sim.levels) - 1, 1)) for i in range(len(sim.levels))]

    tick_years = [1e-40, 1e-30, 1e-20, 1e-10, 1e-4, 1e3, 1e9, 13.8e9]
    tick_tbf = [years_to_tbf(t, sim.t_today) for t in tick_years]
    tick_labels = ["10\u207b\u2074\u2070", "10\u207b\u00b3\u2070", "10\u207b\u00b2\u2070",
                   "10\u207b\u00b9\u2070", "10\u207b\u2074", "10\u00b3", "10\u2079", "now"]
    valid = [(tb, lb) for tb, lb in zip(tick_tbf, tick_labels) if 0 <= tb <= sim.t_bf_max]
    tick_tbf_v, tick_labels_v = zip(*valid) if valid else ([], [])

    fig, ((ax_st, ax_met), (ax_ledger, ax_matter)) = plt.subplots(2, 2, figsize=(17, 13))
    scale_str = ", ".join(f"{lvl.width:g}" for lvl in sim.levels)
    now_note = "auto = peak matter" if sim.t_today_auto else "manual override"
    fig.suptitle(
        "Emergent Structure from a Relaxing Bitstring — Multi-Clock (GR-Compatible)\n"
        f"n={n_bits:g}  scales=[{scale_str}] bits  |  conservation error={sim.conservation_max_error:.2e}"
        f"  |  now: t={sim.t_today:.2f} ({now_note})",
        fontsize=10, fontweight="bold",
    )

    # --- Panel 1: worldlines (envelope now IS the actual size measure --
    # it contracts as matter grows and re-expands toward the pure relaxed
    # entropy curve as matter decays) ---
    ax_st.set_facecolor("#020205")
    ax_st.set_xlabel("Physical time (years, log scale via t\u2192ln t)")
    ax_st.set_ylabel("Comoving y \u00d7 size_measure(t)")
    ax_st.set_xticks(tick_tbf_v)
    ax_st.set_xticklabels(tick_labels_v, fontsize=7)

    size_env = sim.size_measure
    ax_st.fill_between(t_bf, -size_env / 2, size_env / 2, color="gainsboro", alpha=0.15)
    ax_st.plot(t_bf, size_env / 2, color="white", lw=2.2, alpha=0.9)
    ax_st.plot(t_bf, -size_env / 2, color="white", lw=2.2, alpha=0.9)
    ax_st.plot(t_bf, sim.entropy / 2, color="cyan", lw=1.0, ls=":", alpha=0.7)
    ax_st.plot(t_bf, -sim.entropy / 2, color="cyan", lw=1.0, ls=":", alpha=0.7)

    for level, matter, color in zip(sim.levels, sim.per_scale_matter, colors):
        # Worldlines are driven by MATTER (order-parameter-weighted structure),
        # not raw structure_count, so they correctly die off as the string
        # saturates rather than persisting at a false plateau.
        y_slots, active = build_worldlines(matter, slots_per_scale, seed=int(level.width))
        for i, y0 in enumerate(y_slots):
            mask = active[:, i]
            if mask.any():
                ax_st.plot(t_bf[mask], y0 * size_env[mask], color=color, lw=0.6, alpha=0.35)

    ax_st.plot([], [], color="white", lw=2, label="size_measure (actual universe size)")
    ax_st.plot([], [], color="cyan", lw=1.0, ls=":", label="entropy (pure relaxed, no-matter limit)")
    for level, color in zip(sim.levels, colors):
        ax_st.plot([], [], color=color, lw=1.2,
                   label=f"matter w={level.width:g} (lapse=1/{level.width:g})")
    ax_st.axvline(sim.t_today, color="lime", lw=1.5, ls="--", alpha=0.85, label=f"now (t={sim.t_today:g})")
    ax_st.legend(loc="upper left", facecolor="#111115", edgecolor="gray", labelcolor="white", fontsize=8)

    # --- Panel 2: metrics ---
    ax_met.set_facecolor("#0a0a0a")
    ax_met.set_xlabel("Physical time (years, log scale)")
    ax_met.set_ylabel("Fraction of n (normalized)")
    ax_met.set_xticks(tick_tbf_v)
    ax_met.set_xticklabels(tick_labels_v, fontsize=7)

    ax_met.plot(t_bf, sim.entropy, color="cyan", lw=1.5, ls=":",
                label="entropy = S(t)  [pure relaxed, no-matter limit]")
    ax_met.plot(t_bf, fabric_env, color="gray", lw=1.2, ls="--", label="fabric(t)/n [untouched-only]")
    ax_met.plot(t_bf, sim.size_measure, color="orange", lw=2.2,
                label="size_measure = (unfolded \u2212 pending \u2212 matter)/n\n(actual universe size)")

    for level, color in zip(sim.levels, colors):
        norm = level.structure_count / max(level.structure_count.max(), 1e-12)
        ax_met.plot(t_bf, norm, color=color, lw=1.2, label=f"structure w={level.width:g} [norm]")

    ax_met.set_ylim(-0.05, 1.15)
    ax_met.axvline(sim.t_today, color="lime", lw=1.5, ls="--", alpha=0.85)
    ax_met.legend(loc="upper left", fontsize=7, facecolor="#111115", edgecolor="gray", labelcolor="white")

    # --- Panel 3: proper-time ledger -- differential aging & pending backlog ---
    ax_ledger.set_facecolor("#0a0a0a")
    ax_ledger.set_xlabel("Physical time (years, log scale)")
    ax_ledger.set_ylabel("Own local ticks elapsed (normalized to raw \u03c4_max/width)")
    ax_ledger.set_xticks(tick_tbf_v)
    ax_ledger.set_xticklabels(tick_labels_v, fontsize=7)

    for level, color in zip(sim.levels, colors):
        own_ticks = np.floor(t_bf * n_bits / level.width)
        own_ticks_norm = own_ticks / max(own_ticks.max(), 1.0)
        ax_ledger.plot(t_bf, own_ticks_norm, color=color, lw=1.6,
                       label=f"proper time, w={level.width:g} (lapse={level.lapse:.3f})")
    ax_ledger.plot(t_bf, t_bf / sim.t_bf_max, color="lime", lw=1.2, ls=":", label="raw coordinate \u03c4 (lapse=1)")

    ax_ledger2 = ax_ledger.twinx()
    for level, color in zip(sim.levels, colors):
        ax_ledger2.plot(t_bf, level.pending / n_bits, color=color, lw=0.9, ls="--", alpha=0.6)
    ax_ledger2.set_ylabel("pending / n (dashed, right axis)", color="gray", fontsize=8)
    ax_ledger2.tick_params(axis='y', labelcolor="gray")

    ax_ledger.axvline(sim.t_today, color="lime", lw=1.5, ls="--", alpha=0.5)
    ax_ledger.legend(loc="upper left", fontsize=7, facecolor="#111115", edgecolor="gray", labelcolor="white")
    ax_ledger.set_title("Differential aging: proper time (solid) vs pending backlog (dashed)", fontsize=8)

    # --- Panel 4: matter vs raw structure -- the fix itself, made visible ---
    ax_matter.set_facecolor("#0a0a0a")
    ax_matter.set_xlabel("Physical time (years, log scale)")
    ax_matter.set_ylabel("Entity count (number of emergent structures)")
    ax_matter.set_xticks(tick_tbf_v)
    ax_matter.set_xticklabels(tick_labels_v, fontsize=7)

    total_structure_count = sum(lvl.structure_count for lvl in sim.levels)
    ax_matter.plot(t_bf, total_structure_count, color="gray", lw=1.4, ls=":",
                   label="raw structure_count (bump fraction, uncorrected)\n"
                         "-- artifact: never actually vanishes")
    ax_matter.plot(t_bf, sim.total_matter, color="gold", lw=2.4,
                   label=f"total matter = structure \u00d7 \u03b7(\u03c4)^{sim.matter_power:g}\n"
                         "(\u03b7 = departure from equilibrium, \u21920 as string saturates)")
    for level, matter, color in zip(sim.levels, sim.per_scale_matter, colors):
        ax_matter.plot(t_bf, matter, color=color, lw=1.0, alpha=0.8,
                       label=f"matter w={level.width:g}")

    peak_idx = int(np.argmax(sim.total_matter))
    ax_matter.axvline(sim.t_today, color="lime", lw=1.5, ls="--", alpha=0.85,
                      label=f"now = peak matter (t={sim.t_today:.1f})" if sim.t_today_auto else "now (manual)")
    ax_matter.plot(t_bf[peak_idx], sim.total_matter[peak_idx], "o", color="lime", ms=6)
    ax_matter.legend(loc="upper right", fontsize=6.5, facecolor="#111115", edgecolor="gray", labelcolor="white")
    ax_matter.set_title(
        "Matter is a diagnostic reweighting of structure -- the exact bit\n"
        "conservation cascade above is untouched by this panel", fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="white")
    print(f"Saved \u2192 {output_path}")
    print(f"Telescoping conservation check (sum of pending+fabric+structure over all "
          f"scales, plus coarsest scale's promoted mass, vs ground-truth unfolded_bits), "
          f"max abs error: {sim.conservation_max_error:.3e}")
    for level in sim.levels:
        print(f"  scale w={level.width:g}: lapse=1/{level.width:g}={level.lapse:.4f}, "
              f"max pending/n={float(np.max(level.pending))/sim.n_bits:.4e}")
    now_source = "auto-detected (peak of total matter)" if sim.t_today_auto else "manual override"
    print(f"t_today = {sim.t_today:.4f} [{now_source}]")
    print(f"total_matter: start={sim.total_matter[0]:.4f}, peak={sim.total_matter.max():.4f}, "
          f"end={sim.total_matter[-1]:.4f}  (end/peak={sim.total_matter[-1]/max(sim.total_matter.max(),1e-12):.4f})")


# ===========================================================================
# CLI
# ===========================================================================

def parse_scales(raw: str) -> list[float]:
    """Parse a comma-separated list of coherence scales from the CLI.

    Args:
        raw: Comma-separated widths, e.g. ``"6,12,20"``.

    Returns:
        The parsed widths as floats, in the given (finest-to-coarsest) order.
    """
    return [float(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-clock, GR-compatible emergent-structure demonstrator: a "
                    "relaxing bitstring decomposed, at any set of coherence scales, into "
                    "exact retarded falling/bump/rising fractions, each with its own "
                    "proper-time lapse and an explicit pending (in-transit) ledger term "
                    "that restores exact conservation."
    )
    parser.add_argument("--n_bits", type=float, default=184.0)
    parser.add_argument("--t_bf_max", type=float, default=None,
                         help="Max raw bit-flip time, in units of n. Default: ln(n).")
    parser.add_argument("--sat_fraction", type=float, default=0.99)
    parser.add_argument("--k_rate", type=float, default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--t_today", type=float, default=None,
                         help="Raw bit-flip ticks = present day. Default: auto-detected as the "
                              "peak of total matter (where the number of emergent structures "
                              "supporting an observer is highest).")
    parser.add_argument("--matter_power", type=float, default=1.0,
                         help="Exponent p in eta(tau)**p used to reweight structure into "
                              "physically honest matter content. Higher = sharper matter->"
                              "radiation handoff.")
    parser.add_argument(
        "--scales", type=str, default="6,12,20",
        help="Comma-separated coherence scales (window widths, in bits), finest to coarsest.",
    )
    parser.add_argument("--slots", type=int, default=50, help="Worldline slots per scale.")
    parser.add_argument("--output", type=str, default="emergent_structure_relativistic.png")

    args = parser.parse_args()
    scales = parse_scales(args.scales)

    sim = run_simulation(
        n_bits=args.n_bits, scales=scales, steps=args.steps, t_bf_max=args.t_bf_max,
        sat_fraction=args.sat_fraction, k_rate=args.k_rate, t_today=args.t_today,
        matter_power=args.matter_power,
    )
    plot_results(sim, slots_per_scale=args.slots, output_path=args.output)


if __name__ == "__main__":
    main()
