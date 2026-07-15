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


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_results(sim: SimulationResult, slots_per_scale: int, output_path: str) -> None:
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
        "Emergent Structure from a Relaxing Bitstring — Multi-Clock (FIXED units)\n"
        f"n={n_bits:g}  scales=[{scale_str}] bits  |  k_rate={sim.k_rate:g} "
        f"(TRUE Ehrenfest={TRUE_K_RATE:g})  |  conservation error={sim.conservation_max_error:.2e}"
        f"  |  now: t={sim.t_today:.2f} ({now_note})  |  p-saturation at t_bf_max: {sim.saturation_reached:.4f}",
        fontsize=9, fontweight="bold",
    )

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

    ax_met.set_facecolor("#0a0a0a")
    ax_met.set_xlabel("Physical time (years, log scale)")
    ax_met.set_ylabel("Fraction of n (normalized)")
    ax_met.set_xticks(tick_tbf_v)
    ax_met.set_xticklabels(tick_labels_v, fontsize=7)
    ax_met.plot(t_bf, sim.entropy, color="cyan", lw=1.5, ls=":", label="entropy = S(t)")
    ax_met.plot(t_bf, fabric_env, color="gray", lw=1.2, ls="--", label="fabric(t)/n")
    ax_met.plot(t_bf, sim.size_measure, color="orange", lw=2.2, label="size_measure")
    for level, color in zip(sim.levels, colors):
        norm = level.structure_count / max(level.structure_count.max(), 1e-12)
        ax_met.plot(t_bf, norm, color=color, lw=1.2, label=f"structure w={level.width:g} [norm]")
    ax_met.set_ylim(-0.05, 1.15)
    ax_met.axvline(sim.t_today, color="lime", lw=1.5, ls="--", alpha=0.85)
    ax_met.legend(loc="upper left", fontsize=7, facecolor="#111115", edgecolor="gray", labelcolor="white")

    ax_ledger.set_facecolor("#0a0a0a")
    ax_ledger.set_xlabel("Physical time (years, log scale)")
    ax_ledger.set_ylabel("Own local ticks elapsed (normalized)")
    ax_ledger.set_xticks(tick_tbf_v)
    ax_ledger.set_xticklabels(tick_labels_v, fontsize=7)
    for level, color in zip(sim.levels, colors):
        own_ticks = np.floor(t_bf / level.width)
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

    ax_matter.set_facecolor("#0a0a0a")
    ax_matter.set_xlabel("Physical time (years, log scale)")
    ax_matter.set_ylabel("Entity count (number of emergent structures)")
    ax_matter.set_xticks(tick_tbf_v)
    ax_matter.set_xticklabels(tick_labels_v, fontsize=7)
    total_structure_count = sum(lvl.structure_count for lvl in sim.levels)
    ax_matter.plot(t_bf, total_structure_count, color="gray", lw=1.4, ls=":",
                   label="raw structure_count (uncorrected)")
    ax_matter.plot(t_bf, sim.total_matter, color="gold", lw=2.4,
                   label=f"total matter = structure \u00d7 \u03b7(\u03c4)^{sim.matter_power:g}")
    for level, matter, color in zip(sim.levels, sim.per_scale_matter, colors):
        ax_matter.plot(t_bf, matter, color=color, lw=1.0, alpha=0.8, label=f"matter w={level.width:g}")
    peak_idx = int(np.argmax(sim.total_matter))
    ax_matter.axvline(sim.t_today, color="lime", lw=1.5, ls="--", alpha=0.85,
                      label=f"now = peak matter (t={sim.t_today:.1f})" if sim.t_today_auto else "now (manual)")
    ax_matter.plot(t_bf[peak_idx], sim.total_matter[peak_idx], "o", color="lime", ms=6)
    ax_matter.legend(loc="upper right", fontsize=6.5, facecolor="#111115", edgecolor="gray", labelcolor="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="white")
    print(f"Saved \u2192 {output_path}")
    print(f"Telescoping conservation check, max abs error: {sim.conservation_max_error:.3e}")
    print(f"k_rate used: {sim.k_rate:g}  (true Ehrenfest value: {TRUE_K_RATE:g})")
    print(f"p-saturation reached at t_bf_max: {sim.saturation_reached:.6f} (1.0 = fully equilibrated)")
    for level in sim.levels:
        distinct = len(np.unique(level.tau_local))
        print(f"  scale w={level.width:g}: lapse=1/{level.width:g}={level.lapse:.4f}, "
              f"distinct retarded ticks over run = {distinct} (out of {len(sim.t_bf)} grid points), "
              f"max pending/n={float(np.max(level.pending))/sim.n_bits:.4e}")
    now_source = "auto-detected (peak of total matter)" if sim.t_today_auto else "manual override"
    print(f"t_today = {sim.t_today:.4f} [{now_source}]")


# ===========================================================================
# CLI
# ===========================================================================

def parse_scales(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-clock, GR-compatible emergent-structure demonstrator (FIXED units)."
    )
    parser.add_argument("--n_bits", type=float, default=184.0)
    parser.add_argument("--t_bf_max", type=float, default=None,
                         help="Max raw bit-flip time, in units of n. Default: ln(n). "
                              "Pure display-range choice, decoupled from k_rate.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--t_today", type=float, default=None)
    parser.add_argument("--matter_power", type=float, default=1.0)
    parser.add_argument("--scales", type=str, default="6,12,20")
    parser.add_argument("--slots", type=int, default=50)
    parser.add_argument("--output", type=str, default="emergent_structure_relativistic.png")

    args = parser.parse_args()
    scales = parse_scales(args.scales)

    sim = run_simulation(
        n_bits=args.n_bits, scales=scales, steps=args.steps, t_bf_max=args.t_bf_max,
        t_today=args.t_today,
        matter_power=args.matter_power,
    )
    plot_results(sim, slots_per_scale=args.slots, output_path=args.output)


if __name__ == "__main__":
    main()
