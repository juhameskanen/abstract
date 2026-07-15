"""
Emergent Structure from a Relaxing Bitstring — Multi-Clock (GR-Compatible) Version
====================================================================================

This is your original driver script (plotting + CLI), refactored so all the
core model math lives in `multiclock.py` and is imported here rather than
duplicated. Behavior is unchanged from the original single-file version.
"""

from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from multiclock import (
    TRUE_K_RATE,
    SimulationResult,
    build_worldlines,
    run_simulation,
    years_to_tbf,
)

FloatArray = NDArray[np.float64]


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
