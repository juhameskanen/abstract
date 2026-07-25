"""Plotting/CLI driver for the general complex-wavefunction backend."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multiclock import build_worldlines
from wavefunction_model import (
    CompressedComplexWavefunction,
    PhaseCodecConfig,
    SpectralPhaseCodec,
    run_wavefunction_cosmology,
)


def parse_scales(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one scale is required")
    return values


def _phase_boundaries(result) -> tuple[float, float, float]:
    matter = result.total_matter_bits
    peak_idx = int(np.argmax(matter))
    peak = float(matter[peak_idx])
    indices = np.arange(len(matter))
    rising = np.where((indices < peak_idx) & (matter >= 0.20 * peak))[0]
    falling = np.where((indices > peak_idx) & (matter <= 0.50 * peak))[0]
    start = float(result.tau_norm[rising[0]]) if rising.size else float(result.tau_norm[0])
    peak_t = float(result.tau_norm[peak_idx])
    recovery = float(result.tau_norm[falling[0]]) if falling.size else peak_t
    return start, peak_t, recovery


def plot_results(result, output_path: str, slots_per_scale: int = 50) -> None:
    x = result.tau_norm
    d = result.diagnostics
    start_matter, peak_matter, recovery_start = _phase_boundaries(result)
    cmap = plt.get_cmap("plasma")
    colors = [
        cmap(0.15 + 0.7 * i / max(len(result.levels) - 1, 1))
        for i in range(len(result.levels))
    ]

    fig, ((ax_space, ax_budget), (ax_matter, ax_quantum)) = plt.subplots(
        2, 2, figsize=(17, 13)
    )
    phase_status = "PASS" if d.three_stage_detected else "CHECK"
    fig.suptitle(
        "General complex wavefunction -> Born statistical fabric -> emergent microstructures\n"
        f"n={result.n_bits}, scales={[lvl.width for lvl in result.levels]}, "
        f"phase terms={result.codec.term_count}, clock={result.clock_mode}, "
        f"matter power={result.matter_power:g}, three-stage diagnostic={phase_status}, "
        f"ledger error={result.conservation_max_error:.2e}",
        fontsize=10,
        fontweight="bold",
    )

    # Panel 1: geometric interpretation and matter worldlines.
    ax_space.set_facecolor("#020205")
    ax_space.fill_between(
        x,
        -result.size_measure / 2,
        result.size_measure / 2,
        color="gainsboro",
        alpha=0.18,
    )
    ax_space.plot(x, result.size_measure / 2, color="white", lw=2.2)
    ax_space.plot(x, -result.size_measure / 2, color="white", lw=2.2)
    ax_space.plot(
        x,
        result.entropy_fraction / 2,
        color="cyan",
        lw=1.0,
        ls=":",
        alpha=0.8,
    )
    ax_space.plot(
        x,
        -result.entropy_fraction / 2,
        color="cyan",
        lw=1.0,
        ls=":",
        alpha=0.8,
    )
    for level, color in zip(result.levels, colors):
        y_slots, active = build_worldlines(
            level.matter_count, slots_per_scale, seed=int(level.width)
        )
        for slot_idx, y0 in enumerate(y_slots):
            mask = active[:, slot_idx]
            if np.any(mask):
                ax_space.plot(
                    x[mask],
                    y0 * result.size_measure[mask],
                    color=color,
                    lw=0.55,
                    alpha=0.35,
                )
    ax_space.plot([], [], color="white", lw=2.2, label="observable resolution R_Q")
    ax_space.plot([], [], color="cyan", lw=1.0, ls=":", label="Born-entropy ceiling")
    for level, color in zip(result.levels, colors):
        ax_space.plot([], [], color=color, lw=1.2, label=f"microstructures w={level.width}")
    ax_space.set_xlabel("Internal coordinate tau = raw flips / n")
    ax_space.set_ylabel("Comoving coordinate x R_Q(tau)")
    ax_space.legend(loc="upper left", fontsize=7, facecolor="#111115", labelcolor="white")

    # Panel 2: finite information budget.
    ax_budget.set_facecolor("#0a0a0a")
    ax_budget.plot(x, result.entropy_fraction, color="cyan", lw=1.6, ls=":", label="Born entropy / n")
    ax_budget.plot(x, result.no_matter_size, color="silver", lw=1.5, ls="--", label="free expansion (pending only)")
    ax_budget.plot(x, result.size_measure, color="orange", lw=2.4, label="R_Q with matter backreaction")
    ax_budget.fill_between(
        x,
        result.size_measure,
        result.no_matter_size,
        where=result.no_matter_size >= result.size_measure,
        color="gold",
        alpha=0.14,
        label="resolution bound into matter",
    )
    ax_budget.set_xlabel("Internal coordinate tau")
    ax_budget.set_ylabel("Fraction of n")
    ax_budget.set_ylim(min(-0.05, float(np.min(result.size_measure)) - 0.03), 1.05)
    ax_budget.legend(loc="lower right", fontsize=8, facecolor="#111115", labelcolor="white")

    # Panel 3: hump-shaped microstructure probabilities and matter allocation.
    ax_matter.set_facecolor("#0a0a0a")
    for level, color in zip(result.levels, colors):
        ax_matter.plot(
            x,
            level.f_bump,
            color=color,
            lw=1.2,
            ls=":",
            label=f"hump probability w={level.width}",
        )
        ax_matter.plot(
            x,
            level.matter_bits / result.n_bits,
            color=color,
            lw=1.8,
            label=f"matter bits/n w={level.width}",
        )
    ax_matter.plot(
        x,
        result.total_matter_bits / result.n_bits,
        color="gold",
        lw=2.6,
        label="total matter bits / n",
    )
    ax_matter.set_xlabel("Internal coordinate tau")
    ax_matter.set_ylabel("Probability or fraction of n")
    ax_matter.legend(loc="upper right", fontsize=7, facecolor="#111115", labelcolor="white")

    # Panel 4: expansion-rate proxy plus genuinely quantum diagnostics.
    ax_quantum.set_facecolor("#0a0a0a")
    h = np.asarray(result.hubble_proxy)
    h_limit = np.nanpercentile(np.abs(h[np.isfinite(h)]), 98) if np.any(np.isfinite(h)) else 1.0
    h_clip = np.clip(h, -h_limit, h_limit)
    ax_quantum.plot(x, h_clip, color="lime", lw=1.8, label="H_Q = (1/R_Q) dR_Q/d(raw tick)")
    ax_quantum.axhline(0.0, color="gray", lw=0.8)
    ax_quantum.set_xlabel("Internal coordinate tau")
    ax_quantum.set_ylabel("Hubble proxy (clipped at 98th percentile)", color="lime")
    ax_quantum.tick_params(axis="y", labelcolor="lime")
    ax_q2 = ax_quantum.twinx()
    ax_q2.plot(x, result.qubit_entanglement, color="magenta", lw=1.5, label="one-bit entanglement S")
    phase_norm = result.phase_field_rms / max(float(np.max(result.phase_field_rms)), 1e-12)
    ax_q2.plot(x, phase_norm, color="deepskyblue", lw=1.1, ls="--", label="phase-field RMS (normalized)")
    ax_q2.set_ylabel("Quantum diagnostics", color="magenta")
    ax_q2.tick_params(axis="y", labelcolor="magenta")
    lines1, labels1 = ax_quantum.get_legend_handles_labels()
    lines2, labels2 = ax_q2.get_legend_handles_labels()
    ax_quantum.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        fontsize=7,
        facecolor="#111115",
        labelcolor="white",
    )

    for ax in (ax_space, ax_budget, ax_matter, ax_quantum):
        ax.axvline(start_matter, color="yellow", lw=1.0, ls=":", alpha=0.8)
        ax.axvline(peak_matter, color="lime", lw=1.4, ls="--", alpha=0.9)
        ax.axvline(recovery_start, color="deepskyblue", lw=1.0, ls="-.", alpha=0.8)
        ax.axvspan(x[0], start_matter, color="white", alpha=0.025)
        ax.axvspan(start_matter, recovery_start, color="gold", alpha=0.035)
        ax.axvspan(recovery_start, x[-1], color="deepskyblue", alpha=0.025)

    fig.text(
        0.5,
        0.012,
        "Vertical markers: matter loading begins (yellow), peak matter (green), late recovery (blue). "
        "The phase codec changes the complex state and entanglement but not the fabric-basis Born probabilities.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "General complex-wavefunction cosmology. Born sampling yields the "
            "statistical fabric; hump microstructures backreact on resolution."
        )
    )
    parser.add_argument("--n_bits", type=int, default=184)
    parser.add_argument("--t_bf_max", type=float, default=None, help="Maximum normalized time; default ln(n).")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--scales", type=str, default="6,12,20")
    parser.add_argument("--matter_power", type=float, default=0.0)
    parser.add_argument("--clock_mode", choices=("block", "shared"), default="block")
    parser.add_argument("--phase_strength", type=float, default=0.9)
    parser.add_argument("--spatial_modes", type=int, default=3)
    parser.add_argument("--temporal_modes", type=int, default=3)
    parser.add_argument("--pair_range", type=int, default=1)
    parser.add_argument("--phase_seed", type=int, default=0)
    parser.add_argument("--phase_topology", choices=("ring", "line"), default="ring")
    parser.add_argument("--samples", type=int, default=4096, help="Born samples used for a numerical shadow check.")
    parser.add_argument("--slots", type=int, default=50)
    parser.add_argument("--output", type=str, default="general_wavefunction_cosmology.png")
    args = parser.parse_args()

    scales = parse_scales(args.scales)
    phase_config = PhaseCodecConfig(
        strength=args.phase_strength,
        spatial_modes=args.spatial_modes,
        temporal_modes=args.temporal_modes,
        pair_range=args.pair_range,
        seed=args.phase_seed,
        topology=args.phase_topology,
    )
    result = run_wavefunction_cosmology(
        n_bits=args.n_bits,
        scales=scales,
        steps=args.steps,
        t_bf_max=args.t_bf_max,
        matter_power=args.matter_power,
        clock_mode=args.clock_mode,
        phase_config=phase_config,
    )
    plot_results(result, args.output, slots_per_scale=args.slots)

    peak_idx = int(np.argmax(result.total_matter_bits))
    peak_t = float(result.t_bf[peak_idx])
    codec = SpectralPhaseCodec(result.n_bits, phase_config)
    state = CompressedComplexWavefunction(
        n_bits=result.n_bits,
        tau_raw=peak_t,
        tau_max=float(result.t_bf[-1]),
        codec=codec,
    )
    samples = state.sample(args.samples, np.random.default_rng(args.phase_seed + 1009))
    sampled_p = float(np.mean(samples))
    diag = result.diagnostics

    print(f"Saved -> {args.output}")
    print(f"Born shadow at peak matter: analytic p={state.p:.6f}, sampled p={sampled_p:.6f}, error={abs(sampled_p-state.p):.3e}")
    print(f"Complex codec: {phase_config.term_count} generated coefficients, phase strength={phase_config.strength:g}")
    print(f"One-bit entanglement at peak matter: S={result.qubit_entanglement[peak_idx]:.6f} bits")
    print(f"Finite-budget conservation error: {result.conservation_max_error:.3e} bits")
    print("Three-stage profile:")
    print(f"  rapid-growth median rate       = {diag.early_growth_rate:.6e}")
    print(f"  matter-loading median rate     = {diag.matter_loading_rate:.6e}")
    print(f"  matter-induced slowdown        = {diag.slowdown_strength:.6e}")
    print(f"  late-recovery median rate      = {diag.recovery_rate:.6e}")
    print(f"  late positive Hubble proxy     = {diag.recovery_hubble_proxy:.6e}")
    print(f"  peak/end resolution suppression= {diag.peak_suppression:.6e} / {diag.end_suppression:.6e}")
    print(f"  diagnostic                     = {'PASS' if diag.three_stage_detected else 'CHECK'}")


if __name__ == "__main__":
    main()
