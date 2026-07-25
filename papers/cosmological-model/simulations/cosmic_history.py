"""CLI and plots for the observer-conditioned compressed quantum-history model.

Place this file next to ``quantum_history.py`` in
``papers/cosmological-model/simulations`` and run, for example:

    python cosmic_history.py --observer gaussian:9 --output quantum_history.png

The default bit count and scale hierarchy match the existing cosmological
simulation drivers.  The internal history, however, is the exact discrete
one-bit-flip process conditioned on an observer record, then lifted to a
complex phase-coded wavefunction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantum_history import (
    MatterLevelSpec,
    ObserverConditionedEhrenfestHistory,
    ObserverRecord,
    SpectralPhaseCodec,
    SpectralPhaseCodecConfig,
    parse_compositions,
    run_quantum_history_simulation,
)


def parse_int_list(raw: str) -> list[int]:
    values = [int(piece.strip()) for piece in raw.split(",") if piece.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    return values


def parse_observer(
    raw: str,
    *,
    tick: int,
    start: int,
    n_bits: int,
    threshold: float,
) -> ObserverRecord:
    value = raw.strip().lower()
    if value in {"none", "identity", "off"}:
        return ObserverRecord.none(tick=tick)
    if value.startswith("pattern:"):
        record = ObserverRecord.contiguous(
            raw.split(":", maxsplit=1)[1], tick=tick, start=start, label="pixel-observer"
        )
    elif value.startswith("gaussian:"):
        width = int(raw.split(":", maxsplit=1)[1])
        record = ObserverRecord.gaussian_blob(
            width,
            tick=tick,
            start=start,
            threshold=threshold,
        )
    else:
        # A bare binary string is accepted for convenience.
        record = ObserverRecord.contiguous(raw, tick=tick, start=start, label="pixel-observer")

    if record.indices and max(record.indices) >= n_bits:
        raise ValueError("observer record extends beyond the bitstring")
    return record


def plot_result(result, output_path: str) -> None:
    ticks = result.ticks
    n_bits = result.n_bits
    observer_tick = result.observer.tick

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_size, ax_patterns, ax_quantum, ax_weights = axes.ravel()

    observer_text = (
        "identity observer"
        if result.observer.width == 0
        else f"{result.observer.label}: {''.join(map(str, result.observer.pattern))} at tick {observer_tick}"
    )
    fig.suptitle(
        "Observer-conditioned compressed quantum history\n"
        f"n={n_bits}, {observer_text}, P(observer)={result.observer_evidence:.3e}, "
        f"codec MDL proxy={result.codec_description_bits:.1f} bits",
        fontsize=11,
        fontweight="bold",
    )

    # 1. Emergent screen entropy / finite-budget size proxy.
    ax_size.plot(ticks, result.born_entropy_bits / n_bits, lw=2.0, label="Born screen entropy / n")
    ax_size.plot(ticks, result.size_measure, lw=2.2, label="size = (entropy - matter bits) / n")
    ax_size.plot(
        ticks,
        np.clip(2.0 * result.mean_one_fraction, 0.0, 1.0),
        lw=1.2,
        ls=":",
        label="mean contrast proxy 2<E[K]>/n",
    )
    ax_size.axvline(observer_tick, lw=1.2, ls="--", label="observer record")
    ax_size.set_xlabel("Internal elementary flip tick")
    ax_size.set_ylabel("Normalized quantity")
    ax_size.set_ylim(-0.03, max(1.05, float(np.max(result.size_measure)) * 1.05))
    ax_size.legend(fontsize=8)
    ax_size.set_title("Born-visible statistical universe")

    # 2. Matter-pattern probabilities.
    for level in result.levels:
        label = level.spec.resolved_label
        ax_patterns.plot(ticks, level.match_probability, lw=1.2, ls=":", label=f"match {label}")
        ax_patterns.plot(
            ticks,
            level.persistent_probability,
            lw=1.8,
            label=f"persistent {label}",
        )
    ax_patterns.axvline(observer_tick, lw=1.2, ls="--")
    ax_patterns.set_xlabel("Internal elementary flip tick")
    ax_patterns.set_ylabel("Born probability")
    ax_patterns.set_title("Bulk compositions from the conditional wavefunction")
    ax_patterns.legend(fontsize=7)

    # 3. Complex residual and matter budget.
    ax_quantum.plot(
        ticks,
        result.phase_residual_rms,
        lw=2.0,
        label="DCT/Walsh phase-residual RMS",
    )
    ax_quantum.set_xlabel("Internal elementary flip tick")
    ax_quantum.set_ylabel("Phase residual (radians)")
    ax_quantum.axvline(observer_tick, lw=1.2, ls="--")
    ax_quantum2 = ax_quantum.twinx()
    ax_quantum2.plot(
        ticks,
        result.total_matter_bits / n_bits,
        lw=2.0,
        ls="-.",
        label="matter bits / n",
    )
    ax_quantum2.set_ylabel("Matter allocation / n")
    lines_1, labels_1 = ax_quantum.get_legend_handles_labels()
    lines_2, labels_2 = ax_quantum2.get_legend_handles_labels()
    ax_quantum.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=8)
    ax_quantum.set_title("Compression residual invisible to pixel Born counts")

    # 4. P(K_t | observer) snapshots.
    for tick, distribution in result.total_weight_snapshots.items():
        ax_weights.plot(np.arange(len(distribution)), distribution, lw=1.7, label=f"tick {tick}")
    ax_weights.set_xlabel("Total ones K")
    ax_weights.set_ylabel("P(K | observer)")
    ax_weights.set_title("Observer-conditioned Hamming-weight marginals")
    ax_weights.legend(fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=160, facecolor="white")
    print(f"Saved -> {output_path}")


def validate_born_sampling(
    *,
    n_bits: int,
    max_tick: int,
    observer: ObserverRecord,
    phase_config: SpectralPhaseCodecConfig,
    samples: int,
    seed: int,
) -> None:
    codec = SpectralPhaseCodec(n_bits, phase_config)
    history = ObserverConditionedEhrenfestHistory(n_bits, max_tick, observer, codec)
    frame = history.frame(observer.tick)
    rng = np.random.default_rng(seed)
    bitstrings = frame.sample_bitstrings(samples, rng)
    amplitudes = frame.amplitudes(bitstrings)

    sampled_one_fraction = float(np.mean(bitstrings))
    analytic_one_fraction = frame.mean_one_fraction
    print("Born-sampling validation at the observer frame")
    print(f"  samples: {samples}")
    print(f"  analytic mean one fraction: {analytic_one_fraction:.8f}")
    print(f"  sampled mean one fraction:  {sampled_one_fraction:.8f}")
    print(f"  absolute error:             {abs(sampled_one_fraction - analytic_one_fraction):.3e}")
    print(f"  sampled amplitude |psi| range: [{np.min(np.abs(amplitudes)):.3e}, {np.max(np.abs(amplitudes)):.3e}]")

    if n_bits <= 16:
        entropy = frame.half_chain_entanglement_entropy(max_qubits=16)
        norm = float(np.vdot(frame.exact_statevector(max_qubits=16), frame.exact_statevector(max_qubits=16)).real)
        print(f"  exact statevector norm:     {norm:.12f}")
        print(f"  half-chain entanglement:    {entropy:.8f} bits")
    else:
        print("  exact statevector skipped (n_bits > 16); scalable count-space bridge used")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Observer-conditioned complex quantum-history backend. Born sampling "
            "reproduces the conditioned statistical bitstring universe exactly."
        )
    )
    parser.add_argument("--n_bits", type=int, default=184)
    parser.add_argument(
        "--t_bf_max",
        type=float,
        default=None,
        help="Maximum internal time in units of n. Default: ln(n), matching the existing drivers.",
    )
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--scales", type=parse_int_list, default=parse_int_list("6,12,20"))
    parser.add_argument(
        "--compositions",
        type=str,
        default=None,
        help="a:b per scale, e.g. '2:4,4:8,6:14'. Defaults match dicke_layer.default_composition.",
    )
    parser.add_argument("--matter_power", type=float, default=1.0)
    parser.add_argument("--hierarchy", choices=("parallel", "cascade"), default="parallel")
    parser.add_argument("--survival", choices=("literal", "none"), default="literal")

    parser.add_argument(
        "--observer",
        type=str,
        default="gaussian:9",
        help="none, a bare bitstring, pattern:0101, or gaussian:WIDTH",
    )
    parser.add_argument("--observer_tick", type=int, default=None)
    parser.add_argument("--observer_start", type=int, default=0)
    parser.add_argument("--observer_threshold", type=float, default=0.45)

    parser.add_argument("--phase_topology", choices=("ring", "open-chain", "seeded-sparse", "none"), default="ring")
    parser.add_argument("--phase_strength", type=float, default=0.9)
    parser.add_argument("--temporal_modes", type=int, default=3)
    parser.add_argument("--spatial_modes", type=int, default=3)
    parser.add_argument("--sparse_edges", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--output", type=str, default="observer_conditioned_quantum_history.png")
    args = parser.parse_args()

    normalized_max = args.t_bf_max if args.t_bf_max is not None else float(np.log(args.n_bits))
    max_tick = max(1, int(round(args.n_bits * normalized_max)))
    observer_tick = args.observer_tick if args.observer_tick is not None else min(args.n_bits, max_tick)
    observer = parse_observer(
        args.observer,
        tick=observer_tick,
        start=args.observer_start,
        n_bits=args.n_bits,
        threshold=args.observer_threshold,
    )
    levels: list[MatterLevelSpec] = parse_compositions(args.scales, args.compositions)
    phase_config = SpectralPhaseCodecConfig(
        temporal_modes=args.temporal_modes,
        spatial_modes=args.spatial_modes,
        phase_strength=args.phase_strength,
        topology=args.phase_topology,
        sparse_edges=args.sparse_edges,
        seed=args.seed,
    )

    result = run_quantum_history_simulation(
        n_bits=args.n_bits,
        max_tick=max_tick,
        steps=args.steps,
        observer=observer,
        levels=levels,
        phase_config=phase_config,
        matter_power=args.matter_power,
        hierarchy=args.hierarchy,
        survival=args.survival,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plot_result(result, args.output)
    validate_born_sampling(
        n_bits=args.n_bits,
        max_tick=max_tick,
        observer=observer,
        phase_config=phase_config,
        samples=args.samples,
        seed=args.seed + 1,
    )
    print(f"Observer evidence P(O): {result.observer_evidence:.12e}")
    print(f"Phase codec description-length proxy: {result.codec_description_bits:.2f} bits")
    print(f"Maximum normalized Born entropy: {np.max(result.born_entropy_bits) / args.n_bits:.8f}")
    print(f"Maximum matter allocation/n: {np.max(result.total_matter_bits) / args.n_bits:.8f}")


if __name__ == "__main__":
    main()
