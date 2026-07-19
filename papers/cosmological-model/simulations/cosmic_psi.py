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
    combinatorial_entropy_bits,
    run_simulation,
    years_to_tbf,
)
import dicke_layer as dl
from dicke_cascade import LevelSpec, run_cascade_series, run_parallel_series

FloatArray = NDArray[np.float64]


def quantum_parallel_series(n_bits: int, t_bf: FloatArray, levels: list[LevelSpec]):
    """Independent-per-level counterpart to quantum_cascade_series, using
    run_parallel_series instead of run_cascade_series: every level reads
    off the SAME shared (n_bits, k(t)) rather than a shrinking leftover
    substrate, and nothing is multiplied across levels. Appropriate for
    levels meant to represent coexisting categories (e.g. dark matter +
    several visible fermion species) rather than a nested formation
    hierarchy -- see run_parallel_series's docstring for why chaining is
    the wrong choice for that case.
    """
    k_vals = dl.k_of_tau(n_bits, t_bf)
    k_int = np.clip(np.round(k_vals).astype(int), 0, n_bits)
    cascade = run_parallel_series(n_bits, k_int.astype(float), levels, mode="class")
    matter_bits = {}
    for res in cascade:
        matter_bits[res.spec.width] = (
            res.n_windows_available * res.spec.width * res.cumulative_persistent_prob
        )
    entropy_bits = n_bits * combinatorial_entropy_bits(n_bits, k_int.astype(float))
    return cascade, matter_bits, entropy_bits, k_int


def quantum_cascade_series(n_bits: int, t_bf: FloatArray, levels: list[LevelSpec]):
    """Always uses dicke_cascade_v2's 'class' mode (counting-equation: any
    arrangement of a excitations in the w-window counts, matching
    multiclock.family_fractions_exact's count-only fall/bump/rise split).

    'specific' mode (one exact bit ordering) is deliberately NOT offered
    here: it answers a different question (how much distinguishing
    information is in one particular microstate) that has no counterpart
    in the classical statistical-shadow model at all -- the verified
    classical<->quantum correspondence (window_marginal <-> 
    family_fractions_exact) is a counts-only object throughout. Chaining
    'specific' probabilities across a multi-level cascade also compounds
    the C(w,a) combinatorial suppression multiplicatively level over
    level, so the total is dominated by whichever scale is listed first
    regardless of the rest of --scales -- see the module docstring in
    dicke_cascade_v2.py, which already flags this. dl.pattern_probability
    itself is still available in dicke_layer.py for other purposes (e.g.
    a Solomonoff/description-length treatment of specific configurations),
    just not wired into this spacetime/matter pipeline.
    """
    k_vals = dl.k_of_tau(n_bits, t_bf)
    k_int = np.clip(np.round(k_vals).astype(int), 0, n_bits)
    cascade = run_cascade_series(n_bits, k_int.astype(float), levels, mode="class")
    matter_bits = {}
    for res in cascade:
        matter_bits[res.spec.width] = (
            res.n_windows_available * res.spec.width * res.cumulative_persistent_prob
        )
    entropy_bits = n_bits * combinatorial_entropy_bits(n_bits, k_int.astype(float))
    return cascade, matter_bits, entropy_bits, k_int


def plot_results_cascade(sim: SimulationResult, levels: list[LevelSpec],
                          slots_per_scale: int, output_path: str, parallel: bool = False) -> None:
    t_bf = sim.t_bf
    n_bits = sim.n_bits
    widths = [lvl.width for lvl in levels]

    series_fn = quantum_parallel_series if parallel else quantum_cascade_series
    cascade, matter_bits, entropy_bits, k_int = series_fn(n_bits, t_bf, levels)
    total_matter_bits = sum(matter_bits[w] for w in widths)
    size_measure_q = np.clip((entropy_bits - total_matter_bits) / n_bits, 0.0, None)

    cmap = plt.get_cmap("plasma")
    colors = [cmap(0.15 + 0.7 * i / max(len(levels) - 1, 1)) for i in range(len(levels))]

    peak_idx = int(np.argmax(total_matter_bits))
    t_today_q = float(t_bf[peak_idx])

    tick_years = [1e-40, 1e-30, 1e-20, 1e-10, 1e-4, 1e3, 1e9, 13.8e9]
    tick_tbf = [years_to_tbf(t, t_today_q) for t in tick_years]
    tick_labels = ["10\u207b\u2074\u2070", "10\u207b\u00b3\u2070", "10\u207b\u00b2\u2070",
                   "10\u207b\u00b9\u2070", "10\u207b\u2074", "10\u00b3", "10\u2079", "now"]
    valid = [(tb, lb) for tb, lb in zip(tick_tbf, tick_labels) if 0 <= tb <= sim.t_bf_max]
    tick_tbf_v, tick_labels_v = zip(*valid) if valid else ([], [])

    fig, ((ax_st, ax_met), (ax_pat, ax_matter)) = plt.subplots(2, 2, figsize=(17, 13))
    comp_str = ", ".join(f"w={lvl.width}:(a={lvl.a},b={lvl.b})" for lvl in levels)
    surv_str = ", ".join(f"w={lvl.width}:{res.survival_prob:.3g}" for lvl, res in zip(levels, cascade))
    mode_label = "PARALLEL (independent, non-chained)" if parallel else "CASCADED (chained)"
    fig.suptitle(
        f"PSI-LAYER, {mode_label} (counting-equation / class probabilities)\n"
        f"n={n_bits:g}  compositions=[{comp_str}]  |  survival probs=[{surv_str}]\n"
        f"matter_bits peak/entropy_bits peak = "
        f"{total_matter_bits.max():.4f}/{entropy_bits.max():.3f} = "
        f"{total_matter_bits.max()/max(entropy_bits.max(),1e-12):.5f}",
        fontsize=9, fontweight="bold",
    )

    ax_st.set_facecolor("#020205")
    ax_st.set_xlabel("Physical time (years, log scale)")
    ax_st.set_ylabel("Comoving y x size_measure_q(t)")
    ax_st.set_xticks(tick_tbf_v); ax_st.set_xticklabels(tick_labels_v, fontsize=7)
    ax_st.fill_between(t_bf, -size_measure_q/2, size_measure_q/2, color="gainsboro", alpha=0.15)
    ax_st.plot(t_bf, size_measure_q/2, color="white", lw=2.2, alpha=0.9)
    ax_st.plot(t_bf, -size_measure_q/2, color="white", lw=2.2, alpha=0.9)
    ax_st.plot(t_bf, entropy_bits/n_bits/2, color="cyan", lw=1.0, ls=":", alpha=0.7)
    ax_st.plot(t_bf, -entropy_bits/n_bits/2, color="cyan", lw=1.0, ls=":", alpha=0.7)
    for lvl, color in zip(levels, colors):
        y_slots, active = build_worldlines(matter_bits[lvl.width]/lvl.width, slots_per_scale, seed=int(lvl.width))
        for i, y0 in enumerate(y_slots):
            mask = active[:, i]
            if mask.any():
                ax_st.plot(t_bf[mask], y0*size_measure_q[mask], color=color, lw=0.6, alpha=0.35)
    ax_st.axvline(t_today_q, color="lime", lw=1.5, ls="--", alpha=0.85)

    ax_met.set_facecolor("#0a0a0a")
    ax_met.set_xlabel("Physical time (years, log scale)")
    ax_met.set_ylabel("probability")
    ax_met.set_xticks(tick_tbf_v); ax_met.set_xticklabels(tick_labels_v, fontsize=7)
    persist_label = "independent" if parallel else "chained"
    for lvl, res, color in zip(levels, cascade, colors):
        ax_met.plot(t_bf, res.match_prob, color=color, lw=1.2, ls=":", label=f"w={lvl.width} match_prob alone")
        ax_met.plot(t_bf, res.cumulative_persistent_prob, color=color, lw=1.8, label=f"w={lvl.width} {persist_label}")
    ax_met.axvline(t_today_q, color="lime", lw=1.5, ls="--", alpha=0.85)
    ax_met.legend(loc="upper right", fontsize=7, facecolor="#111115", edgecolor="gray", labelcolor="white")

    ax_pat.set_facecolor("#0a0a0a")
    ax_pat.set_xlabel("Physical time (years, log scale)")
    ax_pat.set_ylabel("S_vN (bits)")
    ax_pat.set_xticks(tick_tbf_v); ax_pat.set_xticklabels(tick_labels_v, fontsize=7)
    for lvl, res, color in zip(levels, cascade, colors):
        ax_pat.plot(t_bf, res.entanglement_entropy, color=color, lw=1.6, label=f"w={lvl.width}")
    ax_pat.legend(loc="lower right", fontsize=8, facecolor="#111115", edgecolor="gray", labelcolor="white")

    ax_matter.set_facecolor("#0a0a0a")
    ax_matter.set_xlabel("Physical time (years, log scale)")
    ax_matter.set_ylabel("Bits")
    ax_matter.set_xticks(tick_tbf_v); ax_matter.set_xticklabels(tick_labels_v, fontsize=7)
    ax_matter.plot(t_bf, entropy_bits, color="cyan", lw=1.8, ls=":", label="entropy_bits(t)")
    ax_matter.plot(t_bf, total_matter_bits, color="gold", lw=2.4, label="total matter_bits(t)")
    for lvl, color in zip(levels, colors):
        ax_matter.plot(t_bf, matter_bits[lvl.width], color=color, lw=1.0, alpha=0.85, label=f"w={lvl.width}")
    ax_matter.plot(t_bf[peak_idx], total_matter_bits[peak_idx], "o", color="lime", ms=6)
    ax_matter.legend(loc="upper right", fontsize=7, facecolor="#111115", edgecolor="gray", labelcolor="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="white")
    print(f"Saved -> {output_path}")


def parse_levels(scales_raw, compositions_raw):
    widths = [int(x) for x in scales_raw.split(",") if x.strip()]
    if compositions_raw is None:
        comps = [dl.default_composition(w) for w in widths]
    else:
        comps = [(int(p.split(":")[0]), int(p.split(":")[1])) for p in compositions_raw.split(",")]
    return [LevelSpec(width=w, a=a, b=b) for w, (a, b) in zip(widths, comps)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Psi-layer cascade (dicke_cascade_v2) companion to the classical multi-clock demonstrator."
    )
    parser.add_argument("--n_bits", type=float, default=184.0)
    parser.add_argument("--t_bf_max", type=float, default=None,
                         help="Max raw bit-flip time, in units of n. Default: ln(n), "
                              "same convention as emergent_structure_relativistic.py.")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--t_today", type=float, default=None)
    parser.add_argument("--matter_power", type=float, default=1.0)
    parser.add_argument("--scales", type=str, default="6,12,20")
    parser.add_argument("--compositions", type=str, default=None,
                         help="a:b per scale, comma-separated, e.g. '1:5,2:10,3:17'. "
                              "Default: dl.default_composition(w) per width.")
    parser.add_argument("--slots", type=int, default=50)
    parser.add_argument("--parallel", action="store_true",
                         help="Treat levels as independent/coexisting (no chaining across "
                              "levels) instead of a nested formation hierarchy. Use this for "
                              "e.g. dark matter + several visible fermion species that coexist "
                              "rather than one nesting inside another's leftover substrate.")
    parser.add_argument("--output", type=str, default="cascade.png")

    args = parser.parse_args()
    levels = parse_levels(args.scales, args.compositions)

    sim = run_simulation(
        n_bits=args.n_bits, scales=[lvl.width for lvl in levels], steps=args.steps,
        t_bf_max=args.t_bf_max, t_today=args.t_today, matter_power=args.matter_power,
    )

    plot_results_cascade(sim, levels, slots_per_scale=args.slots, output_path=args.output,
                          parallel=args.parallel)


if __name__ == "__main__":
    main()
