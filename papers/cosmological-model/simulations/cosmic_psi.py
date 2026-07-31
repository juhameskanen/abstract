from __future__ import annotations

import argparse

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from multiclock import (
    SimulationResult,
    build_worldlines,
    combinatorial_entropy_bits,
    run_simulation,
    years_to_tbf,
)
import dicke_layer as dl
from dicke_cascade import (
    LevelSpec,
    run_cascade_series_retarded,
    run_parallel_series_retarded,
)

FloatArray = NDArray[np.float64]


def quantum_parallel_series(n_bits: int, t_bf: FloatArray, levels: list[LevelSpec]):
    """Independent-per-level counterpart to quantum_cascade_series, using
    run_parallel_series_retarded instead of run_cascade_series_retarded:
    every level reads off the SAME shared n_bits pool rather than a
    shrinking leftover substrate, and nothing is multiplied across
    levels. Appropriate for levels meant to represent coexisting
    categories (e.g. dark matter + several visible fermion species)
    rather than a nested formation hierarchy -- see
    run_parallel_series's docstring for why chaining is the wrong
    choice for that case.

    Each level still gets its own retarded clock (tau_local =
    width*floor(tau/width), lapse=1/width) and its own pending backlog,
    exactly like quantum_cascade_series -- substrate chaining and clock
    retardation are independent axes. A width-20 structure ticks its
    own proper time slower whether or not it happens to be chained
    inside another level's leftover, so "independent/coexisting" must
    not mean "no time dilation."
    """
    cascade = run_parallel_series_retarded(n_bits, t_bf, levels, mode="class")
    matter_bits = {}
    pending_bits = np.zeros_like(t_bf)
    for res in cascade:
        matter_bits[res.spec.width] = (
            res.n_windows_available * res.spec.width * res.cumulative_persistent_prob
        )
        pending_bits = pending_bits + res.pending
    k_vals = dl.k_of_tau(n_bits, t_bf)
    k_int = np.clip(np.round(k_vals).astype(int), 0, n_bits)
    entropy_bits = n_bits * combinatorial_entropy_bits(n_bits, k_int.astype(float))
    return cascade, matter_bits, entropy_bits, pending_bits, k_int


def quantum_cascade_series(n_bits: int, t_bf: FloatArray, levels: list[LevelSpec]):
    """Retarded-clock cascade: each level's match probability and entropy
    are evaluated on ITS OWN lagged clock (tau_local = width * floor(tau /
    width), lapse = 1/width), exactly mirroring multiclock.retard() and
    wavefunction_model.local_clock -- the "heavier structures advance
    their own proper time more slowly" mechanism.  Each level also
    carries a pending backlog (entropy known to the raw clock but not yet
    caught up to by that level's own lagged clock), summed into
    pending_bits and subtracted in the caller's size_measure_q, matching
    the classical/wavefunction ledger equation exactly.

    Always uses 'class' mode (counting-equation: any arrangement of a
    excitations in the w-window counts) -- see dicke_cascade.py's
    run_cascade_series docstring for why 'specific' mode is not used for
    the matter signal.

    Matter is the raw hump-family match probability with no extra
    order-parameter weighting -- there is no matter_power free parameter
    here anymore, matching the classical and wavefunction backends.
    """
    cascade = run_cascade_series_retarded(n_bits, t_bf, levels, mode="class")
    matter_bits = {}
    pending_bits = np.zeros_like(t_bf)
    for res in cascade:
        matter_bits[res.spec.width] = (
            res.n_windows_available * res.spec.width * res.cumulative_persistent_prob
        )
        pending_bits = pending_bits + res.pending
    k_vals = dl.k_of_tau(n_bits, t_bf)
    k_int = np.clip(np.round(k_vals).astype(int), 0, n_bits)
    entropy_bits = n_bits * combinatorial_entropy_bits(n_bits, k_int.astype(float))
    return cascade, matter_bits, entropy_bits, pending_bits, k_int


def plot_results_cascade(sim: SimulationResult, levels: list[LevelSpec],
                          slots_per_scale: int, output_path: str, parallel: bool = False) -> None:
    t_bf = sim.t_bf
    n_bits = sim.n_bits
    widths = [lvl.width for lvl in levels]

    if parallel:
        cascade, matter_bits, entropy_bits, pending_bits, k_int = quantum_parallel_series(
            n_bits, t_bf, levels
        )
    else:
        cascade, matter_bits, entropy_bits, pending_bits, k_int = quantum_cascade_series(
            n_bits, t_bf, levels
        )
    total_matter_bits = sum(matter_bits[w] for w in widths)
    size_measure_q = np.clip((entropy_bits - pending_bits - total_matter_bits) / n_bits, 0.0, None)

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
    parser.add_argument("--anim", action="store_true",
                         help="Render the time-dilation animation (GIF) instead of the static plot.")
    parser.add_argument("--frames", type=int, default=150, help="Animation frames (--anim only).")
    parser.add_argument("--fps", type=int, default=20, help="Animation fps (--anim only).")
    from render_3d import add_3d_cli_args
    add_3d_cli_args(parser)

    args = parser.parse_args()
    levels = parse_levels(args.scales, args.compositions)

    sim = run_simulation(
        n_bits=args.n_bits, scales=[lvl.width for lvl in levels], steps=args.steps,
        t_bf_max=args.t_bf_max, t_today=args.t_today,
    )

    size_measure_q = None
    anim_levels = None
    mode_label = "parallel" if args.parallel else "cascaded"
    if args.anim or args.three_d:
        from anim_common import LevelAnimSpec
        series_fn = quantum_parallel_series if args.parallel else quantum_cascade_series
        cascade, matter_bits, entropy_bits, pending_bits, _k_int = series_fn(
            sim.n_bits, sim.t_bf, levels
        )
        total_matter_bits = sum(matter_bits[w] for w in [lvl.width for lvl in levels])
        size_measure_q = np.clip(
            (entropy_bits - pending_bits - total_matter_bits) / sim.n_bits, 0.0, None
        )
        anim_levels = [
            LevelAnimSpec(width=lvl.width, tau_local=res.tau_local, lapse=res.lapse,
                           matter_series=matter_bits[lvl.width])
            for lvl, res in zip(levels, cascade)
        ]

    if args.anim:
        from anim_common import render_time_dilation_animation
        output = args.output if args.output.endswith((".gif", ".mp4")) else "time_dilation_dicke.gif"
        render_time_dilation_animation(
            sim.t_bf, sim.t_bf_max, size_measure_q, anim_levels, output,
            n_slots=args.slots, frames=args.frames, fps=args.fps,
            title=f"Dicke/psi-layer backend ({mode_label}): worldlines growing at different rates",
        )
    else:
        plot_results_cascade(sim, levels, slots_per_scale=args.slots, output_path=args.output,
                              parallel=args.parallel)

    if args.three_d:
        from render_3d import dispatch_3d
        dispatch_3d(
            args, sim.t_bf, size_measure_q, anim_levels,
            n_bits=sim.n_bits, k_rate=sim.k_rate,
            title=f"Dicke/psi-layer backend ({mode_label}): particle population density "
                  f"(illustrative 3D + spherical symmetry)",
        )


if __name__ == "__main__":
    main()
