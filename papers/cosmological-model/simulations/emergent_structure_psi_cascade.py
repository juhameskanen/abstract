"""
Emergent Structure from a Relaxing Bitstring -- PSI-LAYER, CASCADED
====================================================================

This supersedes emergent_structure_psi.py's per-scale treatment, which
checked each scale's pattern_probability against the SAME global (n_bits,
k) independently -- i.e. as if "DM", "nu" and "baryon" windows were drawn
from the full, unconsumed substrate at every scale, with no accounting for
one scale's matched bits being unavailable to the next.

Here, each level's substrate is dicke_cascade_v2's EXACT leftover after
earlier levels: level i's Hilbert space is (n_bits - sum of earlier
widths) modes, with (k - sum of earlier levels' `a`) excitations, using the
verified conditioning |D_n^k> -> |D_w^j> (x) |D_{n-w}^{k-j}> factorization.
On top of that, a level's contribution to "matter" additionally requires
it to SURVIVE (dicke_cascade_v2.survival_probability) for `width` further
ticks, not just match at a single instant.

So each level's honest "matter" signal is cumulative_persistent_prob(t):
the Born-rule probability of matching this level's composition AND every
earlier level's composition in sequence AND all of those surviving long
enough to be stable substrate -- not the independent single-level
match_prob emergent_structure_psi.py used.

Everything this script needs from dicke_cascade_v2 is read off
CascadeSeriesResult -- no probability arrays are recomputed here.

FLAGGED, CARRIED OVER FROM dicke_cascade_v2 (read before trusting numbers):
  - Per-level compositions (a, b) remain a modeling choice, not derived.
  - The survival probability used is the CONSERVATIVE literal-freeze
    notion (zero tolerance for any touch inside the window), a lower
    bound on true persistence, not the tight answer.
  - delta = width (survival checked over as many ticks as the structure's
    own width) is a motivated choice, not a forced consequence.
This script does not resolve any of these; it only wires the (flagged)
cascade machinery into the same 4-panel plot layout used elsewhere in this
project, so the effect of chaining + persistence can actually be seen.
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
    combinatorial_entropy_bits,
    run_simulation,
    years_to_tbf,
)
import dicke_layer as dl
from dicke_cascade_v2 import LevelSpec, run_cascade_series

FloatArray = NDArray[np.float64]


# ===========================================================================
# CASCADE-DRIVEN QUANTITIES
# ===========================================================================

def quantum_cascade_series(n_bits: int, t_bf: FloatArray, levels: list[LevelSpec], mode: str = "specific"):
    """Exact Born-rule time series through the persistence-aware cascade.

    Args:
        mode: "specific" (one exact bit ordering) or "class" (whole (a,b)
            composition class, any ordering -- the macroscopically
            relevant "how much matter has formed" signal). See
            dicke_cascade_v2.run_cascade_series / dicke_layer.class_probability.

    Returns:
        cascade: list[CascadeSeriesResult], one per level (in cascade order)
        matter_bits: dict[width] -> array, bits committed to that level
            once chaining + survival are both accounted for
        entropy_bits: array, the substrate's OWN total combinatorial
            entropy (unrelated to any level's composition -- same
            quantity emergent_structure_psi.py already used)
        k_int: array, the mean-field excitation count feeding the cascade
    """
    k_vals = dl.k_of_tau(n_bits, t_bf)
    k_int = np.clip(np.round(k_vals).astype(int), 0, n_bits)

    cascade = run_cascade_series(n_bits, k_int.astype(float), levels, mode=mode)

    matter_bits = {}
    for res in cascade:
        # bits committed to this level = (disjoint copies available) x
        # (width per copy) x (probability of reaching + surviving this level)
        matter_bits[res.spec.width] = (
            res.n_windows_available * res.spec.width * res.cumulative_persistent_prob
        )

    entropy_bits = n_bits * combinatorial_entropy_bits(n_bits, k_int.astype(float))
    return cascade, matter_bits, entropy_bits, k_int


# ===========================================================================
# PLOTTING -- same 4-panel layout as emergent_structure_psi.py, but every
# matter-related array now comes from the cascade, not independent scales.
# ===========================================================================

def plot_results_cascade(sim: SimulationResult, levels: list[LevelSpec],
                          slots_per_scale: int, output_path: str, mode: str = "specific") -> None:
    t_bf = sim.t_bf
    n_bits = sim.n_bits
    widths = [lvl.width for lvl in levels]

    cascade, matter_bits, entropy_bits, k_int = quantum_cascade_series(n_bits, t_bf, levels, mode=mode)
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
    fig.suptitle(
        "Emergent Structure from a Relaxing Bitstring -- PSI-LAYER, CASCADED (chained substrate + persistence)\n"
        f"n={n_bits:g}  mode={mode}  compositions=[{comp_str}]  |  survival probs=[{surv_str}]\n"
        f"matter_bits peak/entropy_bits peak = "
        f"{total_matter_bits.max():.4f}/{entropy_bits.max():.3f} = "
        f"{total_matter_bits.max()/max(entropy_bits.max(),1e-12):.5f}",
        fontsize=8.5, fontweight="bold",
    )

    # --- ax_st: spacetime envelope, built entirely from cascaded size_measure_q ---
    ax_st.set_facecolor("#020205")
    ax_st.set_xlabel("Physical time (years, log scale via t\u2192ln t)")
    ax_st.set_ylabel("Comoving y \u00d7 size_measure_q(t)  [CASCADED]")
    ax_st.set_xticks(tick_tbf_v)
    ax_st.set_xticklabels(tick_labels_v, fontsize=7)

    ax_st.fill_between(t_bf, -size_measure_q / 2, size_measure_q / 2, color="gainsboro", alpha=0.15)
    ax_st.plot(t_bf, size_measure_q / 2, color="white", lw=2.2, alpha=0.9)
    ax_st.plot(t_bf, -size_measure_q / 2, color="white", lw=2.2, alpha=0.9)
    ax_st.plot(t_bf, entropy_bits / n_bits / 2, color="cyan", lw=1.0, ls=":", alpha=0.7)
    ax_st.plot(t_bf, -entropy_bits / n_bits / 2, color="cyan", lw=1.0, ls=":", alpha=0.7)

    for lvl, color in zip(levels, colors):
        y_slots, active = build_worldlines(matter_bits[lvl.width] / lvl.width, slots_per_scale, seed=int(lvl.width))
        for i, y0 in enumerate(y_slots):
            mask = active[:, i]
            if mask.any():
                ax_st.plot(t_bf[mask], y0 * size_measure_q[mask], color=color, lw=0.6, alpha=0.35)

    ax_st.plot([], [], color="white", lw=2, label="size_measure_q = (entropy_bits \u2212 matter_bits)/n  [CASCADED]")
    ax_st.plot([], [], color="cyan", lw=1.0, ls=":", label="entropy_bits/n  (exact combinatorial entropy)")
    for lvl, color in zip(levels, colors):
        ax_st.plot([], [], color=color, lw=1.2, label=f"w={lvl.width} pattern a={lvl.a},b={lvl.b} (chained+persistent)")
    ax_st.axvline(t_today_q, color="lime", lw=1.5, ls="--", alpha=0.85, label=f"now = peak total matter_bits (t={t_today_q:.1f})")
    ax_st.legend(loc="upper left", facecolor="#111115", edgecolor="gray", labelcolor="white", fontsize=7.5)

    # --- ax_met: match_prob (single-level, as if reached) vs cumulative_persistent_prob (chained+surviving) ---
    ax_met.set_facecolor("#0a0a0a")
    ax_met.set_xlabel("Physical time (years, log scale)")
    ax_met.set_ylabel("probability")
    ax_met.set_xticks(tick_tbf_v)
    ax_met.set_xticklabels(tick_labels_v, fontsize=7)
    for lvl, res, color in zip(levels, cascade, colors):
        ax_met.plot(t_bf, res.match_prob, color=color, lw=1.2, ls=":",
                    label=f"w={lvl.width} match_prob (this level alone)")
        ax_met.plot(t_bf, res.cumulative_persistent_prob, color=color, lw=1.8,
                    label=f"w={lvl.width} cumulative+persistent (chained)")
    ax_met.axvline(t_today_q, color="lime", lw=1.5, ls="--", alpha=0.85)
    ax_met.set_title("Chaining + persistence pulls the deeper levels down hard (dotted -> solid)", fontsize=8.5)
    ax_met.legend(loc="upper right", fontsize=7, facecolor="#111115", edgecolor="gray", labelcolor="white")

    # --- ax_pat: entanglement entropy per level, on its own (shrinking) substrate ---
    ax_pat.set_facecolor("#0a0a0a")
    ax_pat.set_xlabel("Physical time (years, log scale)")
    ax_pat.set_ylabel("$S_{vN}$ (bits)")
    ax_pat.set_xticks(tick_tbf_v)
    ax_pat.set_xticklabels(tick_labels_v, fontsize=7)
    for lvl, res, color in zip(levels, cascade, colors):
        ax_pat.plot(t_bf, res.entanglement_entropy, color=color, lw=1.6,
                    label=f"w={lvl.width} (n_substrate={res.n_substrate})")
    ax_pat.axvline(t_today_q, color="lime", lw=1.5, ls="--", alpha=0.5)
    ax_pat.set_title("Window entanglement entropy on each level's OWN leftover substrate", fontsize=8.5)
    ax_pat.legend(loc="lower right", fontsize=8, facecolor="#111115", edgecolor="gray", labelcolor="white")

    # --- ax_matter: bits committed to structure, honestly, vs total entropy ---
    ax_matter.set_facecolor("#0a0a0a")
    ax_matter.set_xlabel("Physical time (years, log scale)")
    ax_matter.set_ylabel("Bits")
    ax_matter.set_xticks(tick_tbf_v)
    ax_matter.set_xticklabels(tick_labels_v, fontsize=7)
    ax_matter.plot(t_bf, entropy_bits, color="cyan", lw=1.8, ls=":", label="entropy_bits(t)  (total, for scale reference)")
    ax_matter.plot(t_bf, total_matter_bits, color="gold", lw=2.4, label="total matter_bits(t)  (cascaded, no correction factor)")
    for lvl, color in zip(levels, colors):
        ax_matter.plot(t_bf, matter_bits[lvl.width], color=color, lw=1.0, alpha=0.85, label=f"matter_bits w={lvl.width}")
    ax_matter.plot(t_bf[peak_idx], total_matter_bits[peak_idx], "o", color="lime", ms=6)
    ax_matter.set_title("Matter vs entropy at TRUE relative scale -- note the axis, no rescaling applied", fontsize=8.5)
    ax_matter.legend(loc="upper right", fontsize=7, facecolor="#111115", edgecolor="gray", labelcolor="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="white")

    print(f"Saved \u2192 {output_path}")
    print()
    print("Honest disclosure (computed this run, not asserted):")
    print(f"  entropy_bits peak       = {entropy_bits.max():.4f}")
    print(f"  total_matter_bits peak  = {total_matter_bits.max():.4f}  (at t={t_bf[peak_idx]:.1f})")
    ratio = total_matter_bits.max() / max(entropy_bits.max(), 1e-12)
    print(f"  peak matter / peak entropy = {ratio:.6f}")
    for lvl, res in zip(levels, cascade):
        shape = dl.pattern_shape(lvl.a, lvl.b)
        print(f"  level w={lvl.width}: composition (a={lvl.a},b={lvl.b}) -> {shape}, "
              f"n_substrate={res.n_substrate}, survival_prob={res.survival_prob:.4g}, "
              f"match_prob peak={res.match_prob.max():.6g}, "
              f"cumulative_persistent peak={res.cumulative_persistent_prob.max():.6g}, "
              f"matter_bits peak={matter_bits[lvl.width].max():.4f}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_levels(scales_raw: str, compositions_raw: str | None) -> list[LevelSpec]:
    widths = [int(x) for x in scales_raw.split(",") if x.strip()]
    if compositions_raw is None:
        comps = [dl.default_composition(w) for w in widths]
        print("Using DEFAULT compositions (a=w//3, forced a<b) -- a flagged modeling choice:")
        for w, (a, b) in zip(widths, comps):
            print(f"  w={w}: a={a}, b={b}")
    else:
        comps = []
        for pair in compositions_raw.split(","):
            a_str, b_str = pair.split(":")
            comps.append((int(a_str), int(b_str)))
        if len(comps) != len(widths):
            raise ValueError(
                f"--compositions must have one a:b pair per scale ({len(widths)} scales, {len(comps)} given)"
            )
    return [LevelSpec(width=w, a=a, b=b) for w, (a, b) in zip(widths, comps)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cascaded psi-layer driver: chained substrate (exact factorization) "
                    "+ persistence-weighted matter, replacing the independent-scale version."
    )
    parser.add_argument("--n_bits", type=int, default=184)
    parser.add_argument("--t_bf_max", type=float, default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--scales", type=str, default="6,12,20")
    parser.add_argument("--compositions", type=str, default=None,
                         help="e.g. '1:5,2:10,3:17' (a:b per scale). Default: a=w//3, forced a<b.")
    parser.add_argument("--mode", type=str, default="specific", choices=["specific", "class"],
                         help="'specific' = one exact bit ordering (original dicke_layer.pattern_probability, "
                              "answers 'how much distinguishing quantum information'); "
                              "'class' = whole (a,b) composition class, any ordering (dicke_layer.class_probability, "
                              "the macroscopically relevant 'how much matter has formed' signal, matching the "
                              "classical layer's count-based threshold). 'specific' suppresses the matter signal by "
                              "C(width,a) per level, compounding across the cascade -- see module docstrings.")
    parser.add_argument("--slots", type=int, default=50)
    parser.add_argument("--output", type=str, default="emergent_structure_psi_cascade.png")

    args = parser.parse_args()
    levels = parse_levels(args.scales, args.compositions)

    # Reuse multiclock.run_simulation purely for its shared bedrock: t_bf
    # grid, t_bf_max, n_bits. Not for size_measure or matter -- those are
    # recomputed above from the cascade.
    sim = run_simulation(
        n_bits=args.n_bits, scales=[lvl.width for lvl in levels],
        steps=args.steps, t_bf_max=args.t_bf_max, matter_power=1.0,
    )
    plot_results_cascade(sim, levels, slots_per_scale=args.slots, output_path=args.output, mode=args.mode)


if __name__ == "__main__":
    main()
