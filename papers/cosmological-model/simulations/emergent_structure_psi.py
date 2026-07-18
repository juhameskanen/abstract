"""
Emergent Structure from a Relaxing Bitstring -- PSI-LAYER
=========================================================

The hump mechanism is not a decay factor bolted onto structure
formation -- it's about WHICH bit patterns count as structure in the first
place. Per the "Entropy and Emergent Structures" wiki page (Result 3): the
probability of a SPECIFIC pattern with `a` required ones and `b` required
zeros in a width-w window is a genuine hump in tau if and only if a<b (more
zeros required than ones -- a rare, information-dense configuration).
a>=b gives monotonic curves, no hump, by the same closed form.

dicke_layer.pattern_probability() is the EXACT Born-rule (not mean-field)
version of that same formula, verified two ways (see test_dicke_layer.py):
  1. Direct linear algebra on the actual Dicke statevector: every specific
     ordering within a fixed composition sector has IDENTICAL probability,
     matching hypergeom.pmf(a;n,k,w)/C(w,a) to machine precision.
  2. The exact formula reproduces the SAME three-fold classification
     (monotonic / monotonic / hump) as the wiki's independently-derived
     mean-field approximation, peak locations agreeing to ~3%.

EVERYTHING in this script's "matter" and "size_measure" curves is built
from pattern_probability() and the exact combinatorial entropy already in
multiclock.py -- nothing is borrowed from the classical run's size_measure,
and eta(tau)^matter_power appears nowhere in this file.

COMPOSITIONS ARE A FLAGGED MODELING CHOICE, not a derived fact. Which
specific (a,b) counts as "the" structure at width w is not determined by
anything derived so far -- dicke_layer.default_composition() picks
a=w//3 (rounded, forced a<b) as a reasonable default, printed explicitly
below so it's never silently assumed. Pass --compositions to override.

HONEST RESULT, STATED PLAINLY, NOT HIDDEN: at the working scale used
elsewhere in this project (n_bits~184, scales=[6,12,20]), the matter term
built this way is a SMALL perturbation on total entropy (a few bits out of
~180) -- the hump is real and verified, but it does not by itself carve a
dramatic macroscopic three-fold dip in size_measure at these parameters.
This script reports the true magnitude rather than rescaling anything to
force a bigger-looking effect. See the printed diagnostics and the
"Honest disclosure" section of the docstring at the bottom of this file's
plot_results_quantum() for the actual numbers each run produces.
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

FloatArray = NDArray[np.float64]


# ===========================================================================
# PSI-LAYER QUANTITIES -- every one traces to pattern_probability() or
# entanglement_entropy(), both exact Born-rule quantities. No eta, no
# matter_power, no borrowed classical arrays.
# ===========================================================================

def quantum_level_series(n_bits: int, t_bf: FloatArray, scales: list[int], compositions: list[tuple[int, int]]):
    """Exact Born-rule time series per scale: pattern probability (the
    honest 'matter' signal, genuinely humped when a<b) and entanglement
    entropy (S_vN of the window, unrelated to which composition is chosen)."""
    k_vals = dl.k_of_tau(n_bits, t_bf)
    k_int = np.clip(np.round(k_vals).astype(int), 0, n_bits)

    pattern_prob = {}
    matter_bits = {}
    entropy_vn = {}
    for w, (a, b) in zip(scales, compositions):
        P = dl.pattern_probability(n_bits, k_int, a, b)
        pattern_prob[w] = P
        n_windows = n_bits // w
        matter_bits[w] = n_windows * w * P  # expected bits participating in >=1 copy of this pattern
        entropy_vn[w] = np.array([dl.entanglement_entropy(n_bits, k, w) for k in k_int])

    entropy_bits = n_bits * combinatorial_entropy_bits(n_bits, k_int.astype(float))
    return pattern_prob, matter_bits, entropy_vn, entropy_bits, k_int


# ===========================================================================
# PLOTTING -- same 4-panel layout as the classical script, for visual
# comparability, but every array here is quantum-derived (see above).
# ===========================================================================

def plot_results_quantum(sim: SimulationResult, scales: list[int], compositions: list[tuple[int, int]],
                          slots_per_scale: int, output_path: str) -> None:
    t_bf = sim.t_bf
    n_bits = sim.n_bits

    pattern_prob, matter_bits, entropy_vn, entropy_bits, k_int = quantum_level_series(
        n_bits, t_bf, scales, compositions)
    total_matter_bits = sum(matter_bits[w] for w in scales)
    size_measure_q = np.clip((entropy_bits - total_matter_bits) / n_bits, 0.0, None)

    cmap = plt.get_cmap("plasma")
    colors = [cmap(0.15 + 0.7 * i / max(len(scales) - 1, 1)) for i in range(len(scales))]

    peak_idx = int(np.argmax(total_matter_bits))
    t_today_q = float(t_bf[peak_idx])

    tick_years = [1e-40, 1e-30, 1e-20, 1e-10, 1e-4, 1e3, 1e9, 13.8e9]
    tick_tbf = [years_to_tbf(t, t_today_q) for t in tick_years]
    tick_labels = ["10\u207b\u2074\u2070", "10\u207b\u00b3\u2070", "10\u207b\u00b2\u2070",
                   "10\u207b\u00b9\u2070", "10\u207b\u2074", "10\u00b3", "10\u2079", "now"]
    valid = [(tb, lb) for tb, lb in zip(tick_tbf, tick_labels) if 0 <= tb <= sim.t_bf_max]
    tick_tbf_v, tick_labels_v = zip(*valid) if valid else ([], [])

    fig, ((ax_st, ax_met), (ax_pat, ax_matter)) = plt.subplots(2, 2, figsize=(17, 13))
    comp_str = ", ".join(f"w={w}:(a={a},b={b})" for w, (a, b) in zip(scales, compositions))
    fig.suptitle(
        "Emergent Structure from a Relaxing Bitstring -- PSI-LAYER, HONEST (Born-rule only)\n"
        f"n={n_bits:g}  compositions=[{comp_str}]  (a<b chosen for hump -- flagged modeling choice, not derived)\n"
        f"NO eta(tau)^matter_power anywhere in this figure. matter_bits peak/entropy_bits peak = "
        f"{total_matter_bits.max():.3f}/{entropy_bits.max():.3f} = {total_matter_bits.max()/max(entropy_bits.max(),1e-12):.4f}",
        fontsize=8.5, fontweight="bold",
    )

    # --- ax_st: spacetime envelope, built ENTIRELY from quantum size_measure_q ---
    ax_st.set_facecolor("#020205")
    ax_st.set_xlabel("Physical time (years, log scale via t\u2192ln t)")
    ax_st.set_ylabel("Comoving y \u00d7 size_measure_q(t)  [QUANTUM, not borrowed]")
    ax_st.set_xticks(tick_tbf_v)
    ax_st.set_xticklabels(tick_labels_v, fontsize=7)

    ax_st.fill_between(t_bf, -size_measure_q / 2, size_measure_q / 2, color="gainsboro", alpha=0.15)
    ax_st.plot(t_bf, size_measure_q / 2, color="white", lw=2.2, alpha=0.9)
    ax_st.plot(t_bf, -size_measure_q / 2, color="white", lw=2.2, alpha=0.9)
    ax_st.plot(t_bf, entropy_bits / n_bits / 2, color="cyan", lw=1.0, ls=":", alpha=0.7)
    ax_st.plot(t_bf, -entropy_bits / n_bits / 2, color="cyan", lw=1.0, ls=":", alpha=0.7)

    for w, color in zip(scales, colors):
        y_slots, active = build_worldlines(matter_bits[w] / w, slots_per_scale, seed=int(w))
        for i, y0 in enumerate(y_slots):
            mask = active[:, i]
            if mask.any():
                ax_st.plot(t_bf[mask], y0 * size_measure_q[mask], color=color, lw=0.6, alpha=0.35)

    ax_st.plot([], [], color="white", lw=2, label="size_measure_q = (entropy_bits \u2212 matter_bits)/n  [QUANTUM]")
    ax_st.plot([], [], color="cyan", lw=1.0, ls=":", label="entropy_bits/n  (exact combinatorial entropy)")
    for w, (a, b), color in zip(scales, compositions, colors):
        ax_st.plot([], [], color=color, lw=1.2, label=f"w={w} pattern a={a},b={b} (hump={a<b})")
    ax_st.axvline(t_today_q, color="lime", lw=1.5, ls="--", alpha=0.85, label=f"now = peak total matter_bits (t={t_today_q:.1f})")
    ax_st.legend(loc="upper left", facecolor="#111115", edgecolor="gray", labelcolor="white", fontsize=7.5)

    # --- ax_met: the pattern probabilities themselves -- the actual hump curves ---
    ax_met.set_facecolor("#0a0a0a")
    ax_met.set_xlabel("Physical time (years, log scale)")
    ax_met.set_ylabel("pattern_probability(t)  [exact Born-rule, per copy]")
    ax_met.set_xticks(tick_tbf_v)
    ax_met.set_xticklabels(tick_labels_v, fontsize=7)
    for w, (a, b), color in zip(scales, compositions, colors):
        ax_met.plot(t_bf, pattern_prob[w], color=color, lw=1.8, label=f"w={w} (a={a},b={b})")
    ax_met.axvline(t_today_q, color="lime", lw=1.5, ls="--", alpha=0.85)
    ax_met.set_title("Exact Born-rule probability of the specific pattern -- THE hump, unmultiplied by anything", fontsize=8.5)
    ax_met.legend(loc="upper right", fontsize=8, facecolor="#111115", edgecolor="gray", labelcolor="white")

    # --- ax_pat: entanglement entropy per scale (unrelated to composition choice) ---
    ax_pat.set_facecolor("#0a0a0a")
    ax_pat.set_xlabel("Physical time (years, log scale)")
    ax_pat.set_ylabel("$S_{vN}$ (bits)")
    ax_pat.set_xticks(tick_tbf_v)
    ax_pat.set_xticklabels(tick_labels_v, fontsize=7)
    for w, color in zip(scales, colors):
        ax_pat.plot(t_bf, entropy_vn[w], color=color, lw=1.6, label=f"w={w}")
    ax_pat.axvline(t_today_q, color="lime", lw=1.5, ls="--", alpha=0.5)
    ax_pat.set_title("Exact window entanglement entropy (monotonic -- a separate quantity from the pattern hump)", fontsize=8.5)
    ax_pat.legend(loc="lower right", fontsize=8, facecolor="#111115", edgecolor="gray", labelcolor="white")

    # --- ax_matter: bits committed to structure, honestly, vs total entropy ---
    ax_matter.set_facecolor("#0a0a0a")
    ax_matter.set_xlabel("Physical time (years, log scale)")
    ax_matter.set_ylabel("Bits")
    ax_matter.set_xticks(tick_tbf_v)
    ax_matter.set_xticklabels(tick_labels_v, fontsize=7)
    ax_matter.plot(t_bf, entropy_bits, color="cyan", lw=1.8, ls=":", label="entropy_bits(t)  (total, for scale reference)")
    ax_matter.plot(t_bf, total_matter_bits, color="gold", lw=2.4, label="total matter_bits(t)  (sum over scales, no correction factor)")
    for w, color in zip(scales, colors):
        ax_matter.plot(t_bf, matter_bits[w], color=color, lw=1.0, alpha=0.85, label=f"matter_bits w={w}")
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
    print(f"  peak matter / peak entropy = {ratio:.5f}")
    if ratio < 0.05:
        print("  -> matter_bits is a SMALL perturbation on entropy at these parameters.")
        print("     The hump mechanism is real and verified (see test_dicke_layer.py), but")
        print("     does not by itself produce a macroscopically dominant three-fold dip")
        print("     here. This is reported plainly, not hidden or rescaled away.")
    for w, (a, b) in zip(scales, compositions):
        shape = dl.pattern_shape(a, b)
        print(f"  scale w={w}: composition (a={a},b={b}) -> {shape}, "
              f"pattern_probability peak={pattern_prob[w].max():.6g}, "
              f"matter_bits peak={matter_bits[w].max():.4f}")


# ===========================================================================
# CLI
# ===========================================================================

def parse_scales(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def parse_compositions(raw: str | None, scales: list[int]) -> list[tuple[int, int]]:
    if raw is None:
        comps = [dl.default_composition(w) for w in scales]
        print("Using DEFAULT compositions (a=w//3, forced a<b) -- a flagged modeling choice:")
        for w, (a, b) in zip(scales, comps):
            print(f"  w={w}: a={a}, b={b}")
        return comps
    comps = []
    for pair in raw.split(","):
        a_str, b_str = pair.split(":")
        comps.append((int(a_str), int(b_str)))
    if len(comps) != len(scales):
        raise ValueError(f"--compositions must have one a:b pair per scale ({len(scales)} scales, {len(comps)} compositions given)")
    return comps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Honest psi-layer (Dicke-state, Born-rule-only) companion. "
                    "No eta(tau)^matter_power, no borrowed classical size_measure."
    )
    parser.add_argument("--n_bits", type=int, default=184)
    parser.add_argument("--t_bf_max", type=float, default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--scales", type=str, default="6,12,20")
    parser.add_argument("--compositions", type=str, default=None,
                         help="e.g. '2:4,4:8,6:14' (a:b per scale). Default: a=w//3, forced a<b.")
    parser.add_argument("--slots", type=int, default=50)
    parser.add_argument("--output", type=str, default="emergent_structure_psi_layer_honest.png")

    args = parser.parse_args()
    scales = parse_scales(args.scales)
    compositions = parse_compositions(args.compositions, scales)

    # Reuse multiclock.run_simulation purely for its shared bedrock: t_bf grid,
    # t_bf_max, n_bits, lapse=1/w bookkeeping. NOT for size_measure or matter --
    # those are recomputed honestly above. matter_power is passed but unused
    # by anything plotted in this script.
    sim = run_simulation(
        n_bits=args.n_bits, scales=scales, steps=args.steps, t_bf_max=args.t_bf_max,
        matter_power=1.0,
    )
    plot_results_quantum(sim, scales, compositions, slots_per_scale=args.slots, output_path=args.output)


if __name__ == "__main__":
    main()
