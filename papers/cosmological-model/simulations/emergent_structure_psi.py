"""
Emergent Structure from a Relaxing Bitstring -- PSI-LAYER (Dicke-state) version
================================================================================

Same driver, same 4-panel figure, same styling and axes as
emergent_structure_relativistic.py -- the classical D-layer "statistical
shadow" script. The only thing that changes is WHERE the per-scale
"how much structure has formed at width w" numbers come from.

Classical: multiclock.run_simulation()'s level_state recursion on the
           actual bitstring.
Here:      the exact Born-rule answer for the SAME question, computed
           directly on the shared Dicke state |D_n^k>, using
           dicke_layer.py's SINGLE-LEVEL primitives.

Why single-level dicke_layer and NOT the nested dicke_cascade: per
dicke_layer.py's own docstring, "species/scales are confirmed to share
the SAME n and the SAME k(tau) -- exactly as multiclock.level_state's
classical cascade already assumes." That is a parallel, independent-per-
width picture (every scale looks at the SAME total (n, k), just a
different window w) -- which is exactly what these diagrams are built on.
dicke_cascade.py's disjoint-mode-peeling recursion is a DIFFERENT,
still-unverified alternative model (N actually shrinks level to level);
mixing it in here would silently change what the diagrams mean, so it's
deliberately left out of this script.

Explicit mapping used (spelled out, not hidden in the code):

  fabric(t) at scale w        = n_bits * P(j=0 | n_bits, k(t), w)
        Born-rule probability a w-mode window shows zero excitations.

  structure_count(t) at w     = n_bits * (1 - P(j=0 | n_bits, k(t), w))
        raw quantum structure-formation probability, UNCORRECTED.

  per_scale_matter(t) at w    = structure_count(t) at w, unchanged.
        This is deliberately NOT run through multiclock's
        eta(tau)^matter_power evaporation correction: that correction has
        no established quantum-side derivation yet (it's explicitly one
        of the open items on the list -- explain or eliminate
        matter_power -- so it's left out here rather than faked in).

  S_vN(t) at w                 = dicke_layer.entanglement_entropy(n,k(t),w)
        plotted ALONGSIDE (not replacing) the classical entropy = H(p(t))
        curve in the metrics panel -- that side-by-side comparison is the
        whole point of having a psi-layer version at all.

Everything else -- size_measure, t_today, the tick labels, and the
differential-aging ledger panel (lapse=1/w is pure scale-width
bookkeeping, not a classical-vs-quantum question) -- comes straight from
multiclock.run_simulation(), unchanged: those are properties of n and
k(tau) alone, shared bedrock underneath both pictures.
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
import dicke_layer as dl

FloatArray = NDArray[np.float64]


# ===========================================================================
# PSI-LAYER PER-SCALE QUANTITIES (single-level, same (n,k) for every w)
# ===========================================================================

def quantum_level_series(n_bits: float, t_bf: FloatArray, scales: list[float]):
    """Exact Born-rule fabric/structure/entropy time series per scale,
    all sharing the SAME clock trajectory k(tau) as the classical run
    (same TRUE_K_RATE, same n_bits -- reused, not re-derived)."""
    n_int = int(round(n_bits))
    k_vals = n_bits * (0.5 * (1 - np.exp(-TRUE_K_RATE * t_bf / n_bits)))
    k_int = np.clip(np.round(k_vals).astype(int), 1, n_int - 1)

    fabric = {}
    structure = {}
    entropy_vn = {}
    for w in scales:
        w_int = int(round(w))
        p_empty = np.array([dl.window_marginal(n_int, k, w_int)[0] for k in k_int])
        fabric[w] = n_bits * p_empty
        structure[w] = n_bits * (1.0 - p_empty)
        entropy_vn[w] = np.array([dl.entanglement_entropy(n_int, k, w_int) for k in k_int])
    return fabric, structure, entropy_vn


# ===========================================================================
# PLOTTING -- same 4-panel layout as emergent_structure_relativistic.py
# ===========================================================================

def plot_results_quantum(sim: SimulationResult, slots_per_scale: int, output_path: str) -> None:
    t_bf = sim.t_bf
    n_bits = sim.n_bits
    scales = [lvl.width for lvl in sim.levels]

    fabric_q, structure_q, entropy_vn_q = quantum_level_series(n_bits, t_bf, scales)
    per_scale_matter_q = [structure_q[w] for w in scales]   # no matter_power correction (open item)
    total_matter_q = sum(per_scale_matter_q)
    fabric_env_q = sum(fabric_q[w] for w in scales) / (n_bits * len(scales))

    cmap = plt.get_cmap("plasma")
    colors = [cmap(0.15 + 0.7 * i / max(len(sim.levels) - 1, 1)) for i in range(len(sim.levels))]

    tick_years = [1e-40, 1e-30, 1e-20, 1e-10, 1e-4, 1e3, 1e9, 13.8e9]
    tick_tbf = [years_to_tbf(t, sim.t_today) for t in tick_years]
    tick_labels = ["10\u207b\u2074\u2070", "10\u207b\u00b3\u2070", "10\u207b\u00b2\u2070",
                   "10\u207b\u00b9\u2070", "10\u207b\u2074", "10\u00b3", "10\u2079", "now"]
    valid = [(tb, lb) for tb, lb in zip(tick_tbf, tick_labels) if 0 <= tb <= sim.t_bf_max]
    tick_tbf_v, tick_labels_v = zip(*valid) if valid else ([], [])

    fig, ((ax_st, ax_met), (ax_ledger, ax_matter)) = plt.subplots(2, 2, figsize=(17, 13))
    scale_str = ", ".join(f"{w:g}" for w in scales)
    now_note = "auto = peak matter" if sim.t_today_auto else "manual override"
    fig.suptitle(
        "Emergent Structure from a Relaxing Bitstring -- PSI-LAYER (Dicke-state, Born-rule)\n"
        f"n={n_bits:g}  scales=[{scale_str}] bits  |  k_rate={sim.k_rate:g} "
        f"(TRUE Ehrenfest={TRUE_K_RATE:g})  |  matter_power correction: NOT APPLIED (open item)"
        f"  |  now: t={sim.t_today:.2f} ({now_note})  |  p-saturation at t_bf_max: {sim.saturation_reached:.4f}",
        fontsize=9, fontweight="bold",
    )

    # --- ax_st: spacetime / worldlines, using QUANTUM per-scale matter ---
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

    for w, matter, color in zip(scales, per_scale_matter_q, colors):
        y_slots, active = build_worldlines(matter, slots_per_scale, seed=int(w))
        for i, y0 in enumerate(y_slots):
            mask = active[:, i]
            if mask.any():
                ax_st.plot(t_bf[mask], y0 * size_env[mask], color=color, lw=0.6, alpha=0.35)

    ax_st.plot([], [], color="white", lw=2, label="size_measure (actual universe size)")
    ax_st.plot([], [], color="cyan", lw=1.0, ls=":", label="entropy = H(p(t)) (flat-state binary entropy)")
    for w, color in zip(scales, colors):
        ax_st.plot([], [], color=color, lw=1.2, label=f"quantum matter w={w:g} (Born-rule, lapse=1/{w:g})")
    ax_st.axvline(sim.t_today, color="lime", lw=1.5, ls="--", alpha=0.85, label=f"now (t={sim.t_today:g})")
    ax_st.legend(loc="upper left", facecolor="#111115", edgecolor="gray", labelcolor="white", fontsize=8)

    # --- ax_met: metrics, classical entropy/fabric/size_measure PLUS
    #     the quantum S_vN(t) per scale overlaid for direct comparison ---
    ax_met.set_facecolor("#0a0a0a")
    ax_met.set_xlabel("Physical time (years, log scale)")
    ax_met.set_ylabel("Fraction of n / bits (mixed, see legend)")
    ax_met.set_xticks(tick_tbf_v)
    ax_met.set_xticklabels(tick_labels_v, fontsize=7)
    ax_met.plot(t_bf, sim.entropy, color="cyan", lw=1.5, ls=":", label="classical entropy = H(p(t))")
    ax_met.plot(t_bf, fabric_env_q, color="gray", lw=1.2, ls="--", label="quantum fabric(t)/n [mean over scales]")
    ax_met.plot(t_bf, sim.size_measure, color="orange", lw=2.2, label="size_measure")
    for w, color in zip(scales, colors):
        norm = structure_q[w] / max(structure_q[w].max(), 1e-12)
        ax_met.plot(t_bf, norm, color=color, lw=1.2, ls="--", label=f"quantum structure w={w:g} [norm]")
        ax_met.plot(t_bf, entropy_vn_q[w], color=color, lw=1.6, label=f"$S_{{vN}}$ w={w:g} (bits)")
    ax_met.axvline(sim.t_today, color="lime", lw=1.5, ls="--", alpha=0.85)
    ax_met.legend(loc="upper left", fontsize=6.5, facecolor="#111115", edgecolor="gray", labelcolor="white", ncol=2)

    # --- ax_ledger: differential aging -- UNCHANGED, pure scale bookkeeping ---
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
    ax_ledger.set_title("Differential aging: proper time (solid) vs pending backlog (dashed) -- scale bookkeeping, unchanged", fontsize=8)

    # --- ax_matter: entity counts, QUANTUM structure/matter, no matter_power ---
    ax_matter.set_facecolor("#0a0a0a")
    ax_matter.set_xlabel("Physical time (years, log scale)")
    ax_matter.set_ylabel("Bits (expected count, Born-rule)")
    ax_matter.set_xticks(tick_tbf_v)
    ax_matter.set_xticklabels(tick_labels_v, fontsize=7)
    total_structure_q = sum(structure_q[w] for w in scales)
    ax_matter.plot(t_bf, total_structure_q, color="gray", lw=1.4, ls=":",
                   label="raw quantum structure_count (uncorrected, summed over scales)")
    ax_matter.plot(t_bf, total_matter_q, color="gold", lw=2.4,
                   label="total quantum matter = raw structure_count (NO matter_power correction -- open item)")
    for w, matter, color in zip(scales, per_scale_matter_q, colors):
        ax_matter.plot(t_bf, matter, color=color, lw=1.0, alpha=0.8, label=f"quantum matter w={w:g}")
    peak_idx = int(np.argmax(total_matter_q))
    ax_matter.axvline(sim.t_today, color="lime", lw=1.5, ls="--", alpha=0.85,
                      label="now (from classical run, for reference)")
    ax_matter.plot(t_bf[peak_idx], total_matter_q[peak_idx], "o", color="lime", ms=6)
    ax_matter.legend(loc="upper right", fontsize=6.5, facecolor="#111115", edgecolor="gray", labelcolor="white")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor="white")
    print(f"Saved \u2192 {output_path}")
    print(f"k_rate used: {sim.k_rate:g}  (true Ehrenfest value: {TRUE_K_RATE:g})")
    print("NOTE: matter_power evaporation correction NOT applied to the quantum matter curves "
          "(no established quantum-side derivation yet -- open item).")
    for w in scales:
        print(f"  scale w={w:g}: peak quantum structure_count={structure_q[w].max():.3f}, "
              f"peak S_vN={entropy_vn_q[w].max():.4f} bits")


# ===========================================================================
# CLI -- same flags as emergent_structure_relativistic.py
# ===========================================================================

def parse_scales(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Psi-layer (Dicke-state, Born-rule) companion to the classical multi-clock demonstrator."
    )
    parser.add_argument("--n_bits", type=float, default=184.0)
    parser.add_argument("--t_bf_max", type=float, default=None)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--t_today", type=float, default=None)
    parser.add_argument("--matter_power", type=float, default=1.0)
    parser.add_argument("--scales", type=str, default="6,12,20")
    parser.add_argument("--slots", type=int, default=50)
    parser.add_argument("--output", type=str, default="emergent_structure_psi_layer.png")

    args = parser.parse_args()
    scales = parse_scales(args.scales)

    sim = run_simulation(
        n_bits=args.n_bits, scales=scales, steps=args.steps, t_bf_max=args.t_bf_max,
        t_today=args.t_today,
        matter_power=args.matter_power,
    )
    plot_results_quantum(sim, slots_per_scale=args.slots, output_path=args.output)


if __name__ == "__main__":
    main()
