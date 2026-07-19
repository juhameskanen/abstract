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


def quantum_cascade_series(n_bits: int, t_bf: FloatArray, levels: list[LevelSpec], mode: str = "specific"):
    k_vals = dl.k_of_tau(n_bits, t_bf)
    k_int = np.clip(np.round(k_vals).astype(int), 0, n_bits)
    cascade = run_cascade_series(n_bits, k_int.astype(float), levels, mode=mode)
    matter_bits = {}
    for res in cascade:
        matter_bits[res.spec.width] = (
            res.n_windows_available * res.spec.width * res.cumulative_persistent_prob
        )
    entropy_bits = n_bits * combinatorial_entropy_bits(n_bits, k_int.astype(float))
    return cascade, matter_bits, entropy_bits, k_int


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
        f"PSI-LAYER, CASCADED (mode={mode})\n"
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
    for lvl, res, color in zip(levels, cascade, colors):
        ax_met.plot(t_bf, res.match_prob, color=color, lw=1.2, ls=":", label=f"w={lvl.width} match_prob alone")
        ax_met.plot(t_bf, res.cumulative_persistent_prob, color=color, lw=1.8, label=f"w={lvl.width} chained")
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


if __name__ == "__main__":
    levels = parse_levels("6,12,20", None)
    sim = run_simulation(n_bits=184, scales=[l.width for l in levels], steps=3000, matter_power=1.0)
    plot_results_cascade(sim, levels, slots_per_scale=50, output_path="cascade_class.png", mode="class")
    plot_results_cascade(sim, levels, slots_per_scale=50, output_path="cascade_specific.png", mode="specific")
