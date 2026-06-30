"""
Emergent Spacetime Worldlines — Faithful Implementation
===================================================================
Honors Paper III (Emergent Spacetime Structures from Zero-Entropy
Initial States) and Paper IV (Space-time Curvature as an
Information-Theoretic Structure).

v3 revision: the universe's bit-flip clock saturates in n*ln(n) ticks
and unfolds a maximum spatial resolution of 2^n (the framework's own
stated statistics), and the spacetime fabric IS the leftover unfolded
entropy after subtracting what structures have consumed — not a
fixed budget that structures merely dent. Bit-flip time t_bf is
plotted directly (no years-axis stretch): converting it to "years"
via t_phys = t0*exp(t_bf) over a range of ~n*ln(n) ticks overflows
float64 by a wide margin (exp(709) is already the float64 ceiling),
which is itself a sign that bit-flip time, not years, is the
fundamental coordinate here — consistent with the framework's static,
timeless picture.

Copyright 2026 - Juha Meskanen, The IAME Collaboration
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===========================================================================
# CORE COUNTING-EQUATION MACHINERY
# ===========================================================================

def entropy_saturation(t_bf, k_rate, I_max=1.0):
    """
    S(t) = I_max * (1 - exp(-k*t))
    Paper III, sec. 'Comparison to General Relativity'. The only
    entropy curve sanctioned by the papers.
    """
    return I_max * (1.0 - np.exp(-k_rate * t_bf))


def lognormal_abundance(t_bf, peak_t_bf, sigma_ln, amplitude, k_rate):
    """
    Representative lognormal-SHAPED abundance curve N_k(t_bf), gated by
    the entropy gradient dS/dt so that no choice of peak/sigma can let
    structure survive into the saturated (white-noise) regime: once
    dS/dt -> 0, the driving force for structure formation vanishes.
    peak_t_bf, sigma_ln, amplitude are illustrative/representative
    (Paper III: lognormal FORM is filter-invariant, shape parameters
    are not derived).
    """
    t = np.maximum(t_bf, 1e-12)
    mu_ln = np.log(peak_t_bf) + sigma_ln ** 2
    shape = np.exp(-(np.log(t) - mu_ln) ** 2 / (2 * sigma_ln ** 2)) / (t * sigma_ln * np.sqrt(2 * np.pi))
    gate = np.exp(-k_rate * t_bf)  # normalized dS/dt, the only driving force
    gated = shape * gate
    return gated / gated.max() * amplitude


# ===========================================================================
# SIMULATION
# ===========================================================================

def run_simulation(args):
    n = args.n_bits

    # --- Statistical timescale: saturation at n*ln(n) bit-flips ---
    t_bf_max = args.t_bf_max if args.t_bf_max is not None else n * np.log(n)
    # k_rate chosen so S reaches `sat_fraction` of full saturation exactly
    # at t_bf_max, rather than an arbitrary independent rate constant.
    k_rate = args.k_rate if args.k_rate is not None else (
        -np.log(1.0 - args.sat_fraction) / t_bf_max
    )

    t_bf = np.linspace(0.0, t_bf_max, args.steps)
    S = entropy_saturation(t_bf, k_rate)

    # --- Total bits UNFOLDED so far (the rope length released by entropy) ---
    unfolded = S * n  # Paper III: zero entropy -> zero unfolded rope length

    # --- Matter abundance curves N_k(t_bf): representative lognormals,
    #     peak locations given as FRACTIONS of t_bf_max so they
    #     automatically rescale with the n*ln(n) timescale.
    N1 = lognormal_abundance(t_bf, args.peak1_frac * t_bf_max, args.sigma1, args.amp1, k_rate)
    N2 = lognormal_abundance(t_bf, args.peak2_frac * t_bf_max, args.sigma2, args.amp2, k_rate)

    # --- Bits consumed and entity counts, Paper IV 'conservation' eq. ---
    m_bits = args.w1 * N1 + args.w2 * N2   # bits withdrawn from the unfolded rope
    k_count = N1 + N2                       # addressable composite entities

    # m(t) can only consume what has actually unfolded by time t. If this
    # ever clamps, it means structures are demanding more bits than the
    # rope has released — the "more knots than rope length" collapse
    # artifact — so warn loudly rather than silently absorbing it.
    overshoot = m_bits - unfolded
    if np.any(overshoot > 0):
        frac_clamped = np.mean(overshoot > 0)
        print(f"WARNING: structure demand exceeds unfolded entropy budget "
              f"on {frac_clamped:.1%} of the timeline (max overshoot "
              f"{overshoot.max():.2f} bits). Fabric is being clamped to "
              f"zero there -- reduce amp1/amp2/w1/w2 or push peak1_frac/"
              f"peak2_frac later.")
    m_bits = np.minimum(m_bits, unfolded)

    # --- THE BOLD CURVE: spacetime fabric = unfolded rope minus knots ---
    fabric = np.clip(unfolded - m_bits, 0.0, n)

    # --- Full counting-equation resolution (Paper IV, additive form),
    #     shown as a secondary/derived curve for fidelity to the paper's
    #     exact definition; fabric above is the headline quantity.
    R = np.clip((fabric + k_count) / n, 0.0, 1.0)

    return dict(t_bf=t_bf, t_bf_max=t_bf_max, k_rate=k_rate, S=S,
                unfolded=unfolded, N1=N1, N2=N2, m_bits=m_bits,
                k_count=k_count, fabric=fabric, R=R)


# ===========================================================================
# WORLDLINE CONSTRUCTION (comoving coordinates, homogeneous space)
# ===========================================================================

def build_worldlines(N_k, n_slots, seed):
    """
    Each potential entity occupies a FIXED comoving y-slot, evenly
    spaced across [-0.5, 0.5] (homogeneity -> no preferred position).
    Physical y-position = y_comoving * (fabric(t)/n) is applied by the
    caller. Slot activation follows a fixed random rank order so a
    slot's worldline is a genuine persistent segment, not independently
    resampled points per frame.
    """
    rng = np.random.default_rng(seed)
    y_comoving = np.linspace(-0.5, 0.5, n_slots)
    slot_rank = rng.permutation(n_slots)

    N_frac = N_k / max(N_k.max(), 1e-12)
    active_count = np.clip((N_frac * n_slots).astype(int), 0, n_slots)
    active_mask = slot_rank[None, :] < active_count[:, None]
    return y_comoving, active_mask


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_results(sim, args):
    n = args.n_bits
    t_bf = sim["t_bf"]
    envelope = sim["fabric"] / n  # 0 at the singularity, ->1 as fabric saturates

    y1_comoving, active1 = build_worldlines(sim["N1"], args.slots_l1, seed=1)
    y2_comoving, active2 = build_worldlines(sim["N2"], args.slots_l2, seed=2)

    fig, (ax_spacetime, ax_metrics) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Emergent Spacetime Worldlines — Counting-Equation Implementation",
                 fontsize=13, fontweight="bold")

    # --- Left panel: spacetime diagram, bit-flip time on x ---
    ax_spacetime.set_facecolor("#020205")
    ax_spacetime.set_xlabel("Bit-flip time  t  (Planck ticks; saturates at n·ln(n) ≈ "
                             f"{sim['t_bf_max']:.0f})")
    ax_spacetime.set_ylabel("Comoving y × fabric(t)/n   (spatial extent)")

    # The fabric IS the headline curve: the rope unfolding then partially
    # consumed by structure, drawn boldest as requested.
    ax_spacetime.fill_between(t_bf, -envelope / 2, envelope / 2, color="gainsboro",
                               alpha=0.18, label="spacetime fabric  (unfolded − consumed)")
    ax_spacetime.plot(t_bf, envelope / 2, color="white", lw=2.2, alpha=0.9)
    ax_spacetime.plot(t_bf, -envelope / 2, color="white", lw=2.2, alpha=0.9)

    for i, y0 in enumerate(y1_comoving):
        mask = active1[:, i]
        if mask.any():
            ax_spacetime.plot(t_bf[mask], y0 * envelope[mask], color="cyan", lw=0.6, alpha=0.35)
    for i, y0 in enumerate(y2_comoving):
        mask = active2[:, i]
        if mask.any():
            ax_spacetime.plot(t_bf[mask], y0 * envelope[mask], color="magenta", lw=0.9, alpha=0.45)

    ax_spacetime.plot([], [], color="white", lw=2.2, label="spacetime fabric boundary")
    ax_spacetime.plot([], [], color="cyan", lw=1.5, label="L1 neutrino worldlines")
    ax_spacetime.plot([], [], color="magenta", lw=1.5, label="L2 hadron worldlines")
    ax_spacetime.legend(loc="upper left", facecolor="#111115", edgecolor="gray",
                         labelcolor="white", fontsize=8)
    ax_spacetime.text(0.02, 0.02,
                       f"max spatial resolution at fabric=n: 2^{n:.0f} ≈ {2.0**n:.2e} Planck lengths",
                       transform=ax_spacetime.transAxes, color="gray", fontsize=7)

    # --- Right panel: metrics, bit-flip time on x ---
    ax_metrics.set_xlabel("Bit-flip time  t  (Planck ticks)")
    ax_metrics.set_ylabel("Fraction of n")

    ax_metrics.plot(t_bf, sim["S"], color="red", linestyle="--", lw=1.5,
                     label="S(t)/I_max  entropy saturation [exact]")
    ax_metrics.plot(t_bf, sim["unfolded"] / n, color="silver", linestyle=":", lw=1.5,
                     label="unfolded(t)/n = S(t)  (rope released)")
    ax_metrics.plot(t_bf, sim["fabric"] / n, color="white", lw=3.0,
                     label="fabric(t)/n  (unfolded − consumed)  [bold]")
    ax_metrics.plot(t_bf, sim["R"], color="orange", lw=1.8,
                     label="R(t) = (fabric+k)/n  [Paper IV full eq.]")
    ax_metrics.plot(t_bf, sim["N1"] / max(sim["N1"].max(), 1e-12), color="cyan",
                     lw=1.2, alpha=0.7, label="N1(t) [normalized]")
    ax_metrics.plot(t_bf, sim["N2"] / max(sim["N2"].max(), 1e-12), color="magenta",
                     lw=1.2, alpha=0.7, label="N2(t) [normalized]")

    ax_metrics.set_facecolor("#0a0a0a")
    ax_metrics.set_ylim(-0.05, 1.15)
    ax_metrics.legend(loc="upper left", fontsize=8, facecolor="#111115",
                       edgecolor="gray", labelcolor="white")
    ax_metrics.set_title(f"k_rate={sim['k_rate']:.5f} (S reaches {args.sat_fraction:.0%} "
                          f"of saturation at t={sim['t_bf_max']:.0f})")

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, facecolor="white")
    print(f"Saved figure to {args.output}")
    print(f"t_bf_max (n·ln n) = {sim['t_bf_max']:.2f}")
    print(f"k_rate = {sim['k_rate']:.6f}")
    print(f"fabric(t)/n range: [{(sim['fabric']/n).min():.4f}, {(sim['fabric']/n).max():.4f}]")
    print(f"max m(t)/n consumed: {(sim['m_bits']/n).max():.4f}")
    print(f"2^n (max spatial resolution, Planck lengths) = {2.0**n:.4e}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(description="Faithful emergent-spacetime worldline demonstrator.")
    p.add_argument("--n_bits", type=float, default=184.0,
                    help="Total bit budget n (default: framework's established n≈184).")
    p.add_argument("--t_bf_max", type=float, default=None,
                    help="Max bit-flip time. Default: n*ln(n) (the framework's own saturation tick count).")
    p.add_argument("--sat_fraction", type=float, default=0.99,
                    help="Fraction of full entropy saturation reached at t_bf_max (sets k_rate if not given explicitly).")
    p.add_argument("--k_rate", type=float, default=None,
                    help="Override: explicit saturation rate k in S(t)=I_max(1-exp(-kt)). Default: derived from sat_fraction and t_bf_max.")
    p.add_argument("--steps", type=int, default=3000)

    # Lognormal abundance shape parameters (illustrative). Peaks given
    # as FRACTIONS of t_bf_max so they auto-scale with n*ln(n).
    p.add_argument("--peak1_frac", type=float, default=0.20, help="L1 abundance peak, as fraction of t_bf_max.")
    p.add_argument("--sigma1", type=float, default=0.5)
    p.add_argument("--amp1", type=float, default=10.0, help="L1 peak entity count.")
    p.add_argument("--peak2_frac", type=float, default=0.45, help="L2 abundance peak, as fraction of t_bf_max.")
    p.add_argument("--sigma2", type=float, default=0.6)
    p.add_argument("--amp2", type=float, default=8.0, help="L2 peak entity count.")

    # Bit-widths per composite entity (illustrative)
    p.add_argument("--w1", type=float, default=2.0, help="Bits per L1 (neutrino) entity.")
    p.add_argument("--w2", type=float, default=5.0, help="Bits per L2 (hadron) entity.")

    # Worldline rendering
    p.add_argument("--slots_l1", type=int, default=60)
    p.add_argument("--slots_l2", type=int, default=40)

    p.add_argument("--output", type=str, default="spacetime_worldlines.png")
    args = p.parse_args()

    sim = run_simulation(args)
    plot_results(sim, args)


if __name__ == "__main__":
    main()
