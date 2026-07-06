#!/usr/bin/env python3
"""
Ehrenfest Pattern-Probability Family Classification
=====================================================

For a bit pattern of length L with k ones (weight k), the probability of
finding that EXACT pattern at a position in the relaxing bitstring is

    P(k | tau) = p(tau)^k * (1 - p(tau))^(L-k),      p(tau) = 0.5*(1-exp(-2 tau))

Since all patterns of the same weight k give the identical curve (bits are
i.i.d.), there are only L+1 distinct curves for a given L, not 2^L. Each
one falls into exactly one of three shape families, determined purely by
comparing k to L/2 (found by solving d/dp[ln P] = 0 -> p* = k/L):

  FALLING  : k = 0            -> P = (1-p)^L, strictly decreasing
  RISING   : k >= L/2          -> p* = k/L outside reachable range (0,0.5),
                                   so P rises monotonically, never turns over
  BUMP     : 0 < k < L/2       -> P rises, peaks at tau* = -0.5*ln(1 - 2k/L),
                                   then falls back toward equilibrium 0.5^L
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def p_analytic(tau):
    return 0.5 * (1.0 - np.exp(-2.0 * tau))


def pattern_curve(k, L, tau):
    p = p_analytic(tau)
    return (p ** k) * ((1.0 - p) ** (L - k))


def classify(k, L):
    if k == 0:
        return "falling"
    elif k >= L / 2:
        return "rising"
    else:
        return "bump"


def peak_tau(k, L):
    """tau* where P(k|tau) peaks, for bump-family k only."""
    ratio = 2.0 * k / L
    if ratio >= 1.0:
        return None
    return -0.5 * np.log(1.0 - ratio)


def plot_families(L, max_tau, outfile, show_all=True, k_subset=None):
    tau_grid = np.linspace(0.0, max_tau, 600)

    fig, ax = plt.subplots(figsize=(10, 6.5))

    k_values = k_subset if k_subset is not None else range(0, L + 1)

    falling_ks = [k for k in k_values if classify(k, L) == "falling"]
    bump_ks    = [k for k in k_values if classify(k, L) == "bump"]
    rising_ks  = [k for k in k_values if classify(k, L) == "rising"]

    cmap_bump   = cm.get_cmap("autumn_r")
    cmap_rising = cm.get_cmap("winter_r")

    handles = []

    # Falling family (k=0) - single curve, always exists
    for k in falling_ks:
        curve = pattern_curve(k, L, tau_grid)
        line, = ax.plot(tau_grid, curve, color="black", lw=2.5, zorder=5,
                        label=f"FALLING  k={k}  (only k=0)")
        handles.append(line)

    # Bump family - color gradient by k/L (closer to L/2 = darker/later peak)
    for k in bump_ks:
        frac = k / (L / 2.0)  # 0..~1
        colour = cmap_bump(0.15 + 0.75 * frac)
        curve = pattern_curve(k, L, tau_grid)
        line, = ax.plot(tau_grid, curve, color=colour, lw=1.3, alpha=0.85)
        tp = peak_tau(k, L)
        if tp is not None and tp <= max_tau:
            ax.plot(tp, pattern_curve(k, L, np.array([tp]))[0], "o",
                    color=colour, ms=4, zorder=4)

    # Rising family - color gradient by k/L (closer to 1 = brighter)
    for k in rising_ks:
        frac = (k - L / 2.0) / (L / 2.0) if L > 0 else 0
        colour = cmap_rising(0.15 + 0.75 * max(frac, 0))
        curve = pattern_curve(k, L, tau_grid)
        line, = ax.plot(tau_grid, curve, color=colour, lw=1.3, alpha=0.85)

    # Family legend proxies
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="black", lw=2.5,
               label=f"FALLING (k=0, {len(falling_ks)} curve)"),
        Line2D([0], [0], color=cmap_bump(0.6), lw=2,
               label=f"BUMP (0<k<L/2, {len(bump_ks)} curves) — peak marked \u25CF"),
        Line2D([0], [0], color=cmap_rising(0.6), lw=2,
               label=f"RISING (k\u2265L/2, {len(rising_ks)} curves)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=9.5, framealpha=0.9)

    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$ = flip count / n  (normalized bit-flip time)", fontsize=11)
    ax.set_ylabel(r"$P(\mathrm{exact\ pattern}\mid\tau) = p(\tau)^k (1-p(\tau))^{L-k}$   (log scale)",
                  fontsize=11)
    ax.set_title(
        f"Pattern-Probability Shape Families  (L={L})\n"
        r"classified by $k$ vs $L/2$: falling ($k{=}0$), bump ($0{<}k{<}L/2$), rising ($k{\geq}L/2$)",
        fontsize=11)
    ax.axvline(1.0, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax.grid(alpha=0.2, which="both")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved -> {outfile}")

    print(f"\nL={L}: {len(falling_ks)} falling, {len(bump_ks)} bump, {len(rising_ks)} rising "
          f"(out of {L+1} distinct weight-curves, {2**L} total patterns)")
    print("Sample bump peak times (tau* where curve turns over):")
    for k in bump_ks[:8]:
        tp = peak_tau(k, L)
        print(f"  k={k:3d}  (p*=k/L={k/L:.3f})   tau* = {tp:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=12, help="pattern length (default 12)")
    ap.add_argument("--max-tau", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="family_curves.png")
    args = ap.parse_args()
    plot_families(args.L, args.max_tau, args.out)


if __name__ == "__main__":
    main()
