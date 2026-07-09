#!/usr/bin/env python3
"""Pattern-probability shape families under the exact finite-n Ehrenfest walk (hypergeometric version).

QUESTION THIS SCRIPT ANSWERS
-----------------------------
The BINOMIAL companion script (family_mass_binomial.py) answers "which
weight-class family (falling / bump / rising) of an L-bit window is typical
at relaxation time tau" using the idealized *i.i.d.* model: each of the n
bits is flipped independently with probability p(tau) = 1/2(1-exp(-2*tau)),
so a window's composition follows Binomial(L, p(tau)).

This script asks the SAME question under the actual, EXACT finite-n
combinatorial structure of the Ehrenfest monotone walk

    U: 000...0  ->  111...1        (n bits, k = number of 1s = discrete step)

instead of the continuous i.i.d. approximation. At each discrete step k,
the "ensemble of the walk at that step" is taken to be ALL C(n,k)
equally-likely arrangements of k ones among n positions (the natural
typicality-preserving population at that entropy stage -- i.e. we do not
track *which* bits flipped, only how many). A fixed L-bit window's
composition then follows Hypergeometric(n, k, L): sampling L positions
WITHOUT replacement from a population of n bits of which k are 1.

So this is a consistency / robustness check: does the BUMP-dominance
result from the binomial (i.i.d., "with replacement") toy model survive
once we use the exact, finite-n, without-replacement combinatorics that
actually describes a specific bitstring walking from all-zero to
all-one? Concretely it answers two things:

  1. (family_mass / entropy) At a given step k (equivalently k/n, or the
     normalized Shannon entropy S(k)/S_max = log2 C(n,k) / log2 C(n,n/2)),
     which family -- falling (all-zero window), bump (mixed window), or
     rising (majority-one window) -- holds the majority of exact
     probability mass? (Answer, as before: bump dominates almost the
     entire relaxation from zero entropy to equilibrium.)

  2. (patterns) Is that mass-level dominance still consistent with every
     INDIVIDUAL specific window pattern's probability curve being a
     simple, unremarkable function of k -- i.e. is "bump wins" purely a
     multiplicity/entropy effect (many mixed patterns, each individually
     unremarkable) rather than any single pattern becoming intrinsically
     more likely?

Domain note: only k = 0 .. n/2 is plotted (zero entropy -> equilibrium).
Walking further, k = n/2 .. n, would double back down to zero entropy at
the all-ones string k=n, which is a distinct (and here uninteresting)
regime -- this script only tracks the first half of the walk.

Two outputs are produced, mirroring the binomial script:
  - plot_patterns_exact: individual specific-window-pattern probability
                          curves P(pattern | k), one per weight class j,
                          grouped/colored by family, on a log scale.
                          (Exact-combinatorics analogue of
                          family_binomial_annotated.plot_families.)
  - plot_family_mass_exact: total exact probability MASS in each family as
                          a function of k (with normalized Shannon entropy
                          S(k) overlaid on a twin axis).
                          (Exact-combinatorics analogue of
                          family_binomial_annotated.plot_family_mass.)
"""
import argparse
from math import comb, lgamma, log
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def classify(j: int, L: int) -> str:
    """Classify a window weight j (out of L bits) into a shape family.

    Families:
        "falling": j == 0 (all-zero window; no 1s sampled)
        "rising":  j >= L/2 (majority-ones window)
        "bump":    0 < j < L/2 (mixed window)

    Args:
        j: Number of 1s observed in the L-bit window.
        L: Window length in bits.

    Returns:
        One of "falling", "rising", or "bump".
    """
    if j == 0:
        return "falling"
    elif j >= L / 2:
        return "rising"
    else:
        return "bump"


def entropy_curve(n: int, ks: IntArray) -> FloatArray:
    """Normalized Shannon entropy of the weight-k ensemble, in bits.

    S(k) = log2 C(n,k), normalized so that S(n/2) = 1 (the equilibrium /
    maximum-entropy weight class). Computed via lgamma for numerical
    stability at large n (avoids overflow from comb() on e.g. n=9182).

    Args:
        n: Total bitstring length.
        ks: Array of weight values k at which to evaluate S(k).

    Returns:
        Array of normalized entropy values S(k)/S_max, same shape as ks.
    """
    def log2_comb(n: int, k: int) -> float:
        if k < 0 or k > n:
            return -np.inf
        return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / log(2)
    s_max = log2_comb(n, n // 2)
    return np.array([log2_comb(n, int(k)) / s_max for k in ks])


def hypergeom_pmf(n: int, k: int, L: int, j: int) -> float:
    """Exact probability that a fixed L-bit window contains exactly j ones.

    P(j ones in window | k ones among n total, arranged uniformly at
    random) = C(k,j) * C(n-k, L-j) / C(n,L). This is the family-mass
    quantity: probability summed over all C(L,j) specific window patterns
    that share weight j.

    Args:
        n: Total bitstring length.
        k: Number of 1s in the full n-bit population at this step.
        L: Window length in bits.
        j: Number of 1s to find in the window.

    Returns:
        Exact hypergeometric probability mass at j (0.0 if C(n,L) is 0,
        which cannot occur for valid L <= n).
    """
    denom = comb(n, L)
    if denom == 0:
        return 0.0
    num = comb(k, j) * comb(n - k, L - j)
    return num / denom


def pattern_prob(n: int, k: int, L: int, j: int) -> float:
    """Exact probability of one SPECIFIC L-bit window pattern with weight j.

    Unlike hypergeom_pmf (which sums over all C(L,j) window patterns
    sharing weight j), this gives the probability of one particular,
    fixed arrangement of those j ones within the window -- the exact,
    without-replacement analogue of the binomial script's pattern_curve.

    Since all C(L,j) specific patterns of a given weight are equally
    likely under the uniform-arrangement population model, this is simply
    the family mass divided by its multiplicity:

        pattern_prob(n,k,L,j) = hypergeom_pmf(n,k,L,j) / C(L,j)

    This is computed via hypergeom_pmf rather than the seemingly more
    direct combinatorial identity C(n-L, k-j) / C(n,k), because the latter
    requires forming a central binomial coefficient with an argument of
    order k (which can be ~n/2, i.e. thousands of digits for large n) --
    prohibitively expensive. Routing through hypergeom_pmf keeps every
    binomial coefficient's smaller argument bounded by L, which is cheap
    regardless of how large n is.

    Args:
        n: Total bitstring length.
        k: Number of 1s in the full n-bit population at this step.
        L: Window length in bits.
        j: Number of 1s in the specific window pattern (0 <= j <= L).

    Returns:
        Exact probability of that one specific window pattern.
    """
    multiplicity = comb(L, j)
    if multiplicity == 0:
        return 0.0
    return hypergeom_pmf(n, k, L, j) / multiplicity


def family_mass_exact(n: int, L: int) -> Tuple[IntArray, Dict[str, FloatArray]]:
    """Exact probability mass in each family, for every step k = 0..n.

    Args:
        n: Total bitstring length.
        L: Window length in bits.

    Returns:
        Tuple of (ks, mass) where ks = np.arange(0, n+1) and mass is a
        dict with keys "falling", "bump", "rising", each an array (same
        shape as ks) of total exact probability mass in that family.
    """
    ks: IntArray = np.arange(0, n + 1)
    mass: Dict[str, FloatArray] = {
        "falling": np.zeros(n + 1),
        "bump": np.zeros(n + 1),
        "rising": np.zeros(n + 1),
    }
    for idx, k in enumerate(ks):
        jmin = max(0, L - (n - k))
        jmax = min(L, k)
        for j in range(jmin, jmax + 1):
            p = hypergeom_pmf(n, int(k), L, j)
            mass[classify(j, L)][idx] += p
    return ks, mass


def plot_patterns_exact(
    n: int,
    L: int,
    outfile: str,
    num_k_points: int = 700,
) -> None:
    """Plot individual specific-window-pattern probability curves, by family.

    Exact-combinatorics analogue of the binomial script's plot_families:
    draws pattern_prob(n, k, L, j) for each weight class j = 0..L over a
    grid of steps k = 0..n/2 (zero entropy -> equilibrium only; see module
    docstring), colored by family (falling / bump / rising) on a log
    scale, with bump-family curves marked at their numerically-located
    peak.

    Args:
        n: Total bitstring length.
        L: Window length in bits.
        outfile: Path to save the output PNG figure.
        num_k_points: Number of k-grid points to sample for each curve
            (a subsample of the full 0..n/2 range is used for tractable
            plotting time at large n; the family-mass plot instead uses
            every integer k).

    Returns:
        None. Writes a PNG to outfile and prints a summary to stdout.
    """
    k_grid: IntArray = np.unique(np.linspace(0, n // 2, num_k_points).astype(np.int64))
    tau_grid: FloatArray = k_grid / n

    j_values: range = range(0, L + 1)
    falling_js: List[int] = [j for j in j_values if classify(j, L) == "falling"]
    bump_js: List[int] = [j for j in j_values if classify(j, L) == "bump"]
    rising_js: List[int] = [j for j in j_values if classify(j, L) == "rising"]
    cmap_bump = cm.get_cmap("autumn_r")
    cmap_rising = cm.get_cmap("winter_r")

    fig, ax = plt.subplots(figsize=(10, 6.5))

    def curve_for(j: int) -> FloatArray:
        return np.array([pattern_prob(n, int(k), L, j) for k in k_grid])

    for j in falling_js:
        curve = curve_for(j)
        ax.plot(tau_grid, curve, color="black", lw=2.5, zorder=5)

    for j in bump_js:
        frac = j / (L / 2.0)
        colour = cmap_bump(0.15 + 0.75 * frac)
        curve = curve_for(j)
        ax.plot(tau_grid, curve, color=colour, lw=1.3, alpha=0.85)
        peak_idx = int(np.argmax(curve))
        if 0 < peak_idx < len(curve) - 1:
            ax.plot(tau_grid[peak_idx], curve[peak_idx], "o", color=colour, ms=4, zorder=4)

    for j in rising_js:
        frac = (j - L / 2.0) / (L / 2.0) if L > 0 else 0
        colour = cmap_rising(0.15 + 0.75 * max(frac, 0))
        curve = curve_for(j)
        ax.plot(tau_grid, curve, color=colour, lw=1.3, alpha=0.85)

    legend_elems = [
        Line2D([0], [0], color="black", lw=2.5,
               label=f"FALLING (j=0, {len(falling_js)} curve)"),
        Line2D([0], [0], color=cmap_bump(0.6), lw=2,
               label=f"BUMP (0<j<L/2, {len(bump_js)} curves) — peak marked \u25CF"),
        Line2D([0], [0], color=cmap_rising(0.6), lw=2,
               label=f"RISING (j\u2265L/2, {len(rising_js)} curves)"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=9.5, framealpha=0.9)
    ax.set_yscale("log")
    ax.set_xlabel(r"$k/n$  (fraction of bits flipped so far; equilibrium at $k/n=0.5$)", fontsize=11)
    ax.set_ylabel(r"$P(\mathrm{exact\ window\ pattern}\mid k)$   (log scale)", fontsize=11)
    ax.set_title(
        f"Exact Pattern-Probability Shape Families  (n={n}, window L={L})\n"
        r"Hypergeometric($n,k,L$), classified by $j$ vs $L/2$: "
        r"falling ($j{=}0$), bump ($0{<}j{<}L/2$), rising ($j{\geq}L/2$)",
        fontsize=11)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved -> {outfile}")
    print(f"\nn={n}, L={L}: {len(falling_js)} falling, {len(bump_js)} bump, "
          f"{len(rising_js)} rising (out of {L+1} distinct weight-curves)")


def plot_family_mass_exact(n: int, L: int, outfile: str) -> None:
    """Plot which family holds the majority of exact probability mass, with entropy overlay.

    Exact-combinatorics analogue of the binomial script's plot_family_mass:
    plots total hypergeometric probability mass in the falling/bump/rising
    families as functions of k/n over k = 0..n/2 (zero entropy ->
    equilibrium), with the normalized Shannon entropy S(k) overlaid on a
    twin y-axis, saves the figure, and prints a family-crossover report
    plus the uniform-in-k time-averaged family composition.

    Args:
        n: Total bitstring length.
        L: Window length in bits.
        outfile: Path to save the output PNG figure.

    Returns:
        None. Writes a PNG to outfile and prints summary tables to stdout.
    """
    ks_full, mass_full = family_mass_exact(n, L)
    half = n // 2 + 1
    ks: IntArray = ks_full[:half]
    mass: Dict[str, FloatArray] = {fam: mass_full[fam][:half] for fam in mass_full}
    tau: FloatArray = ks / n  # k/n, ranges 0 -> 0.5 at equilibrium
    S: FloatArray = entropy_curve(n, ks)  # normalized Shannon entropy, 0 -> 1

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(tau, mass["falling"], color="black", lw=2, label="FALLING (j=0)")
    ax.plot(tau, mass["bump"], color="darkorange", lw=2, label="BUMP (0<j<L/2)")
    ax.plot(tau, mass["rising"], color="steelblue", lw=2, label="RISING (j>=L/2)")
    ax.axhline(0.5, color="gray", lw=0.7, ls=":")
    ax.set_xlabel(r"$k/n$  (fraction of bits flipped so far; equilibrium reached at $k/n=0.5$)")
    ax.set_ylabel("exact probability mass in family (hypergeometric)")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1.02)

    ax2 = ax.twinx()
    ax2.plot(tau, S, color="mediumseagreen", lw=2, ls="--", label=r"entropy $S(k)/S_{max}$")
    ax2.set_ylabel(r"normalized Shannon entropy $S(k)/S_{max} = \log_2\binom{n}{k}/\log_2\binom{n}{n/2}$",
                   color="mediumseagreen")
    ax2.tick_params(axis="y", labelcolor="mediumseagreen")
    ax2.set_ylim(0, 1.02)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    ax.set_title(
        f"Exact finite-n family dominance vs. entropy, zero entropy -> equilibrium\n"
        f"(n={n}, window length L={L}) -- Hypergeometric(n,k,L), k=0..n/2"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved -> {outfile}")

    # dominant family + crossover report
    prev_winner = None
    for idx, k in enumerate(ks):
        vals = {fam: mass[fam][idx] for fam in mass}
        winner = max(vals, key=vals.get)
        if winner != prev_winner:
            print(f"  k={k:4d} (k/n={k/n:.3f})  falling={vals['falling']:.4f}  "
                  f"bump={vals['bump']:.4f}  rising={vals['rising']:.4f}  -> {winner} takes over")
            prev_winner = winner

    # time-averaged (uniform over k=0..n/2) composition -- the literal
    # "typical composition of me across zero-entropy -> equilibrium" number
    avg = {fam: mass[fam].mean() for fam in mass}
    print(f"\n  UNIFORM-IN-k AVERAGE over k=0..n/2 (n={n}, L={L}):")
    print(f"    falling={avg['falling']:.4f}  bump={avg['bump']:.4f}  rising={avg['rising']:.4f}")


def main() -> None:
    """Parse CLI args and generate both the per-pattern and family-mass exact plots."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9182, help="total bitstring length")
    ap.add_argument("--L", type=int, default=30, help="window/pattern length")
    ap.add_argument("--L-min", type=int, default=3, dest="L_min",
                     help="minimum allowed pattern length (default 3). "
                          "Bump requires 0<j<L/2 to have an integer "
                          "solution: L=1 gives 0<j<0.5 (empty), L=2 gives "
                          "0<j<1 (empty). L=3 is the smallest L for which "
                          "the bump family exists at all (j=1 works). "
                          "NOTE: this only guards --L (errors if L<L_min); "
                          "it does not itself change the plotted curve. "
                          "To get a slower/flatter bump onset, pass a "
                          "SMALLER --L directly -- larger L makes bump "
                          "take over FASTER, not slower (a bigger window "
                          "is more likely to catch an early 1-bit).")
    ap.add_argument("--out", type=str, default="family_hypergeometric.png",
                     help="output file for the per-pattern-curve plot")
    ap.add_argument("--mass-out", type=str, default="family_mass_hypergeometric.png",
                     help="output file for the 'which family is typical' (mass + entropy) plot")
    args = ap.parse_args()
    if args.L < args.L_min:
        raise SystemExit(
            f"--L={args.L} is below --L-min={args.L_min}. "
            f"L<3 has no bump family at all (0<j<L/2 has no integer solution)."
        )
    plot_patterns_exact(args.n, args.L, args.out)
    print()
    plot_family_mass_exact(args.n, args.L, args.mass_out)


if __name__ == "__main__":
    main()
