#!/usr/bin/env python3
"""
Ehrenfest Bit-Flip Relaxation: Entropy, Pattern Curves, and Weight-Class Heatmap
==================================================================================

Three complementary views of the same Ehrenfest relaxation process, all driven
by the single continuum bit-flip-probability law:

    p(tau) = 0.5 * (1 - exp(-2*tau)),      tau = flip_count / n

1) ENTROPY CURVE  (--mode patterns, left axis, always drawn in that mode)
   S(tau)/n = H2(p(tau))  — universal, independent of n.

2) SPECIFIC PATTERN CURVES  (--mode patterns, right axis)
   For a literal bit pattern like "10000000000001", the probability of
   finding THAT EXACT pattern at a random position in the string, as a
   function of tau. Rare patterns (many 1s in a long string) stay
   exponentially small even at equilibrium: (1/2)^len(pattern).

3) HAMMING-WEIGHT-CLASS HEATMAP  (--mode heatmap)   <-- new
   This answers a different question: "if I were an emergent observer built
   on ONE of the 2^L possible length-L strings, which family (by weight k =
   number of 1-bits) would I most likely find myself in?"

   This is NOT the same as (2). A specific string's probability is
       P(exact string | tau) = p^k (1-p)^(L-k)
   which is maximized by the all-zeros (or all-ones) string -- individually
   "most probable" but a poor stand-in for typicality, since it ignores
   multiplicity.

   The typicality question groups all C(L,k) strings that share weight k
   into one macrostate. The probability of landing in that macrostate is
   the binomial law:
       P(weight = k | tau) = C(L,k) * p(tau)^k * (1-p(tau))^(L-k)
   This is already a proper distribution over k for each tau (sums to 1),
   and it is sharply peaked near k ~ L*p(tau), increasingly so as L grows.
   The heatmap's ridge (the k that maximizes P at each tau) traces out the
   most likely weight-class of an emergent observer as the process unfolds.

Usage examples
--------------
    python ehrenfest_entropy_v3.py --mode patterns
    python ehrenfest_entropy_v3.py --mode patterns --patterns 010 110 1010
    python ehrenfest_entropy_v3.py --mode heatmap --L 40 --max-tau 3
    python ehrenfest_entropy_v3.py --mode both --L 30 --patterns 0000 1111 0101
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.special import gammaln


# ── Core continuum model ───────────────────────────────────────────────────

def p_analytic(tau):
    """Expected fraction of 1-bits at normalized time tau = flip_count/n."""
    return 0.5 * (1.0 - np.exp(-2.0 * tau))


def binary_entropy(p):
    """Shannon binary entropy H2(p) in bits, safe at boundaries."""
    p = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)


def entropy_fraction(tau):
    """Universal entropy saturation curve S(tau)/n = H2(p(tau))."""
    return binary_entropy(p_analytic(tau))


def pattern_probability(pattern: str, tau):
    """
    Probability of observing the EXACT `pattern` at a random position in the
    bitstring at normalized time tau. Depends on where the 1s are only
    through the count (all bits are i.i.d.), so in fact this equals
    p^k (1-p)^(L-k) with k = number of 1s -- see weight_class_log_prob for
    the multiplicity-aware version.
    """
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    p = p_analytic(tau)
    prob = np.ones_like(p)
    for ch in pattern:
        if ch == '1':
            prob *= p
        elif ch == '0':
            prob *= (1.0 - p)
        else:
            raise ValueError(f"Pattern characters must be '0' or '1', got '{ch}'")
    return prob


def log_binom(L, k):
    """log C(L,k), stable for large L via log-gamma (avoids overflow)."""
    return gammaln(L + 1) - gammaln(k + 1) - gammaln(L - k + 1)


def weight_class_log_prob(L: int, tau_grid: np.ndarray):
    """
    Returns a (L+1, len(tau_grid)) array: log P(weight=k | tau) for
    k = 0..L, computed in log-space so it stays numerically stable for
    large L (this is exactly the kind of astronomically-large-n regime
    the black-hole entropy scripts already need to handle).

    P(weight=k | tau) = C(L,k) p^k (1-p)^(L-k)
    log P = log C(L,k) + k*log(p) + (L-k)*log(1-p)

    Each column (fixed tau) already sums to 1 over k (it's the binomial
    pmf) -- no renormalization needed.
    """
    p = p_analytic(tau_grid)                       # shape (T,)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    k = np.arange(L + 1)                            # shape (K,)
    logC = log_binom(L, k)                          # shape (K,)

    log_p = np.log(p)                               # shape (T,)
    log_q = np.log(1.0 - p)                          # shape (T,)

    # Broadcast to (K, T)
    log_probs = (logC[:, None]
                 + np.outer(k, log_p)
                 + np.outer(L - k, log_q))
    return log_probs


# ── Plot 1: entropy + specific pattern curves (original view) ─────────────

_PATTERN_COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#ffe119",
]


def plot_patterns(max_tau: float, patterns: list[str], outfile: str):
    tau_grid = np.linspace(0.0, max_tau, 500)

    fig, ax_ent = plt.subplots(figsize=(9, 5.5))
    ax_pat = ax_ent.twinx()

    s_grid = entropy_fraction(tau_grid)
    ax_ent.plot(tau_grid, s_grid, color="black", lw=2.5, zorder=3,
                label=r"$S(\tau)/n$ — entropy (left axis)")
    ax_ent.set_ylabel(r"$S(\tau)/n$  (Shannon entropy, fraction of max)", fontsize=11)
    ax_ent.set_ylim(-0.03, 1.10)
    ax_ent.set_xlabel(r"$\tau = \mathrm{flip\ count}/n$  (normalized bit-flip time)",
                      fontsize=11)

    pat_handles = []
    for i, pat in enumerate(patterns):
        colour = _PATTERN_COLOURS[i % len(_PATTERN_COLOURS)]
        L = len(pat)
        p_eq = 0.5 ** L
        p_curve = pattern_probability(pat, tau_grid)
        line, = ax_pat.plot(tau_grid, p_curve, color=colour, lw=1.8,
                            label=f'"{pat}"  (len={L}, eq={p_eq:.2e})')
        pat_handles.append(line)
        ax_pat.axhline(p_eq, color=colour, lw=0.8, ls="--", alpha=0.5)

    if patterns:
        max_prob = max(pattern_probability(p, np.array([0.5]))[0] for p in patterns)
        min_eq = min(0.5 ** len(p) for p in patterns)
        if max_prob / (min_eq + 1e-30) > 100:
            ax_pat.set_yscale("log")
            ax_pat.set_ylabel("Pattern probability  (log scale, right axis)", fontsize=11)
        else:
            ax_pat.set_ylabel("Pattern probability  (right axis)", fontsize=11)
            ax_pat.set_ylim(bottom=0)

    ent_handle, = ax_ent.plot([], [], color="black", lw=2.5, label=r"$S(\tau)/n$ (left axis)")
    ax_ent.legend(handles=[ent_handle] + pat_handles, loc="upper left",
                  fontsize=8.5, framealpha=0.9)

    ax_ent.axvline(1.0, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax_ent.text(1.02, 0.05, r"$\tau=1$", color="gray", fontsize=8,
                transform=ax_ent.get_xaxis_transform())

    title = "Universal Ehrenfest Entropy Saturation"
    if patterns:
        title += f"\n+ Specific Pattern Probabilities: {', '.join(patterns)}"
    ax_ent.set_title(title, fontsize=11)
    ax_ent.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved → {outfile}")


# ── Plot 2: Hamming-weight-class heatmap (typicality view) ────────────────

def plot_heatmap(L: int, max_tau: float, n_tau: int, outfile: str,
                  log_color: bool):
    """
    Heatmap of P(weight=k | tau) over k in [0, L], tau in [0, max_tau].
    Overlaid with the ridge k*(tau) = argmax_k P(weight=k|tau) and the
    mean-field trajectory k = L * p(tau) for comparison.
    """
    tau_grid = np.linspace(1e-6, max_tau, n_tau)   # avoid log(0) at tau=0
    log_probs = weight_class_log_prob(L, tau_grid)  # (K, T)
    probs = np.exp(log_probs)                       # sums to 1 per column

    k_vals = np.arange(L + 1)
    ridge_k = k_vals[np.argmax(probs, axis=0)]       # most likely k at each tau
    mean_k = L * p_analytic(tau_grid)                # mean-field E[k]

    fig, ax = plt.subplots(figsize=(9.5, 6))

    if log_color:
        # floor tiny probabilities so LogNorm doesn't choke on exact zeros
        floor = probs[probs > 0].min() if np.any(probs > 0) else 1e-300
        plot_probs = np.clip(probs, floor, None)
        norm = LogNorm(vmin=max(floor, 1e-12), vmax=probs.max())
        label = r"$P(\mathrm{weight}=k \mid \tau)$  (log scale)"
    else:
        plot_probs = probs
        norm = None
        label = r"$P(\mathrm{weight}=k \mid \tau)$"

    mesh = ax.pcolormesh(tau_grid, k_vals, plot_probs, shading="auto",
                          cmap="magma", norm=norm)
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(label, fontsize=10)

    ax.plot(tau_grid, ridge_k, color="cyan", lw=1.8,
            label=r"ridge: most likely $k$ (mode of binomial)")
    ax.plot(tau_grid, mean_k, color="white", lw=1.2, ls="--", alpha=0.8,
            label=r"mean-field: $k = L\,p(\tau)$")

    ax.axvline(1.0, color="gray", lw=0.8, ls=":", alpha=0.7)
    ax.text(1.02, L * 0.02, r"$\tau=1$", color="gray", fontsize=8)

    ax.set_xlabel(r"$\tau = \mathrm{flip\ count}/n$  (normalized bit-flip time)",
                  fontsize=11)
    ax.set_ylabel(r"Hamming weight $k$  (number of 1-bits, out of $L$)", fontsize=11)
    ax.set_title(
        f"Typicality of an Emergent Observer: Weight-Class Probability (L={L})\n"
        r"$P(\mathrm{weight}=k\mid\tau) = \binom{L}{k} p(\tau)^k (1-p(\tau))^{L-k}$",
        fontsize=11)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved → {outfile}")

    # Sanity check: each column should sum to 1
    col_sums = probs.sum(axis=0)
    print(f"\nColumn-sum check (should all be ~1.0): "
          f"min={col_sums.min():.6f}, max={col_sums.max():.6f}")
    print(f"At tau={max_tau:.2f}: mode k={ridge_k[-1]} (of L={L}), "
          f"mean k={mean_k[-1]:.2f}, "
          f"std = sqrt(L p(1-p)) = {np.sqrt(L * p_analytic(max_tau) * (1-p_analytic(max_tau))):.2f}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ehrenfest entropy / pattern curves / weight-class typicality heatmap.")
    parser.add_argument("--mode", choices=["patterns", "heatmap", "both"],
                        default="both",
                        help="Which view(s) to generate (default: both)")
    parser.add_argument("--patterns", nargs="*", default=["0000", "1111", "0101"],
                        help="Specific bit patterns for the 'patterns' view "
                             "(default: 0000 1111 0101)")
    parser.add_argument("--max-tau", type=float, default=3.0,
                        help="Maximum normalized time (default: 3.0)")
    parser.add_argument("--L", type=int, default=40,
                        help="String length for the weight-class heatmap (default: 40)")
    parser.add_argument("--n-tau", type=int, default=400,
                        help="Number of tau samples across the heatmap (default: 400)")
    parser.add_argument("--linear-color", action="store_true",
                        help="Use linear color scale on heatmap instead of log "
                             "(log is default; probabilities span many orders of magnitude for large L)")
    parser.add_argument("--out-patterns", type=str, default="patterns_curve.png")
    parser.add_argument("--out-heatmap", type=str, default="weight_heatmap.png")
    args = parser.parse_args()

    for pat in (args.patterns or []):
        if not all(c in "01" for c in pat):
            parser.error(f"Pattern '{pat}' contains characters other than 0 and 1.")

    if args.mode in ("patterns", "both"):
        plot_patterns(max_tau=args.max_tau, patterns=args.patterns or [],
                      outfile=args.out_patterns)

    if args.mode in ("heatmap", "both"):
        plot_heatmap(L=args.L, max_tau=args.max_tau, n_tau=args.n_tau,
                    outfile=args.out_heatmap, log_color=not args.linear_color)


if __name__ == "__main__":
    main()
