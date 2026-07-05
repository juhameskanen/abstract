#!/usr/bin/env python3
"""
Ehrenfest Bit-Flip Relaxation: Entropy Curve and Pattern Emergence
===================================================================

Plots two things on the same figure:

LEFT AXIS  — Shannon entropy S(tau)/n (fraction of maximum).
             This is the universal Ehrenfest entropy saturation curve,
             independent of n.

RIGHT AXIS — For each pattern supplied via --patterns, the probability
             of finding that exact pattern at a uniformly random position
             in the bitstring, as a function of normalized time tau.

             At tau=0 (all zeros): P(pattern) = 1 if pattern is all-zeros,
             0 otherwise.
             At tau→∞ (white noise): P(pattern) = (1/2)^len(pattern),
             which is tiny for long patterns.

             A complex pattern like "10000000000001" reaches equilibrium
             probability (1/2)^14 ≈ 6e-5, NOT anything close to 1.
             The earlier version wrongly plotted global entropy for all
             patterns, making every pattern look identical and wrong.

Model
-----
Continuum Ehrenfest limit: each bit is independently 1 with probability
    p(tau) = 0.5 * (1 - exp(-2*tau)),   tau = flip_count / n.

Pattern probability at time tau, for pattern b_0 b_1 ... b_{L-1}:
    P(pattern | tau) = prod_i  p(tau)^b_i * (1-p(tau))^(1-b_i)

This assumes independent bits, which is exact in the continuum limit
and an excellent approximation for n >> len(pattern).

Simulation overlay
------------------
For verification, direct Ehrenfest simulation is run and pattern
occurrences are counted directly in the bitstring state, confirming
the analytic curves.

Usage examples
--------------
    python ehrenfest_entropy.py
    python ehrenfest_entropy.py --patterns 010 110 1010 10000000000001
    python ehrenfest_entropy.py --n 500 2000 --max-tau 4 --out fig.png
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Analytic curves ───────────────────────────────────────────────────────────

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
    Probability of observing `pattern` at a random position in the
    bitstring at normalized time tau.

    Pattern is a string of '0' and '1' characters, e.g. "010" or "1010".
    Each bit position is independently 1 with probability p(tau), so:

        P(pattern | tau) = prod_i  p^b_i * (1-p)^(1-b_i)

    where b_i in {0,1} are the bits of the pattern.

    Note: this equals (1/2)^len(pattern) at full equilibrium (tau→∞),
    which is tiny for long patterns.
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


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate_ehrenfest(n: int, max_tau: float, patterns: list[str],
                       rng=None, n_sample_points: int = 200):
    """
    Run direct Ehrenfest simulation and return, for each tau sample point:
      - tau values
      - measured p (fraction of 1-bits)
      - measured pattern probabilities (dict pattern -> array)

    Pattern probability is measured as the fraction of all windows of
    length len(pattern) in the current bitstring that match the pattern.
    """
    if rng is None:
        rng = np.random.default_rng()

    total_flips = int(max_tau * n)
    sample_every = max(1, total_flips // n_sample_points)

    state = np.zeros(n, dtype=np.int8)
    m = 0

    tau_list   = []
    p_list     = []
    pat_counts = {pat: [] for pat in patterns}

    for flip in range(total_flips):
        # Single Ehrenfest flip
        idx = rng.integers(0, n)
        if state[idx] == 0:
            state[idx] = 1; m += 1
        else:
            state[idx] = 0; m -= 1

        if (flip + 1) % sample_every == 0:
            tau_list.append((flip + 1) / n)
            p_list.append(m / n)
            # Count pattern occurrences
            for pat in patterns:
                L = len(pat)
                pat_arr = np.array([int(c) for c in pat], dtype=np.int8)
                n_windows = n - L + 1
                if n_windows <= 0:
                    pat_counts[pat].append(0.0)
                    continue
                # Sliding window match count
                count = 0
                for start in range(n_windows):
                    if np.array_equal(state[start:start + L], pat_arr):
                        count += 1
                pat_counts[pat].append(count / n_windows)

    return (np.array(tau_list), np.array(p_list),
            {p: np.array(v) for p, v in pat_counts.items()})


def verify(n_values=(200, 2000), max_tau=3.0, seed=0):
    """Print max deviation between simulation and analytic p(tau)."""
    rng = np.random.default_rng(seed)
    check_taus = np.linspace(max_tau / 12, max_tau, 12)
    print(f"\n{'n':>6}  {'max |p_sim - p_analytic|':>26}")
    for n in n_values:
        tau_sim, p_sim, _ = simulate_ehrenfest(n, max_tau, [], rng=rng,
                                                n_sample_points=100)
        idxs = [int(np.argmin(np.abs(tau_sim - t))) for t in check_taus]
        dev  = float(np.max(np.abs(p_sim[idxs] - p_analytic(check_taus))))
        print(f"{n:>6}  {dev:>26.5f}")


# ── Plotting ──────────────────────────────────────────────────────────────────

# Visually distinct colours for up to 8 patterns
_PATTERN_COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#ffe119",
]


def plot(max_tau: float, patterns: list[str], n_sim: int,
         run_sim: bool, outfile: str, seed: int):
    """
    Two-axis figure:
      Left  — entropy fraction S(tau)/n  (black, bold)
      Right — pattern probabilities       (colours, one curve per pattern)
    """
    tau_grid = np.linspace(0.0, max_tau, 500)

    fig, ax_ent = plt.subplots(figsize=(9, 5.5))
    ax_pat = ax_ent.twinx()

    # ── LEFT: entropy curve ──────────────────────────────────────────────
    s_grid = entropy_fraction(tau_grid)
    ax_ent.plot(tau_grid, s_grid, color="black", lw=2.5, zorder=3,
                label=r"$S(\tau)/n$ — entropy (left axis)")
    ax_ent.set_ylabel(r"$S(\tau)/n$  (Shannon entropy, fraction of max)",
                      fontsize=11)
    ax_ent.set_ylim(-0.03, 1.10)
    ax_ent.set_xlabel(r"$\tau = \mathrm{flip\ count}/n$  "
                      r"(normalized bit-flip time)", fontsize=11)

    # ── RIGHT: pattern probability curves ───────────────────────────────
    pat_handles = []

    for i, pat in enumerate(patterns):
        colour = _PATTERN_COLOURS[i % len(_PATTERN_COLOURS)]
        L      = len(pat)
        p_eq   = 0.5 ** L  # equilibrium probability

        # Analytic curve
        p_curve = pattern_probability(pat, tau_grid)
        line, = ax_pat.plot(tau_grid, p_curve, color=colour, lw=1.8,
                            label=f'"{pat}"  (len={L}, eq={p_eq:.2e})')
        pat_handles.append(line)

        # Equilibrium asymptote (dashed, same colour, faint)
        ax_pat.axhline(p_eq, color=colour, lw=0.8, ls="--", alpha=0.5)

    # Simulation overlay if requested
    if run_sim and patterns:
        rng = np.random.default_rng(seed)
        tau_sim, _, pat_sim = simulate_ehrenfest(
            n_sim, max_tau, patterns, rng=rng, n_sample_points=150)
        for i, pat in enumerate(patterns):
            colour = _PATTERN_COLOURS[i % len(_PATTERN_COLOURS)]
            ax_pat.scatter(tau_sim, pat_sim[pat], s=12, color=colour,
                           alpha=0.4, zorder=2)

    # Right-axis label and scale
    if patterns:
        # Use log scale if range spans more than 2 orders of magnitude
        all_probs = [pattern_probability(p, np.array([max_tau]))[0]
                     for p in patterns]
        max_prob  = max(pattern_probability(p, np.array([0.5]))[0]
                        for p in patterns)
        min_eq    = min(0.5 ** len(p) for p in patterns)
        if max_prob / (min_eq + 1e-30) > 100:
            ax_pat.set_yscale("log")
            ax_pat.set_ylabel(
                "Pattern probability  (log scale, right axis)", fontsize=11)
        else:
            ax_pat.set_ylabel(
                "Pattern probability  (right axis)", fontsize=11)
        ax_pat.set_ylim(bottom=0)

    # ── Legend ────────────────────────────────────────────────────────────
    ent_handle, = ax_ent.plot([], [], color="black", lw=2.5,
                               label=r"$S(\tau)/n$ (left axis)")
    if run_sim and patterns:
        sim_handle = plt.scatter([], [], s=12, color="gray", alpha=0.6,
                                 label=f"simulated (n={n_sim})")
        all_handles = [ent_handle] + pat_handles + [sim_handle]
    else:
        all_handles = [ent_handle] + pat_handles

    ax_ent.legend(handles=all_handles, loc="upper left",
                  fontsize=8.5, framealpha=0.9)

    # Annotations
    ax_ent.axvline(1.0, color="gray", lw=0.8, ls=":", alpha=0.6)
    ax_ent.text(1.02, 0.05, r"$\tau=1$", color="gray", fontsize=8,
                transform=ax_ent.get_xaxis_transform())

    title_parts = ["Universal Ehrenfest Entropy Saturation"]
    if patterns:
        title_parts.append(f"+ Pattern Probabilities: {', '.join(patterns)}")
    ax_ent.set_title("\n".join(title_parts), fontsize=11)

    ax_ent.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved → {outfile}")

    # Print equilibrium table
    if patterns:
        print("\nEquilibrium probabilities (tau → ∞, white-noise limit):")
        for pat in patterns:
            L   = len(pat)
            p_eq = 0.5 ** L
            print(f"  P('{pat}') = (1/2)^{L} = {p_eq:.6e}")
        print("\nNote: complex patterns become exponentially rare at equilibrium.")
        print("They do NOT track the entropy curve.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ehrenfest entropy curve + pattern emergence probabilities.")
    parser.add_argument("--patterns", nargs="*", default=["010", "110", "1010"],
                        help="Bit patterns to plot (default: 010 110 1010). "
                             "Each must contain only '0' and '1'.")
    parser.add_argument("--max-tau", type=float, default=3.0,
                        help="Maximum normalized time (default: 3.0)")
    parser.add_argument("--n", type=int, nargs="+", default=[200, 2000],
                        help="Sizes for the verification check (default: 200 2000)")
    parser.add_argument("--sim", action="store_true",
                        help="Overlay direct simulation dots on the pattern curves")
    parser.add_argument("--sim-n", type=int, default=500,
                        help="Bitstring length for simulation overlay (default: 500)")
    parser.add_argument("--out", type=str, default="entropy_curve.png",
                        help="Output filename (default: entropy_curve.png)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip verification printout")
    args = parser.parse_args()

    # Validate patterns
    for pat in (args.patterns or []):
        if not all(c in "01" for c in pat):
            parser.error(f"Pattern '{pat}' contains characters other than 0 and 1.")

    if not args.no_verify:
        verify(n_values=args.n, max_tau=args.max_tau, seed=args.seed)

    plot(max_tau=args.max_tau,
         patterns=args.patterns or [],
         n_sim=args.sim_n,
         run_sim=args.sim,
         outfile=args.out,
         seed=args.seed)


if __name__ == "__main__":
    main()
    
