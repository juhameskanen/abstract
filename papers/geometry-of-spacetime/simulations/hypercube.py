#!/usr/bin/env python3
"""
Simulates a bitstring of n bits (packed to use exactly n/8 bytes of real
memory -- e.g. 4 GB of RAM holds a 4*8 = 32 Gbit string), starting all-zero,
mutated by flipping random bits in batches.

"Size of the observer's universe" at bit-flip-count t is taken to be the
number of pairs of currently-active (1-valued) knots:

    size(t) = C(k(t), 2) = k(t) * (k(t) - 1) / 2

where k(t) is the current Hamming weight (population count) of the string.
This is computed purely from the *current* state -- no memory of which bits
were ever flipped before is used or required.

The simulation runs for n/2 total bit-flips (grouped into batches of
--flips-per-cycle each), which is where the Ehrenfest process is expected
to approximately saturate (mean-field relaxation time ~ n/2).

Because n can be enormous (billions of bits), we do NOT recompute the full
population count every cycle (that would be O(n) per cycle -- far too slow).
Instead we track k incrementally: for each batch, we read the pre-flip value
of every bit about to be toggled, count how many were 1 (-> becoming 0) vs
0 (-> becoming 1), and update k by the net delta. This is O(batch size) per
cycle, independent of n.

Usage:
    python bitflip_size.py --gb 4 --flips-per-cycle 5000000
    python bitflip_size.py --bits 100000 --flips-per-cycle 200   # small test run
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def run(n_bits: int, flips_per_cycle: int, seed: int | None, log_every: int):
    rng = np.random.default_rng(seed)

    n_bytes = n_bits // 8
    print(f"[+] Allocating bitstring: n = {n_bits:,} bits "
          f"({n_bytes / (1024**3):.3f} GB packed, all-zero start)")
    bitstring = np.zeros(n_bytes, dtype=np.uint8)

    total_flips_target = n_bits // 2  # relaxation-time budget, ~n/2
    n_cycles = max(1, total_flips_target // flips_per_cycle) * 3
    print(f"[+] Target total flips: {total_flips_target:,}  "
          f"-> {n_cycles:,} cycles of {flips_per_cycle:,} flips each")

    k = 0  # current Hamming weight (exact int, arbitrary precision)
    t_history = []
    k_history = []
    size_history = []  # C(k,2)

    record_stride = max(1, n_cycles // log_every)

    for cycle in range(n_cycles):
        # pick flips_per_cycle uniformly random global bit positions
        idx = rng.integers(0, n_bits, size=flips_per_cycle, dtype=np.int64)
        byte_idx = idx >> 3
        bit_pos = (idx & 7).astype(np.uint8)
        mask = (np.uint8(1) << bit_pos)

        # read pre-flip values (gather) to know which bits are currently 1
        cur = bitstring[byte_idx]
        bit_val = (cur >> bit_pos) & 1
        ones_hit = int(bit_val.sum())

        # net change in Hamming weight this cycle
        # (assumes negligible within-batch index collisions; valid when
        #  flips_per_cycle << n, true for any realistic batch size here)
        delta_k = flips_per_cycle - 2 * ones_hit
        k += delta_k

        # apply the flips in place (unbuffered, handles rare duplicate
        # indices correctly via ufunc.at)
        np.bitwise_xor.at(bitstring, byte_idx, mask)

        if cycle % record_stride == 0 or cycle == n_cycles - 1:
            t = (cycle + 1) * flips_per_cycle
            size = k * (k - 1) // 2  # exact C(k,2), arbitrary-precision int
            t_history.append(t)
            k_history.append(k)
            size_history.append(size)

    print(f"[+] Done. Final k = {k:,} (n/2 = {n_bits//2:,}), "
          f"final size = {size_history[-1]:.3e}")
    return np.array(t_history, dtype=np.float64), \
           np.array(k_history, dtype=np.float64), \
           np.array(size_history, dtype=np.float64)


def plot(t_hist, k_hist, size_hist, n_bits, flips_per_cycle, out_path=None):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'crimson'
    ax1.set_xlabel("Bit-flips (t)", fontsize=11)
    ax1.set_ylabel(r"Size $= \binom{k}{2} = k(k-1)/2$", color=color, fontsize=11)
    ax1.plot(t_hist, size_hist, color=color, linewidth=2.5,
              label="Size (pairwise knot connections)")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2 = ax1.twinx()
    color = 'royalblue'
    ax2.set_ylabel("Hamming weight k(t) (bits set to 1)", color=color, fontsize=11)
    ax2.plot(t_hist, k_hist, color=color, linewidth=1.5, linestyle="--",
              label="k(t) = current entropy")
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Size of hypercube universe vs. bit-flips\n"
              f"n = {n_bits:,} bits, {flips_per_cycle:,} flips/cycle "
              f"(log scale on size axis)", fontsize=10)
    fig.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=130)
        print(f"[+] Saved plot to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate bit-flip-driven 'size' growth: size = C(k,2)."
    )
    parser.add_argument("--gb", type=float, default=4.0,
                         help="Bitstring size in GB of packed memory "
                              "(n = gb * 1024^3 * 8 bits). Default: 4 GB.")
    parser.add_argument("--bits", type=int, default=None,
                         help="Override: exact number of bits (must be a "
                              "multiple of 8). Takes precedence over --gb. "
                              "Useful for quick small test runs.")
    parser.add_argument("--flips-per-cycle", type=int, default=5_000_000,
                         help="Number of random bit-flips per batch/cycle. "
                              "Larger = faster but coarser time resolution.")
    parser.add_argument("--log-points", type=int, default=2000,
                         help="Approximate number of points to record for "
                              "the plot (independent of flips-per-cycle).")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=str, default=None,
                         help="Save plot to this path instead of showing it.")
    args = parser.parse_args()

    n_bits = args.bits if args.bits is not None else int(args.gb * 1024**3 * 8)
    if n_bits % 8 != 0:
        n_bits -= n_bits % 8  # keep byte-aligned

    t_hist, k_hist, size_hist = run(
        n_bits=n_bits,
        flips_per_cycle=args.flips_per_cycle,
        seed=args.seed,
        log_every=args.log_points,
    )
    plot(t_hist, k_hist, size_hist, n_bits, args.flips_per_cycle, out_path=args.out)
