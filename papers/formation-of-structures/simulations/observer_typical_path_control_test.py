#!/usr/bin/env python3
"""
Control test for observer_typical_path_v5.py's core claim (header item 3):
that the *specific* Hamming-weight-ordered truncation of the Hadamard basis
is what produces the increasing-entropy trend and the emerge-then-fade
observer arc -- not just "any" progressive truncation.

If a RANDOM ordering (same set of basis coefficients, same phases, just
shuffled) produces statistically similar entropy trends and similar
emergence arcs, then the "MDL/compression-level ordering IS the arrow of
time" claim is not actually doing the work the theory says it's doing --
you'd get the same qualitative behavior from any monotonically-growing
subset of coefficients, and the specific tie to Kolmogorov/Hamming-weight
complexity would be decorative rather than load-bearing.

We compare three orderings of the same D=2^n basis, same phase draws:
  - hamming : popcount-then-index (the theory's proposed ordering)
  - random  : a fixed random permutation (control -- same growth in size,
              no complexity structure)
  - reverse : popcount descending (anti-control -- most complex first,
              should be the worst case if the theory is right)

For each ordering x each of several independent phase seeds, we measure:
  (a) Spearman rank correlation of k vs S(rho_A) -- how strongly entropy
      trends upward with k, regardless of local noise
  (b) whether the emerge-then-fade observer arc appears at all, at a fixed
      hidden-qubit budget (same budget across orderings, so this isn't
      biased by re-running find_minimal_n separately per ordering)
"""

import math
import sys
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, ".")
from observer_typical_path_v6 import (
    popcount, fwht, partial_trace, von_neumann_entropy,
    morton_maps, coarse_cell_geometry, solve_typical_path,
)


def ordering_hamming(n_qubits, rng=None):
    indices = np.arange(2 ** n_qubits)
    return np.array(sorted(indices, key=lambda x: (popcount(x), x)))


def ordering_reverse_hamming(n_qubits, rng=None):
    indices = np.arange(2 ** n_qubits)
    return np.array(sorted(indices, key=lambda x: (-popcount(x), x)))


def ordering_random(n_qubits, rng):
    indices = np.arange(2 ** n_qubits)
    perm = indices.copy()
    rng.shuffle(perm)
    return perm


def ordering_natural(n_qubits, rng=None):
    # identity -- arguably the single shortest-description rule of all
    return np.arange(2 ** n_qubits)


def ordering_reverse_natural(n_qubits, rng=None):
    return np.arange(2 ** n_qubits)[::-1]


ORDERINGS = {
    "hamming": ordering_hamming,
    "reverse_hamming": ordering_reverse_hamming,
    "natural": ordering_natural,
    "reverse_natural": ordering_reverse_natural,
    "random": ordering_random,
}


def run_one(order_name, order_fn, n_qubits, n_a, spatial_qubits, n_per_axis,
            phase_seed, order_seed, num_frames,
            min_record_bits, s_max, presence, max_candidates, jump_cost,
            idx_to_xy, cell_centers):
    d = 2 ** n_qubits
    hidden_qubits = n_qubits - spatial_qubits
    n_b = n_qubits - n_a

    order_rng = np.random.default_rng(order_seed)
    basis_order = order_fn(n_qubits, order_rng)

    phase_rng = np.random.default_rng(phase_seed)
    static_phases = np.exp(1j * phase_rng.uniform(0, 2 * np.pi, size=d))

    k_steps = np.unique(np.linspace(1, d, num=min(num_frames, d), dtype=int))
    s_vals, cell_probs_by_frame = [], []
    for k in k_steps:
        psi_h = np.zeros(d, dtype=complex)
        sel = basis_order[:k]
        psi_h[sel] = static_phases[sel]
        psi_h /= np.linalg.norm(psi_h)
        psi_comp = fwht(psi_h)

        rho_a = partial_trace(psi_comp, n_a, n_b)
        s_a = von_neumann_entropy(rho_a)
        s_vals.append(s_a)
        cell_probs_by_frame.append(np.real(np.diag(rho_a)))

    rho, _ = spearmanr(k_steps, s_vals)

    path, path_probs, _ = solve_typical_path(cell_probs_by_frame, cell_centers,
                                              max_candidates, jump_cost)
    n_cells = 2 ** n_a
    uniform_p = 1.0 / n_cells
    presence_floor = presence * uniform_p
    visible = [
        (min_record_bits <= s_vals[i] < s_max) and (path_probs[i] >= presence_floor)
        for i in range(len(k_steps))
    ]
    emerges = any(visible)
    fades_again = emerges and not visible[-1]

    return {
        "order": order_name, "phase_seed": phase_seed,
        "spearman_k_vs_S": float(rho),
        "S_range": (float(min(s_vals)), float(max(s_vals))),
        "emerges": emerges, "fades_again": fades_again,
        "visible_frac": float(np.mean(visible)) if visible else 0.0,
    }


def main():
    observer = 6
    n_per_axis = observer // 2
    spatial_qubits = 2 * n_per_axis
    resolution = 2 ** n_per_axis
    hidden_qubits = 6          # fixed budget across orderings (n=12 found reliable for hamming at observer=6)
    n_qubits = spatial_qubits + hidden_qubits
    min_record_bits = max(0.05 * observer, 0.05)
    s_max = 0.98 * observer
    presence = 1.5
    max_candidates = 48
    jump_cost = 0.0
    num_frames = 20
    n_phase_seeds = 15
    order_seed = 7  # fixes the random permutation ordering itself

    idx_to_xy, _ = morton_maps(n_per_axis)
    cell_centers, _ = coarse_cell_geometry(observer, resolution)

    print(f"control test: observer={observer}, n={n_qubits} (hidden={hidden_qubits} fixed budget), "
          f"{n_phase_seeds} phase seeds x {len(ORDERINGS)} orderings\n")

    phase_seeds = [int(x) for x in np.random.default_rng(123).integers(0, 2**31 - 1, size=n_phase_seeds)]

    all_records = []
    for order_name, order_fn in ORDERINGS.items():
        for ps in phase_seeds:
            rec = run_one(order_name, order_fn, n_qubits, observer, spatial_qubits, n_per_axis,
                          ps, order_seed, num_frames, min_record_bits, s_max, presence,
                          max_candidates, jump_cost, idx_to_xy, cell_centers)
            all_records.append(rec)

    print(f"{'ordering':<10} {'mean spearman(k,S)':<20} {'emerges-and-fades':<20} {'emerges (any)':<15}")
    for order_name in ORDERINGS:
        recs = [r for r in all_records if r["order"] == order_name]
        mean_rho = np.mean([r["spearman_k_vs_S"] for r in recs])
        std_rho = np.std([r["spearman_k_vs_S"] for r in recs])
        n_fade = sum(r["fades_again"] for r in recs)
        n_emerge = sum(r["emerges"] for r in recs)
        print(f"{order_name:<10} {mean_rho:.3f} (std {std_rho:.3f}){'':<3} "
              f"{n_fade}/{n_phase_seeds}{'':<14} {n_emerge}/{n_phase_seeds}")




if __name__ == "__main__":
    main()
