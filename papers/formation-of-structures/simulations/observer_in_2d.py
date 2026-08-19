#!/usr/bin/env python3
"""
Observer as a shortest-description-length path in a growing-entropy 2D
complex-valued spectrally compressed universe -- v5.

The ultimate goal: define the observer and the rest follows as typical (most probable) outcome
due to solomonof induction with "spectral complexity" measure.

This POC should demonstrate the following:

1. The universe is static and timeless, with information content of n bits.

2. The information (bits) can arranged to represent both code (the most trivial continuous
periodic function sin) and data (frequencies and phases of modes) to describe all possible
representations of observer.

3. The wavefunction of the universe can be decomposed to nearly infinite
ensembles, which can be arranged into increasing entropy order. This is a TREND, not a rule
imposed frame-to-frame: no consecutive pair of ensembles is forced to have higher entropy than
the last. What we assume is only the compression-level ordering (Hamming-weight-truncated
Hadamard coefficients, see get_hamming_ordered_hadamard_basis); whether that ordering happens
to produce a broadly increasing entropy trend -- and how noisy that trend is -- is something the
simulation reports (--seed-sweep), not something asserted or enforced. We will later show the
trend itself emerges from Solomonoff induction rather than being hand-picked.

4. Time emerges as walk through these ensembles. We are interested in walks
that contain the specified observer (as universes without the observer cannot be observed by the observer).

5. Those walks that has minimal description length dominate the measure. So the
observer most likely find themselves within one of the most compressible (minimal wavefunction) walks.
These are walks with smooth predictable dynamics, as such information compresses best.

6. The compression codec is the wavefunction, complex-valued,
MPEG-like, over ensembles (over time). MDL is measured over the entire WALK (path), not per frame!

7. Walk through all possible paths from zero entropy (Wootters style internal emergent time); shortest
description length dominates the measure. Implemented here as an exact Viterbi DP over the branch
graph -- NOTE: this finds the single maximum-a-posteriori path, i.e. an approximation to the full
2^-L-weighted Solomonoff measure over ALL paths, not the measure itself. Treat it as that:
 - state at frame j = a coarse cell c (candidate observer location)
 - emission cost(c, j)  = -log2 P(c | k_j)          (Born/Shannon code length)
 - transition cost(c', c) = lambda * ||pos(c') - pos(c)||  (motion-vector cost -- exactly what
   MPEG pays for a moving region: cheap if it's roughly the same place, expensive if it teleports)
   "Presence" (--presence) now means: is the winning path's cell probability at this frame far
   enough above the uniform baseline to represent real compression gain, not noise. Frames failing
   that (or outside the entropy validity band) are marked not-visible.

8. Spectral - geometric mapping. The raster probability amplitudes define the observer. At each
frame the observer's visual IS a masked contour of the REAL |psi(x,y,z,t)|^2 raster, cropped to the
winning path's coarse-cell block (plus one block of padding for context) and contoured at real,
data-derived levels. If the true amplitude structure there is a ring, a ring is what renders.
NOTE (still true in v5): the geometry rendered here is a flat 2D raster. The header's aspiration
in items 10-11 below (3+1 D, spherical symmetry, singularity at zero information) is NOT what is
implemented yet -- this is the next piece of work.

9. Spectral complexity: this is Kolmogorov complexity in principle, but as the actual algorithm
(the sin() function) cost is per class and therefore its impact is vanishingly small and constant.
The actual dominant complexity emerges from the data -- the informational cost of frequencies and
phases.

10. Spectral -> geometry mapping (NOT implemented yet -- next focus area). Observer should sense
living in 3+1 geometric space with spherical symmetry (observational fact, to be derived, not
assumed). High dimensional Hilbert space can be mapped to geometric space in infinitely many ways.
Why three, why spherical? We assume those emerge from Solomonoff induction, as the simplest minimal
solutions with shortest description length.

11. Zero information (or as close to zero as the minimal wavefunction allows) should therefore map
to geometric singularity -- any geometric interpretation of "one state" should then map to a
geometric point. Its classical statistical "bitstring" equivalence is all-zero bitstring. All
possible ways to map, slice, express or interpret the bitstring as "positions" yield the same
geometric object -- a point. Similarly, when starting out from zero-mode, one-mode etc. wavefunction
ensembles, their geometric projections should collapse to a geometric point (geometry with no
micro structures).

Expected outcome:
- universe starts minimally simple, just black image, the probability amplitudes should be
practically zero everywhere, no emergent micro structures, let alone observer.
- as entropy increases probability amplitudes rise gradually and the heat map begins to glow
minimally trivial structures that get more complex with time, soon observer emerges and later
vanishes as entropy goes up and emergent structures vanish to make room for structureless "noise"
- observer does not jump chaotically, it emerges, moves smoothly as the probability amplitudes peak,
and vanish. For most of the time the universe contains no observer.

ON PARAMETER COUNT (design direction, not yet fully realized): the goal is a ONE-parameter theory
-- --observer defines the observer, and the rest (--resolution, --n, --min-record-bits, --s-max)
should follow rather than be tunable. --presence and --jump-cost are still hand-set thresholds
(honestly flagged in group (1)/(3) below) and remain candidates for being derived away in a later
version. Everything in group (4) is pure rendering/optimization and carries no theoretical content.
"""

import argparse
import math
import sys
from collections import Counter

import numpy as np
from scipy.ndimage import zoom as ndimage_zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter

CONTOUR_UPSAMPLE = 6


# ---------------------------------------------------------------------------
# Core spectral / quantum machinery (unchanged, validated)
# ---------------------------------------------------------------------------

def popcount(x: int) -> int:
    return bin(x).count('1')


def get_hamming_ordered_hadamard_basis(n_qubits: int) -> np.ndarray:
    indices = np.arange(2 ** n_qubits)
    sorted_indices = sorted(indices, key=lambda x: (popcount(x), x))
    return np.array(sorted_indices)


def fwht(a: np.ndarray) -> np.ndarray:
    """Fast Walsh-Hadamard transform, orthonormal, O(D log D)."""
    a = a.astype(complex).copy()
    d = len(a)
    h = 1
    while h < d:
        for i in range(0, d, h * 2):
            x = a[i:i + h].copy()
            y = a[i + h:i + 2 * h].copy()
            a[i:i + h] = x + y
            a[i + h:i + 2 * h] = x - y
        h *= 2
    return a / np.sqrt(d)


def partial_trace(state_vector, n_a, n_b):
    dim_a, dim_b = 2 ** n_a, 2 ** n_b
    psi_tensor = state_vector.reshape((dim_a, dim_b))
    return np.dot(psi_tensor, psi_tensor.conj().T)


def von_neumann_entropy(rho, tol=1e-12):
    eigenvalues = np.linalg.eigvalsh(rho)
    eigs = eigenvalues[eigenvalues > tol]
    if len(eigs) == 0:
        return 0.0
    return float(-np.sum(eigs * np.log2(eigs)))


# ---------------------------------------------------------------------------
# Spatial embedding: Morton (Z-order) interleave of x/y bits.
# ---------------------------------------------------------------------------

def morton_maps(n_bits_per_axis: int):
    side = 2 ** n_bits_per_axis
    idx_to_xy, xy_to_idx = {}, {}
    for x in range(side):
        for y in range(side):
            idx = 0
            for b in range(n_bits_per_axis - 1, -1, -1):
                idx = (idx << 1) | ((x >> b) & 1)
                idx = (idx << 1) | ((y >> b) & 1)
            idx_to_xy[idx] = (x, y)
            xy_to_idx[(x, y)] = idx
    return idx_to_xy, xy_to_idx


def probs_to_grid(spatial_probs, idx_to_xy, side):
    grid = np.zeros((side, side))
    for idx, p in enumerate(spatial_probs):
        x, y = idx_to_xy[idx]
        grid[y, x] = p
    return grid


def build_smooth_field(grid: np.ndarray, upsample: int = CONTOUR_UPSAMPLE) -> np.ndarray:
    smooth = ndimage_zoom(grid, upsample, order=3, mode="nearest")
    return np.clip(smooth, 0, None)


def region_mask(shape, box, side, upsample, pad_blocks=1):
    x0, y0, bp = box
    xlo = max(0, x0 - pad_blocks * bp)
    ylo = max(0, y0 - pad_blocks * bp)
    xhi = min(side, x0 + bp + pad_blocks * bp)
    yhi = min(side, y0 + bp + pad_blocks * bp)
    uxlo, uxhi = int(round(xlo * upsample)), int(round(xhi * upsample))
    uylo, uyhi = int(round(ylo * upsample)), int(round(yhi * upsample))
    mask = np.ones(shape, dtype=bool)
    mask[uylo:uyhi, uxlo:uxhi] = False
    return mask


def coarse_cell_geometry(observer_bits: int, side: int):
    n_a_per_axis = observer_bits // 2
    block_side = 2 ** n_a_per_axis
    block_px = side // block_side
    idx_to_bxy, _ = morton_maps(n_a_per_axis)
    centers, boxes = {}, {}
    for idx, (bx, by) in idx_to_bxy.items():
        x0, y0 = bx * block_px, by * block_px
        centers[idx] = ((bx + 0.5) * block_px, (by + 0.5) * block_px)
        boxes[idx] = (x0, y0, block_px)
    return centers, boxes


# ---------------------------------------------------------------------------
# Shortest-description-length path search (Viterbi DP over the branch graph)
# ---------------------------------------------------------------------------

def solve_typical_path(cell_probs_by_frame, cell_centers, max_candidates, jump_cost_weight):
    n_frames = len(cell_probs_by_frame)
    eps = 1e-300

    candidates_per_frame = [
        np.argsort(probs)[::-1][:max_candidates] for probs in cell_probs_by_frame
    ]

    dp = {int(c): -math.log2(max(cell_probs_by_frame[0][c], eps)) for c in candidates_per_frame[0]}
    backptr = [None] * n_frames

    for j in range(1, n_frames):
        probs_j = cell_probs_by_frame[j]
        prev_items = list(dp.items())
        dp_next = {}
        bp_j = {}
        for c in candidates_per_frame[j]:
            c = int(c)
            emit = -math.log2(max(probs_j[c], eps))
            cx, cy = cell_centers[c]
            best_cost, best_prev = math.inf, None
            for pc, pcost in prev_items:
                pcx, pcy = cell_centers[pc]
                dist = math.hypot(cx - pcx, cy - pcy)
                total = pcost + jump_cost_weight * dist + emit
                if total < best_cost:
                    best_cost, best_prev = total, pc
            dp_next[c] = best_cost
            bp_j[c] = best_prev
        dp = dp_next
        backptr[j] = bp_j

    end_c = min(dp, key=dp.get)
    total_cost = dp[end_c]

    path = [None] * n_frames
    path[-1] = end_c
    for j in range(n_frames - 1, 0, -1):
        path[j - 1] = backptr[j][path[j]]

    path_probs = [float(cell_probs_by_frame[j][path[j]]) for j in range(n_frames)]
    return path, path_probs, total_cost


# ---------------------------------------------------------------------------
# CLI + validation
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Observer as the shortest-description-length path through a "
                    "growing-entropy 2D universe. Define the observer; the universe "
                    "size needed for it to emerge is found automatically. "
                    "Run with no arguments for sane defaults.")

    obs_def = p.add_argument_group("(1) observer definition -- this IS the theory's one input")
    obs_def.add_argument("--observer", type=int, default=8,
        help="Qubits in the observer's memory register (must be even, <=16). Default: 8.")
    obs_def.add_argument("--min-record-bits", type=float, default=None,
        help="Entropy floor, in bits, below which rho_A holds no record. "
             "Default: 5%% of --observer's bit capacity.")
    obs_def.add_argument("--s-max", type=float, default=None,
        help="Entropy ceiling, in bits, above which rho_A is thermalized. "
             "Default: 98%% of --observer's bit capacity.")
    obs_def.add_argument("--presence", type=float, default=1.5,
        help="Winning path's cell probability must be >= presence * uniform to count "
             "as visible this frame. Default: 1.5x.")

    emergent = p.add_argument_group("(2) emergent -- derived from (1), not independent inputs")
    emergent.add_argument("--resolution", type=int, default=None,
        help="Spatial raster side (power of 2). Default: 2^(observer/2).")
    emergent.add_argument("--n", type=int, default=None,
        help="Total wavefunction qubits. Default: found by SEARCH.")

    prior = p.add_argument_group("(3) optional continuity prior -- off by default")
    prior.add_argument("--jump-cost", type=float, default=0.0,
        help="Bits-per-pixel weight on spatial continuity across frames. Default: 0.0.")

    perf = p.add_argument_group("(4) performance / validation only -- no physical meaning")
    perf.add_argument("--max-candidates", type=int, default=48,
        help="Beam width (candidate cells/frame) for the exact DP. Default: 48.")
    perf.add_argument("--output", type=str, default="observer_typical_path_v5")
    perf.add_argument("--seed", type=int, default=42,
        help="Which realization of the random static phases to draw (single-run mode), "
             "or the seed used to GENERATE the seed list for --seed-sweep.")
    perf.add_argument("--frames", type=int, default=120,
        help="Frames sampled along k for the FINAL rendered animation (single-run mode).")
    perf.add_argument("--scan-frames", type=int, default=16,
        help="Coarser frame count used while SEARCHING for the minimal --n, and for every "
             "run inside --seed-sweep (cheap existence check, not the final render).")
    perf.add_argument("--seed-sweep", type=int, default=0,
        help="If > 0, skip the single-seed render entirely and instead run the minimal-n "
             "search across this many random seeds, reporting how often the emerge-then-fade "
             "observer arc appears. This is the statistical check for whether the arc is "
             "'typical' rather than an artifact of one --seed.")
    return p.parse_args()


def resolve_defaults(args):
    if args.observer % 2 != 0:
        sys.exit(f"--observer must be even. Got {args.observer}.")
    if args.observer <= 0 or args.observer > 16:
        sys.exit(f"--observer must be in (0,16]. Got {args.observer}.")

    if args.resolution is None:
        args.resolution = 2 ** (args.observer // 2)

    res = args.resolution
    if res < 2 or (res & (res - 1)) != 0:
        sys.exit(f"--resolution must be a power of 2 (>=2). Got {res}.")
    n_per_axis = int(math.log2(res))
    spatial_qubits = 2 * n_per_axis
    if spatial_qubits < args.observer:
        sys.exit(f"--resolution={res} gives only {spatial_qubits} spatial qubits, which is "
                  f"smaller than --observer={args.observer}. Raise --resolution or lower --observer.")

    if args.min_record_bits is None:
        args.min_record_bits = max(0.05 * args.observer, 0.05)

    if args.presence <= 0:
        sys.exit(f"--presence must be > 0. Got {args.presence}.")
    if args.jump_cost < 0:
        sys.exit(f"--jump-cost must be >= 0. Got {args.jump_cost}.")
    if args.max_candidates < 1:
        sys.exit(f"--max-candidates must be >= 1. Got {args.max_candidates}.")
    if args.seed_sweep < 0:
        sys.exit(f"--seed-sweep must be >= 0. Got {args.seed_sweep}.")

    s_max = args.s_max if args.s_max is not None else 0.98 * args.observer
    if args.min_record_bits >= s_max:
        sys.exit(f"--min-record-bits ({args.min_record_bits}) must be < --s-max ({s_max:.2f}).")

    max_hidden = 20 - spatial_qubits
    if max_hidden < 1:
        sys.exit(f"--observer={args.observer} already uses all {spatial_qubits} spatial "
                  f"qubits available up to the n<=20 tractability ceiling; no room left "
                  f"for any hidden/environment qubits, so no record can ever form. "
                  f"Lower --observer.")

    return n_per_axis, spatial_qubits, s_max, max_hidden


# ---------------------------------------------------------------------------
# Core sweep: build the compression sweep for one candidate universe size,
# solve the typical path, and assemble per-frame results.
# ---------------------------------------------------------------------------

def simulate_universe(n_qubits, n_per_axis, spatial_qubits, s_max, args, num_frames,
                       idx_to_xy, cell_centers, build_smooth=True):
    d = 2 ** n_qubits
    side = 2 ** n_per_axis
    n_a = args.observer
    n_b = n_qubits - n_a
    hidden_qubits = n_qubits - spatial_qubits

    hadamard_order = get_hamming_ordered_hadamard_basis(n_qubits)
    rng = np.random.default_rng(args.seed)
    static_phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=d))

    k_steps = np.unique(np.linspace(1, d, num=min(num_frames, d), dtype=int))
    uniform_spatial = 1.0 / (2 ** spatial_qubits)

    frames = []
    for k in k_steps:
        psi_h = np.zeros(d, dtype=complex)
        sel = hadamard_order[:k]
        psi_h[sel] = static_phases[sel]
        psi_h /= np.linalg.norm(psi_h)
        psi_comp = fwht(psi_h)
        probs = np.abs(psi_comp) ** 2

        if hidden_qubits > 0:
            spatial_probs = probs.reshape(2 ** spatial_qubits, 2 ** hidden_qubits).sum(axis=1)
        else:
            spatial_probs = probs
        grid = probs_to_grid(spatial_probs, idx_to_xy, side)
        contrast = np.clip(grid - uniform_spatial, 0, None)
        smooth_contrast = build_smooth_field(contrast) if build_smooth else None

        rho_a = partial_trace(psi_comp, n_a, n_b)
        s_a = von_neumann_entropy(rho_a)
        cell_probs = np.real(np.diag(rho_a))

        frames.append({"k": k, "grid": grid, "contrast": contrast,
                        "smooth_contrast": smooth_contrast, "s_a": s_a, "cell_probs": cell_probs})

    cell_probs_by_frame = [f["cell_probs"] for f in frames]
    path, path_probs, total_cost = solve_typical_path(
        cell_probs_by_frame, cell_centers, args.max_candidates, args.jump_cost)

    n_cells = 2 ** n_a
    uniform_p = 1.0 / n_cells
    presence_floor = args.presence * uniform_p
    uniform_bits = math.log2(n_cells)

    results = []
    for f, c, p in zip(frames, path, path_probs):
        entropy_ok = args.min_record_bits <= f["s_a"] < s_max
        gain = uniform_bits - (-math.log2(max(p, 1e-300)))
        visible = entropy_ok and (p >= presence_floor)
        results.append({**f, "cell": c, "p": p, "gain": gain, "visible": visible})

    ratio_arr = np.array([r["p"] / uniform_p for r in results])
    max_ratio = float(ratio_arr.max())
    localization = (np.clip((ratio_arr - 1.0) / (max_ratio - 1.0), 0.0, 1.0)
                     if max_ratio > 1.0 else np.zeros_like(ratio_arr))
    band_width = max(s_max - args.min_record_bits, 1e-9)
    for i, r in enumerate(results):
        u = np.clip((r["s_a"] - args.min_record_bits) / band_width, 0.0, 1.0)
        r["emergence"] = float(math.sin(math.pi * u) * localization[i])

    return results, total_cost, uniform_spatial


# ---------------------------------------------------------------------------
# Minimal-n search: single-run wrapper (prints + exits) and a silent core
# usable by both the single-run CLI path and the seed sweep.
# ---------------------------------------------------------------------------

def search_minimal_n(args, n_per_axis, spatial_qubits, s_max, max_hidden,
                      idx_to_xy, cell_centers, verbose=True):
    """
    Search upward in hidden/environment qubits for the smallest total
    wavefunction at which the observer definition is satisfied AND later
    fades again (see module docstring / v4 rationale). Never exits --
    returns (n_found_or_None, results_at_that_n_or_None, status_str) so it
    can be called repeatedly across seeds without killing the sweep.
    """
    if verbose:
        print(f"searching for the smallest universe where the observer both emerges AND "
              f"later vanishes again (observer={args.observer} qubits, record band "
              f"[{args.min_record_bits:.2f}, {s_max:.2f}) bits, presence>={args.presence}x, "
              f"seed={args.seed})...")
    for hidden in range(1, max_hidden + 1):
        n_try = spatial_qubits + hidden
        results, _, _ = simulate_universe(n_try, n_per_axis, spatial_qubits, s_max, args,
                                           args.scan_frames, idx_to_xy, cell_centers,
                                           build_smooth=False)
        emerges = any(r["visible"] for r in results)
        fades_again = emerges and not results[-1]["visible"]
        if verbose:
            status = ("emerges and fades" if fades_again else
                       "emerges but never fades (bath too small to thermalize it)" if emerges else
                       "never emerges")
            print(f"  n={n_try:2d} (hidden={hidden:2d})  ->  {status}")
        if fades_again:
            return n_try, results, "emerges and fades"
    # loop finished without ever fading again; report the last hidden-qubit
    # count tried so the caller can distinguish "emerges but never fades"
    # (bath too small to thermalize) from "never emerges at all"
    if emerges:
        return None, results, "emerges but never fades"
    return None, None, "never emerges (up to n=20)"


def find_minimal_n(args, n_per_axis, spatial_qubits, s_max, max_hidden, idx_to_xy, cell_centers):
    """Single-run CLI wrapper: same search, but exits with a clear message on failure."""
    n_found, results, status = search_minimal_n(
        args, n_per_axis, spatial_qubits, s_max, max_hidden, idx_to_xy, cell_centers, verbose=True)
    if n_found is None:
        sys.exit(f"No universe up to n=20 satisfies this observer definition (status: {status}). "
                  f"Try a smaller --observer, a looser --presence, or a wider "
                  f"[--min-record-bits, --s-max) band.")
    return n_found


# ---------------------------------------------------------------------------
# Multi-seed validation sweep
# ---------------------------------------------------------------------------

def run_seed_sweep(args, n_per_axis, spatial_qubits, s_max, max_hidden, idx_to_xy, cell_centers):
    """
    Statistical check for the word 'typical': run the minimal-n search across
    many independent phase realizations (seeds) and report how often the
    emerge -> persist -> fade observer arc actually appears, how big a
    universe it took, and where in the sweep the emergence peaked. A single
    --seed run can't distinguish 'this is what typical observers look like'
    from 'this is what seed 42 looks like' -- this can.
    """
    n_seeds = args.seed_sweep
    gen = np.random.default_rng(args.seed)
    seeds = [int(x) for x in gen.integers(0, 2 ** 31 - 1, size=n_seeds)]

    print(f"seed sweep: {n_seeds} independent phase realizations, observer={args.observer} qubits, "
          f"scan_frames={args.scan_frames}, presence>={args.presence}x, jump_cost={args.jump_cost}")
    print(f"(seed list generated from --seed {args.seed}; base run parameters otherwise unchanged)\n")

    records = []
    saved_seed = args.seed
    for i, s in enumerate(seeds):
        args.seed = s
        n_found, results, status = search_minimal_n(
            args, n_per_axis, spatial_qubits, s_max, max_hidden, idx_to_xy, cell_centers,
            verbose=False)
        rec = {"seed": s, "n": n_found, "status": status}
        if results is not None:
            vis = [r for r in results if r["visible"]]
            if vis:
                rec["window_frac"] = len(vis) / len(results)
                peak = max(results, key=lambda r: r["emergence"])
                rec["peak_k_frac"] = peak["k"] / results[-1]["k"]
                rec["peak_s"] = peak["s_a"]
        records.append(rec)
        print(f"  [{i + 1:3d}/{n_seeds}] seed={s:11d}  n={str(n_found):>4}  status={status}")
    args.seed = saved_seed

    succ = [r for r in records if r["status"] == "emerges and fades"]
    frac_succ = len(succ) / n_seeds
    print(f"\nseed sweep summary ({n_seeds} seeds, observer={args.observer}):")
    print(f"  emerge-then-fade observer arc: {len(succ)}/{n_seeds}  ({100 * frac_succ:.0f}%)")

    if succ:
        ns = np.array([r["n"] for r in succ], dtype=float)
        print(f"  minimal n required: mean={ns.mean():.2f}  std={ns.std():.2f}  "
              f"range=[{int(ns.min())},{int(ns.max())}]  mode={Counter(ns.astype(int)).most_common(1)[0][0]}")
        wf = np.array([r["window_frac"] for r in succ if "window_frac" in r])
        if len(wf):
            print(f"  visible-window width (frac. of scan frames): mean={wf.mean():.2f}  std={wf.std():.2f}")
        pf = np.array([r["peak_k_frac"] for r in succ if "peak_k_frac" in r])
        if len(pf):
            print(f"  emergence peak position (frac. of sweep, 0=start 1=end): "
                  f"mean={pf.mean():.2f}  std={pf.std():.2f}")

    other_status = Counter(r["status"] for r in records if r["status"] != "emerges and fades")
    for st, cnt in other_status.items():
        print(f"  {st}: {cnt}/{n_seeds}")

    if succ and len(succ) >= 3:
        ns = np.array([r["n"] for r in succ], dtype=float)
        pf = np.array([r["peak_k_frac"] for r in succ if "peak_k_frac" in r])
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        n_bins = np.arange(ns.min() - 0.5, ns.max() + 1.5, 1)
        axes[0].hist(ns, bins=n_bins, color="steelblue", edgecolor="white")
        axes[0].set_xlabel("minimal n (qubits)")
        axes[0].set_ylabel("seeds")
        axes[0].set_title(f"minimal universe size\n({len(succ)}/{n_seeds} seeds succeeded)")
        if len(pf):
            axes[1].hist(pf, bins=12, range=(0, 1), color="mediumorchid", edgecolor="white")
            axes[1].set_xlabel("emergence peak (frac. of sweep)")
            axes[1].set_ylabel("seeds")
            axes[1].set_title("where in the sweep the observer\nis most 'in focus'")
        plt.tight_layout()
        hist_path = f"{args.output}_seed_sweep.png"
        plt.savefig(hist_path, dpi=130)
        plt.close(fig)
        print(f"\nSaved seed-sweep histograms to {hist_path}")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    n_per_axis, spatial_qubits, s_max, max_hidden = resolve_defaults(args)
    idx_to_xy, _ = morton_maps(n_per_axis)
    cell_centers, cell_boxes = coarse_cell_geometry(args.observer, args.resolution)

    print(f"observer definition: {args.observer} qubits, record band "
          f"[{args.min_record_bits:.2f}, {s_max:.2f}) bits, presence>={args.presence}x uniform")
    print(f"resolution (emergent) = {args.resolution}x{args.resolution}  "
          f"(exactly {2 * n_per_axis} spatial qubits for a {args.observer}-qubit observer)")

    if args.seed_sweep > 0:
        run_seed_sweep(args, n_per_axis, spatial_qubits, s_max, max_hidden, idx_to_xy, cell_centers)
        return

    if args.n is None:
        n_qubits = find_minimal_n(args, n_per_axis, spatial_qubits, s_max, max_hidden, idx_to_xy, cell_centers)
        print(f"n (emergent) = {n_qubits}  (D={2**n_qubits:,}) -- smallest universe in which "
              f"this observer definition is satisfiable")
    else:
        n_qubits = args.n
        if n_qubits < spatial_qubits:
            sys.exit(f"--n={n_qubits} is too small: need >= {spatial_qubits} spatial qubits.")
        if n_qubits > 20:
            sys.exit(f"--n={n_qubits} needs D=2^{n_qubits}={2**n_qubits:,}; too large for this "
                      f"dense implementation.")
        print(f"n (forced by --n) = {n_qubits}  (D={2**n_qubits:,})")

    print(f"jump-cost (continuity prior) = {args.jump_cost}"
          + ("  [pure Solomonoff: no continuity assumption]" if args.jump_cost == 0 else "")
          + f"   max-candidates = {args.max_candidates}")

    results, total_cost, uniform_spatial = simulate_universe(
        n_qubits, n_per_axis, spatial_qubits, s_max, args, args.frames,
        idx_to_xy, cell_centers, build_smooth=True)

    d = 2 ** n_qubits
    side = args.resolution
    global_max_contrast = max(r["contrast"].max() for r in results)
    global_max_smooth_contrast = max(r["smooth_contrast"].max() for r in results)
    print(f"contrast (|P - uniform|) range across sweep: 0.0 -> {global_max_contrast:.5f}   "
          f"(uniform baseline = {uniform_spatial:.5f})")
    print(f"total path description length = {total_cost:.2f} bits over {len(results)} frames")
    vis_ks = [r["k"] for r in results if r["visible"]]
    if vis_ks:
        print(f"observer visible window: k={vis_ks[0]}..{vis_ks[-1]} ({len(vis_ks)}/{len(results)} frames)")
    else:
        print("observer never visible on the winning path at this --presence -- try lowering it.")
    peak_i = int(np.argmax([r["emergence"] for r in results]))
    print(f"emergence probability peaks at k={results[peak_i]['k']} "
          f"(S={results[peak_i]['s_a']:.2f} bits, emergence={results[peak_i]['emergence']:.2f})")

    # --- Render ---
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1], hspace=0.55, wspace=0.3)
    ax_grid = fig.add_subplot(gs[:, 0])
    ax_bits = fig.add_subplot(gs[0, 1])
    ax_entropy = fig.add_subplot(gs[1, 1])
    ax_emergence = fig.add_subplot(gs[2, 1])

    im = ax_grid.imshow(results[0]["contrast"], cmap="inferno", origin="lower",
                         interpolation="bicubic", vmin=0, vmax=global_max_contrast)
    contour_holder = {"cs": None}
    ax_grid.set_title("Universe: contrast |ψ(x,y)|² − uniform  (observer = real raster contour)")
    ax_grid.set_xticks([]); ax_grid.set_yticks([])

    k_arr = [r["k"] for r in results]
    gain_arr = [r["gain"] for r in results]
    ax_bits.plot(k_arr, gain_arr, color="gray", lw=1)
    ax_bits.axhline(0.0, color="red", ls="--", lw=0.8, label="uniform baseline")
    marker, = ax_bits.plot([k_arr[0]], [gain_arr[0]], "o", color="cyan", ms=7)
    ax_bits.set_xlabel("k", fontsize=8)
    ax_bits.set_ylabel("gain (bits)", fontsize=8)
    ax_bits.set_title("Compression gain vs. uniform", fontsize=9)
    ax_bits.legend(fontsize=7)
    ax_bits.tick_params(labelsize=7)

    s_arr = [r["s_a"] for r in results]
    ax_entropy.plot(k_arr, s_arr, color="darkorange", lw=1.5)
    ax_entropy.axhline(args.min_record_bits, color="gray", ls="--", lw=0.8, label="min record")
    ax_entropy.axhline(s_max, color="gray", ls=":", lw=0.8, label="thermalized")
    marker_entropy, = ax_entropy.plot([k_arr[0]], [s_arr[0]], "o", color="cyan", ms=7)
    ax_entropy.set_xlabel("k", fontsize=8)
    ax_entropy.set_ylabel("S(ρ_A) bits", fontsize=8)
    ax_entropy.set_title("Entropy: minimal → equilibrium", fontsize=9)
    ax_entropy.legend(fontsize=7)
    ax_entropy.tick_params(labelsize=7)

    emergence_arr = [r["emergence"] for r in results]
    ax_emergence.plot(k_arr, emergence_arr, color="mediumorchid", lw=1.5)
    ax_emergence.fill_between(k_arr, emergence_arr, color="mediumorchid", alpha=0.15)
    marker_emergence, = ax_emergence.plot([k_arr[0]], [emergence_arr[0]], "o", color="cyan", ms=7)
    ax_emergence.set_xlabel("k", fontsize=8)
    ax_emergence.set_ylabel("emergence prob.", fontsize=8)
    ax_emergence.set_ylim(-0.05, 1.05)
    ax_emergence.set_title("P(observer emergence) -- heuristic", fontsize=9)
    ax_emergence.tick_params(labelsize=7)

    title = fig.suptitle("")

    fixed_levels = [0.25 * global_max_smooth_contrast,
                     0.55 * global_max_smooth_contrast,
                     0.85 * global_max_smooth_contrast]
    contour_extent = (-0.5, side - 0.5, -0.5, side - 0.5)

    def masked_smooth_field(r):
        mask = region_mask(r["smooth_contrast"].shape, cell_boxes[r["cell"]], side, CONTOUR_UPSAMPLE)
        return np.ma.masked_array(r["smooth_contrast"], mask=mask)

    def update(i):
        r = results[i]
        im.set_data(r["contrast"])

        if contour_holder["cs"] is not None:
            contour_holder["cs"].remove()
            contour_holder["cs"] = None

        if r["visible"]:
            field = masked_smooth_field(r)
            levels = [lv for lv in fixed_levels if lv < field.max()]
            if levels:
                cs = ax_grid.contour(field, levels=levels, extent=contour_extent,
                                      cmap="cool", linewidths=1.3, alpha=0.9)
                contour_holder["cs"] = cs

        marker.set_data([r["k"]], [r["gain"]])
        marker_entropy.set_data([r["k"]], [r["s_a"]])
        marker_emergence.set_data([r["k"]], [r["emergence"]])
        if r["visible"]:
            status = "OBSERVER (typical path)"
        elif r["s_a"] < args.min_record_bits:
            status = "no record"
        elif r["s_a"] >= s_max:
            status = "thermalized"
        else:
            status = "path too diffuse (below --presence)"
        title.set_text(f"k={r['k']}/{d}   S(ρ_A)={r['s_a']:.2f}   "
                        f"P(path cell)={r['p']:.4f}   gain={r['gain']:+.2f} bits   [{status}]")
        return im, marker, marker_entropy, marker_emergence

    anim = FuncAnimation(fig, update, frames=len(results), interval=110, blit=False)
    gif_path = f"{args.output}.gif"
    anim.save(gif_path, writer=PillowWriter(fps=9))
    plt.close(fig)
    print("Saved animation to", gif_path)

    fig2, axes = plt.subplots(1, 6, figsize=(20, 4))
    preview_idxs = np.linspace(0, len(results) - 1, 6, dtype=int)
    for ax, i in zip(axes, preview_idxs):
        r = results[i]
        ax.imshow(r["contrast"], cmap="inferno", origin="lower", interpolation="bicubic",
                   vmin=0, vmax=global_max_contrast)
        if r["visible"]:
            field = masked_smooth_field(r)
            levels = [lv for lv in fixed_levels if lv < field.max()]
            if levels:
                ax.contour(field, levels=levels, extent=contour_extent,
                           cmap="cool", linewidths=1.3, alpha=0.9)
        ax.set_title(f"k={r['k']}\nS={r['s_a']:.2f}  gain={r['gain']:+.2f}b", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    preview_path = f"{args.output}_preview.png"
    plt.savefig(preview_path, dpi=130)
    plt.close(fig2)
    print("Saved static preview to", preview_path)


if __name__ == "__main__":
    main()
