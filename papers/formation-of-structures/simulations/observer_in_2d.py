#!/usr/bin/env python3
"""
Observer as a shortest-description-length path in a growing-entropy 2D
complex-valued spectrally compressed universe -- v4.
The ultimate goal: define the observer and the rest follows as typical (most probable) outcome
due to solomonof induction with "spectral complexity" measure. 

This POC should demonstrate the following:

The universe is static and timeless, described by the total wavefunction,
 with information content of n bits. 

The wavefunction of the universe can be decomposed to nearly infinite
ensembles, which can be arranged into increasing entropy order.

Time emerges as walk through these ensembles. We are interested in walks
that contain the specified observer (as universes without the observer cannot be observed by the observer)

Those walks that has minimal description length dominate the measure. So the
observer most likely find themselves within one of the most compressible (minimal wavefunction) walks.
These are walks with smooth predictable dynamics, as such information compresses best.

The compression codec is the wavefunction, complex-valued,
MPEG-like, over ensembles (over time). MDL is measured over the entire WALK (path), not per frame!

"Walk through all possible paths from zero entropy; shortest
description length dominates the measure." This is a
literal Viterbi dynamic program over the branch graph:
 - state at frame j = a coarse cell c (candidate observer location)
 - emission cost(c, j)  = -log2 P(c | k_j)          (Born/Shannon
                                                             code length)
       - transition cost(c', c) = lambda * ||pos(c') - pos(c)||  (motion-
         vector cost -- exactly what MPEG pays for a moving region: cheap
         if it's roughly the same place, expensive if it teleports)
   The path minimizing total accumulated cost is the one with shortest
   description length, i.e. the one a Solomonoff-style measure 2^-L
   weights highest -- "the path the observer most likely finds
   themselves on." This replaces BOTH the old fidelity gate and the
   old multi-branch blob cloud with a single selected trajectory,
   found by exact DP (not sampling).
   "Presence" (--presence) now means: is the winning path's cell
   probability at this frame far enough above the uniform baseline
   to represent real compression gain, not noise. Frames failing that
   (or outside the entropy validity band) are marked not-visible.

Spectral - geometric mapping. thed raster probability 
amplitudes define the observer." At each frame the observer's
visual IS a masked contour of the REAL |psi(x,y)|^2 raster, cropped
to the winning path's coarse-cell block (plus one block of padding
for context) and contoured at real, data-derived levels. If the
true amplitude structure there is a ring, a ring is what renders.

Spectral complexity: this is Kolmogorov complexity in principle, but as the
actual algorithm (the sin() function) cost is constant, per class, its impact
is vanishingly small. The  actual dominant complexity emerges from the data - the
informational cost of frequencies and phases. 

Missing proper spectral -> geometry mapping. Zero information (or as close zero as minimal wavefunciton allows) should
map to geometric singularity. Any geometric interpretation of "one state" should then map to
a geometric point. Its classical statistical "bitstring" equivalence is all-zero bitstring.
All possible ways to map or interpret the bitstring as "positiosn" yield the same geometric object - a point.
Similary, when starting out from zero mode, one mode etc. wavefunction ensembles, their
geometric projections should collapse to geometric point (geometry with no micro structures).

Expected outcome;
- universe starts minimally simple, just black image, the probability amplitudes should be
zero everywhere, no emergent micro structuures, let alone observer.
- as entropy increases probability amplitudes rise gradually and the heat map begins to glow minimally
trivial structures that get more complex with time, soon observer memerges and later vanishes as
entropy goes up and emergent structures vanish to make room for structureless "noise"
- observer does not jump chaotically, it emerges, moves smootly, and vanish.


Observer as a shortest-description-length path in a growing-entropy 2D
complex-valued spectrally compressed universe -- v4 (render fix).

"""

import argparse
import math
import sys

import numpy as np
from scipy.ndimage import zoom as ndimage_zoom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation, PillowWriter

# Upsampling factor used to build the contour field. Matches (in spirit) the
# 'bicubic' interpolation matplotlib applies to the *displayed* background so
# that contour lines are drawn on geometry as smooth as what the eye compares
# them against, instead of raw marching-squares on the coarse observer-cell grid.
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
    """Cubic-spline upsample of a raw raster field, order-3 (bicubic-like),
    for use as the CONTOUR field so it matches the smoothness of the
    bicubic-interpolated background image. Cubic splines can ring slightly
    below zero near sharp edges, so we clip back to a physical (>=0) field."""
    smooth = ndimage_zoom(grid, upsample, order=3, mode="nearest")
    return np.clip(smooth, 0, None)


def region_mask(shape, box, side, upsample, pad_blocks=1):
    """Boolean mask (True = masked OUT) for the padded observer-cell box,
    expressed in upsampled-pixel coordinates. Used with np.ma.masked_array
    so the contour simply isn't drawn outside the region -- no hard zero
    cliff is introduced at the boundary the way masked_region() used to do."""
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
    """
    jump_cost_weight (lambda) has units of BITS PER PIXEL. Total path cost is
    a single MDL code length in bits:

        L(path) = sum_j [ -log2 P(cell_j | k_j) ]        <- amplitude code
                + sum_j [ lambda * ||pos_j - pos_{j-1}|| ]  <- motion code

    The first term is not a free choice -- it's the Born-rule/Shannon code
    length straight from the wavefunction. The second term is *also* an MDL
    code length in the same spirit (a smoothly-moving path is cheaper to
    describe than a teleporting one, exactly like a video codec's motion
    vectors), but bits and pixels are different units, so lambda is the
    necessary exchange rate between them. It cannot be derived from nothing
    -- some exchange rate must be chosen for the two code lengths to be
    commensurable at all -- but it CAN be made scale-consistent rather than
    an arbitrary magic number: see resolve_defaults()'s derivation of
    --jump-cost, which pins lambda to grid geometry instead of hand-tuning
    it per --resolution/--n combination.
    """
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
    """
    Reclassified around the observer-centric theory: define the OBSERVER,
    let everything else follow as the typical/most-compressible outcome.

    (1) OBSERVER DEFINITION -- together these ARE what "an observer" means
        in this model. None of them is derived from the wavefunction; they
        are the recognition criteria we, the ones asking the question,
        impose in order to say "a record formed here." That's unavoidable
        for the same reason a particle detector needs a sigma-level -- but
        it's honest to keep them as one deliberate bundle rather than
        pretending some are "physics" and others are "just thresholds."
            --observer          size of the observer's memory register (qubits)
            --min-record-bits   entropy floor: below this, rho_A carries no
                                 distinguishable record at all
            --s-max              entropy ceiling: above this, rho_A is
                                 thermalized/mixed, record washed out
            --presence           how far the winning path's cell probability
                                 must clear the uniform baseline to count as
                                 "there" rather than noise

    (2) EMERGENT -- NOT free parameters at all. These follow from (1):
            --resolution   derived as EXACTLY enough spatial qubits to give
                            the observer register a well-defined position
                            (side = 2^(observer/2)) -- no unused context
                            padding unless you override it by hand.
            --n             derived by SEARCH: the smallest total wavefunction
                            (fewest hidden/environment qubits beyond the
                            observer's own footprint) at which the observer
                            definition in (1) is actually satisfiable
                            somewhere along the compression sweep. This is
                            "sufficiently large for the observer to emerge,"
                            taken literally, as an output rather than a guess.

    (3) OPTIONAL CONTINUITY PRIOR -- OFF by default, and deliberately not
        part of (1) or (2):
            --jump-cost   Pure Solomonoff induction over the wavefunction
                           gives you exactly sum_j [-log2 P(cell_j | k_j)]
                           as the path's description length -- nothing about
                           frame-to-frame spatial proximity. Any term that
                           rewards nearby cells across frames is an EXTRA
                           assumption (real observers persist continuously)
                           that the wavefunction's own measure does not
                           supply. So the theory-pure default is 0: each
                           frame's cell is chosen independently by minimal
                           description length alone, and whether the
                           resulting path *happens* to look continuous
                           becomes something to check, not something built
                           in. Set > 0 only if you want to explicitly test a
                           continuity-prior variant against the pure one.

    (4) PERFORMANCE ONLY -- no physical or definitional meaning; changes
        runtime and rendering fidelity, not the answer.
            --max-candidates, --frames, --seed, --output, --scan-frames
    """
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
        help="Spatial raster side (power of 2). Default: 2^(observer/2) -- exactly enough "
             "to give the observer a well-defined position, no more. Override to add "
             "spatial context padding around the observer's footprint.")
    emergent.add_argument("--n", type=int, default=None,
        help="Total wavefunction qubits. Default: found by SEARCH -- the smallest universe "
             "(fewest hidden/environment qubits) at which the observer definition in (1) "
             "is satisfiable anywhere in the compression sweep. Set explicitly to skip the "
             "search and force a specific universe size.")

    prior = p.add_argument_group("(3) optional continuity prior -- off by default")
    prior.add_argument("--jump-cost", type=float, default=0.0,
        help="Bits-per-pixel weight on spatial continuity across frames. Default: 0.0 "
             "(pure Solomonoff induction on the wavefunction alone -- no external "
             "assumption that the observer persists continuously). See parse_args "
             "docstring category (3) before setting this above zero.")

    perf = p.add_argument_group("(4) performance only -- no physical meaning")
    perf.add_argument("--max-candidates", type=int, default=48,
        help="Beam width (candidate cells/frame) for the exact DP. Default: 48.")
    perf.add_argument("--output", type=str, default="observer_typical_path_v4")
    perf.add_argument("--seed", type=int, default=42,
        help="Which realization of the random static phases to draw. Like --presence, an "
             "unavoidable free choice -- every run necessarily picks *some* phase "
             "realization -- rather than something derivable from the definition alone.")
    perf.add_argument("--frames", type=int, default=120,
        help="Frames sampled along k for the FINAL rendered animation.")
    perf.add_argument("--scan-frames", type=int, default=16,
        help="Coarser frame count used only while SEARCHING for the minimal --n (cheap "
             "existence check per candidate universe size, not the final render).")
    return p.parse_args()


def resolve_defaults(args):
    """
    Resolve category (1)+(2) parameters that were left unset. --observer is
    the one true independent input; --resolution follows from it directly
    (a formula, not a search). --n is intentionally NOT resolved here -- it
    requires actually running sweeps to find the minimal universe that
    supports the observer definition, so that search lives in
    find_minimal_n() / main(), after this function has fixed everything
    else the search needs to hold constant.
    """
    if args.observer % 2 != 0:
        sys.exit(f"--observer must be even. Got {args.observer}.")
    if args.observer <= 0 or args.observer > 16:
        sys.exit(f"--observer must be in (0,16]. Got {args.observer}.")

    if args.resolution is None:
        args.resolution = 2 ** (args.observer // 2)  # exactly enough grid for the register

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

    s_max = args.s_max if args.s_max is not None else 0.98 * args.observer
    if args.min_record_bits >= s_max:
        sys.exit(f"--min-record-bits ({args.min_record_bits}) must be < --s-max ({s_max:.2f}).")

    return n_per_axis, spatial_qubits, s_max


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Core sweep: build the compression sweep for one candidate universe size,
# solve the typical path, and assemble per-frame results. Used twice: cheaply
# during find_minimal_n()'s search, and once at full fidelity for the render.
# ---------------------------------------------------------------------------

def simulate_universe(n_qubits, n_per_axis, spatial_qubits, s_max, args, num_frames,
                       idx_to_xy, cell_centers, build_smooth=True):
    d = 2 ** n_qubits
    side = 2 ** n_per_axis
    n_a = args.observer
    n_b = n_qubits - n_a
    hidden_qubits = n_qubits - spatial_qubits  # qubits summed out of the VISUAL raster

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


def find_minimal_n(args, n_per_axis, spatial_qubits, s_max, idx_to_xy, cell_centers):
    """
    Literalizes 'sufficiently large for the observer to emerge': search
    upward in hidden/environment qubits (n = spatial_qubits + hidden) for
    the SMALLEST total wavefunction at which the observer definition
    (--observer, --min-record-bits, --s-max, --presence) is actually
    satisfied by at least one frame of the winning path, somewhere in the
    compression sweep. n is an OUTPUT of the observer definition, not an
    independent input to it.

    Success criterion is deliberately stronger than "appears at all": the
    theory's actual prediction (see module docstring) is emerge -> persist
    -> VANISH as entropy keeps climbing past the record band into
    thermalization. A bath of just 1-2 hidden qubits can satisfy "appears
    somewhere" trivially and then just stay visible for the rest of the
    sweep (S(rho_A) plateaus well under s_max because a tiny bath can't
    carry the observer's reduced state up to full mixing) -- that's a real
    but degenerate case, not the emerge-and-fade arc the theory predicts.
    So "found" here requires the winning path to be visible at at least one
    frame AND not still visible at the final frame -- i.e. enough bath
    exists for the record to also wash back out again before the sweep ends.

    hidden=0 means n_b (qubits traced out of rho_A) = 0 too, so rho_A stays
    exactly pure (S=0) at every k -- no record can ever form -- which is why
    the search always has to start above 0.
    """
    max_hidden = 20 - spatial_qubits
    if max_hidden < 1:
        sys.exit(f"--observer={args.observer} already uses all {spatial_qubits} spatial "
                  f"qubits available up to the n<=20 tractability ceiling; no room left "
                  f"for any hidden/environment qubits, so no record can ever form. "
                  f"Lower --observer.")

    print(f"searching for the smallest universe where the observer both emerges AND "
          f"later vanishes again (observer={args.observer} qubits, record band "
          f"[{args.min_record_bits:.2f}, {s_max:.2f}) bits, presence>={args.presence}x)...")
    for hidden in range(1, max_hidden + 1):
        n_try = spatial_qubits + hidden
        results, _, _ = simulate_universe(n_try, n_per_axis, spatial_qubits, s_max, args,
                                           args.scan_frames, idx_to_xy, cell_centers,
                                           build_smooth=False)
        emerges = any(r["visible"] for r in results)
        fades_again = emerges and not results[-1]["visible"]
        status = ("emerges and fades" if fades_again else
                   "emerges but never fades (bath too small to thermalize it)" if emerges else
                   "never emerges")
        print(f"  n={n_try:2d} (hidden={hidden:2d})  ->  {status}")
        if fades_again:
            return n_try
    sys.exit(f"No universe up to n=20 satisfies this observer definition. Try a smaller "
              f"--observer, a looser --presence, or a wider [--min-record-bits, --s-max) band.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    n_per_axis, spatial_qubits, s_max = resolve_defaults(args)
    idx_to_xy, _ = morton_maps(n_per_axis)
    cell_centers, cell_boxes = coarse_cell_geometry(args.observer, args.resolution)

    print(f"observer definition: {args.observer} qubits, record band "
          f"[{args.min_record_bits:.2f}, {s_max:.2f}) bits, presence>={args.presence}x uniform")
    print(f"resolution (emergent) = {args.resolution}x{args.resolution}  "
          f"(exactly {2 * n_per_axis} spatial qubits for a {args.observer}-qubit observer)")

    if args.n is None:
        n_qubits = find_minimal_n(args, n_per_axis, spatial_qubits, s_max, idx_to_xy, cell_centers)
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

    # Contour levels as fixed fractions of the SAME global scale the background
    # colormap uses (vmin=0, vmax=global_max_contrast). This is what fixes the
    # "40% brightness renders as a full-strength contour" problem: a frame
    # whose local peak is only 40% of the video-wide max now draws faint/no
    # contour, matching how dark it correctly looks against the background.
    fixed_levels = [0.25 * global_max_smooth_contrast,
                     0.55 * global_max_smooth_contrast,
                     0.85 * global_max_smooth_contrast]
    # Pixel-center extent so the upsampled contour field lines up exactly with
    # imshow's default extent for the raw side x side background array.
    contour_extent = (-0.5, side - 0.5, -0.5, side - 0.5)

    def masked_smooth_field(r):
        """Clip the smoothed contrast field to the winning cell's padded box
        via masking (not zeroing). A masked array simply isn't contoured
        outside the mask -- there's no artificial value discontinuity at the
        box edge, so the box's straight sides no longer get traced as if they
        were real amplitude structure."""
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
            # Only draw levels that actually fall under this frame's real
            # (unmasked) peak -- otherwise a genuinely faint frame would still
            # get a full triple-contour outline at levels it never reaches.
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
