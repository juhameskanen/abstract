#!/usr/bin/env python3
"""
Observer as a shortest-description-length path in a growing-entropy 2D
complex-valued spectrally compressed universe.

This header is written to be self-reconstructing: the argument, the reason
for every design choice, and the empirical result that validated the key
mechanism are all recorded here, not just in the code or in conversation.
Someone re-reading only this docstring should be able to (a) restate the
theory's claim, (b) understand why THIS specific construction was chosen
over alternatives, and (c) rebuild the simulation from scratch if the file
were lost.


The argument (one-parameter theory)
-----------------------------------
Define an observer: a finite subsystem (a qubit register, --observer bits)
capable of holding a distinguishable record (non-trivial reduced entropy,
neither pure nor maximally mixed). Ask: within the space of ALL possible
universes (wavefunctions) that could contain such an observer, which ones
does the observer typically find itself in?

Solomonoff induction answers this without any extra physical assumption:
the measure 2^-L(x) over any set of possible objects/paths x is dominated
by whichever ones have short description length L(x). A universe/walk that
is smooth and predictable has a short description (few bits: "keep doing
what you were doing"); a chaotic, jumpy one needs its randomness spelled
out bit by bit and so has near-maximal description length. Therefore an
observer embedded in a "typical" (measure-dominant) universe should
generically find its surroundings smooth, continuous, and predictable --
NOT because continuity is imposed as a law, but because the measure over
all possible universes is overwhelmingly concentrated on the continuous,
compressible ones. This is the one real input of the theory: --observer.
Everything else below is claimed to follow, and where it doesn't yet
(section "STATUS", below), that's flagged explicitly.


WHY A SPECTRAL / MPEG-LIKE CODEC, SPECIFICALLY
------------------------------------------------
The argument above needs SOME compression codec to make "short description
length" concrete and computable. Any codec that is (i) continuous -- small
changes in the underlying data produce small changes in the reconstructed
signal, not jumps -- and (ii) algorithmically minimal -- the decoder itself
costs a near-constant, vanishing number of bits regardless of how much data
it's fed -- would in principle serve the argument above equally well; the
theory does not require THIS codec uniquely. A spectral / Fourier-family
codec (of which the classical MPEG/JPEG progressive-frequency scheme is the
familiar real-world example) is used here for three reasons, and it matters
that these are three DIFFERENT KINDS of reason, not restatements of each
other:
  1. Algorithmically minimal: the decoder is a single fixed transform
     (here, the Fast Walsh-Hadamard Transform, playing the role JPEG's IDCT
     or MPEG's IDCT-per-macroblock plays) -- constant code cost, satisfying
     the "vanishing algorithm cost" requirement below.
  2. Continuous by construction: adding one more spectral coefficient
     perturbs the reconstructed field smoothly, never discontinuously.
  3. Empirically motivated, NOT derived: physical reality already appears
     to be built from wave/field superpositions (this is an appeal to what
     we observe, not a Solomonoff-measure argument) -- a reasonable prior
     for which codec family to explore first, but a different KIND of
     justification than (1)/(2). Keep these threads visibly separate:
     (1)+(2) are in-principle checkable claims about this model; (3) is
     motivation for why this codec family over an unmotivated alternative.


THE ACTUAL MECHANISM: CONTINUITY OF THE SWEEP, NOT ITEM-LEVEL COMPLEXITY
---------------------------------------------------------------------------
An earlier draft of this header described the compression-level ordering
(which basis coefficients get included as the "clock" k advances) as
"Kolmogorov-complexity-ordered" -- lowest-complexity data first. That
framing is WRONG and was empirically falsified (see EMPIRICAL VALIDATION
below); it is kept here, crossed out in spirit, only so nobody re-derives
it and re-runs into the same dead end.

What actually matters is not the complexity of individual included
coefficients, but the CONTINUITY OF THE WALK through spectral space as a
whole: does each step from k to k+1 change the set of "on" coefficients by
a small, bounded amount in spectral-weight terms, or by a large one? A
walk-selection rule with this property produces, at every k, a state whose
partial spectral content is a coherently band-limited (blurred) version of
the final state -- exactly the progressive-JPEG picture. A walk-selection
rule WITHOUT this property can be just as short to describe (e.g. plain
binary counting order) yet still fails, because consecutive steps can flip
many spectral-weight bits at once.

Concretely: order the D=2^n_qubits computational basis states by Hamming
weight (popcount) of their index, ascending, ties broken by index
(get_hamming_ordered_hadamard_basis). Build the wavefunction at compression
level k by taking the first k basis states in this order, giving them
independently drawn random phases (fixed per run, --seed), normalizing, and
applying the FWHT to get the state in the computational/position basis
(fwht). This is a Fourier-family truncation where "frequency" is popcount:
adding one more state at a time changes total spectral weight by at most
one Hamming-weight step, so the walk is maximally continuous in the
relevant sense.


EMPIRICAL VALIDATION (ordering_control_test.py -- reproduce this before
trusting anything else in this file)
---------------------------------------------------------------------------
Tested five walk-selection rules, same D=2^12 basis, same 15 independent
phase draws (--seed) per rule, same observer=6 / hidden=6 budget:

  ordering          mean Spearman(k, S(rho_A))    emerges-and-fades
  hamming (popcount asc)      1.000 (std 0.000)        15/15
  reverse_hamming (desc)      1.000 (std 0.000)        15/15
  natural (binary counting)   1.000 (std 0.000)          4/15
  reverse_natural              1.000 (std 0.000)          0/15
  random (shuffled)          0.79  (std 0.19, noisy)      2/15

Two findings, both load-bearing:
  (a) Entropy-vs-k monotonicity is a WEAK test -- any deterministic,
      non-random walk gets a clean monotone trend, including binary
      counting. It does not by itself distinguish good walk rules from bad
      ones. Random walks fail this too (noisy, non-monotone), confirming
      SOME structure is necessary, just not confirming which.
  (b) Reliable observer emergence (a legible, presence-thresholded,
      persistent-then-fading record) is the STRONG test, and it tracks
      spectral continuity specifically: both popcount directions succeed
      (15/15, direction-symmetric -- consistent with Solomonoff induction
      caring about the SHORTNESS of the ordering rule, which popcount-asc
      and popcount-desc share, not its direction), while binary counting
      -- an equally short rule that is NOT spectrally continuous -- mostly
      or entirely fails (4/15, 0/15 reversed). This is the evidence that
      motivates the "continuity of the sweep" mechanism above over the
      earlier "low complexity first" framing.

Open follow-up, not yet run: test a walk rule that is spectrally graded but
NOT literally popcount (e.g. shuffled within each popcount class, or a
different monotone function of popcount) to determine whether popcount
specifically is required or any comparably continuous grading suffices.


THIS POC DEMONSTRATES
------------------------------------------------------------------------------
1. The universe is static and timeless, with information content of n bits
   (--n, or found by search -- see IMPLEMENTATION SUMMARY).

2. The n bits encode both code (the FWHT decoder -- fixed, vanishing cost)
   and data (which spectral coefficients are on, and their phases) that
   together describe every possible representation of the observer.

3. The wavefunction decomposes into a near-infinite family of ensembles
   (one per compression level k) that trend toward higher entropy as k
   grows -- a TREND, not a rule forced frame-to-frame (see ordering
   discussion above; verified empirically, not assumed).

4. Time emerges as a walk through these ensembles, restricted to walks that
   contain the specified observer (a universe without the observer cannot
   be observed by it).

5. Walks of minimal description length dominate the measure: the observer
   typically finds itself on one of the most compressible walks, i.e. the
   smoothest, most spectrally-continuous ones (see THE ACTUAL MECHANISM).
   This has to be demonstrated by the wavefunction's own behavior, not
   imposed on the path-selection algorithm -- see item 7's note on the
   removed --jump-cost parameter.

6. The codec is the wavefunction itself, complex-valued, spectral/MPEG-like
   over ensembles (over emergent time). MDL is scored over the WHOLE walk,
   not per frame: total path cost is the sum of each frame's own code
   length, and that sum is minimized exactly when every frame independently
   picks its own highest-probability cell -- see item 7.

7. Implemented as per-frame maximum-likelihood cell selection
   (solve_typical_path): at frame j, the winning cell is
   argmax_c P(c | k_j), i.e. the cell minimizing that frame's own
   Born/Shannon code length -log2 P(c | k_j). The reported "total path
   description length" sums these per-frame code lengths over the whole
   walk -- consistent with item 6, since summing independent per-frame
   minima IS the minimum-total-cost walk when there is no cross-frame term.
   "Presence" (--presence): the winning cell's probability must clear the
   uniform baseline by this factor to count as a real record rather than
   noise; frames failing that (or outside the entropy validity band
   [--min-record-bits, --s-max)) are marked not-visible.
   REMOVED (v8): an earlier version added an explicit MPEG-style
   motion-vector cost (lambda * ||pos(c')-pos(c)||, a --jump-cost CLI
   parameter, framed as a Viterbi DP over a branch graph) to penalize
   spatial jumps between frames. This was a hand-coded continuity PRIOR
   imposed on the algorithm -- exactly the kind of predefined rule item 5's
   own argument says should NOT be needed: smoothness is supposed to emerge
   because smooth walks compress best and dominate the measure, not because
   the selection algorithm is penalized for jumping. It was also already
   inert at every setting actually used: default --jump-cost=0.0 makes the
   transition term vanish identically, so the DP was provably reducing to
   this same per-frame argmax the whole time -- removing it changes no
   prior numerical result. Any discontinuity that now shows up in the
   winning path across frames is real information about the physics (the
   probability landscape moved abruptly, or comparably-likely modes are
   competing), not something smoothed away by the algorithm -- treat path
   jumps as a diagnostic to look at, not a bug to suppress.

8. Spectral-geometric mapping (CURRENT, PLACEHOLDER): the observer's visual
   at each frame is a masked contour of the REAL |psi(x,y)|^2 raster,
   cropped to the winning path's cell block and contoured at real,
   data-derived levels -- not a stylized stand-in. BUT the embedding space
   itself (a flat 2D Morton raster) is a placeholder, not a derived result
   -- see STATUS below. This item demonstrates "does a legible, stable
   structure emerge, persist, and fade," which does not require the
   embedding to be correct, only honest about what it is.

9. Spectral complexity: the DECODER (FWHT) cost is constant and vanishing
   per frame; the complexity that actually varies with k is the DATA cost
   -- how many coefficient phases must be specified, roughly linear in k.
   "Kolmogorov complexity of the ordering" was a misleading way to describe
   this (see THE ACTUAL MECHANISM); the fix is to talk about description
   length of the WALK-SELECTION RULE and spectral continuity, not about a
   complexity ranking of individual basis coefficients.

10. Spectral -> geometry mapping (NOT implemented -- next focus area, see
    STATUS). Why 3+1 D, why spherical symmetry, is meant to be derived from
    Solomonoff induction over possible Hilbert-space -> geometric-space
    embeddings (shortest description length again), not assumed.

11. Zero information should map to a geometric singularity/point (also not
    yet implemented as a literal claim -- current 2D raster only shows this
    qualitatively, as "near-zero contrast everywhere").

Expected qualitative outcome (already reproduced by the sim): near-zero k
-> flat/black raster, no structure; as k grows, structure gradually
brightens and localizes; a legible, smoothly-moving observer record
emerges, persists, then dissolves back into thermalized noise as k
approaches D. The observer does not teleport; for most of the sweep no
observer is visible at all.


IMPLEMENTATION SUMMARY (enough to rebuild from scratch)
------------------------------------------------------------------------------
- Basis: n_qubits-qubit computational basis, D=2^n_qubits states.
- Walk order: get_hamming_ordered_hadamard_basis -- sort state indices by
  (popcount, index).
- Per compression level k: zero-init a D-vector, set the first k
  (walk-ordered) entries to independently-drawn uniform-random phases
  (--seed), L2-normalize, apply fwht (orthonormal Fast Walsh-Hadamard
  Transform) to get the computational/position-basis state psi_comp.
- Spatial raster: reshape/marginalize |psi_comp|^2 over spatial_qubits
  (Morton/Z-order interleaved x,y bits, morton_maps) vs. hidden_qubits
  (environment, summed out) -- probs_to_grid.
- Observer register: n_a = --observer qubits, treated as the leading
  spatial_qubits when --resolution is left at its default (2^(observer/2)),
  so diag(partial_trace(psi_comp, n_a, n_b)) is EXACTLY the spatial
  marginal (verified numerically) -- the observer's reduced density matrix
  IS the position distribution in this default configuration.
  von_neumann_entropy(rho_a) = S(rho_A), the record-formation entropy.
- Path selection: solve_typical_path -- per-frame argmax, no DP (see item 7).
- Universe size n: search upward in hidden/environment qubits
  (find_minimal_n / search_minimal_n) for the smallest n at which the
  observer definition (record band + presence) is satisfiable AND later
  fades again -- n is an OUTPUT of the observer definition, not a free
  input, per the one-parameter goal below.
- --seed-sweep N: statistical validation across N independent phase draws,
  reporting how often the emerge-then-fade arc appears (this is the
  'typical' claim, tested rather than asserted).


STATUS / open items
------------------------------------------------------------------------------
- ONE-PARAMETER GOAL (not yet fully realized): --observer should be the
  only theoretically-meaningful input; --resolution and --n already follow
  from it by default. --min-record-bits and --s-max also default to
  functions of --observer (5%/98% of its bit capacity) but are still
  declared in the SAME CLI group as --observer and --presence (group (1),
  "this IS the theory's one input") rather than in group (2) ("emergent"),
  and the 5%/98% constants themselves are unexplained -- this grouping vs.
  STATUS-text mismatch is a known, still-open inconsistency, not yet fixed.
  --presence remains a genuinely independent hand-set threshold. 
- COARSE-CELL DEGENERACY: at default settings, block_px works out to 1 --
  "coarse cells" are currently single pixels, not blocks. Either this is
  fine (decide explicitly) or --resolution needs to be decoupled from
  --observer to get real spatial coarse-graining.
- GEOMETRY MAPPING (items 10-11): not implemented. Candidate first step
  under discussion: test whether the raster's emergent structure is
  isotropic (no axis privileged by the Morton encoding) before attempting
  an MDL-optimal embedding-dimension search.
- ORDERING FOLLOW-UP: popcount-vs-not-popcount spectrally-graded orderings,
  to isolate whether popcount specifically is required (see EMPIRICAL
  VALIDATION open follow-up).
- EMERGENCE-CURVE HONESTY: the plotted "emergence" score is
  sin(pi*u)*localization, where u sweeps 0->1 across the entropy record
  band. Given entropy trends upward with k for almost any deterministic
  walk (see EMPIRICAL VALIDATION), the sin(pi*u) factor's bump SHAPE is
  close to guaranteed by the band definition alone and is not itself
  evidence of anything; only the localization factor (whether the winning
  cell's probability actually beats uniform) carries real information.
  State this next to the plot, not just here.
- SEQUENCY WORDING: ties within a popcount tier are broken by raw index,
  not Gray-code bit-reversal, so this is coarse frequency-band ordering,
  not true Walsh sequency order -- scope "spectral order" language down to
  that.
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
# Shortest-description-length path search (per-frame argmax; see item 7 --
# no DP/branch-graph structure remains as of v8, see STATUS)
# ---------------------------------------------------------------------------

def solve_typical_path(cell_probs_by_frame):
    """
    Per-frame maximum-likelihood cell selection -- the shortest-description-
    length "path" IS simply the highest-probability cell at every frame,
    independently. There is no cross-frame term (no motion-vector / jump
    cost): the theory's claim is that smooth, predictable trajectories
    emerge because they compress best and dominate the measure, which has
    to be demonstrated by the physics (the wavefunction's own spectral
    continuity, see module docstring) producing a smoothly-shifting
    probability landscape -- NOT by an auxiliary hand-coded penalty on
    jumps in the selection algorithm itself. (An earlier version had a
    --jump-cost parameter doing exactly that; removed -- see STATUS.
    Mathematically it was already inert at its default of 0.0: with no
    jump-cost term, the Viterbi DP over the branch graph collapses exactly
    to this independent per-frame argmax, since the transition cost
    contributes nothing and the accumulated path cost is just a running sum
    of per-frame code lengths.)

    Any real discontinuity that shows up in the resulting path -- the
    winning cell jumping between frames -- is now genuine information about
    the physics (the probability landscape itself moved abruptly, or
    several comparably-likely modes are competing) rather than something
    smoothed away by the algorithm.
    """
    n_frames = len(cell_probs_by_frame)
    eps = 1e-300

    path = [int(np.argmax(probs)) for probs in cell_probs_by_frame]
    path_probs = [float(cell_probs_by_frame[j][path[j]]) for j in range(n_frames)]
    total_cost = float(sum(-math.log2(max(p, eps)) for p in path_probs))
    return path, path_probs, total_cost


def path_jump_distances(path, cell_centers):
    """
    Pure diagnostic, computed AFTER path selection and fed into nothing --
    it cannot bias the result, only report on it. For each frame after the
    first, the Euclidean distance (in raster pixels) between this frame's
    winning cell and the previous frame's winning cell. First frame gets
    None (no previous frame to compare to). This is the quantitative
    version of "the observer does not teleport": if that claim is true, the
    distances restricted to the visible window should be small compared to
    the distance between two independently-random cells (see
    random_pair_expected_distance) -- not merely look small in a GIF.
    """
    dists = [None]
    for j in range(1, len(path)):
        cx, cy = cell_centers[path[j]]
        pcx, pcy = cell_centers[path[j - 1]]
        dists.append(math.hypot(cx - pcx, cy - pcy))
    return dists


def random_pair_expected_distance(cell_centers, n_samples=2000, rng=None):
    """Baseline for path_jump_distances: mean distance between two cells
    chosen independently at random, estimated by sampling (exact enumeration
    is O(n_cells^2), too slow once --observer is large). Used so a reported
    jump distance can be judged against "no better than chance" rather than
    presented as small in isolation."""
    rng = rng or np.random.default_rng(0)
    keys = np.array(list(cell_centers.keys()))
    a = rng.choice(keys, size=n_samples)
    b = rng.choice(keys, size=n_samples)
    dists = [math.hypot(cell_centers[int(x)][0] - cell_centers[int(y)][0],
                        cell_centers[int(x)][1] - cell_centers[int(y)][1])
             for x, y in zip(a, b)]
    return float(np.mean(dists))


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

    perf = p.add_argument_group("(3) performance / validation only -- no physical meaning")
    perf.add_argument("--output", type=str, default="observer_typical_path_v8")
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
        cell_probs_by_frame)
    jump_dists = path_jump_distances(path, cell_centers)

    n_cells = 2 ** n_a
    uniform_p = 1.0 / n_cells
    presence_floor = args.presence * uniform_p
    uniform_bits = math.log2(n_cells)

    results = []
    for f, c, p, jd in zip(frames, path, path_probs, jump_dists):
        entropy_ok = args.min_record_bits <= f["s_a"] < s_max
        gain = uniform_bits - (-math.log2(max(p, 1e-300)))
        visible = entropy_ok and (p >= presence_floor)
        results.append({**f, "cell": c, "p": p, "gain": gain, "visible": visible, "jump_dist": jd})

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
          f"scan_frames={args.scan_frames}, presence>={args.presence}x")
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
                vis_jumps = [r["jump_dist"] for r in vis if r["jump_dist"] is not None]
                if vis_jumps:
                    rec["mean_jump_visible"] = float(np.mean(vis_jumps))
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
        mj = np.array([r["mean_jump_visible"] for r in succ if "mean_jump_visible" in r])
        if len(mj):
            random_baseline = random_pair_expected_distance(cell_centers)
            print(f"  mean path jump distance in visible window: mean={mj.mean():.2f}px "
                  f"std={mj.std():.2f}px  (vs. {random_baseline:.2f}px expected under "
                  f"independent random cell choice) -- this is the quantitative version of "
                  f"'the observer does not teleport'")

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

    vis_jumps = [r["jump_dist"] for r in results if r["visible"] and r["jump_dist"] is not None]
    all_jumps = [r["jump_dist"] for r in results if r["jump_dist"] is not None]
    random_baseline = random_pair_expected_distance(cell_centers)
    if vis_jumps:
        print(f"path jump distance in visible window: mean={np.mean(vis_jumps):.2f}px "
              f"max={np.max(vis_jumps):.2f}px  (vs. {random_baseline:.2f}px expected under "
              f"independent random cell choice, side={side}px)")
    print(f"path jump distance over WHOLE sweep: mean={np.mean(all_jumps):.2f}px "
          f"max={np.max(all_jumps):.2f}px  (vs. {random_baseline:.2f}px random baseline) -- "
          f"expect this to be much larger than the visible-window figure, since most of the "
          f"sweep has no real record and the argmax cell is picking noise")

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
