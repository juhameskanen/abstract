"""
Correlated SPACETIME direction field: extends spatial_field.py's approach
from 1D (position only) to 2D (position x time-frame), using BOTH derived
correlations -- pair_correlation.g(r) (space) and the new joint_kernel.py /
persistence.py (space-time jointly) -- as the covariance-shaping kernel.

WHY A FIELD, NOT A PER-PARTICLE WALK: an earlier attempt tried to turn
Joint(r, Delta) into a single-particle transition probability (draw a new
position for "the same" knot each frame, weighted by Joint). That failed
conceptually, not numerically: conditional "probability window at i+r also
matches" values of ~0.2-0.35 hold SIMULTANEOUSLY across dozens of nearby
positions, summing far past 1 -- because the model has no single-particle
exclusivity. It describes a correlated DENSITY FIELD (many positions can
be simultaneously likely), not one tracked object's motion.

The fix: extend spatial_field.py's trick to 2D. Instead of a direction
that's a smooth function of position s alone, build one that's a smooth
function of (s, frame) jointly -- nearby positions AND nearby frames get
correlated directions, matching the derived g(r) and Joint(r,Delta) shapes.
Points are still sampled independently each frame (same Poisson-by-count
process as before, no change there) -- what's new is that a point's
direction now depends smoothly on WHEN it was sampled too, not just where.
Two points close in (s, frame) space get similar directions, which is what
produces the visual appearance of continuity -- with no particle identity,
transition kernel, or exclusivity assumption anywhere.

Same explicit caveat as spatial_field.py: turning a probability-ratio into
a Gaussian-field covariance is an approximate translation, not a unique or
exact one. Validated below by checking the empirical field correlation has
the right SHAPE in both dimensions.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import hypergeom

from persistence import persistence
from joint_kernel import joint_kernel_batch


def build_spacetime_field(n: int, w: int, k: float, a: int, n_frames: int,
                           ticks_per_frame: float, seed: int,
                           r_max: int | None = None, dt_max: int = 3) -> np.ndarray:
    """Return a (n, n_frames, 3) array: field[s, f] is the unit direction
    assigned to chain position s at animation frame f, correlated across
    both nearby s (via g(r), support r<w) and nearby f (via Joint/Persistence,
    truncated at dt_max frames -- both correlations plateau quickly, so a
    small temporal support is a reasonable, explicitly-flagged truncation).
    """
    if r_max is None:
        r_max = w - 1
    k_int = int(round(k))
    marginal = hypergeom.pmf(a, n, k_int, w)

    # spatial-only (dt=0) values are covered by the batched loop below too
    # (validated earlier: joint_kernel at delta=0 exactly matches
    # pair_correlation.pair_correlation's g(r)) -- no separate computation needed.

    # temporal excess-correlation weights at ds=0 (persistence.py; the "same
    # position, later time" axis, which joint_kernel.py doesn't cover since
    # it requires two DISTINCT windows).
    dt_ticks = np.arange(0, dt_max + 1) * ticks_per_frame
    pers_dt = np.array([persistence(n, w, a, int(round(d))) for d in dt_ticks])
    w_time_r0 = np.sqrt(np.clip(pers_dt / marginal - 1.0, 0.0, None))  # r=0 axis

    # full 2D excess-correlation kernel over (ds in [-r_max,r_max], dt in [0,dt_max])
    # (dt<0 mirrors dt>0 by time-symmetry of the stationary process)
    kernel = np.zeros((2 * r_max + 1, dt_max + 1))
    dt_ticks_int = [int(round(dt * ticks_per_frame)) for dt in range(dt_max + 1)]
    for ds in range(1, r_max + 1):
        # batched over all dt at once: one eigendecomposition per r instead of
        # dt_max+1 separate matrix powers (the earlier performance bottleneck)
        jr_batch = joint_kernel_batch(n, w, ds, dt_ticks_int, a, k_int)
        for dtf, dt_tick in enumerate(dt_ticks_int):
            val = np.sqrt(np.clip(jr_batch[dt_tick] / marginal**2 - 1.0, 0.0, None))
            kernel[r_max - ds, dtf] = val
            kernel[r_max + ds, dtf] = val
    for dtf in range(dt_max + 1):
        kernel[r_max, dtf] = w_time_r0[dtf] if dtf > 0 else 0.0  # ds=0 axis (self term at dt=0 excluded)

    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=(n, n_frames, 3))
    field = innovations.copy()

    for di, ds in enumerate(range(-r_max, r_max + 1)):
        for dtf, dt in enumerate(range(0, dt_max + 1)):
            wgt = kernel[di, dtf]
            if wgt <= 0:
                continue
            shifted = np.roll(innovations, shift=(-ds, -dt), axis=(0, 1))
            field += wgt * shifted
            if dt > 0:  # also add the mirrored (+dt) contribution, since we only looped dt>=0
                shifted_back = np.roll(innovations, shift=(-ds, dt), axis=(0, 1))
                field += wgt * shifted_back

    norms = np.linalg.norm(field, axis=2, keepdims=True)
    norms[norms == 0] = 1.0
    return field / norms


def empirical_correlation(field: np.ndarray, ds_values, dt_values) -> np.ndarray:
    n, n_frames, _ = field.shape
    out = np.zeros((len(ds_values), len(dt_values)))
    for i, ds in enumerate(ds_values):
        for j, dt in enumerate(dt_values):
            shifted = np.roll(field, shift=(-ds, -dt), axis=(0, 1))
            out[i, j] = np.mean(np.sum(field * shifted, axis=2))
    return out


if __name__ == "__main__":
    n, w, a, k = 184, 6, 2, 90
    n_frames, ticks_per_frame = 40, 40.0
    field = build_spacetime_field(n, w, k, a, n_frames, ticks_per_frame, seed=11)

    ds_values = [0, 1, 2, 5, 8]
    dt_values = [0, 1, 2, 5]
    emp = empirical_correlation(field, ds_values, dt_values)
    print(f"{'ds\\dt':>6}" + "".join(f"{dt:>8}" for dt in dt_values))
    for i, ds in enumerate(ds_values):
        print(f"{ds:>6}" + "".join(f"{emp[i,j]:>8.3f}" for j in range(len(dt_values))))
