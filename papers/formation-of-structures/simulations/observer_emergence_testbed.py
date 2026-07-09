#!/usr/bin/env python3
"""Observer-emergence testbed: does a spreading, decohering wavefunction produce a bump in blob-likeness vs entropy?

QUESTION THIS SCRIPT ANSWERS
-----------------------------
The bit-level scripts asked which weight-class family dominates a
relaxing bitstring. The abstract mode-occupation testbed asked which
occupation-profile shape dominates an abstract spectral ensemble. This
script asks the same family of question in the most physically literal
setting available: a genuine complex wavefunction psi(x,y) on a 2D grid,
evolving under two competing, physically motivated processes --

    1. free-particle dispersion (unitary; spreads a narrow wavepacket
       into a wider, still-perfectly-Gaussian one -- this GROWS a
       geometric object, G, out of an initially sub-resolution seed)
    2. environmental dephasing (unitary but stochastic; randomizes the
       relative phase of each Fourier mode -- this is a decoherence
       proxy that eventually destroys any clean geometric description)

-- and asks: as the position-space Shannon entropy of |psi|^2 grows from
(near) zero to (near) the grid's maximum, what fraction of the
probability density is still well-described as a single compact,
Gaussian-shaped "observer" (G), versus having become an incompressible,
speckled, noise-like pattern?

Concretely this tests the hypothesis: that observers (compact,
well-bounded geometric objects, here approximated as Gaussian blobs) are
NOT most abundant at either entropy extreme -- not at zero entropy
(nothing has enough room/resolution to exist as an extended object yet)
and not at maximum entropy (the wavefunction has degenerated into pure,
incompressible speckle, the higher-dimensional analogue of the
Porter-Thomas-typical flat state from the earlier abstract mode-mixing
testbed) -- but peaks somewhere in between, tracing the same bump/
lognormal-like universal shape found in the bit-level and mode-occupation
experiments, this time with an explicit, literal observer interpretation.

WHY NOT JUST DECAY A HAND-SEEDED PERFECT BLOB?
------------------------------------------------
A naive design -- seed a perfect Gaussian at t=0 and just let it decohere
-- would only ever show a MONOTONIC FALL in blob-quality, never a rise,
because the blob is already fully formed at the start by construction.
That would beg the question. The design here instead starts from a
near-delta (sub-resolution, arguably not yet a valid extended geometric
object at all) seed and lets the blob GROW into existence via ordinary
unitary dispersion -- a real, independently-derivable physical process,
with a known closed-form prediction for how its width should grow in the
absence of decoherence (see `analytic_width` below) -- while dephasing
independently and gradually destroys the clean Gaussian shape. The
resulting rise (as the object resolves and grows) then fall (as noise
overwhelms it) is not scripted in by hand; it is a genuine competition
between two rates in the same simulation, and had to be checked
numerically rather than assumed (an earlier back-of-envelope combinatorial
argument -- "a blob is always cheaper to describe than noise, so it
should just keep winning" -- turned out to conflate the cost of
DESCRIBING an outcome with the PROBABILITY of a decohering physical
process actually producing that outcome; those are not the same thing,
which is exactly why this needs to be simulated rather than argued from
a cost functional alone).

WHY NO IMPORTANCE-REWEIGHTING LAYER THIS TIME
------------------------------------------------
The earlier abstract mode-occupation testbed had to bolt on an
exp(-kappa * cost) reweighting of independently-generated candidate
paths, which turned out to be fragile (weight collapse / degenerate
effective sample size). Here, the simulated dynamics (free dispersion +
stochastic dephasing, both unitary/norm-preserving) already constitute a
genuine, correctly-normalized physical process. So this script just runs
many independent realizations directly and measures what fraction of
them are blob-like at each entropy checkpoint -- no separate reweighting
is needed or used.

WHAT "BLOB-LIKE" MEANS HERE (approximate, first cut)
------------------------------------------------------
At a given step, fit a single 2D Gaussian to |psi(x,y)|^2 by its weighted
mean and covariance, then compute what fraction of the total probability
mass lies within the Mahalanobis-distance-squared <= 2 ellipse of that
fit. For a PERFECT 2D Gaussian this fraction is EXACTLY 1 - exp(-1) =~
0.632 regardless of its size or elongation (a convenient scale-invariant
theoretical reference). A speckled/noise-like density will enclose much
LESS than this reference fraction within the same-shaped ellipse, because
its mass is not concentrated the way a true Gaussian's is. blob_quality
is defined as (enclosed fraction) / 0.632, capped at 1; a path is
classified "blob" when blob_quality exceeds a threshold. This is a crude,
approximate proxy -- like the shape classifier in the abstract testbed,
treat it as a first cut to inspect (via the saved example-density plots),
not a final, principled measure.

CAVEATS
-------
- Grid is periodic (FFT-based), so spreading that wraps around the grid
  is a simulation artifact, not physics. CONFIRMED empirically: on a
  200x200 grid, position-space entropy plateaus around
  S/S_max =~ 0.96 (never reaching 1.0) and the "peakiness" of the density
  (max/mean) stops monotonically decreasing and starts fluctuating once
  the fitted width approaches roughly half the grid size -- a signature
  of the wavepacket's tail wrapping around the torus and self-interfering
  (a revival-like effect), not genuine thermalization. This produces a
  partial, artifactual RECOVERY of blob_quality at high tau that should
  NOT be read as a second wave of observer re-emergence. Increasing the
  grid size pushes this artifact to later tau and weakens it (verified:
  going from 80x80 to 200x200 dropped the high-tau blob_quality recovery
  well below the "blob" classification threshold at decoherence_rate=0.1,
  where the smaller grid had recovered past it) but does not eliminate it
  -- doing so fully would require non-periodic (absorbing/open) boundary
  conditions instead, which is a natural next step but changes what
  "equilibrium" means (dispersing to infinity rather than filling a box).
  Until that is implemented, treat results above roughly tau=0.85-0.9 as
  suspect, and the dip itself (tau =~ 0.35-0.85 in the default
  configuration) as the trustworthy part of the curve.
- The dephasing channel is a simplified stochastic unravelling of
  decoherence (independent random per-mode phase kicks), not a full
  open-quantum-systems density-matrix treatment.
- As before, this is a testbed: mechanism parameters (dispersion rate,
  dephasing rate, grid size) are swappable via CLI, and the specific
  blob-quality metric is one reasonable choice among several that could
  be tried. An earlier version of blob_quality (enclosed probability mass
  within a Mahalanobis ellipse) turned out to be blind to decoherence
  entirely -- it only measures overall spread (a 2nd-moment quantity),
  and dephasing adds fine speckle without changing the overall spread
  much. The current version (pixel-level total-variation distance to the
  best-fit Gaussian) is sensitive to that speckle and is what actually
  produced the results below; if you swap in a different blob-quality
  metric, re-check that it is actually sensitive to the kind of
  structure-loss your chosen mechanism produces before trusting its curve.
"""
import argparse
from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

EPS = 1e-12
MAHALANOBIS_R2 = 2.0
THEORETICAL_ENCLOSED_FRAC = 1.0 - np.exp(-MAHALANOBIS_R2 / 2.0)  # = 0.6321... for r2=2


# ---------------------------------------------------------------------------
# Wavefunction evolution
# ---------------------------------------------------------------------------

def init_wavefunction(H: int, W: int, sigma0: float) -> ComplexArray:
    """Initialize a narrow, near-delta Gaussian wavepacket at the grid center.

    Args:
        H: Grid height (pixels).
        W: Grid width (pixels).
        sigma0: Initial position-space standard deviation (pixels); kept
            small so the seed is sub-resolution / not yet a well-formed
            extended object.

    Returns:
        Complex amplitude grid of shape (H, W), normalized so
        sum(|psi|^2) = 1.
    """
    y, x = np.mgrid[0:H, 0:W]
    cy, cx = H / 2.0, W / 2.0
    psi = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (4.0 * sigma0 ** 2)).astype(np.complex128)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
    return psi


def _k_squared_grid(H: int, W: int) -> FloatArray:
    """Squared angular-wavenumber grid for the free-particle propagator.

    Args:
        H: Grid height.
        W: Grid width.

    Returns:
        Array of shape (H, W): kx^2 + ky^2 at each FFT frequency.
    """
    ky = 2 * np.pi * np.fft.fftfreq(H).reshape(H, 1)
    kx = 2 * np.pi * np.fft.fftfreq(W).reshape(1, W)
    return kx ** 2 + ky ** 2


def evolve_wavefunction_steps(
    H: int, W: int, num_steps: int, dt: float,
    decoherence_rate: float, sigma0: float, rng: np.random.Generator,
) -> Generator[ComplexArray, None, None]:
    """Evolve a wavepacket under free dispersion + stochastic dephasing.

    At each step: apply the exact free-particle propagator phase
    exp(-i*(kx^2+ky^2)*dt/2) in Fourier space (unitary; grows the
    wavepacket), then multiply each Fourier mode by an independent random
    phase exp(i*N(0, decoherence_rate^2 * dt)) (also unitary; a simplified
    dephasing/decoherence proxy).

    Args:
        H: Grid height.
        W: Grid width.
        num_steps: Number of steps to yield after the initial seed.
        dt: Time step size (natural units, hbar = m = 1).
        decoherence_rate: Standard deviation scale of the per-step random
            phase kick applied to each Fourier mode.
        sigma0: Initial position-space width of the seed wavepacket.
        rng: NumPy random Generator.

    Yields:
        psi_0 (the seed), psi_1, ..., psi_num_steps: complex amplitude
        grids of shape (H, W), each normalized.
    """
    psi = init_wavefunction(H, W, sigma0)
    yield psi.copy()
    k_sq = _k_squared_grid(H, W)
    dispersion_phase = np.exp(-1j * k_sq * dt / 2.0)
    for _ in range(num_steps):
        psi_k = np.fft.fft2(psi)
        psi_k *= dispersion_phase
        if decoherence_rate > 0:
            random_phase = rng.normal(0.0, decoherence_rate * np.sqrt(dt), size=psi_k.shape)
            psi_k *= np.exp(1j * random_phase)
        psi = np.fft.ifft2(psi_k)
        norm = np.sqrt(np.sum(np.abs(psi) ** 2))
        psi /= norm
        yield psi.copy()


def analytic_width(sigma0: float, t: float) -> float:
    """Closed-form free-particle Gaussian wavepacket width at time t.

    sigma(t) = sigma0 * sqrt(1 + (t / (2 * sigma0^2))^2), the standard
    result for a free (hbar = m = 1) Gaussian wavepacket's position-space
    standard deviation, used as a decoherence_rate=0 sanity check against
    the simulated width from fit_gaussian_2d.

    Args:
        sigma0: Initial width.
        t: Elapsed time (num_steps * dt).

    Returns:
        Predicted position-space standard deviation at time t.
    """
    return sigma0 * np.sqrt(1.0 + (t / (2.0 * sigma0 ** 2)) ** 2)


# ---------------------------------------------------------------------------
# Measurements: entropy, Gaussian fit, blob-quality
# ---------------------------------------------------------------------------

def shannon_entropy_2d(prob2d: FloatArray) -> float:
    """Shannon entropy (nats) of a 2D probability density.

    Args:
        prob2d: Probability array (any shape, sums to ~1).

    Returns:
        -sum(p * log(p)) over all cells, p clamped away from 0.
    """
    p = np.clip(prob2d, EPS, None)
    return float(-np.sum(p * np.log(p)))


@dataclass
class GaussianFit:
    """Weighted-moment Gaussian fit to a 2D probability density."""
    mean: FloatArray   # shape (2,): (x, y)
    cov: FloatArray    # shape (2, 2)


def fit_gaussian_2d(prob2d: FloatArray) -> GaussianFit:
    """Fit a single 2D Gaussian to a probability density via weighted moments.

    Args:
        prob2d: Probability array of shape (H, W), sums to ~1.

    Returns:
        GaussianFit with the weighted mean (x, y) and 2x2 covariance
        matrix of the density.
    """
    H, W = prob2d.shape
    y, x = np.mgrid[0:H, 0:W]
    p = prob2d
    mx = np.sum(x * p)
    my = np.sum(y * p)
    dx = x - mx
    dy = y - my
    cxx = np.sum(dx * dx * p)
    cyy = np.sum(dy * dy * p)
    cxy = np.sum(dx * dy * p)
    cov = np.array([[cxx, cxy], [cxy, cyy]])
    return GaussianFit(mean=np.array([mx, my]), cov=cov)


def blob_quality(prob2d: FloatArray, fit: GaussianFit) -> float:
    """Pixel-level closeness of the actual density to its best-fit Gaussian.

    An earlier version of this metric checked only how much probability
    mass fell inside a Mahalanobis-distance ellipse -- a second-moment
    (overall-spread) check. That turned out to be blind to decoherence:
    random per-mode dephasing adds fine-grained speckle ON TOP of an
    envelope whose overall spread barely changes, so the ellipse-mass
    metric stayed pinned near 1.0 (then plateaued around 0.83) no matter
    how strong the decoherence was -- it was measuring the wrong thing.

    This version instead computes the total-variation distance between
    the ACTUAL discrete density and the density predicted by a smooth 2D
    Gaussian PDF with the same fitted mean/covariance, evaluated
    pixel-by-pixel. Fine speckle shows up directly as per-pixel
    mismatch, which a moment-based check cannot see.

    Args:
        prob2d: Probability array of shape (H, W), sums to ~1.
        fit: GaussianFit from fit_gaussian_2d for the same density.

    Returns:
        blob_quality = 1 - (total variation distance), in [0, 1]. 1.0
        means the density looks exactly like a smooth Gaussian at pixel
        resolution; values near 0 mean the density is dominated by
        structure the smooth Gaussian fit does not predict (speckle).
    """
    H, W = prob2d.shape
    y, x = np.mgrid[0:H, 0:W]
    cov = fit.cov + np.eye(2) * EPS
    try:
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
    except np.linalg.LinAlgError:
        return 0.0
    if det_cov <= 0:
        return 0.0
    dx = x - fit.mean[0]
    dy = y - fit.mean[1]
    maha_sq = (
        inv_cov[0, 0] * dx * dx
        + 2 * inv_cov[0, 1] * dx * dy
        + inv_cov[1, 1] * dy * dy
    )
    gaussian_pdf = np.exp(-0.5 * maha_sq) / (2 * np.pi * np.sqrt(det_cov))
    gaussian_prob = gaussian_pdf / np.sum(gaussian_pdf)  # discretize/normalize over the grid
    tv_distance = 0.5 * float(np.sum(np.abs(prob2d - gaussian_prob)))
    return float(max(0.0, 1.0 - tv_distance))


# ---------------------------------------------------------------------------
# Monte Carlo ensemble + entropy-checkpoint alignment
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """One realization's recorded entropy and blob-quality history."""
    entropy_frac: FloatArray
    blob_quality: FloatArray
    example_density: Optional[FloatArray] = None


def _first_crossing_index(entropy_frac: FloatArray, tau: float) -> Optional[int]:
    """Index of the first step at which normalized entropy reaches tau.

    Args:
        entropy_frac: Array of entropy values normalized to [0, 1].
        tau: Target normalized entropy level.

    Returns:
        First index i with entropy_frac[i] >= tau, or None if never reached.
    """
    hits = np.flatnonzero(entropy_frac >= tau)
    return int(hits[0]) if hits.size > 0 else None


def run_ensemble(
    H: int, W: int, num_steps: int, dt: float, decoherence_rate: float,
    sigma0: float, num_realizations: int, seed: int = 0,
) -> List[Trajectory]:
    """Simulate many independent realizations and record their trajectories.

    Args:
        H: Grid height.
        W: Grid width.
        num_steps: Steps per realization.
        dt: Time step size.
        decoherence_rate: Dephasing strength.
        sigma0: Initial wavepacket width.
        num_realizations: Number of independent Monte Carlo realizations.
        seed: RNG seed.

    Returns:
        List of Trajectory objects, one per realization.
    """
    rng = np.random.default_rng(seed)
    h_max = np.log(H * W)
    trajectories: List[Trajectory] = []
    for _ in range(num_realizations):
        entropy_list: List[float] = []
        quality_list: List[float] = []
        for psi in evolve_wavefunction_steps(H, W, num_steps, dt, decoherence_rate, sigma0, rng):
            prob = np.abs(psi) ** 2
            entropy_list.append(shannon_entropy_2d(prob) / h_max)
            fit = fit_gaussian_2d(prob)
            quality_list.append(blob_quality(prob, fit))
        trajectories.append(Trajectory(
            entropy_frac=np.array(entropy_list),
            blob_quality=np.array(quality_list),
        ))
    return trajectories


def aggregate_by_entropy_checkpoint(
    trajectories: List[Trajectory], tau_grid: FloatArray, blob_thresh: float = 0.75,
) -> Tuple[FloatArray, FloatArray]:
    """Aggregate mean blob-quality and blob-classified fraction at each tau.

    Args:
        trajectories: Output of run_ensemble.
        tau_grid: Normalized-entropy checkpoints in (0, 1].
        blob_thresh: blob_quality threshold above which a realization is
            classified "blob" at that checkpoint.

    Returns:
        Tuple (mean_quality, blob_fraction), each an array matching
        tau_grid, with NaN at checkpoints no realization reached.
    """
    mean_quality = np.full_like(tau_grid, np.nan)
    blob_fraction = np.full_like(tau_grid, np.nan)
    for t_idx, tau in enumerate(tau_grid):
        vals = []
        for traj in trajectories:
            idx = _first_crossing_index(traj.entropy_frac, tau)
            if idx is not None:
                vals.append(traj.blob_quality[idx])
        if vals:
            vals_arr = np.array(vals)
            mean_quality[t_idx] = vals_arr.mean()
            blob_fraction[t_idx] = float(np.mean(vals_arr > blob_thresh))
    return mean_quality, blob_fraction


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_observer_abundance(
    tau_grid: FloatArray, mean_quality: FloatArray, blob_fraction: FloatArray,
    decoherence_rate: float, outfile: str,
) -> None:
    """Plot mean blob-quality and blob-classified fraction vs normalized entropy.

    Args:
        tau_grid: Normalized-entropy checkpoints.
        mean_quality: Mean blob_quality at each checkpoint (from
            aggregate_by_entropy_checkpoint).
        blob_fraction: Fraction of realizations classified "blob" at each
            checkpoint.
        decoherence_rate: Value used, for the plot title.
        outfile: Path to save the PNG.

    Returns:
        None. Writes a PNG to outfile.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(tau_grid, mean_quality, color="darkorange", lw=2, label="mean blob-quality")
    ax.plot(tau_grid, blob_fraction, color="steelblue", lw=2, ls="--",
            label=f"fraction classified 'blob' (>{0.75:.2f})")
    ax.set_xlabel(r"$S(\tau)/S_{max}$  (normalized position-space entropy)")
    ax.set_ylabel("value")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Observer (blob) abundance vs. entropy\n"
        f"free dispersion + dephasing (decoherence_rate={decoherence_rate})"
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved -> {outfile}")


def plot_example_densities(
    H: int, W: int, num_steps: int, dt: float, decoherence_rate: float,
    sigma0: float, seed: int, outfile: str,
) -> None:
    """Plot |psi|^2 snapshots at low/medium/high entropy for one example realization.

    Args:
        H, W: Grid dimensions.
        num_steps: Steps to run.
        dt: Time step size.
        decoherence_rate: Dephasing strength.
        sigma0: Initial width.
        seed: RNG seed for this example realization.
        outfile: Path to save the PNG.

    Returns:
        None. Writes a PNG to outfile.
    """
    rng = np.random.default_rng(seed)
    h_max = np.log(H * W)
    snapshots: List[Tuple[float, FloatArray]] = []
    for psi in evolve_wavefunction_steps(H, W, num_steps, dt, decoherence_rate, sigma0, rng):
        prob = np.abs(psi) ** 2
        snapshots.append((shannon_entropy_2d(prob) / h_max, prob))

    targets = [0.1, 0.4, 0.7, 0.95]
    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 4.2))
    for ax, target in zip(axes, targets):
        idx = _first_crossing_index(np.array([s for s, _ in snapshots]), target)
        if idx is None:
            idx = len(snapshots) - 1
        tau_actual, prob = snapshots[idx]
        ax.imshow(prob, cmap="magma")
        ax.set_title(f"tau≈{tau_actual:.2f}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Example density snapshots (decoherence_rate={decoherence_rate})")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved -> {outfile}")


def check_analytic_width(H: int, W: int, sigma0: float, dt: float, num_steps: int) -> None:
    """Print a decoherence_rate=0 sanity check against the closed-form width formula.

    Args:
        H, W: Grid dimensions.
        sigma0: Initial width.
        dt: Time step size.
        num_steps: Steps to run.

    Returns:
        None. Prints a comparison table to stdout.
    """
    rng = np.random.default_rng(0)
    print("Sanity check: simulated width vs. closed-form free-particle spreading (decoherence_rate=0)")
    print(f"{'step':>6} {'t':>8} {'sim width':>12} {'analytic width':>16}")
    for i, psi in enumerate(evolve_wavefunction_steps(H, W, num_steps, dt, 0.0, sigma0, rng)):
        if i % max(1, num_steps // 8) != 0:
            continue
        prob = np.abs(psi) ** 2
        fit = fit_gaussian_2d(prob)
        sim_width = float(np.sqrt(np.trace(fit.cov) / 2.0))
        t = i * dt
        print(f"{i:6d} {t:8.2f} {sim_width:12.3f} {analytic_width(sigma0, t):16.3f}")


def main() -> None:
    """Parse CLI args, run the ensemble, and save all diagnostic plots."""
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--H", type=int, default=200)
    ap.add_argument("--W", type=int, default=200)
    ap.add_argument("--sigma0", type=float, default=1.5)
    ap.add_argument("--dt", type=float, default=0.5)
    ap.add_argument("--num-steps", type=int, default=500, dest="num_steps")
    ap.add_argument("--num-realizations", type=int, default=50, dest="num_realizations")
    ap.add_argument("--decoherence-rate", type=float, default=0.1, dest="decoherence_rate")
    ap.add_argument("--num-checkpoints", type=int, default=20, dest="num_checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="observer_abundance.png")
    ap.add_argument("--examples-out", type=str, default="observer_example_densities.png",
                     dest="examples_out")
    ap.add_argument("--skip-sanity-check", action="store_true", dest="skip_sanity_check")
    args = ap.parse_args()

    if not args.skip_sanity_check:
        check_analytic_width(args.H, args.W, args.sigma0, args.dt, args.num_steps)
        print()

    trajectories = run_ensemble(
        args.H, args.W, args.num_steps, args.dt, args.decoherence_rate,
        args.sigma0, args.num_realizations, seed=args.seed,
    )
    tau_grid = np.linspace(0.02, 0.98, args.num_checkpoints)
    mean_quality, blob_fraction = aggregate_by_entropy_checkpoint(trajectories, tau_grid)
    plot_observer_abundance(tau_grid, mean_quality, blob_fraction, args.decoherence_rate, args.out)
    print()
    plot_example_densities(
        args.H, args.W, args.num_steps, args.dt, args.decoherence_rate,
        args.sigma0, args.seed + 1000, args.examples_out,
    )


if __name__ == "__main__":
    main()
