"""
Shared 3D-particle-density renderer, usable by all three cosmology backends.

WHAT THIS IS AND IS NOT
------------------------
The simulation itself is metric-agnostic: it only defines a relaxation
across an internal information ledger (bit-flip coordinate time, a
per-level lapse/retarded clock, and size_measure as a resolution
scalar). It says nothing about the number of spatial dimensions or the
topology of any metric.

Embedding this relaxation into 3D space with spherical symmetry -- as
this module does for visualization -- is a rendering choice, not a
consequence of the model. Only the following are derived quantities
pulled from the run:
  - size_measure(tau)     -- the FRW-like background radius R(tau).
                              Radius IS coordinate time here: R(tau) is
                              monotonic, so every sampled point's radius
                              is simply a record of when it was sampled.
  - lvl.matter_series      -- per-scale structure/matter count at each
                              tau, used ONLY as the (relative) sampling
                              rate for how many points to draw at that
                              tau -- i.e. a population-density readout,
                              not a particle count with tracked identity.

Everything else here is an arbitrary placement choice made only for
the picture: the isotropic random direction assigned to each sampled
point. Marker size (by level lapse) is likewise cosmetic.

WHAT CHANGED (and why)
-----------------------
An earlier version of this module assigned each comoving "slot" a
fixed direction and let it stay active/inactive across many frames,
then optionally drew a trail connecting its positions over time. That
implied persistent particle identity and literal motion. But the
underlying model (`build_worldlines` / matter_series) only ever
defines a population COUNT at each tau via a hypergeometric-style
structure-count statistic -- there is no equation of motion and no
tracked identity anywhere in multiclock.py. Drawing continuous tracks
therefore asserted something the model does not compute.

This version instead draws, independently at every tau in the run, a
fresh stochastic (Poisson) number of points whose count is driven by
that level's matter_series(tau), and permanently deposits them at
radius R(tau) -- then never revisits or moves them again. The result
is a single accumulated point cloud: a density map of "how much of
this scale's structure existed near this moment in cosmic history,"
honestly rendered as population density rather than as tracked
individual worldlines.

DIRECTION IS NO LONGER PURE NOISE, BUT IS STILL AN APPROXIMATION
-------------------------------------------------------------------
Each sampled point is also assigned a chain position s (uniform over
[0, n_bits) -- positions are exchangeable in the underlying model, so
this part is exact, not a placeholder). Its 3D direction is then a
LOOKUP into a spatially-correlated direction field built once per
level from pair_correlation.py's derived g(r): nearby chain positions
get nearby directions, positions >= w apart get independent ones,
matching the exact short-range/plateau shape derived analytically.

This is still a real approximation, flagged honestly: g(r) is a ratio
of match probabilities, not a linear field covariance, so turning it
into a Gaussian-field smoothing kernel (spatial_field.py) is a
reasonable but not unique translation -- validated to have the right
SHAPE (strong correlation at small separations, falling off), but it
does not reproduce g(r)'s exact hard cutoff at r=w (the smoothing
leaves a longer decaying tail instead of the derived model's sharp
plateau). The field is also built once per level using a single
representative k (that level's own peak-matter time), not the full
time-varying k(tau) -- another explicit simplification. Time
persistence (derived in persistence.py) is NOT yet wired in here: a
knot's position s is still drawn independently at each sampling time,
with no correlation across nearby t yet.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from multiclock import ground_truth_pool
from dicke_layer import default_composition
from spatial_field import build_direction_field

CAVEAT = (
    "ILLUSTRATIVE: 3D embedding + spherical symmetry assumed for rendering only "
    "-- not derived from or asserted by the model. Points are a stochastic "
    "population-density sample (radius = coordinate time via size_measure, "
    "direction is arbitrary/isotropic); they are NOT tracked particles or "
    "worldlines -- there is no persistent identity or equation of motion here."
)


def _sphere_wireframe(r: float, n: int = 16):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _sample_level_cloud(matter_series: np.ndarray, R: np.ndarray, rate_per_step: float,
                         direction_field: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stochastically sample points for one level, accumulated over the full run.

    At every tau index t, draw a Poisson count with mean
    `rate_per_step * frac(t)` (frac = matter_series normalized to its own
    max). Each point gets a chain position s drawn uniformly over
    [0, n_bits) (positions are exchangeable in the model -- this part is
    exact) and its direction is a LOOKUP into `direction_field[s]` (the
    spatially-correlated field from spatial_field.build_direction_field),
    not a fresh independent draw. It's placed at radius R[t] permanently.
    No point, once placed, is ever moved or re-sampled.

    Returns (points[N,3], creation_idx[N]) where creation_idx records the
    tau-index each point belongs to, so callers can reveal the cloud
    cumulatively (up to a given tau) for the snapshot grid / animation.
    """
    rng = np.random.default_rng(seed)
    denom = max(float(np.max(matter_series)), 1e-12)
    frac = np.clip(matter_series / denom, 0.0, None)
    counts = rng.poisson(frac * rate_per_step)
    total = int(counts.sum())
    if total == 0:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int)
    creation_idx = np.repeat(np.arange(len(matter_series)), counts)
    n_bits = direction_field.shape[0]
    chain_positions = rng.integers(0, n_bits, size=total)
    vecs = direction_field[chain_positions]
    points = vecs * R[creation_idx, None]
    return points, creation_idx


def _build_level_clouds(levels, t_bf, R, rate_per_step: float, n_bits: int, k_rate: float):
    cmap = plt.get_cmap("plasma")
    colors = [
        cmap(0.15 + 0.7 * i / max(len(levels) - 1, 1)) for i in range(len(levels))
    ]
    level_clouds = []
    for lvl, color in zip(levels, colors):
        w = int(round(lvl.width))
        # Representative k: this level's own peak-matter time (explicit
        # simplification -- the field is built once, not re-fit per tau).
        peak_idx = int(np.argmax(lvl.matter_series))
        _, k_local = ground_truth_pool(lvl.tau_local[peak_idx:peak_idx + 1], n_bits, k_rate)
        k_ref = float(k_local[0])
        a, _b = default_composition(w)
        field = build_direction_field(n_bits, w, k_ref, a, seed=w + 97)

        points, creation_idx = _sample_level_cloud(
            lvl.matter_series, R, rate_per_step, field, seed=w
        )
        level_clouds.append((points, creation_idx, color, lvl))
    return level_clouds


def render_3d_particles(
    t_bf: np.ndarray,
    size_measure: np.ndarray,
    levels: list,
    output: str,
    n_bits: int,
    k_rate: float,
    animate: bool = False,
    n_particles: int = 3,
    frames: int = 150,
    fps: int = 20,
    title: str = "Particle population density (illustrative 3D + spherical symmetry)",
) -> None:
    """Render each level's population density as an accumulated 3D point cloud.

    `levels` is a list of anim_common.LevelAnimSpec (or any object with
    .width, .matter_series, .lapse) -- the same backend-agnostic
    description already used by --anim.

    `n_particles` is now the max EXPECTED points sampled per timestep, per
    scale (Poisson rate), not a fixed particle-per-scale count -- total
    points over the whole run scale with roughly steps * n_particles * (mean
    normalized matter fraction).

    animate=False -> static multi-snapshot grid, each panel showing the
                      cloud accumulated up to that point in cosmic history.
    animate=True  -> rotating GIF/MP4 that grows the cloud over the run,
                      background sphere expanding alongside it.
    """
    R = np.asarray(size_measure, dtype=float)
    lim = float(np.max(np.abs(R))) * 1.05 if np.max(np.abs(R)) > 0 else 1.0
    level_clouds = _build_level_clouds(levels, t_bf, R, rate_per_step=float(n_particles),
                                        n_bits=int(round(n_bits)), k_rate=k_rate)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if not animate:
        _render_snapshots(t_bf, R, level_clouds, lim, output, title)
    else:
        _render_animation(t_bf, R, level_clouds, lim, output, frames, fps, title)


def _render_snapshots(t_bf, R, level_clouds, lim, output, title):
    n = len(t_bf)
    fracs = [0.02, 0.05, 0.12, 0.30, 0.6, 1.0]
    idxs = [int(f * (n - 1)) for f in fracs]

    fig = plt.figure(figsize=(15, 9))
    for panel, idx in enumerate(idxs):
        ax = fig.add_subplot(2, 3, panel + 1, projection="3d")
        x, y, z = _sphere_wireframe(R[idx], n=18)
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, lw=0.4)
        for points, creation_idx, color, lvl in level_clouds:
            mask = creation_idx <= idx
            if not np.any(mask):
                continue
            marker_size = 6 + 30 * lvl.lapse
            ax.scatter(
                points[mask, 0], points[mask, 1], points[mask, 2],
                color=color, s=marker_size, alpha=0.5, depthshade=True,
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_title(f"t={t_bf[idx]:.0f}  (R={R[idx]:.3f})", fontsize=9)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()

    fig.suptitle(f"{title}\n{CAVEAT}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output, dpi=130)
    plt.close(fig)
    print(f"saved -> {output}")


def _render_animation(t_bf, R, level_clouds, lim, output, frames, fps, title):
    n_frames_total = len(t_bf)
    stride = max(1, n_frames_total // frames)
    frame_indices = np.arange(0, n_frames_total, stride)
    if frame_indices[-1] != n_frames_total - 1:
        frame_indices = np.append(frame_indices, n_frames_total - 1)

    fig = plt.figure(figsize=(9, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markersize=8, label=f"w={lvl.width:g}  (lapse=1/{lvl.width:g})")
        for _, _, color, lvl in level_clouds
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              facecolor="#111115", edgecolor="gray", labelcolor="white")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    time_text = fig.text(0.5, 0.03, "", ha="center", fontsize=10, fontweight="bold")
    caption = fig.text(0.5, 0.005, CAVEAT, ha="center", fontsize=6.5, wrap=True)

    def update(frame_idx):
        idx = frame_indices[frame_idx]
        ax.cla()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
                  facecolor="#111115", edgecolor="gray", labelcolor="white")

        x, y, z = _sphere_wireframe(R[idx], n=16)
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.2, lw=0.3)
        for points, creation_idx, color, lvl in level_clouds:
            mask = creation_idx <= idx
            if np.any(mask):
                marker_size = 6 + 30 * lvl.lapse
                ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2],
                           color=color, s=marker_size, alpha=0.5)
        ax.view_init(elev=20, azim=frame_idx * 1.2)
        time_text.set_text(f"coordinate time \u03c4 = {t_bf[idx]:8.1f}")
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(frame_indices), blit=False)
    writer = animation.PillowWriter(fps=fps)
    ani.save(output, writer=writer, dpi=100)
    plt.close(fig)
    print(f"saved -> {output}  (frames={len(frame_indices)}, stride={stride}, fps={fps})")


# ===========================================================================
# Shared CLI wiring, so each backend script (cosmic_wavefunction.py,
# cosmic_d.py, cosmic_psi.py) adds --3d with one line instead of
# duplicating argparse/dispatch code.
# ===========================================================================

def add_3d_cli_args(parser) -> None:
    """Add --3d and --n_particles to an argparse parser.

    Assumes --anim, --frames, --fps, and --output are already added by the
    caller (all three backend scripts already have these).
    """
    parser.add_argument("--3d", dest="three_d", action="store_true",
                         help="Add a 3D view: each level's population density "
                              "rendered as a stochastically-sampled point cloud "
                              "in an illustrative 3D + spherically symmetric "
                              "embedding (radius = coordinate time; NOT tracked "
                              "particles/worldlines -- flagged in the output). "
                              "Alone this produces a static multi-snapshot grid; "
                              "combined with --anim it produces a rotating "
                              "GIF/MP4 that grows the cloud over the run instead.")
    parser.add_argument("--n_particles", type=int, default=3,
                         help="Max expected points sampled per timestep, per "
                              "scale, in the 3D view (Poisson rate; --3d only). "
                              "Total points over the whole run scale with "
                              "roughly steps * n_particles * mean matter "
                              "fraction.")


def dispatch_3d(args, t_bf: np.ndarray, size_measure: np.ndarray, levels: list,
                 title: str, n_bits: int, k_rate: float) -> None:
    """Call render_3d_particles from a backend's main(), reading the shared
    --3d/--anim/--frames/--fps/--n_particles args. No-op if --3d wasn't passed.

    n_bits/k_rate should be the SAME sim.n_bits/sim.k_rate the backend's own
    run_simulation() call produced, so the correlated direction field uses
    the exact same k(tau) the physics pipeline uses -- no separate/drifting
    copy of the relaxation parameters.
    """
    if not getattr(args, "three_d", False):
        return
    out_path = Path(args.output)
    suffix = ".gif" if args.anim else ".png"
    threed_output = str(out_path.with_name(f"{out_path.stem}_3d{suffix}"))
    render_3d_particles(
        t_bf, size_measure, levels, threed_output,
        n_bits=n_bits, k_rate=k_rate,
        animate=args.anim, n_particles=args.n_particles,
        frames=args.frames, fps=args.fps,
        title=title,
    )
