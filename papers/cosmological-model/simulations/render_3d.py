"""
Shared 3D-particle renderer, usable by all three cosmology backends.

WHAT THIS IS AND IS NOT
------------------------
The simulation itself is metric-agnostic: it only defines a relaxation
across an internal information ledger (bit-flip coordinate time, a
per-level lapse/retarded clock, and size_measure as a resolution
scalar). It says nothing about the number of spatial dimensions or the
topology of any metric.

Embedding worldlines into 3D space with spherical symmetry -- as this
module does for visualization -- is a rendering choice, not a
consequence of the model. This generalizes (and replaces) the original
one-off snapshot script, which already flagged this honestly. Only the
following are derived quantities pulled from the run:
  - size_measure(tau)              -- the FRW-like background radius
  - lvl.tau_local, lvl.lapse       -- each level's own retarded clock
  - lvl.matter_series              -- drives which comoving "slots" are
                                       active at a given tau (via the
                                       same build_worldlines() used by
                                       the existing 2D worldline panel
                                       and by --anim)

Everything else here is an arbitrary placement choice made only for
the picture:
  - the 3D unit direction assigned to each comoving slot
  - the comoving radius fraction assigned to each slot
  - marker size as a function of level lapse

None of this is derived from or asserted as part of the theory. It is
flagged in every figure this module produces.
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

from multiclock import build_worldlines

CAVEAT = (
    "ILLUSTRATIVE: 3D embedding + spherical symmetry assumed for rendering only "
    "-- not derived from or asserted by the model. Particle directions/comoving "
    "radii are arbitrary placement choices; only each level's active-slot "
    "fraction (from matter_series) and the background size_measure(tau) are "
    "quantities produced by the run."
)


def _sphere_wireframe(r: float, n: int = 16):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _assign_3d_directions(n_slots: int, seed: int) -> np.ndarray:
    """Arbitrary unit direction * comoving-radius-fraction per slot.

    Flagged: this is a placement choice for the picture, exactly like
    the fixed comoving_pos/comoving_r dicts in the original script,
    generalized to any number of slots and made deterministic per level
    via `seed` (so re-running with the same scales reproduces the same
    picture) instead of being hand-picked per scale.
    """
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=(n_slots, 3))
    vec /= np.linalg.norm(vec, axis=1, keepdims=True)
    radius_frac = rng.uniform(0.15, 1.0, size=n_slots)
    return vec * radius_frac[:, None]


def _first_active_indices(active: np.ndarray) -> np.ndarray:
    """First timestep index each particle (column) goes active, or -1 if never."""
    n_particles = active.shape[1]
    out = np.full(n_particles, -1, dtype=int)
    for p in range(n_particles):
        idxs = np.flatnonzero(active[:, p])
        if idxs.size:
            out[p] = idxs[0]
    return out


def _trail_points(direction, R, first_idx, upto_idx, max_points):
    """Subsampled path of a single particle from its activation to `upto_idx`.

    Not a new derived quantity: it is the same (direction * R(t)) position
    used for the marker itself, just evaluated at intermediate t rather than
    only at the current frame -- i.e. "where this particle's marker would
    have been plotted at earlier times," subsampled for render speed.
    """
    if first_idx < 0 or first_idx > upto_idx:
        return None
    span = upto_idx - first_idx + 1
    k = min(max_points, span)
    idxs = np.unique(np.linspace(first_idx, upto_idx, k).astype(int))
    return direction[None, :] * R[idxs, None]


def _build_level_data(levels, n_particles):
    cmap = plt.get_cmap("plasma")
    colors = [
        cmap(0.15 + 0.7 * i / max(len(levels) - 1, 1)) for i in range(len(levels))
    ]
    level_data = []
    for lvl, color in zip(levels, colors):
        # Reuses the SAME derived active-slot logic as the existing 2D
        # worldline panel and the --anim renderer -- only the embedding
        # (2D comoving y vs 3D direction) differs.
        _, active = build_worldlines(lvl.matter_series, n_particles, seed=int(lvl.width))
        directions = _assign_3d_directions(n_particles, seed=int(lvl.width) + 97)
        first_active = _first_active_indices(active)
        level_data.append((directions, active, first_active, color, lvl))
    return level_data


def render_3d_particles(
    t_bf: np.ndarray,
    size_measure: np.ndarray,
    levels: list,
    output: str,
    animate: bool = False,
    n_particles: int = 40,
    frames: int = 150,
    fps: int = 20,
    tracks: bool = False,
    track_points: int = 60,
    title: str = "Worldlines as particles (illustrative 3D + spherical symmetry)",
) -> None:
    """Render worldlines as particles in a 3D comoving embedding.

    `levels` is a list of anim_common.LevelAnimSpec (or any object with
    .width, .tau_local, .lapse, .matter_series) -- the same backend-
    agnostic description already used by --anim, so each CLI script can
    build it once and pass it to both renderers.

    animate=False -> static multi-snapshot grid (like the original script).
    animate=True  -> rotating GIF/MP4, particles moving outward with the
                      background as it expands.
    tracks=True   -> also draw each particle's history path (trail) from
                      the moment it became active up to the current frame,
                      subsampled to `track_points`. Slower, especially
                      combined with --anim and many particles.
    """
    R = np.asarray(size_measure, dtype=float)
    lim = float(np.max(np.abs(R))) * 1.05 if np.max(np.abs(R)) > 0 else 1.0
    level_data = _build_level_data(levels, n_particles)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if not animate:
        _render_snapshots(t_bf, R, level_data, lim, output, title, tracks, track_points)
    else:
        _render_animation(t_bf, R, level_data, lim, output, frames, fps, title,
                           tracks, track_points)


def _draw_trails(ax, level_data, R, upto_idx, track_points):
    for directions, active, first_active, color, lvl in level_data:
        for p in range(directions.shape[0]):
            if first_active[p] < 0 or first_active[p] > upto_idx:
                continue
            pts = _trail_points(directions[p], R, first_active[p], upto_idx, track_points)
            if pts is not None and len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, alpha=0.35, lw=0.8)


def _render_snapshots(t_bf, R, level_data, lim, output, title, tracks, track_points):
    n = len(t_bf)
    fracs = [0.02, 0.05, 0.12, 0.30, 0.6, 1.0]
    idxs = [int(f * (n - 1)) for f in fracs]

    fig = plt.figure(figsize=(15, 9))
    for panel, idx in enumerate(idxs):
        ax = fig.add_subplot(2, 3, panel + 1, projection="3d")
        x, y, z = _sphere_wireframe(R[idx], n=18)
        ax.plot_wireframe(x, y, z, color="gray", alpha=0.25, lw=0.4)
        if tracks:
            _draw_trails(ax, level_data, R, idx, track_points)
        for directions, active, first_active, color, lvl in level_data:
            mask = active[idx]
            if not np.any(mask):
                continue
            pts = directions[mask] * R[idx]
            marker_size = 10 + 60 * lvl.lapse
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                color=color, s=marker_size, alpha=0.85, depthshade=True,
            )
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_title(f"t={t_bf[idx]:.0f}  (R={R[idx]:.3f})", fontsize=9)
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()

    track_note = " Trails show each particle's path since activation." if tracks else ""
    fig.suptitle(f"{title}\n{CAVEAT}{track_note}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(output, dpi=130)
    plt.close(fig)
    print(f"saved -> {output}")


def _render_animation(t_bf, R, level_data, lim, output, frames, fps, title, tracks, track_points):
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
        for _, _, _, color, lvl in level_data
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              facecolor="#111115", edgecolor="gray", labelcolor="white")

    fig.suptitle(title, fontsize=11, fontweight="bold")
    time_text = fig.text(0.5, 0.03, "", ha="center", fontsize=10, fontweight="bold")
    track_note = " Trails show each particle's path since activation." if tracks else ""
    caption = fig.text(0.5, 0.005, CAVEAT + track_note, ha="center", fontsize=6.5, wrap=True)

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
        if tracks:
            _draw_trails(ax, level_data, R, idx, track_points)
        for directions, active, first_active, color, lvl in level_data:
            mask = active[idx]
            if np.any(mask):
                pts = directions[mask] * R[idx]
                marker_size = 10 + 60 * lvl.lapse
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                           color=color, s=marker_size, alpha=0.85)
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
    """Add --3d, --n_particles, --tracks, --track_points to an argparse parser.

    Assumes --anim, --frames, --fps, and --output are already added by the
    caller (all three backend scripts already have these).
    """
    parser.add_argument("--3d", dest="three_d", action="store_true",
                         help="Add a 3D view: worldlines rendered as particles in an "
                              "illustrative 3D + spherically symmetric embedding "
                              "(NOT derived -- flagged in the output). Alone this "
                              "produces a static multi-snapshot grid; combined with "
                              "--anim it produces a rotating GIF/MP4 instead.")
    parser.add_argument("--n_particles", type=int, default=40,
                         help="Particles per scale in the 3D view (--3d only).")
    parser.add_argument("--tracks", action="store_true",
                         help="Draw each particle's history path since it became "
                              "active (--3d only). Slower, especially with --anim.")
    parser.add_argument("--track_points", type=int, default=60,
                         help="Max subsampled points per trail (--tracks only).")


def dispatch_3d(args, t_bf: np.ndarray, size_measure: np.ndarray, levels: list,
                 title: str) -> None:
    """Call render_3d_particles from a backend's main(), reading the shared
    --3d/--anim/--frames/--fps/--tracks/--track_points args. No-op if --3d
    wasn't passed.
    """
    if not getattr(args, "three_d", False):
        return
    out_path = Path(args.output)
    suffix = ".gif" if args.anim else ".png"
    threed_output = str(out_path.with_name(f"{out_path.stem}_3d{suffix}"))
    render_3d_particles(
        t_bf, size_measure, levels, threed_output,
        animate=args.anim, n_particles=args.n_particles,
        frames=args.frames, fps=args.fps,
        tracks=args.tracks, track_points=args.track_points,
        title=title,
    )
