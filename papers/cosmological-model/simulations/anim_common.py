"""
Shared time-dilation animation renderer, usable by all three cosmology
backends (statistical/multiclock, dicke/cascade, wavefunction).

This is cosmic_d_anim.py's logic, generalized: instead of assuming
multiclock.SimulationResult, it takes a small, backend-agnostic
description of "what a level is" (LevelAnimSpec) so each backend's own
CLI script can build that description from whatever result object it
already produces, without duplicating the animation code three times.

Nothing physical is added here -- same two panels as cosmic_d_anim.py:
  - top: comoving worldlines + size_measure envelope, each level's
    worldlines revealed only up to ITS OWN tau_local (not raw tau).
  - bottom: "clock race" -- own_ticks(tau) = floor(tau / width) per
    level, staircased against shared raw tau. Slope 1/width per level;
    the reference line has slope 1 (a hypothetical lapse=1 observer).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from multiclock import build_worldlines


@dataclass
class LevelAnimSpec:
    """Backend-agnostic description of one scale, for animation purposes.

    width:         window width w (bits) -- used only for labeling/color seed.
    tau_local:     array, same length/grid as t_bf; this level's own
                   retarded clock reading at each raw-tau sample
                   (e.g. width * floor(tau_raw / width)).
    matter_series: array, same grid as t_bf; feeds build_worldlines to
                   decide which comoving slots are "on" at each tau.
    lapse:         1/width, used only for the legend label.
    """

    width: float
    tau_local: np.ndarray
    matter_series: np.ndarray
    lapse: float


def render_time_dilation_animation(
    t_bf: np.ndarray,
    t_bf_max: float,
    size_measure: np.ndarray,
    levels: list[LevelAnimSpec],
    output: str,
    n_slots: int = 50,
    frames: int = 150,
    fps: int = 20,
    title: str = "Worldlines growing at different rates (time dilation)",
) -> None:
    n_frames_total = len(t_bf)
    frame_stride = max(1, n_frames_total // frames)
    frame_indices = np.arange(0, n_frames_total, frame_stride)
    if frame_indices[-1] != n_frames_total - 1:
        frame_indices = np.append(frame_indices, n_frames_total - 1)

    cmap = plt.get_cmap("plasma")
    colors = [
        cmap(0.15 + 0.7 * i / max(len(levels) - 1, 1)) for i in range(len(levels))
    ]

    worldline_data = []
    for lvl, color in zip(levels, colors):
        y_slots, active = build_worldlines(lvl.matter_series, n_slots, seed=int(lvl.width))
        worldline_data.append((y_slots, active, color, lvl))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2.0, 1.3]}
    )
    fig.patch.set_facecolor("white")

    ax_top.set_facecolor("#020205")
    ax_top.set_xlim(0, t_bf_max)
    ax_top.set_ylim(-0.6, 0.6)
    ax_top.set_xlabel("raw bit-flip coordinate time  \u03c4")
    ax_top.set_ylabel("comoving y  \u00d7  size_measure(\u03c4)")
    ax_top.set_title(
        "Structure emerging in coordinate time \u2014 each scale lags behind "
        "\u03c4 by its own retarded clock",
        fontsize=9.5, color="black",
    )

    env_top, = ax_top.plot([], [], color="white", lw=2.2, alpha=0.9)
    env_bot, = ax_top.plot([], [], color="white", lw=2.2, alpha=0.9)
    fill_container = {"poly": None}

    worldline_artists = []
    for y_slots, active, color, lvl in worldline_data:
        lines_for_level = [
            ax_top.plot([], [], color=color, lw=0.7, alpha=0.45)[0] for _ in range(len(y_slots))
        ]
        worldline_artists.append(lines_for_level)

    for lvl, color in zip(levels, colors):
        ax_top.plot([], [], color=color, lw=1.4, label=f"scale w={lvl.width:g}  (lapse=1/{lvl.width:g})")
    ax_top.legend(loc="upper left", fontsize=8, facecolor="#111115", edgecolor="gray", labelcolor="white")

    tau_marker = ax_top.axvline(0, color="lime", lw=1.3, ls="--", alpha=0.85)

    own_ticks_full = [np.floor(t_bf / lvl.width) for lvl in levels]
    max_ticks = max(ot.max() for ot in own_ticks_full)

    ax_bot.set_facecolor("#0a0a0a")
    ax_bot.set_xlim(0, t_bf_max)
    ax_bot.set_ylim(0, max_ticks * 1.05)
    ax_bot.set_xlabel("coordinate time  \u03c4  (shared, universal)")
    ax_bot.set_ylabel("own clock reading  (ticks elapsed)")
    ax_bot.set_title(
        "Proper time of each scale:  own\u2009ticks(\u03c4) = floor(\u03c4 / w)\n"
        "same \u03c4 elapsed \u2192 different number of own ticks \u2192 time dilation",
        fontsize=9, color="black",
    )
    ax_bot.tick_params(axis="x", labelsize=8)
    ax_bot.grid(color="#333333", lw=0.5, alpha=0.5)

    race_lines = []
    for lvl, color in zip(levels, colors):
        (ln,) = ax_bot.step([], [], where="post", color=color, lw=2.2,
                             label=f"w={lvl.width:g}  (slope=1/{lvl.width:g})")
        race_lines.append(ln)
    ax_bot.plot([0, t_bf_max], [0, t_bf_max], color="gray", lw=1.0, ls=":", alpha=0.5,
                label="slope=1 reference (coordinate time itself)")
    ax_bot.legend(loc="upper left", fontsize=8, facecolor="#111115", edgecolor="gray", labelcolor="white")

    tick_texts = [
        ax_bot.text(0, 0, "", va="bottom", ha="left", fontsize=8, color=color, fontweight="bold")
        for color in colors
    ]

    time_text = fig.text(0.5, 0.965, "", ha="center", fontsize=11, fontweight="bold")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    def init():
        env_top.set_data([], [])
        env_bot.set_data([], [])
        for lines_for_level in worldline_artists:
            for ln in lines_for_level:
                ln.set_data([], [])
        for ln in race_lines:
            ln.set_data([], [])
        for txt in tick_texts:
            txt.set_text("")
        time_text.set_text("")
        return []

    def update(frame_idx):
        idx = frame_indices[frame_idx]
        tau_now = t_bf[idx]

        env_top.set_data(t_bf[: idx + 1], size_measure[: idx + 1] / 2)
        env_bot.set_data(t_bf[: idx + 1], -size_measure[: idx + 1] / 2)

        if fill_container["poly"] is not None:
            fill_container["poly"].remove()
        fill_container["poly"] = ax_top.fill_between(
            t_bf[: idx + 1], -size_measure[: idx + 1] / 2, size_measure[: idx + 1] / 2,
            color="gainsboro", alpha=0.15,
        )

        for (y_slots, active, color, lvl), lines_for_level in zip(worldline_data, worldline_artists):
            # This scale's own clock has only reached tau_local (<= tau_now),
            # lagging by up to `width` raw ticks and advancing in jumps of
            # `width` -- not every frame. This is the same fix applied to
            # cosmic_wavefunction.py's static plot, generalized here.
            tau_local_now = lvl.tau_local[idx]
            own_idx = int(np.searchsorted(t_bf, tau_local_now, side="right"))
            for i, y0 in enumerate(y_slots):
                mask = active[:own_idx, i]
                if mask.any():
                    xs = t_bf[:own_idx][mask]
                    ys = y0 * size_measure[:own_idx][mask]
                    lines_for_level[i].set_data(xs, ys)

        tau_marker.set_xdata([tau_now, tau_now])

        for ln, ot, lvl, txt in zip(race_lines, own_ticks_full, levels, tick_texts):
            ln.set_data(t_bf[: idx + 1], ot[: idx + 1])
            own_ticks_now = int(ot[idx])
            txt.set_position((tau_now + t_bf_max * 0.01, own_ticks_now))
            txt.set_text(f"{own_ticks_now}")

        time_text.set_text(f"coordinate time \u03c4 = {tau_now:8.1f}  /  {t_bf_max:.1f}")
        return []

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_indices), init_func=init, blit=False,
    )
    writer = animation.PillowWriter(fps=fps)
    ani.save(output, writer=writer, dpi=110)
    plt.close(fig)
    print(f"saved \u2192 {output}  (frames={len(frame_indices)}, stride={frame_stride}, fps={fps})")
