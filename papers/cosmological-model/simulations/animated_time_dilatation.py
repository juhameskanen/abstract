"""
Animated time-dilation demo for the multi-clock bitstring model.

Grounding (nothing new is invented here — every quantity plotted already
exists in `multiclock.py`):

  - lvl.width   : the scale's window size w (bits)
  - lvl.lapse   : 1/w, the derived "clock rate" of that scale relative to
                  raw coordinate time tau
  - lvl.tau_local = retard(tau, w) = w * floor(tau / w)
                  the scale's own elapsed proper time at raw coordinate
                  time tau. This is exactly what a width-w structure's
                  internal clock reads: it only ticks once every w raw
                  bit-flips.

BUGFIX vs. the previous version of this script: the top panel used to reveal
every scale's worldlines up through the CURRENT raw tau, regardless of w --
so once a structure existed, it always kept pace with raw tau and every
scale looked equally fast. The only thing that differed was *when* a slot
turned on (which is entropy/structure-formation timing, already correct),
not how fast it then moved. Fixed below: each scale's worldlines are now
revealed only up to that scale's own retarded time lvl.tau_local, so a
large-w structure's visible tip lags behind raw tau and advances in
visible jumps of size w, while a small-w structure almost keeps pace.
Structures still only switch on when build_worldlines says they exist
(entropy-driven, unchanged) -- nothing here starts at tau=0.

The "worldlines growing at different speeds" panel is literally
tau_local(frame) plotted against a common raw-tau axis for each scale,
next to the tau itself (the lapse=1 reference / "coordinate observer").
Because tau_local jumps in steps of w, a large-w scale's bar visibly
lags and jumps in big chunks while a small-w scale's bar almost keeps
pace with raw tau — i.e. it "ages" faster. That lag is the model's own
predicted time-dilation effect (its analogue of gravitational/velocity
time dilation), not an artistic embellishment.

The top panel reuses the model's own comoving-worldline construction
(build_worldlines) to also show structure emerging, revealed
progressively frame by frame, for physical context.
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import multiclock as mc


def parse_scales(raw: str) -> list[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


parser = argparse.ArgumentParser(
    description="Animated time-dilation view of the multi-clock model. "
                "Takes the SAME simulation args as emergent_structure_relativistic.py, "
                "plus two animation-only knobs (--frames, --fps)."
)
# --- identical to emergent_structure_relativistic.py ---
parser.add_argument("--n_bits", type=float, default=184.0)
parser.add_argument("--t_bf_max", type=float, default=None,
                     help="Max raw bit-flip time, in units of n. Default: ln(n).")
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--t_today", type=float, default=None)
parser.add_argument("--matter_power", type=float, default=1.0)
parser.add_argument("--scales", type=str, default="6,12,20")
parser.add_argument("--slots", type=int, default=50)
parser.add_argument("--output", type=str, default="time_dilation.gif")
# --- animation-only ---
parser.add_argument("--frames", type=int, default=150,
                     help="Target number of animation frames (subsamples --steps grid points).")
parser.add_argument("--fps", type=int, default=20)

args = parser.parse_args()
SCALES = parse_scales(args.scales)
N_SLOTS = args.slots
FRAME_STRIDE = max(1, args.steps // args.frames)
FPS = args.fps

# ---------------------------------------------------------------------
# Run the underlying simulation (identical physics/math to the static plot)
# ---------------------------------------------------------------------
sim = mc.run_simulation(
    n_bits=args.n_bits, scales=SCALES, steps=args.steps, t_bf_max=args.t_bf_max,
    t_today=args.t_today, matter_power=args.matter_power,
)

t_bf = sim.t_bf
n_frames_total = len(t_bf)
frame_indices = np.arange(0, n_frames_total, FRAME_STRIDE)
if frame_indices[-1] != n_frames_total - 1:
    frame_indices = np.append(frame_indices, n_frames_total - 1)

cmap = plt.get_cmap("plasma")
colors = [cmap(0.15 + 0.7 * i / max(len(sim.levels) - 1, 1)) for i in range(len(sim.levels))]

# Precompute comoving worldlines (top panel) exactly as in the static plot
worldline_data = []
for level, matter, color in zip(sim.levels, sim.per_scale_matter, colors):
    y_slots, active = mc.build_worldlines(matter, N_SLOTS, seed=int(level.width))
    worldline_data.append((y_slots, active, color, level))

size_env = sim.size_measure

# ---------------------------------------------------------------------
# Figure layout: top = spacetime/structure panel, bottom = clock race panel
# ---------------------------------------------------------------------
fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2.0, 1.3]}
)
fig.patch.set_facecolor("white")

# --- top panel setup ---
ax_top.set_facecolor("#020205")
ax_top.set_xlim(0, sim.t_bf_max)
ax_top.set_ylim(-0.6, 0.6)
ax_top.set_xlabel("raw bit-flip coordinate time  \u03c4")
ax_top.set_ylabel("comoving y  \u00d7  size_measure(\u03c4)")
ax_top.set_title("Structure emerging in coordinate time \u2014 each scale lags behind \u03c4 by its own retarded clock", fontsize=9.5, color="black")

env_top, = ax_top.plot([], [], color="white", lw=2.2, alpha=0.9)
env_bot, = ax_top.plot([], [], color="white", lw=2.2, alpha=0.9)
fill_container = {"poly": None}

worldline_artists = []
for y_slots, active, color, level in worldline_data:
    lines_for_level = []
    for i in range(len(y_slots)):
        (ln,) = ax_top.plot([], [], color=color, lw=0.7, alpha=0.45)
        lines_for_level.append(ln)
    worldline_artists.append(lines_for_level)

for level, color in zip(sim.levels, colors):
    ax_top.plot([], [], color=color, lw=1.4, label=f"scale w={level.width:g}  (lapse=1/{level.width:g})")
ax_top.legend(loc="upper left", fontsize=8, facecolor="#111115", edgecolor="gray", labelcolor="white")

tau_marker = ax_top.axvline(0, color="lime", lw=1.3, ls="--", alpha=0.85)

# --- bottom panel setup: proper time (own tick count) vs coordinate time ---
# own_ticks(tau) = floor(tau / w) is each scale's own clock reading. Plotting
# it against tau produces a fan of straight lines (staircases) with slope
# 1/w -- literally "worldlines growing at different speeds": the same
# stretch of coordinate time tau produces more ticks for small w (fast
# clock) and fewer ticks for large w (slow / dilated clock).
own_ticks_full = [np.floor(t_bf / lvl.width) for lvl in sim.levels]
max_ticks = max(ot.max() for ot in own_ticks_full)

ax_bot.set_facecolor("#0a0a0a")
ax_bot.set_xlim(0, sim.t_bf_max)
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
for level, color in zip(sim.levels, colors):
    (ln,) = ax_bot.step([], [], where="post", color=color, lw=2.2,
                         label=f"w={level.width:g}  (slope=1/{level.width:g})")
    race_lines.append(ln)
ax_bot.plot([0, sim.t_bf_max], [0, sim.t_bf_max], color="gray", lw=1.0, ls=":", alpha=0.5,
            label="slope=1 reference (coordinate time itself)")
ax_bot.legend(loc="upper left", fontsize=8, facecolor="#111115", edgecolor="gray", labelcolor="white")

tick_texts = []
for i, (level, color) in enumerate(zip(sim.levels, colors)):
    txt = ax_bot.text(0, 0, "", va="bottom", ha="left", fontsize=8, color=color, fontweight="bold")
    tick_texts.append(txt)

time_text = fig.text(0.5, 0.965, "", ha="center", fontsize=11, fontweight="bold")

fig.suptitle(
    "IAME multi-clock model: worldlines growing at different rates (time dilation)",
    fontsize=12, fontweight="bold", y=0.995,
)
fig.tight_layout(rect=[0, 0, 1, 0.94])


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

    # top panel: reveal envelope + worldlines up to current tau
    env_top.set_data(t_bf[: idx + 1], size_env[: idx + 1] / 2)
    env_bot.set_data(t_bf[: idx + 1], -size_env[: idx + 1] / 2)

    if fill_container["poly"] is not None:
        fill_container["poly"].remove()
    fill_container["poly"] = ax_top.fill_between(
        t_bf[: idx + 1], -size_env[: idx + 1] / 2, size_env[: idx + 1] / 2,
        color="gainsboro", alpha=0.15,
    )

    for (y_slots, active, color, level), lines_for_level in zip(worldline_data, worldline_artists):
        # This scale's own clock has only reached tau_local (<= tau_now), lagging
        # by up to w raw ticks and advancing in jumps of w -- not every frame.
        tau_local_now = level.tau_local[idx]
        own_idx = int(np.searchsorted(t_bf, tau_local_now, side="right"))
        for i, y0 in enumerate(y_slots):
            mask = active[:own_idx, i]
            if mask.any():
                xs = t_bf[:own_idx][mask]
                ys = y0 * size_env[:own_idx][mask]
                lines_for_level[i].set_data(xs, ys)

    tau_marker.set_xdata([tau_now, tau_now])

    # bottom panel: own-tick staircases, revealed up to current tau
    for ln, ot, level, txt in zip(race_lines, own_ticks_full, sim.levels, tick_texts):
        ln.set_data(t_bf[: idx + 1], ot[: idx + 1])
        own_ticks_now = int(ot[idx])
        txt.set_position((tau_now + sim.t_bf_max * 0.01, own_ticks_now))
        txt.set_text(f"{own_ticks_now}")

    time_text.set_text(f"coordinate time \u03c4 = {tau_now:8.1f}  /  {sim.t_bf_max:.1f}")

    return []


ani = animation.FuncAnimation(
    fig, update, frames=len(frame_indices), init_func=init, blit=False,
)

writer = animation.PillowWriter(fps=FPS)
ani.save(args.output, writer=writer, dpi=110)
print(f"saved \u2192 {args.output}")
print(f"n_bits={args.n_bits:g}  scales={SCALES}  steps={args.steps}  slots={N_SLOTS}  "
      f"frames={len(frame_indices)} (stride={FRAME_STRIDE})  fps={FPS}")
