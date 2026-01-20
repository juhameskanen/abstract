#!/usr/bin/env python3
"""
Gray-code configuration-space visualizer for the "ontological equivalence" demo.

- Top: wide 1D bitstring (execution trace).
- Bottom: multiple 2D reshapes (different row/col aspect ratios) of the SAME bitstring.
- Traversal modes:
    * 'gray'            : Binary-reflected Gray code (smooth, single-bit flips, visits all 2^N states).
    * 'random_permute'  : A random permutation of all states (visits all states but multi-bit jumps).
    * 'neighbor_walk'   : Random neighbor walk (flip one random bit each step; may revisit states).
- WARNING: frames = 2**N. Choose N accordingly (N <= 10 recommended for interactive runs).
"""
import numpy as np
import math
import shutil
import os
import argparse
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ----------------------- Utility functions --------------------------------
def gray_code_int(i: int) -> int:
    """Binary-reflected Gray code integer for index i."""
    return i ^ (i >> 1)

def int_to_bitarray(x: int, N: int) -> np.ndarray:
    """Return array of N bits (MSB on the left)."""
    return np.array([ (x >> (N - 1 - j)) & 1 for j in range(N) ], dtype=np.uint8)

def all_gray_frames(N: int) -> np.ndarray:
    """Return array shape (2**N, N) containing Gray-code bitframes."""
    frames = 1 << N
    out = np.empty((frames, N), dtype=np.uint8)
    for i in range(frames):
        g = gray_code_int(i)
        out[i] = int_to_bitarray(g, N)
    return out

def all_permuted_frames(N: int, seed: int = None) -> np.ndarray:
    """Return a random permutation of all 2**N states (each state as bit array)."""
    rng = np.random.RandomState(seed)
    total = 1 << N
    perm = rng.permutation(total)
    out = np.empty((total, N), dtype=np.uint8)
    for i, val in enumerate(perm):
        out[i] = int_to_bitarray(val, N)
    return out

def neighbor_random_walk(N: int, steps: int, seed: int = None) -> np.ndarray:
    """Return frames from a random neighbor walk (flip one random bit per step)."""
    rng = np.random.RandomState(seed)
    out = np.zeros((steps, N), dtype=np.uint8)
    state = np.zeros(N, dtype=np.uint8)
    out[0] = state.copy()
    for i in range(1, steps):
        bit = rng.randint(0, N)
        state[bit] ^= 1
        out[i] = state.copy()
    return out

def factor_pairs(n: int):
    """Return factor pairs (r,c) of n, sorted by closeness to square."""
    pairs = []
    for r in range(1, int(math.sqrt(n)) + 1):
        if n % r == 0:
            pairs.append((r, n // r))
    pairs_all = pairs + [(c, r) for r, c in pairs if r != c]
    # de-dupe and sort by how square-ish they are
    pairs_all = sorted({p for p in pairs_all}, key=lambda p: abs(p[0] - p[1]))
    return pairs_all

def choose_shapes(N: int, want: int = 2):
    """Select up to `want` distinct grid shapes (r,c) that multiply to N."""
    pairs = factor_pairs(N)
    if not pairs:
        # prime N fallback
        return [(1, N), (N, 1)][:want]
    chosen = []
    for p in pairs:
        if len(chosen) >= want:
            break
        if not any(abs(p[0]/p[1] - q[0]/q[1]) < 0.05 for q in chosen):
            chosen.append(p)
    # pad if needed
    idx = 0
    while len(chosen) < want and idx < len(pairs):
        if pairs[idx] not in chosen:
            chosen.append(pairs[idx])
        idx += 1
    return chosen[:want]

# ----------------------- Visualization function ---------------------------
def run_visualization(
    N=8,
    mode="gray",
    bottom_views=2,
    fps=20,
    save_mp4=False,
    mp4_path="gray_code_demo.mp4",
    seed=None,
):
    """
    Run the animation.
      - N: bit-length
      - mode: 'gray' | 'random_permute' | 'neighbor_walk'
      - bottom_views: how many 2D reinterpretations to show
      - fps: playback fps (interval = 1000 / fps ms)
      - save_mp4: try to save an mp4 (requires ffmpeg)
    """
    assert mode in ("gray", "random_permute", "neighbor_walk")
    if mode == "neighbor_walk":
        steps = 1 << min(N, 12)  # neighbor walk length; keep bounded if N large
    else:
        steps = 1 << N

    print(f"Preparing visualization: N={N}, mode={mode}, frames={steps}")

    if mode == "gray":
        frames_bits = all_gray_frames(N)
    elif mode == "random_permute":
        frames_bits = all_permuted_frames(N, seed=seed)
    else:  # neighbor_walk
        frames_bits = neighbor_random_walk(N, steps=steps, seed=seed)

    # choose shapes for bottom views
    shapes = choose_shapes(N, want=bottom_views)
    print("Shapes for bottom views:", shapes)

    # build figure
    fig = plt.figure(figsize=(12, 6))
    # top 1D strip (wide)
    ax_top = fig.add_axes([0.05, 0.70, 0.90, 0.25])
    img_top = ax_top.imshow(frames_bits[0:1, :], cmap="gray_r", aspect="auto", vmin=0, vmax=1)
    ax_top.set_title("1D Bitstring (Execution Trace)", fontsize=12)
    ax_top.axis("off")

    # bottom views side-by-side
    bottom_images = []
    left = 0.05
    width_each = 0.9 / len(shapes)
    for i, sh in enumerate(shapes):
        ax = fig.add_axes([left + i * width_each, 0.06, width_each - 0.02, 0.56])
        im = ax.imshow(frames_bits[0].reshape(sh), cmap="gray_r", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"2D view {sh[0]}x{sh[1]}", fontsize=10)
        ax.axis("off")
        bottom_images.append(im)

    plt.suptitle("Configuration-space traversal: same base bits, multiple interpretations", fontsize=14)
    #plt.tight_layout(rect=[0, 0, 1, 0.95])

    # update function
    def update(idx):
        bits = frames_bits[idx % frames_bits.shape[0]]
        img_top.set_data(bits[np.newaxis, :])
        for im, sh in zip(bottom_images, shapes):
            im.set_data(bits.reshape(sh))
        return [img_top] + bottom_images

    interval_ms = int(1000 / fps)
    ani = FuncAnimation(fig, update, frames=frames_bits.shape[0], interval=interval_ms, blit=False, repeat=False)

    # Keep a strong reference until after show()
    global _ani
    _ani = ani

    # show interactive window (or in notebooks, plt.show() may not animate; see environment)
    plt.show()

    if save_mp4:
        if shutil.which("ffmpeg") is None:
            print("ffmpeg not found on PATH; cannot save mp4. Install ffmpeg or set save_mp4=False.")
        else:
            print("Saving mp4 (this can take a while)...")
            writer = FFMpegWriter(fps=fps)
            ani.save(mp4_path, writer=writer)
            print("Saved:", mp4_path)

    return ani

# ----------------------- If run as script -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration-space bitstring visualizer")
    parser.add_argument("--N", type=int, default=10, help="Bit length (frames = 2**N)")
    parser.add_argument("--mode", type=str, default="gray", choices=("gray", "random_permute", "neighbor_walk"))
    parser.add_argument("--bottom", type=int, default=2, help="Number of bottom 2D views")
    parser.add_argument("--fps", type=int, default=20, help="Playback frames per second")
    parser.add_argument("--save", action="store_true", help="Save an mp4 (requires ffmpeg)")
    parser.add_argument("--out", type=str, default="gray_demo.mp4", help="MP4 output path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (for non-gray modes)")
    args = parser.parse_args()

    run_visualization(N=args.N, mode=args.mode, bottom_views=args.bottom, fps=args.fps,
                      save_mp4=args.save, mp4_path=args.out, seed=args.seed)
