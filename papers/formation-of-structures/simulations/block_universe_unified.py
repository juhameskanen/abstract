"""
Block Universe Unified Simulation (PoC)
======================================

This script implements a unified variational simulation combining:

1. Spectral compression (Hilbert-space smoothness)
2. Entropy-driven temporal structure (arrow of time)
3. Geometric compression (MDL-based trajectory smoothness)
4. Observer-field coupling (joint probability principle)

The system optimizes a spacetime wavefunction Ψ(t, x, y) and a set of
observer trajectories x_i(t) simultaneously.

Core hypothesis:
----------------
Observers exist in regions that are jointly compressible in:
    - Spectral representation (Ψ)
    - Geometric trajectory (MDL)

This produces emergent structures, interactions, and proto-dynamics
without explicitly encoding forces or velocities.

Usage:
------
python unified_sim.py --help

Example:
--------
python unified_sim.py --T 80 --H 64 --W 64 --n_obs 3 --iterations 300

Author:
-------
Juha Meskanen
"""

from __future__ import annotations

import argparse
from typing import Tuple

import cv2
import numpy as np
import torch



def build_k2(T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Construct squared frequency grid for spectral regularization.

    Args:
        T: Number of time steps.
        H: Height of spatial grid.
        W: Width of spatial grid.
        device: Torch device.

    Returns:
        Tensor of shape (T, H, W) containing squared frequencies.
    """
    kt = torch.fft.fftfreq(T).reshape(T, 1, 1).repeat(1, H, W)
    ky = torch.fft.fftfreq(H).reshape(1, H, 1).repeat(T, 1, W)
    kx = torch.fft.fftfreq(W).reshape(1, 1, W).repeat(T, H, 1)
    return (kx**2 + ky**2 + kt**2).to(device)


def compute_entropy(p: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy of a probability distribution.

    Args:
        p: Probability tensor.

    Returns:
        Scalar entropy value.
    """
    p = torch.clamp(p, min=1e-12)
    return -torch.sum(p * torch.log(p))


def mdl_loss(traj: torch.Tensor) -> torch.Tensor:
    """
    Compute MDL-inspired trajectory smoothness loss.

    Penalizes deviation from linear extrapolation.

    Args:
        traj: Tensor of shape (T, 2).

    Returns:
        Scalar loss value.
    """
    loss = torch.tensor(0.0, device=traj.device)
    for t in range(2, traj.shape[0]):
        pred = traj[t - 1] + (traj[t - 1] - traj[t - 2])
        loss += torch.sum((traj[t] - pred) ** 2)
    return loss


def bilinear_sample(field: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """
    Sample 2D field at continuous coordinates using bilinear interpolation.

    Args:
        field: Tensor of shape (H, W).
        pos: Tensor of shape (2,) with values in [0, 1].

    Returns:
        Interpolated scalar value.
    """
    H, W = field.shape

    x = pos[0] * (W - 1)
    y = pos[1] * (H - 1)

    x0 = torch.clamp(x.long(), 0, W - 2)
    y0 = torch.clamp(y.long(), 0, H - 2)

    dx = x - x0.float()
    dy = y - y0.float()

    f00 = field[y0, x0]
    f10 = field[y0, x0 + 1]
    f01 = field[y0 + 1, x0]
    f11 = field[y0 + 1, x0 + 1]

    return (
        f00 * (1 - dx) * (1 - dy)
        + f10 * dx * (1 - dy)
        + f01 * (1 - dx) * dy
        + f11 * dx * dy
    )



def run_simulation(args: argparse.Namespace) -> None:
    """
    Execute the unified variational simulation.

    Args:
        args: Parsed command-line arguments.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k2 = build_k2(args.T, args.H, args.W, device)

    # Initialize wavefunction
    psi = torch.randn((args.T, args.H, args.W, 2), device=device).detach()
    psi *= args.psi_scale
    psi.requires_grad_(True)

    # Low-entropy seed
    psi.data[0, args.H // 2, args.W // 2, 0] = 1.0

    # Observer trajectories
    obs = torch.rand((args.n_obs, args.T, 2), device=device).detach()
    obs.requires_grad_(True)

    optimizer = torch.optim.Adam([psi, obs], lr=args.lr)

    h_max = np.log(args.H * args.W)


    for it in range(args.iterations):
        optimizer.zero_grad()

        psi_c = torch.complex(psi[..., 0], psi[..., 1])

        # Spectral loss
        psi_k = torch.fft.fftn(psi_c, dim=(0, 1, 2))
        spec_loss = torch.sum(k2 * torch.abs(psi_k) ** 2)

        entropy_loss = torch.tensor(0.0, device=device)
        coupling_loss = torch.tensor(0.0, device=device)

        for t in range(args.T):
            prob = torch.abs(psi_c[t]) ** 2
            p_norm = prob / (torch.sum(prob) + 1e-12)

            progress = t / (args.T - 1)
            target_h = (
                args.h_start + (args.h_end - args.h_start) * progress
            ) * h_max

            h = compute_entropy(p_norm)
            entropy_loss += (h - target_h) ** 2

            for i in range(args.n_obs):
                density = bilinear_sample(prob, obs[i, t])
                coupling_loss += (density - args.target_density) ** 2

        # Geometric loss
        geom_loss = torch.tensor(0.0, device=device)
        for i in range(args.n_obs):
            geom_loss += mdl_loss(obs[i])

        loss = (
            args.kappa * spec_loss
            + entropy_loss
            + args.alpha * geom_loss
            + args.beta * coupling_loss
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            obs.clamp_(0.0, 1.0)

        if it % args.print_every == 0:
            print(f"Iter {it:04d} | Loss {loss.item():.4f}")


    psi_c = torch.complex(psi[..., 0], psi[..., 1])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (args.W, args.H))

    for t in range(args.T):
        prob = torch.abs(psi_c[t]) ** 2
        frame = prob / (prob.max() + 1e-12)

        frame = frame.detach().cpu().numpy()
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_MAGMA)

        for i in range(args.n_obs):
            x = int(obs[i, t, 0].item() * (args.W - 1))
            y = int(obs[i, t, 1].item() * (args.H - 1))
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

        out.write(frame)

    out.release()
    print(f"Saved to {args.output}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()

    # Grid
    parser.add_argument("--T", type=int, default=300)
    parser.add_argument("--H", type=int, default=256)
    parser.add_argument("--W", type=int, default=256)

    # Observers
    parser.add_argument("--n_obs", type=int, default=3, help="Number of observer trajectories")

    # Optimization
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for optimization")

    # Loss weights
    parser.add_argument("--kappa", type=float, default=0.001, help="Observer influence on spectral loss")
    parser.add_argument("--alpha", type=float, default=0.05, help="Inertial strength of observer trajectories")
    parser.add_argument("--beta", type=float, default=2.0, help="Field smoothness")

    # Entropy
    parser.add_argument("--h_start", type=float, default=0.4, help="Starting relative entropy (0-1)")
    parser.add_argument("--h_end", type=float, default=0.98, help="Ending relative entropy (0-1)")

    # Misc
    parser.add_argument("--psi_scale", type=float, default=0.01, help="Initial scale of wavefunction")
    parser.add_argument("--target_density", type=float, default=0.02, help="Target density for observer coupling")

    # Output
    parser.add_argument("--output", type=str, default="unified_sim.mp4")
    parser.add_argument("--fps", type=int, default=20)

    parser.add_argument("--print_every", type=int, default=20)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation(args)
