"""
Variational Emergent Spacetime (VEST) Solver
===========================================

A global variational optimization over a block-universe spacetime manifold.
This simulation synthesizes wavefunction evolution with an "Observer Filter"
based on bit-pattern matching.

Core Logic:
-----------
1. The solver finds a 3D wavefunction Ψ(t, x, y) that satisfies a resolution-
   invariant entropy trajectory (normalized from 0.0 to 1.0).
2. A simplicity bias (Fourier spectral penalty) acts as "Inertia," forcing
   the emergent structures to move and evolve smoothly.
3. The "Observer Filter" identifies specific bit-patterns, creating a
   statistical pressure for structures to crystallize where patterns emerge.

   

   
Author: Juha Meskanen
"""

import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, Dict


def build_3d_frequency_grid(T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Constructs a squared frequency grid for 3D FFT regularization.
    Used to penalize high-frequency noise (Inertia/Simplicity constraint).
    """
    kt = torch.fft.fftfreq(T).reshape(T, 1, 1).repeat(1, H, W)
    ky = torch.fft.fftfreq(H).reshape(1, H, 1).repeat(T, 1, W)
    kx = torch.fft.fftfreq(W).reshape(1, 1, W).repeat(T, H, 1)
    return (kx**2 + ky**2 + kt**2).to(device)


def get_observer_target(width: int, height: int, pattern: int, bits_per_pixel: int = 8) -> torch.Tensor:
    """
    Creates a spatial mask representing the 'Observer Filter'.
    Pixels matching the bit-pattern provide a target for structural emergence.
    """
    np.random.seed(pattern)
    # Generate a pseudo-random bit-field
    bit_field = np.random.randint(0, 2**bits_per_pixel, (height, width))
    mask = (bit_field == pattern).astype(np.float32)
    return torch.from_numpy(mask)


def compute_shannon_entropy(prob_dist: torch.Tensor) -> torch.Tensor:
    """
    Computes Shannon entropy of a probability distribution.
    """
    p = torch.clamp(prob_dist, min=1e-12)
    return -torch.sum(p * torch.log(p))


def solve_spacetime(
    width: int,
    height: int,
    time_steps: int,
    iterations: int,
    h_start_frac: float,
    h_end_frac: float,
    pattern: int,
    lmbda: float,
    kappa: float
) -> torch.Tensor:
    """
    Variational solver that sculpts the spacetime block.
    
    Args:
        width, height: Spatial resolution.
        time_steps: Temporal resolution.
        iterations: Optimization steps.
        h_start_frac, h_end_frac: Relative entropy targets (0.0 to 1.0).
        pattern: The observer's target bit-pattern.
        lmbda: Coupling strength of the observer filter.
        kappa: Simplicity weight (vacuum viscosity).

    Returns:
        A tensor representing the optimized wavefunction Ψ(t, x, y). 

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_sq = build_3d_frequency_grid(time_steps, height, width, device)
    observer_mask = get_observer_target(width, height, pattern).to(device)

    # Initialize Wavefunction (Real, Imaginary components)
    psi = 0.02 * torch.randn((time_steps, height, width, 2), device=device)
    # Start with a low-entropy seed in the center
    psi[0, height // 2, width // 2, 0] = 1.0
    psi.requires_grad_()

    optimizer = torch.optim.Adam([psi], lr=0.01)
    h_max = np.log(width * height)

    print(f"Running Variational Solver (Device: {device})...")

    for step in range(iterations):
        optimizer.zero_grad()
        psi_c = torch.complex(psi[..., 0], psi[..., 1])

        # 1. Simplicity Prior (Fourier Smoothing / Inertia)
        psi_k = torch.fft.fftn(psi_c, dim=(0, 1, 2))
        complexity_loss = torch.sum(k_sq * torch.abs(psi_k)**2)

        # 2. Entropy and Observer Constraints
        total_constraint_loss = 0.0
        
        for t in range(time_steps):
            prob = torch.abs(psi_c[t])**2
            p_sum = torch.sum(prob) + 1e-12
            p_norm = prob / p_sum

            # Resolution-invariant entropy target
            progress = t / (time_steps - 1)
            target_h = (h_start_frac + (h_end_frac - h_start_frac) * progress) * h_max
            
            current_h = compute_shannon_entropy(p_norm)
            
            # Observer pressure: force density toward the pattern-match mask
            observer_loss = torch.mean((prob - (observer_mask * progress))**2)
            
            total_constraint_loss += (current_h - target_h)**2 + lmbda * observer_loss

        # Total Loss: Simplicity vs. Constraints
        loss = kappa * complexity_loss + total_constraint_loss
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"Step {step:03d} | Total Loss: {loss.item():.4f}")

    return psi.detach()


def save_visual_output(
    wavefunction: torch.Tensor, 
    filename: str, 
    colormap_name: str
) -> None:
    """
    Renders the wavefunction's probability density to a video file.
    """
    psi_c = torch.complex(wavefunction[..., 0], wavefunction[..., 1])
    T, H, W = psi_c.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30, (W, H))

    cmaps: Dict[str, int] = {
        "jet": cv2.COLORMAP_JET,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "magma": cv2.COLORMAP_MAGMA,
        "hot": cv2.COLORMAP_HOT
    }
    color_code = cmaps.get(colormap_name, cv2.COLORMAP_MAGMA)

    print(f"Rendering video to {filename}...")
    for t in range(T):
        density = torch.abs(psi_c[t])**2
        density = (density / (density.max() + 1e-12)).cpu().numpy()
        frame = (density * 255).astype(np.uint8)
        color_frame = cv2.applyColorMap(frame, color_code)
        out.write(color_frame)
    
    out.release()


def main():
    parser = argparse.ArgumentParser(description="Variational Emergent Spacetime")
    parser.add_argument("--lmbda", type=float, default=3.0, 
                    help="Coupling strength of the observer filter (λ)")
    parser.add_argument("--kappa", type=float, default=0.0005, help="Simplicity Weight (Vacuum Viscosity)")
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--time_steps", type=int, default=350)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--pattern", type=int, default=0b0010, help="Observer bit-pattern filter")
    parser.add_argument("--h_start", type=float, default=0.1, help="Starting relative entropy (0-1)")
    parser.add_argument("--h_end", type=float, default=0.97, help="Ending relative entropy (0-1)")
    parser.add_argument("--colormap", type=str, default="magma", choices=["jet", "viridis", "magma", "hot"])
    parser.add_argument("--output", type=str, default="spacetime_evolution.mp4")

    args = parser.parse_args()

    # 1. Optimize the spacetime block
    wf = solve_spacetime(
        args.width, args.height, args.time_steps, 
        args.iterations, args.h_start, args.h_end, args.pattern, args.lmbda, args.kappa
    )

    # 2. Render to video
    save_visual_output(wf, args.output, args.colormap)
    print("Execution complete.")


if __name__ == "__main__":
    main()