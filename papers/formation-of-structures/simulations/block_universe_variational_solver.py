"""
Spacetime Variational Wavefunction Solver (Full 3D Spectral Model)
=================================================================

This module implements a *true block-universe variational solver* where
space and time are treated symmetrically.

The wavefunction Ψ(x, y, t) is optimized globally using a *spacetime spectral
complexity functional*:

    C = sum_{kx, ky, kt} (kx^2 + ky^2 + kt^2) |Ψ(kx, ky, kt)|^2

Key properties:
    - No explicit temporal smoothness term
    - Temporal continuity emerges from spectral penalty in k_t
    - Fully consistent with spectral MDL principle

The solver minimizes:
    - Spacetime spectral complexity
    - Entropy trajectory constraint

Usage:
    python spacetime_variational_solver.py --width 128 --height 128 --time_steps 50 --iterations 500

Author:
    Juha Meskanen
"""

import argparse
import numpy as np
import torch
import cv2


def build_3d_frequency_grid(T: int, H: int, W: int, device: torch.device):
    """Constructs full spacetime frequency grid.

    Args:
        T (int): Time dimension.
        H (int): Height.
        W (int): Width.
        device (torch.device): Device.

    Returns:
        torch.Tensor: Squared frequency magnitude (T, H, W).
    """
    kt = torch.fft.fftfreq(T).reshape(T, 1, 1).repeat(1, H, W)
    ky = torch.fft.fftfreq(H).reshape(1, H, 1).repeat(T, 1, W)
    kx = torch.fft.fftfreq(W).reshape(1, 1, W).repeat(T, H, 1)

    return (kx**2 + ky**2 + kt**2).to(device)


def compute_entropy(prob: torch.Tensor) -> torch.Tensor:
    """Computes Shannon entropy.

    Args:
        prob (torch.Tensor): Probability distribution.

    Returns:
        torch.Tensor: Entropy.
    """
    return -torch.sum(prob * torch.log(prob + 1e-12))


def generate_spacetime_wavefunction(width=128, height=128, time_steps=50, iterations=500):
    """Optimizes full spacetime wavefunction Ψ(x,y,t).

    Args:
        width (int): Spatial width.
        height (int): Spatial height.
        time_steps (int): Number of time steps.
        iterations (int): Optimization steps.

    Returns:
        torch.Tensor: Optimized tensor (T, 2, H, W).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Full 3D spectral metric
    k_sq_3d = build_3d_frequency_grid(time_steps, height, width, device)

    # Learn Ψ(x,y,t) in spatial domain directly
    psi = torch.randn((time_steps, height, width, 2), device=device, requires_grad=True)

    optimizer = torch.optim.Adam([psi], lr=0.01)

    for step in range(iterations):
        optimizer.zero_grad()

        psi_complex = torch.complex(psi[..., 0], psi[..., 1])

        # --- FULL 3D FFT ---
        psi_k = torch.fft.fftn(psi_complex, dim=(0, 1, 2))

        # --- Spacetime Spectral Complexity ---
        complexity = torch.sum(k_sq_3d * torch.abs(psi_k)**2)

        # --- Entropy Constraint (per time slice) ---
        entropy_loss = 0.0

        for t in range(time_steps):
            psi_t = psi_complex[t]
            prob = torch.abs(psi_t)**2
            prob = prob / (torch.sum(prob) + 1e-12)

            target_h = (t / (time_steps - 1)) * np.log(width * height)
            entropy = compute_entropy(prob)

            entropy_loss += (entropy - target_h) ** 2

        # --- Total Loss ---
        loss = 0.05 * complexity + entropy_loss

        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Iteration {step}, Loss: {loss.item():.4f}")

    return psi.detach()


def save_video(wavefunction, filename="genesis_spectral_v3.mp4"):
    """Saves spacetime slices as video.

    Args:
        wavefunction (torch.Tensor): Tensor (T,H,W,2).
        filename (str): Output filename.
    """
    psi_complex = torch.complex(wavefunction[..., 0], wavefunction[..., 1])

    T, H, W = psi_complex.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(filename, fourcc, 30, (W, H), isColor=False)

    for t in range(T):
        frame = torch.abs(psi_complex[t])**2
        frame = frame.cpu().numpy()
        frame = frame / (frame.max() + 1e-12)
        frame = (frame * 255).astype(np.uint8)

        writer.write(frame)

    writer.release()
    print(f"Saved video to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Spacetime Variational Solver")

    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--time_steps", type=int, default=150)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--output", type=str, default="genesis_spectral_v3.mp4")

    args = parser.parse_args()

    wavefunction = generate_spacetime_wavefunction(
        width=args.width,
        height=args.height,
        time_steps=args.time_steps,
        iterations=args.iterations,
    )

    save_video(wavefunction, args.output)


if __name__ == "__main__":
    main()
