"""
Spacetime Entropy-Constrained Wavefunction Solver
================================================

This module implements a variational solver that constructs a 3D spacetime
wavefunction ψ(t, x, y) whose induced probability distributions follow a
prescribed entropy trajectory over time.

Core idea
---------
We optimize a complex-valued field ψ such that:

1. Smoothness constraint:
   The wavefunction is penalized in Fourier space to suppress high-frequency
   components (spacetime regularity).

2. Entropy constraint:
   At each time slice t, the probability distribution

       p(x, y) ∝ |ψ(t, x, y)|²

   is transformed via a temperature-like parameter β(t), and its Shannon entropy
   is forced to follow a target curve:

       H(t) ≈ H_start + (H_end - H_start) * (1 - exp(-k t))

This produces a global spacetime configuration consistent with a prescribed
entropy evolution.

Interpretation
--------------
This is NOT a forward time simulation. Instead, it is a global variational
optimization over spacetime, consistent with a block-universe perspective.

Key properties
--------------
- Numerically stable (log-domain softmax)
- Explicit entropy control via β(t)
- Smooth spacetime via spectral regularization
- Flexible entropy trajectories

Outputs
-------
- Optimized wavefunction ψ(t, x, y)
- Entropy diagnostics vs. target
- Optional visualization as video

Author: (your name)
"""

from typing import Tuple

import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


def build_3d_frequency_grid(T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """
    Construct squared frequency grid for 3D FFT regularization.

    Args:
        T: Number of time steps
        H: Height
        W: Width
        device: Torch device

    Returns:
        Tensor of shape (T, H, W) containing k^2 values.
    """
    kt = torch.fft.fftfreq(T).reshape(T, 1, 1).repeat(1, H, W)
    ky = torch.fft.fftfreq(H).reshape(1, H, 1).repeat(T, 1, W)
    kx = torch.fft.fftfreq(W).reshape(1, 1, W).repeat(T, H, 1)
    return (kx**2 + ky**2 + kt**2).to(device)


def compute_entropy(prob: torch.Tensor) -> torch.Tensor:
    """
    Compute Shannon entropy of a probability distribution.

    Args:
        prob: Probability tensor (must sum to 1)

    Returns:
        Scalar entropy value.
    """
    prob = torch.clamp(prob, min=1e-12)
    return -torch.sum(prob * torch.log(prob))


def generate_spacetime_wavefunction(
    width: int,
    height: int,
    time_steps: int,
    iterations: int,
    entropy_start_fraction: float,
    entropy_end_fraction: float,
    entropy_power: float,
) -> torch.Tensor:
    """
    Optimize a spacetime wavefunction under entropy and smoothness constraints.

    Args:
        width: Spatial width
        height: Spatial height
        time_steps: Number of time slices
        iterations: Optimization steps
        entropy_start_fraction: Initial entropy as fraction of maximum
        entropy_end_fraction: Final entropy as fraction of maximum
        entropy_power: Controls curvature of entropy evolution

    Returns:
        Optimized wavefunction tensor of shape (T, H, W, 2)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    k_sq_3d = build_3d_frequency_grid(time_steps, height, width, device)

    # Stable low-entropy-biased initialization
    psi = 0.01 * torch.randn((time_steps, height, width, 2), device=device)
    psi[0, height // 2, width // 2, 0] = 1.0
    psi.requires_grad_()

    optimizer = torch.optim.Adam([psi], lr=0.002)

    H_max = np.log(width * height)

    for step in range(iterations):
        optimizer.zero_grad()

        psi_complex = torch.complex(psi[..., 0], psi[..., 1])

        # --- Smoothness term ---
        psi_k = torch.fft.fftn(psi_complex, dim=(0, 1, 2))
        complexity = torch.sum(k_sq_3d * torch.abs(psi_k) ** 2)

        entropy_loss = 0.0

        for t in range(time_steps):
            psi_t = psi_complex[t]

            amp = torch.abs(psi_t) ** 2 + 1e-12
            log_amp = torch.log(amp)

            progress = t / (time_steps - 1)
            beta = 0.5 + 3.0 * (progress ** entropy_power)

            prob = torch.softmax(beta * log_amp.flatten(), dim=0)
            prob = prob.reshape(height, width)

            entropy = compute_entropy(prob)

            H_start = entropy_start_fraction * H_max
            H_end = entropy_end_fraction * H_max
            target_h = H_start + (H_end - H_start) * (1 - np.exp(-5 * progress))

            entropy_loss += (entropy - target_h) ** 2

        loss = 0.001 * complexity + 2.0 * entropy_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_([psi], max_norm=1.0)
        optimizer.step()

        if step % 50 == 0:
            print(f"Iteration {step}, Loss: {loss.item():.4f}")

    return psi.detach()


def save_video(wavefunction: torch.Tensor, filename: str = "output.mp4") -> None:
    """
    Save probability density evolution as a grayscale video.

    Args:
        wavefunction: Tensor (T, H, W, 2)
        filename: Output file path
    """
    psi_complex = torch.complex(wavefunction[..., 0], wavefunction[..., 1])
    T, H, W = psi_complex.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(filename, fourcc, 30, (W, H), isColor=False)

    for t in range(T):
        frame = torch.abs(psi_complex[t]) ** 2
        frame = frame.cpu().numpy()
        frame = frame / (frame.max() + 1e-12)
        frame = (frame * 255).astype(np.uint8)
        writer.write(frame)

    writer.release()
    print(f"Saved video to {filename}")


def compute_diagnostics(
    wavefunction: torch.Tensor,
    entropy_start_fraction: float,
    entropy_end_fraction: float,
    entropy_power: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute actual vs target entropy curves.

    Returns:
        Tuple of (actual_entropies, target_entropies)
    """
    psi_complex = torch.complex(wavefunction[..., 0], wavefunction[..., 1])
    T, H, W = psi_complex.shape

    entropies = []
    targets = []

    H_max = np.log(H * W)

    for t in range(T):
        psi_t = psi_complex[t]

        amp = torch.abs(psi_t) ** 2 + 1e-12
        log_amp = torch.log(amp)

        progress = t / (T - 1)
        beta = 0.5 + 3.0 * (progress ** entropy_power)

        prob = torch.softmax(beta * log_amp.flatten(), dim=0)
        prob = prob.reshape(H, W)

        entropy = compute_entropy(prob).item()
        entropies.append(entropy)

        H_start = entropy_start_fraction * H_max
        H_end = entropy_end_fraction * H_max
        target_h = H_start + (H_end - H_start) * (1 - np.exp(-5 * progress))

        targets.append(target_h)

    return np.array(entropies), np.array(targets)


def plot_diagnostics(entropies: np.ndarray, targets: np.ndarray) -> None:
    """
    Plot entropy evolution against target curve.
    """
    plt.figure()
    plt.plot(entropies, label="Actual")
    plt.plot(targets, "--", label="Target")
    plt.legend()
    plt.title("Entropy Evolution")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--time_steps", type=int, default=250)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--entropy_start_fraction", type=float, default=0.5)
    parser.add_argument("--entropy_end_fraction", type=float, default=0.95)
    parser.add_argument("--entropy_power", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()

    wavefunction = generate_spacetime_wavefunction(
        args.width,
        args.height,
        args.time_steps,
        args.iterations,
        args.entropy_start_fraction,
        args.entropy_end_fraction,
        args.entropy_power,
    )

    save_video(wavefunction, args.output)

    entropies, targets = compute_diagnostics(
        wavefunction,
        args.entropy_start_fraction,
        args.entropy_end_fraction,
        args.entropy_power,
    )

    plot_diagnostics(entropies, targets)


if __name__ == "__main__":
    main()