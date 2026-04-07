"""
Spacetime Entropy-Constrained Wavefunction Solver
=================================================

A global variational optimization over spacetime, consistent with a block-universe perspective.
Updated with color gradient support.

Author: Juha Meskanen (Modifications by Gemini)
"""

from typing import Tuple

import argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


def build_3d_frequency_grid(T: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    kt = torch.fft.fftfreq(T).reshape(T, 1, 1).repeat(1, H, W)
    ky = torch.fft.fftfreq(H).reshape(1, H, 1).repeat(T, 1, W)
    kx = torch.fft.fftfreq(W).reshape(1, 1, W).repeat(T, H, 1)
    return (kx**2 + ky**2 + kt**2).to(device)


def compute_entropy(prob: torch.Tensor) -> torch.Tensor:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k_sq_3d = build_3d_frequency_grid(time_steps, height, width, device)

    psi = 0.01 * torch.randn((time_steps, height, width, 2), device=device)
    psi[0, height // 2, width // 2, 0] = 1.0
    psi.requires_grad_()

    optimizer = torch.optim.Adam([psi], lr=0.002)
    H_max = np.log(width * height)

    for step in range(iterations):
        optimizer.zero_grad()
        psi_complex = torch.complex(psi[..., 0], psi[..., 1])

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


def save_video(wavefunction: torch.Tensor, filename: str = "output.mp4", colormap: str = "gray") -> None:
    """
    Save probability density evolution as a video with selectable color gradients.
    """
    psi_complex = torch.complex(wavefunction[..., 0], wavefunction[..., 1])
    T, H, W = psi_complex.shape

    # Define Colormap Mapping
    cmaps = {
        "jet": cv2.COLORMAP_JET,
        "magma": cv2.COLORMAP_MAGMA,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "hot": cv2.COLORMAP_HOT,
        "cool": cv2.COLORMAP_COOL
    }

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Gray is single channel, others are 3-channel BGR
    is_color = colormap != "gray"
    writer = cv2.VideoWriter(filename, fourcc, 30, (W, H), isColor=is_color)

    for t in range(T):
        frame = torch.abs(psi_complex[t]) ** 2
        frame = frame.cpu().numpy()
        
        # Max normalization for visibility
        f_max = frame.max()
        if f_max > 0:
            frame = (frame / f_max * 255).astype(np.uint8)
        else:
            frame = np.zeros((H, W), dtype=np.uint8)

        if colormap == "gray":
            writer.write(frame)
        elif colormap == "stripes":
            # Procedural 'stripes' effect based on intensity mod
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_PARULA)
            frame[frame % 40 < 10] = 0 # Artificial interference stripes
            writer.write(frame)
        elif colormap in cmaps:
            color_frame = cv2.applyColorMap(frame, cmaps[colormap])
            writer.write(color_frame)
        else:
            # Fallback to gray if colormap name is unknown
            writer.write(frame)

    writer.release()
    print(f"Saved video to {filename} with colormap: {colormap}")


def compute_diagnostics(
    wavefunction: torch.Tensor,
    entropy_start_fraction: float,
    entropy_end_fraction: float,
    entropy_power: float,
) -> Tuple[np.ndarray, np.ndarray]:
    psi_complex = torch.complex(wavefunction[..., 0], wavefunction[..., 1])
    T, H, W = psi_complex.shape
    entropies, targets = [], []
    H_max = np.log(H * W)

    for t in range(T):
        psi_t = psi_complex[t]
        amp = torch.abs(psi_t) ** 2 + 1e-12
        log_amp = torch.log(amp)
        progress = t / (T - 1)
        beta = 0.5 + 3.0 * (progress ** entropy_power)
        prob = torch.softmax(beta * log_amp.flatten(), dim=0).reshape(H, W)
        entropy = compute_entropy(prob).item()
        entropies.append(entropy)
        H_start = entropy_start_fraction * H_max
        H_end = entropy_end_fraction * H_max
        target_h = H_start + (H_end - H_start) * (1 - np.exp(-5 * progress))
        targets.append(target_h)

    return np.array(entropies), np.array(targets)


def plot_diagnostics(entropies: np.ndarray, targets: np.ndarray) -> None:
    plt.figure()
    plt.plot(entropies, label="Actual")
    plt.plot(targets, "--", label="Target")
    plt.legend()
    plt.title("Entropy Evolution")
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--time_steps", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--entropy_start_fraction", type=float, default=0.5)
    parser.add_argument("--entropy_end_fraction", type=float, default=0.97)
    parser.add_argument("--entropy_power", type=float, default=2.0)
    parser.add_argument("--output", type=str, default="output_color.mp4")
    parser.add_argument("--colormap", type=str, default="jet", 
                        choices=["gray", "jet", "magma", "viridis", "hot", "cool", "stripes"],
                        help="Select the color gradient for the output video.")

    args = parser.parse_args()

    wavefunction = generate_spacetime_wavefunction(
        args.width, args.height, args.time_steps,
        args.iterations, args.entropy_start_fraction,
        args.entropy_end_fraction, args.entropy_power
    )

    save_video(wavefunction, args.output, colormap=args.colormap)

    entropies, targets = compute_diagnostics(
        wavefunction, args.entropy_start_fraction,
        args.entropy_end_fraction, args.entropy_power
    )
    plot_diagnostics(entropies, targets)


if __name__ == "__main__":
    main()
