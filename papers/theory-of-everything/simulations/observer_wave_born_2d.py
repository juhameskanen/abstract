
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple
from simulation_engine import SimulationEngine

class ObserverWaveBorn2D(SimulationEngine):
    """
    2D observer simulation using Born-rule sampling.

    The observer is a normalized Gaussian wavefunction ψ_O.
    Microstate slices are continuous fields φ_t.

    Acceptance of updates is governed by Born weights:
        P_t = |<ψ_O | φ_t>|^2

    This models quantum-style observer sampling:
    states with higher overlap with the observer wavefunction
    dominate the measure.
    """

    def __init__(
        self,
        time_steps: int,
        size: int,
        lambda_: float,
        window: int = 1,
        sigma: float = 6.0,
        keep: int = 16
    ) -> None:
        super().__init__(time_steps, lambda_, window=window)

        self.L = size
        self.sigma = sigma
        self.keep = keep

        self.psi = self._make_wavefunction()
        self.initialize_history()

    def _make_wavefunction(self) -> np.ndarray:
        """
        Construct normalized observer wavefunction ψ_O(x,y).
        """
        x = np.arange(self.L)
        y = np.arange(self.L)
        X, Y = np.meshgrid(x, y)

        cx = self.L // 2
        cy = self.L // 2

        g = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * self.sigma**2))

        norm = np.sqrt(np.sum(np.abs(g)**2))
        return g / norm

    def initialize_history(self) -> None:
        """
        Initialize random continuous field history.
        """
        self.history = np.random.randn(self.T, self.L, self.L)

    # --- Born amplitude machinery ---

    def _amplitude(self, grid: np.ndarray) -> complex:
        """
        Compute observer amplitude <ψ_O | φ>.
        """
        return np.sum(np.conjugate(self.psi) * grid)

    def _born_weight(self, grid: np.ndarray) -> float:
        """
        Born probability weight |<ψ|φ>|^2.
        """
        amp = self._amplitude(grid)
        return float(np.abs(amp)**2 + 1e-12)

    # --- Optional: keep your compression structure ---
    # This shapes the field but does NOT determine probability directly

    def _wave_compress(self, grid: np.ndarray) -> np.ndarray:
        """
        Fourier compression used as a proposal shaping operator.
        """
        spectrum = np.fft.fft2(grid)

        flat = np.abs(spectrum).flatten()
        idx = np.argsort(flat)[::-1]

        mask = np.zeros_like(flat)
        mask[idx[:self.keep]] = 1
        mask = mask.reshape(spectrum.shape)

        compressed = spectrum * mask
        return np.real(np.fft.ifft2(compressed))

    # --- Born-rule Metropolis step ---

    def metropolis_step(self) -> None:
        """
        Born-rule Metropolis update.

        Accept updates proportional to Born probability ratio.
        """
        t = np.random.randint(0, self.T)
        i = np.random.randint(0, self.L)
        j = np.random.randint(0, self.L)

        old_grid = self.history[t].copy()
        old_w = self._born_weight(old_grid)

        delta = np.random.normal(scale=0.5)
        self.history[t, i, j] += delta

        # optional compression shaping
        self.history[t] = self._wave_compress(self.history[t])

        new_grid = self.history[t]
        new_w = self._born_weight(new_grid)

        ratio = new_w / old_w

        if ratio < 1 and np.random.rand() > ratio:
            self.history[t] = old_grid  # reject

    def observer_projection(self) -> np.ndarray:
        """
        Observer intensity |ψ * φ|.
        """
        return np.abs(self.history * self.psi)

    # --- plotting unchanged ---

    def _setup_plot(self, dpi: int):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=dpi)

        self.micro_im = axes[0].imshow(self.history[0], cmap='magma')
        self.obs_im = axes[1].imshow(self.observer_projection()[0], cmap='viridis')

        axes[0].set_title("Field Amplitude φ")
        axes[1].set_title("Observer Intensity |ψφ|")

        axes[0].axis('off')
        axes[1].axis('off')

        return fig, axes

    def _update_plot(self, frame: int):
        t = frame % self.T

        for _ in range(40):
            self.metropolis_step()

        obs = self.observer_projection()

        self.micro_im.set_data(self.history[t])
        self.obs_im.set_data(obs[t])

        return self.micro_im, self.obs_im



def main() -> None:
    parser = argparse.ArgumentParser(description="Wavefunction Observer Simulation")
    parser.add_argument("--file", type=str, default="observer_wave", help="Output filename")
    parser.add_argument("--steps", type=int, default=200, help="Animation frames")
    parser.add_argument("--time", type=int, default=100, help="History length")
    parser.add_argument("--size", type=int, default=64, help="Grid size")
    parser.add_argument("--lambda_", type=float, default=1.0, help="Gibbs parameter")
    parser.add_argument("--sigma", type=float, default=6.0, help="Observer radius")
    parser.add_argument("--keep", type=int, default=16, help="Fourier modes kept")
    parser.add_argument("--res", type=int, default=120, help="Resolution")
    parser.add_argument('--format', choices=['gif', 'mp4'], default='gif')

    args = parser.parse_args()

    sim = ObserverWaveBorn2D(
        time_steps=args.time,
        size=args.size,
        lambda_=args.lambda_,
        sigma=args.sigma,
        keep=args.keep
    )

    sim.res = args.res
    sim.animate(args.steps, args.file, args.format)

if __name__ == "__main__":
    main()
