"""
observer_gibbs_2d.py

2D Observer-Filtered Gibbs History Simulation
---------------------------------------------

This module demonstrates emergent smooth patterns in a 2D "universe"
under observer conditioning. It extends the 1D bitstring simulation to
2D lattices.

Theory Background
-----------------
- History γ = sequence of 2D binary grids: shape (T, L, L)
- Probability: P(γ | O) ∝ exp(-λ C_O[γ])
- Complexity C_O[γ] = spatiotemporal Hamming distance between consecutive
  grids + optional spatial smoothness within each grid
- Observer O sees coarse-grained block averages (spatial averaging)
- High λ favors smooth histories; the observer sees emergent patterns

Usage
-----
python observer_gibbs_2d.py --lambda_ 2.0 --steps 200 --time 100 --size 32 --window 4 --format gif
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple
from simulation_engine import SimulationEngine

class ObserverGibbs2D(SimulationEngine):
    """2D observer-Gibbs simulation with lattice histories."""

    def __init__(self, time_steps: int, size: int, lambda_: float, window: int = 1, alpha: float = 0.0) -> None:
        super().__init__(time_steps, lambda_, window=window, alpha=alpha)
        self.L = size
        self.initialize_history()

    def initialize_history(self) -> None:
        self.history = np.random.randint(0, 2, size=(self.T, self.L, self.L), dtype=np.int8)
        self.coarse_history = self.observer_projection()

    def complexity(self, t: int) -> float:
        temporal = np.sum(np.abs(self.history[t] - self.history[t-1]))
        if self.alpha > 0:
            spatial = np.sum(np.abs(self.history[t,:-1,:] - self.history[t,1:,:])) + \
                      np.sum(np.abs(self.history[t,:,:-1] - self.history[t,:,1:]))
        else:
            spatial = 0
        return temporal + self.alpha * spatial

    def metropolis_step(self) -> None:
        t = np.random.randint(1, self.T)
        i,j = np.random.randint(0,self.L), np.random.randint(0,self.L)
        before = self.complexity(t)
        self.history[t,i,j] ^= 1
        after = self.complexity(t)
        delta = after - before
        if delta > 0 and np.random.rand() > np.exp(-self.lambda_ * delta):
            self.history[t,i,j] ^= 1  # reject

    def observer_projection(self) -> np.ndarray:
        new_L = self.L // self.window
        coarse = np.zeros((self.T, new_L, new_L))
        for t in range(self.T):
            for i in range(new_L):
                for j in range(new_L):
                    block = self.history[t, i*self.window:(i+1)*self.window,
                                         j*self.window:(j+1)*self.window]
                    coarse[t,i,j] = block.mean()
        return coarse

    def _setup_plot(self, dpi: int):
        fig, axes = plt.subplots(1,2,figsize=(8,4),dpi=dpi)
        self.micro_im = axes[0].imshow(self.history[0], cmap='viridis', vmin=0, vmax=1)
        self.coarse_im = axes[1].imshow(self.coarse_history[0], cmap='viridis', vmin=0, vmax=1)
        axes[0].set_title("Microstate (LxL)"); axes[0].axis('off')
        axes[1].set_title("Observer Projection"); axes[1].axis('off')
        return fig, axes

    def _update_plot(self, frame: int):
        for _ in range(50):
            self.metropolis_step()
        self.coarse_history[:] = self.observer_projection()
        self.micro_im.set_data(self.history[frame])
        self.coarse_im.set_data(self.coarse_history[frame])
        return self.micro_im, self.coarse_im
"""
main_2d.py

Run 2D observer-Gibbs simulation and export heatmap animation.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="2D Observer-Filtered Gibbs Simulation")
    parser.add_argument("--file", type=str, default="observer_2d", help="Output filename")
    parser.add_argument("--steps", type=int, default=200, help="Animation frames")
    parser.add_argument("--time", type=int, default=100, help="History length T")
    parser.add_argument("--size", type=int, default=32, help="Grid size L")
    parser.add_argument("--lambda_", type=float, default=1.0, help="Gibbs parameter λ")
    parser.add_argument("--window", type=int, default=4, help="Observer coarse block size")
    parser.add_argument("--alpha", type=float, default=0.0, help="Spatial smoothness weight")
    parser.add_argument("--res", type=int, default=120, help="Resolution")
    parser.add_argument('--format', choices=['gif', 'mp4'], default='gif', help="Output format")
    args = parser.parse_args()

    sim = ObserverGibbs2D(time_steps=args.time, size=args.size, lambda_=args.lambda_,
                           window=args.window, alpha=args.alpha)
    sim.res = args.res
    sim.animate(steps=args.steps, filename=args.file, fmt=args.format)

if __name__ == "__main__":
    main()
