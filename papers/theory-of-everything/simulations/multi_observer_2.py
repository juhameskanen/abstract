import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple

# -----------------------------
# Gaussian observer wavefunction
# -----------------------------
def make_gaussian(L: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    """Construct a normalized 2D Gaussian centered at (cx, cy)."""
    x = np.arange(L)
    y = np.arange(L)
    X, Y = np.meshgrid(x, y)
    g = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    return g / np.sqrt(np.sum(np.abs(g)**2) + 1e-12)


# -----------------------------
# Multi-observer Born rule simulation
# -----------------------------
class MultiObserverBornSim:
    """
    Universe wavefunction Ψ with multiple Gaussian observers ψ_i.
    Updates follow joint Born probability: P = Π_i |<ψ_i | Ψ>|^2.
    Overlapping observers naturally dominate the measure, producing emergent compression/merger.
    Tracks mean inter-observer distance as a measure of merger over time.
    """

    def __init__(self, L: int = 32, T: int = 100, n_obs: int = 3, sigma: float = 5.0):
        """
        Initialize the universe field and observer wavefunctions.

        Args:
            L (int): Grid size (L x L)
            T (int): Number of time slices
            n_obs (int): Number of observer components
            sigma (float): Gaussian width of observers
        """
        self.L = L
        self.T = T
        self.n_obs = n_obs
        self.sigma = sigma

        # Universe wavefunction: small random field
        self.Psi = np.random.randn(T, L, L) * 0.5

        # Observer wavefunctions: Gaussian blobs, fixed positions
        self.centers: List[Tuple[float, float]] = [
            (np.random.uniform(0, L), np.random.uniform(0, L))
            for _ in range(n_obs)
        ]
        self.psis: List[np.ndarray] = [make_gaussian(L, cx, cy, sigma)
                                       for cx, cy in self.centers]

        # Tracking mean inter-observer distance over frames
        self.mean_distances: List[float] = []

    # -----------------------------
    # Born amplitude and probability
    # -----------------------------
    def amplitude(self, grid: np.ndarray, psi: np.ndarray) -> complex:
        """
        Compute the complex amplitude of psi in the universe field.

        Args:
            grid (np.ndarray): Universe field Ψ[t]
            psi (np.ndarray): Observer wavefunction

        Returns:
            complex: inner product <ψ | Ψ>
        """
        return np.sum(np.conjugate(psi) * grid)

    def born_weight(self, grid: np.ndarray) -> float:
        """
        Compute the joint Born probability over all observers.

        Args:
            grid (np.ndarray): Universe field Ψ[t]

        Returns:
            float: Joint probability P = Π_i |<ψ_i | Ψ>|^2
        """
        w = 1.0
        for psi in self.psis:
            amp = self.amplitude(grid, psi)
            w *= (np.abs(amp)**2 + 1e-12)
        return float(w)

    # -----------------------------
    # Metropolis update of the universe
    # -----------------------------
    def metropolis_step(self) -> None:
        """
        Perform a Metropolis-style update on a random element of Ψ.
        Acceptance is determined by the joint Born probability.
        """
        t = np.random.randint(0, self.T)
        i = np.random.randint(0, self.L)
        j = np.random.randint(0, self.L)

        old_grid = self.Psi[t].copy()
        old_w = self.born_weight(old_grid)

        delta = np.random.normal(scale=0.5)
        self.Psi[t, i, j] += delta

        new_w = self.born_weight(self.Psi[t])
        ratio = new_w / old_w

        if ratio < 1 and np.random.rand() > ratio:
            # Reject update
            self.Psi[t, i, j] = old_grid[i, j]

    # -----------------------------
    # Observer-projected intensity
    # -----------------------------
    def observer_intensity(self, t: int) -> np.ndarray:
        """
        Compute the observer-projected intensity at time t.

        Args:
            t (int): Time slice index

        Returns:
            np.ndarray: |Σ_i ψ_i * Ψ[t]|
        """
        return np.abs(sum(psi * self.Psi[t] for psi in self.psis))

    # -----------------------------
    # Inter-observer distance metrics
    # -----------------------------
    def observer_distances(self, t: int) -> np.ndarray:
        """
        Compute pairwise overlap distances between observers at time t.

        Args:
            t (int): Time slice index

        Returns:
            np.ndarray: n_obs x n_obs symmetric matrix of distances
        """
        n = self.n_obs
        D = np.zeros((n, n))
        amplitudes = [self.amplitude(self.Psi[t], psi) for psi in self.psis]

        for i in range(n):
            for j in range(i + 1, n):
                ai, aj = amplitudes[i], amplitudes[j]
                norm = np.sqrt(np.abs(ai)**2 * np.abs(aj)**2) + 1e-12
                D[i, j] = 1 - np.abs(ai * np.conj(aj)) / norm
                D[j, i] = D[i, j]
        return D

    def mean_observer_distance(self, t: int) -> float:
        """
        Compute mean pairwise observer distance at time t.

        Args:
            t (int): Time slice index

        Returns:
            float: Mean distance over all observer pairs
        """
        D = self.observer_distances(t)
        n = self.n_obs
        return np.sum(np.triu(D, k=1)) / (n * (n - 1) / 2)

    # -----------------------------
    # Animation
    # -----------------------------
    def animate(self, steps: int = 200, res: int = 120, filename: str = "multi_observer_born.gif"):
        """
        Animate the universe and observer intensities, tracking mean distance.

        Args:
            steps (int): Number of animation frames
            res (int): DPI / resolution
            filename (str): Output filename (gif)
        """
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=res)
        micro_im = axes[0].imshow(self.Psi[0], cmap="magma")
        obs_im = axes[1].imshow(self.observer_intensity(0), cmap="viridis")

        axes[0].set_title("Universe Ψ")
        axes[1].set_title("Observer Intensity")
        axes[0].axis("off")
        axes[1].axis("off")

        self.mean_distances = []

        def update(frame: int):
            for _ in range(40):
                self.metropolis_step()
            t = frame % self.T
            micro_im.set_data(self.Psi[t])
            obs_im.set_data(self.observer_intensity(t))
            # track mean distance
            self.mean_distances.append(self.mean_observer_distance(t))
            return micro_im, obs_im

        ani = animation.FuncAnimation(fig, update, frames=steps, blit=True)
        ani.save(filename, writer="pillow", fps=12)
        print(f"Saved animation to {filename}")

        # Plot mean observer distance over frames
        plt.figure()
        plt.plot(self.mean_distances)
        plt.xlabel("Frame")
        plt.ylabel("Mean observer distance")
        plt.title("Emergent Observer Merger")
        plt.show()


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-observer Born-rule PoC simulation")
    parser.add_argument("--steps", type=int, default=400, help="Number of animation frames")
    parser.add_argument("--T", type=int, default=100, help="Number of time slices in the universe field")
    parser.add_argument("--size", type=int, default=64, help="Spatial size of the universe grid (L x L)")
    parser.add_argument("--obs", type=int, default=3, help="Number of Gaussian observer components")
    parser.add_argument("--sigma", type=float, default=10.0, help="Gaussian width of observers")
    parser.add_argument("--file", type=str, default="multi_observer_born.gif", help="Output filename (GIF)")
    args = parser.parse_args()

    sim = MultiObserverBornSim(L=args.size, T=args.T, n_obs=args.obs, sigma=args.sigma)
    sim.animate(steps=args.steps, filename=args.file)


if __name__ == "__main__":
    main()
