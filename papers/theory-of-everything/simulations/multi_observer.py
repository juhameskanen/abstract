import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple

# -----------------------------
# Helper: Gaussian observer component
# -----------------------------
def make_gaussian(L: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    x = np.arange(L)
    y = np.arange(L)
    X, Y = np.meshgrid(x, y)
    g = np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2)
               - ((Y - cy)**2 + (X - cx)**2) / (2 * sigma**2))
    return g / np.sqrt(np.sum(g**2) + 1e-12)


# -----------------------------
# PoC simulation class
# -----------------------------
class MultiObserverCompressionSim:
    """
    Universe wavefunction Ψ with emergent Gaussian observers.
    Observers merge if shared components reduce total compression.
    """

    def __init__(self, L: int = 32, T: int = 100, n_obs: int = 3, sigma: float = 5.0):
        self.L = L
        self.T = T
        self.n_obs = n_obs
        self.sigma = sigma

        # initialize universe wavefunction: small random field
        self.Psi = np.random.randn(T, L, L) * 0.5

        # candidate observer components: random centers
        self.centers: List[Tuple[float, float]] = [
            (np.random.uniform(0, L), np.random.uniform(0, L))
            for _ in range(n_obs)
        ]
        self.psis: List[np.ndarray] = [make_gaussian(L, cx, cy, sigma)
                                       for cx, cy in self.centers]

    def compress_error(self, t: int) -> float:
        """
        Compute reconstruction error of Psi[t] using observer components.
        """
        approx = np.zeros_like(self.Psi[t])
        for psi in self.psis:
            # project Psi[t] onto psi
            coeff = np.sum(self.Psi[t] * psi)
            approx += coeff * psi
        residual = self.Psi[t] - approx
        return np.mean(residual**2)

    def metropolis_step(self) -> None:
        """
        Metropolis-style update:
        - small random change to Ψ
        - accept if compression improves or probabilistically
        """
        t = np.random.randint(0, self.T)
        i = np.random.randint(0, self.L)
        j = np.random.randint(0, self.L)

        old_error = self.compress_error(t)
        delta = np.random.normal(scale=0.5)
        self.Psi[t, i, j] += delta
        new_error = self.compress_error(t)

        if new_error > old_error and np.random.rand() > np.exp(-(old_error - new_error)):
            # reject
            self.Psi[t, i, j] -= delta

    def update_components(self) -> None:
        """
        Emergent component merging:
        - merge similar Gaussian components to reduce reconstruction error
        """
        merged: List[np.ndarray] = []
        keep_centers: List[Tuple[float, float]] = []

        skip = set()
        for i, psi_i in enumerate(self.psis):
            if i in skip:
                continue
            new_psi = psi_i.copy()
            cx_i, cy_i = self.centers[i]
            for j, psi_j in enumerate(self.psis):
                if j <= i or j in skip:
                    continue
                # if overlap is high, merge
                overlap = np.sum(psi_i * psi_j)
                if overlap > 0.1:  # threshold
                    new_psi += psi_j
                    skip.add(j)
            new_psi /= np.sqrt(np.sum(new_psi**2) + 1e-12)
            merged.append(new_psi)
            keep_centers.append((cx_i, cy_i))
        self.psis = merged
        self.centers = keep_centers

    def animate(self, steps: int = 200, res: int = 120, filename: str = "multi_observer.gif"):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=res)
        micro_im = axes[0].imshow(self.Psi[0], cmap="magma")
        # observer intensity = sum of |ψ_i * Ψ|
        obs_im = axes[1].imshow(np.abs(sum(psi * self.Psi[0] for psi in self.psis)), cmap="viridis")

        axes[0].set_title("Universe Ψ")
        axes[1].set_title("Observer Intensity")
        axes[0].axis("off")
        axes[1].axis("off")

        def update(frame: int):
            for _ in range(50):
                self.metropolis_step()
            self.update_components()
            t = frame % self.T
            micro_im.set_data(self.Psi[t])
            obs_intensity = np.abs(sum(psi * self.Psi[t] for psi in self.psis))
            obs_im.set_data(obs_intensity)
            return micro_im, obs_im

        ani = animation.FuncAnimation(fig, update, frames=steps, blit=True)
        ani.save(filename, writer="pillow", fps=12)
        print(f"Saved animation to {filename}")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--obs", type=int, default=3)
    parser.add_argument("--sigma", type=float, default=15.0)
    parser.add_argument("--file", type=str, default="multi_observer.gif")
    args = parser.parse_args()

    sim = MultiObserverCompressionSim(L=args.size, T=args.T, n_obs=args.obs, sigma=args.sigma)
    sim.animate(steps=args.steps, filename=args.file)

if __name__ == "__main__":
    main()
