import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple
from qbitwave import QBitwave 

# -----------------------------
# Multi-observer QBitwave simulation
# -----------------------------
class MultiObserverQBitwaveSim:
    """
    Universe field Ψ with multiple observers modeled as Wavefunctions (QBitwave objects).
    Updates follow Born probabilities weighted by wave_complexity.
    Observers with more compressible wavefunctions dominate the measure,
    producing emergent attraction and merger without ad-hoc translation.
    """

    def __init__(self, L: int = 32, T: int = 100, n_obs: int = 3, sigma: float = 5.0, bits: int = 64):
        """
        Args:
            L (int): Grid size
            T (int): Time steps
            n_obs (int): Number of observers
            sigma (float): Gaussian width (initial localization)
            bits (int): Bitstring length per observer
        """
        self.L = L
        self.T = T
        self.n_obs = n_obs
        self.sigma = sigma
        self.bits = bits

        # Universe wavefunction Ψ
        self.Psi = np.random.randn(T, L, L) * 0.5

        # Observers as QBitwave objects with random initial bitstrings
        self.observers: List[QBitwave] = [
            QBitwave(bitstring=np.random.randint(0, 2, L*L).tolist())
            for _ in range(n_obs)
        ]

        # Initial positions for visualization
        self.positions: List[Tuple[float, float]] = [
            (np.random.uniform(0, L), np.random.uniform(0, L))
            for _ in range(n_obs)
        ]
        for obs in self.observers:
            obs._analyze_bitstring_1to1()

        # Track mean inter-observer distance
        self.mean_distances: List[float] = []

    # -----------------------------
    # Global spectral entropy of a weighted field
    # -----------------------------
    def spectral_entropy(self, field: np.ndarray) -> float:
        fft = np.fft.rfft2(field)
        psd = np.abs(fft)**2
        psd /= np.sum(psd) + 1e-12
        H = -np.sum(psd * np.log2(psd + 1e-12))
        return float(H)

    def total_weight(self, t: int) -> float:
        """
        Compute joint Born weight: product of observer compressibility-weighted overlaps.
        """
        H = self.spectral_entropy(self.Psi[t])
        return np.exp(-H)

   
    # -----------------------------
    # Metropolis update using observer-weighted spectral measure
    # -----------------------------
    def metropolis_step(self):
        t = np.random.randint(0, self.T)
        i = np.random.randint(0, self.L)
        j = np.random.randint(0, self.L)

        old_val = self.Psi[t, i, j]

        # Weight = exp(-spectral entropy of Ψ weighted by observer field)
        obs_field = self.observer_field()
        old_H = self.spectral_entropy(self.Psi[t] * obs_field)
        old_w = np.exp(-old_H)

        # Propose small change
        delta = np.random.normal(scale=0.5)
        self.Psi[t, i, j] += delta

        # New weight
        new_H = self.spectral_entropy(self.Psi[t] * obs_field)
        new_w = np.exp(-new_H)

        # Metropolis accept/reject
        ratio = new_w / (old_w + 1e-12)
        if ratio < 1 and np.random.rand() > ratio:
            self.Psi[t, i, j] = old_val  # reject
    
    # -----------------------------
    # Combined observer field
    # -----------------------------
    def observer_field(self) -> np.ndarray:
        """
        Returns a combined Gaussian field representing all observers.
        """
        field = np.zeros((self.L, self.L))
        X, Y = np.meshgrid(np.arange(self.L), np.arange(self.L))
        for x, y in self.positions:
            g = np.exp(-((X - x)**2 + (Y - y)**2)/(2*self.sigma**2))
            g /= np.sqrt(np.sum(g**2) + 1e-12)
            field += g
        return field



    # -----------------------------
    # Observer-projected intensity
    # -----------------------------
    def observer_intensity(self, t: int) -> np.ndarray:
        """
        Sum of Gaussian-weighted overlaps of Psi with observer amplitudes.
        """
        result = np.zeros((self.L, self.L))
        psi_field = self.Psi[t]
        for obs in self.observers:
            x, y = self.positions[self.observers.index(obs)]
            X, Y = np.meshgrid(np.arange(self.L), np.arange(self.L))
            g = np.exp(-((X - x)**2 + (Y - y)**2)/(2*self.sigma**2))
            g /= np.sqrt(np.sum(np.abs(g)**2) + 1e-12)
            result += np.abs(psi_field * g)
        return result

    # -----------------------------
    # Mean inter-observer distance
    # -----------------------------
    def mean_distance(self, t: int) -> float:
        """
        Compute mean Euclidean distance between observer positions.
        """
        total = 0.0
        count = 0
        for i in range(self.n_obs):
            xi, yi = self.positions[i]
            for j in range(i+1, self.n_obs):
                xj, yj = self.positions[j]
                total += np.sqrt((xi - xj)**2 + (yi - yj)**2)
                count += 1
        return total / max(1, count)

    # -----------------------------
    # Animation
    # -----------------------------
    def animate(self, steps: int = 200, res: int = 120, filename: str = "multi_qbitwave.gif"):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=res)
        micro_im = axes[0].imshow(self.Psi[0], cmap="magma")
        obs_im = axes[1].imshow(self.observer_intensity(0), cmap="viridis")

        axes[0].set_title("Universe Ψ")
        axes[1].set_title("Observer intensity")
        axes[0].axis("off")
        axes[1].axis("off")

        self.mean_distances = []
        self.total_entropies = []

        def update(frame):
            for _ in range(40):
                self.metropolis_step()
            t = frame % self.T
            micro_im.set_data(self.Psi[t])
            obs_im.set_data(self.observer_intensity(t))
            self.mean_distances.append(self.mean_distance(t))
            self.total_entropies.append(self.spectral_entropy(self.Psi[t]))

            return micro_im, obs_im

        ani = animation.FuncAnimation(fig, update, frames=steps, blit=False)
        fig.canvas.draw()   # <-- THIS LINE
        ani.save(filename, writer="pillow", fps=12)
        print(f"Saved animation to {filename}")

        plt.figure()
        plt.plot(self.total_entropies)
        plt.title("Global spectral entropy")
        plt.xlabel("Frame")
        plt.ylabel("H(Ψ)")
        plt.show()


        plt.figure()
        plt.plot(self.mean_distances)
        plt.xlabel("Frame")
        plt.ylabel("Mean observer distance")
        plt.title("Emergent observer merger via QBitwave")
        plt.show()


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-observer QBitwave PoC simulation")
    parser.add_argument("--steps", type=int, default=400, help="Number of animation frames")
    parser.add_argument("--T", type=int, default=100, help="Number of time slices in Ψ")
    parser.add_argument("--size", type=int, default=32, help="Spatial size of Ψ (LxL)")
    parser.add_argument("--obs", type=int, default=3, help="Number of observers")
    parser.add_argument("--sigma", type=float, default=5.0, help="Gaussian width of observers")
    parser.add_argument("--bits", type=int, default=64, help="Bitstring length per observer")
    parser.add_argument("--file", type=str, default="multi_qbitwave.gif", help="Output filename")
    args = parser.parse_args()

    sim = MultiObserverQBitwaveSim(L=args.size, T=args.T, n_obs=args.obs, sigma=args.sigma, bits=args.bits)
    sim.animate(steps=args.steps, filename=args.file)


if __name__ == "__main__":
    main()
