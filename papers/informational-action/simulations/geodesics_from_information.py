import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter  # The 'Fluidity' tool
from qbitwave import QBitwave

class GeodesicSurface:
    def __init__(self, size: int = 64, basis_size: int = 8):
        self.size = size
        self.basis_size = basis_size
        # High-stability ratio
        self.window = basis_size * 4 
        
        self.x = np.linspace(0, 10, size)
        self.y = np.linspace(0, 10, size)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.surface = np.zeros((size, size))

    def generate_surface(self, noise_level: float = 0.05):
        self.surface = np.zeros((self.size, self.size))
        centers = [(2, 2, 2.0), (7, 7, -1.5), (5, 8, 1.0), (8, 4, -0.8)]
        for cx, cy, amp in centers:
            self.surface += amp * np.exp(-((self.X-cx)**2 + (self.Y-cy)**2)/2)
        self.surface += np.random.normal(0, noise_level, (self.size, self.size))

    def _get_qbit_metrics(self, segment):
        local_mean = np.mean(segment)
        # Planck Jitter: Essential for breaking grid-locking
        jitter = np.random.normal(0, 0.005, segment.shape) 
        bits = (segment + jitter > local_mean).astype(int).tolist()
        qb = QBitwave(bitstring=bits, fixed_basis_size=self.basis_size)
        return qb.wave_complexity(), qb.amplitudes

    def find_hawking_path_2d(self, start_row: int = None):
        if start_row is None: start_row = self.size // 2
        path_indices = [start_row]
        row = start_row
        for col in range(self.size - 1):
            candidates = [r for r in [row-1, row, row+1] if 0 <= r < self.size]
            grads = [self.surface[r, col+1] - self.surface[row, col] for r in candidates]
            row = candidates[np.argmin(np.abs(grads))]
            path_indices.append(row)
        return path_indices

    def find_qbitwave_path_2d_complex(self, start_row: int = None):
        if start_row is None: start_row = self.size // 2
        path_indices = [start_row]
        row = start_row
        prev_amplitudes = None

        for col in range(self.size - 1):
            candidates = [r for r in [row-1, row, row+1] if 0 <= r < self.size]
            scores, candidate_amps = [], []

            for r in candidates:
                start_c = col + 1 - self.window
                if start_c < 0:
                    segment = np.concatenate([np.zeros(abs(start_c)), self.surface[r, 0:col+2]])
                else:
                    segment = self.surface[r, start_c:col+2]
                
                segment = segment[:self.window]
                entropy, current_amps = self._get_qbit_metrics(segment)
                candidate_amps.append(current_amps)
                
                if prev_amplitudes is not None and prev_amplitudes.size == current_amps.size:
                    corr = np.abs(np.vdot(prev_amplitudes, current_amps))
                    norm = (np.linalg.norm(prev_amplitudes) * np.linalg.norm(current_amps)) + 1e-12
                    coherence_cost = 1.0 - (corr / norm)
                else:
                    coherence_cost = 0.0

                scores.append(entropy + 2.0 * coherence_cost)

            best_idx = np.argmin(scores)
            row = candidates[best_idx]
            prev_amplitudes = candidate_amps[best_idx]
            path_indices.append(row)
        return path_indices
    

    def visualize(self):
        h_path = self.find_hawking_path_2d()
        q_path = self.find_qbitwave_path_2d_complex()

        fig = plt.figure(figsize=(14, 7))

        # --- 3D Surface ---
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(self.X, self.Y, self.surface, cmap='viridis', alpha=0.5, antialiased=True)

        # Hawking path
        ax1.plot(self.x, self.y[h_path], self.surface[h_path, np.arange(self.size)] + 0.05,
                 color='red', linewidth=3, label='Euclidean (Min Action)')

        # QBitwave path
        ax1.plot(self.x, self.y[q_path], self.surface[q_path, np.arange(self.size)] + 0.08,
                 color='blue', linestyle='--', linewidth=3, label='Informational (Min Entropy)')

        ax1.set_title(f"3D Geodesics (Surface size {self.size})")
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Amplitude")
        ax1.legend()

        # --- 2D Cross-section ---
        ax2 = fig.add_subplot(122)
        ax2.plot(self.x, self.surface[h_path, np.arange(self.size)], color='red', alpha=0.7, label='Euclidean Action')
        ax2.plot(self.x, self.surface[q_path, np.arange(self.size)], color='blue', linestyle='--', label='Informational Action')
        ax2.set_title("Path Comparison")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Amplitude / Metric Depth")
        ax2.legend()

        plt.tight_layout()
        plt.show()


class HeatmapGeodesic(GeodesicSurface):
    def compute_cost_maps(self):
        action_map = np.zeros_like(self.surface)
        entropy_map = np.zeros_like(self.surface)
        
        print(f"Applying Fluid-Field Dynamics (Scipy-Blur active)...")

        for c in range(self.window, self.size):
            for r in range(self.size):
                action_map[r, c] = (self.surface[r, c] - self.surface[r, c-1])**2
                
                segment = self.surface[r, c - self.window : c]
                # High-fidelity sampling for the map
                samples = []
                for offset in np.linspace(-0.03, 0.03, 5):
                    h, _ = self._get_qbit_metrics(segment + offset)
                    samples.append(h)
                entropy_map[r, c] = np.mean(samples)

        # Apply Gaussian Blur to the Entropy Map to simulate Quantum Delocalization
        # sigma=2.0 provides a beautiful, fluid field.
        entropy_map = gaussian_filter(entropy_map, sigma=2.0)
        action_map = gaussian_filter(action_map, sigma=1.0) # Light blur for Hawking too

        # Normalize
        action_map = (action_map - np.min(action_map)) / (np.max(action_map) + 1e-9)
        entropy_map = (entropy_map - np.min(entropy_map)) / (np.max(entropy_map) + 1e-9)

        return np.exp(-4.0 * action_map), np.exp(-2.5 * entropy_map)

    def visualize_with_heatmap(self):
        h_path = self.find_hawking_path_2d()
        q_path = self.find_qbitwave_path_2d_complex()
        action_prob, entropy_prob = self.compute_cost_maps()

        fig = plt.figure(figsize=(16, 8))
        for i, (prob, path, title, col, cmap) in enumerate([
            (action_prob, h_path, "Euclidean (Geometric Action)", 'red', 'Reds'),
            (entropy_prob, q_path, "Informational (Spectral Entropy)", 'blue', 'Blues')
        ]):
            ax = fig.add_subplot(1, 2, i+1, projection='3d')
            ax.contourf(self.X, self.Y, prob, zdir='z', offset=-2, cmap=cmap, alpha=0.6)
            ax.plot_surface(self.X, self.Y, self.surface, cmap='viridis', alpha=0.3, antialiased=True)
            ax.plot(self.x, self.y[path], self.surface[path, np.arange(self.size)] + 0.05, 
                    color=col, linewidth=3, label=f'{title} Path')
            ax.set_title(title)
            ax.set_zlim(-2, 3)
            ax.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--basis_size', type=int, default=8)
    parser.add_argument('--noise', type=float, default=0.01)
    args = parser.parse_args()


    sim = GeodesicSurface(size=args.size, basis_size=args.basis_size)
    sim.generate_surface(noise_level=args.noise)
    sim.visualize()


    sim = HeatmapGeodesic(size=args.size, basis_size=args.basis_size)
    sim.generate_surface(noise_level=args.noise)
    sim.visualize_with_heatmap()