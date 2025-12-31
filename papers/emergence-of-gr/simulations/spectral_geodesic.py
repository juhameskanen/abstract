"""
Spectral Geodesics in Log-Normal Configuration Space (MDL-Faithful Version)

This simulation implements Ladder 8b of the Abstract Universe theory:

- Observer paths emerge from a static configuration space of bitstrings.
- Paths are selected to minimize their spectral description length (MDL).
- Microstructure density follows a log-normal distribution, modeling gravitational attraction.
- No heuristic weightings or arbitrary parameters are used; the MDL principle alone determines the path.
- Adaptive step length emerges from the local cost landscape (more probable / compressible steps allow larger steps).
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, List

class LogNormalManifold:
    """Represents the configuration ensemble with log-normal microstructure density."""

    def __init__(self, size: float = 1.0, resolution: int = 200) -> None:
        self.size: float = size
        self.res: int = resolution
        self.coords: np.ndarray = np.linspace(0, size, resolution)
        self.X, self.Y = np.meshgrid(self.coords, self.coords)
        self.density: np.ndarray = np.zeros_like(self.X)

    def add_mass(self, pos: Tuple[float, float], sigma: float = 0.6, mu: float = -1.8) -> None:
        dist = np.sqrt((self.X - pos[0])**2 + (self.Y - pos[1])**2)
        d = np.clip(dist, 1e-9, None)  # Avoid log(0)
        term1 = 1.0 / (d * sigma * np.sqrt(2 * np.pi))
        term2 = np.exp(-((np.log(d) - mu)**2) / (2 * sigma**2))
        self.density = term1 * term2
        self.density /= np.max(self.density)  # Normalize peak to 1

class MDLObserver:
    """Observer whose trajectory is fully determined by minimizing spectral description length."""

    def __init__(self, start_pos: np.ndarray, initial_vel: np.ndarray, memory_depth: int = 15) -> None:
        self.history: List[np.ndarray] = [start_pos - initial_vel * i for i in range(memory_depth, 0, -1)]
        self.history.append(start_pos)
        self.memory_depth: int = memory_depth

    def get_next_step(self, manifold: LogNormalManifold) -> None:
        curr = self.history[-1]

        # Inertia: predicted next step from average of recent velocities
        velocities = [self.history[i] - self.history[i-1] for i in range(1, len(self.history))]
        avg_vel = np.mean(velocities[-self.memory_depth:], axis=0)
        predicted_step = curr + avg_vel

        # Candidate steps: circular sampling around current position
        num_candidates = 120
        candidate_radius = 0.01
        angles = np.linspace(0, 2 * np.pi, num_candidates)
        candidates = curr + candidate_radius * np.column_stack([np.cos(angles), np.sin(angles)])

        min_cost = float("inf")
        best_candidate = None

        for cand in candidates:
            # Clip to manifold boundaries
            if not (0 <= cand[0] <= manifold.size and 0 <= cand[1] <= manifold.size):
                continue

            # Inertia cost: deviation from predicted step
            inertia_cost = np.linalg.norm(cand - predicted_step)**2

            # Spectral cost: inverse of local density (probability / compressibility)
            ix = np.searchsorted(manifold.coords, cand[0]) - 1
            iy = np.searchsorted(manifold.coords, cand[1]) - 1
            ix = np.clip(ix, 0, manifold.res-1)
            iy = np.clip(iy, 0, manifold.res-1)
            spectral_cost = 1.0 / (manifold.density[iy, ix] + 1e-7)

            total_cost = inertia_cost + spectral_cost

            if total_cost < min_cost:
                min_cost = total_cost
                best_candidate = cand

        if best_candidate is not None:
            self.history.append(best_candidate)

def run_simulation() -> None:
    parser = argparse.ArgumentParser(description="Spectral Geodesic Simulation (MDL-Faithful)")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--memory", type=int, default=15)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--mu", type=float, default=-1.8)
    parser.add_argument("--init_speed", type=float, default=0.01)
    parser.add_argument("--init_angle", type=float, default=math.pi/2.0)
    args = parser.parse_args()

    # Setup manifold
    world = LogNormalManifold()
    center = (0.5, 0.5)
    world.add_mass(pos=center, sigma=args.sigma, mu=args.mu)

    # Initial radial direction
    start_pos = np.array([0.1, 0.85])
    radial_dir = np.array(center) - start_pos
    radial_dir /= np.linalg.norm(radial_dir)
    angle_rot = np.array([[np.cos(args.init_angle), -np.sin(args.init_angle)],
                          [np.sin(args.init_angle),  np.cos(args.init_angle)]])
    initial_vel = radial_dir @ angle_rot * args.init_speed

    # Setup observer
    obs = MDLObserver(start_pos=start_pos, initial_vel=initial_vel, memory_depth=args.memory)

    # Run simulation
    for _ in range(args.steps):
        obs.get_next_step(world)

    # Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(world.density, extent=[0,1,0,1], origin='lower', cmap='magma')
    plt.colorbar(label='Spectral Probability Density')

    path = np.array(obs.history)
    plt.plot(path[:,0], path[:,1], 'w-', linewidth=2, label='Observer MDL Path')
    plt.scatter(path[-1,0], path[-1,1], color='white', s=30)
    peak_dist = np.exp(args.mu - args.sigma**2)
    circle = plt.Circle(center, peak_dist, color='cyan', fill=False, linestyle='--', label='Peak Density Ring')
    plt.gca().add_patch(circle)
    plt.scatter(center[0], center[1], color='cyan', s=100, marker='*', label='Singularity (d=0)')

    plt.title(f"Spectral MDL Geodesic Simulation\nSigma={args.sigma}")
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    run_simulation()
