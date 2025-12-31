"""
Spectral Geodesics in Log-Normal Configuration Space

This simulation implements Ladder 8b of the Abstract Universe theory:

- Observer paths emerge from a static configuration space of bitstrings.
- The probability of each configuration is encoded in a spectral wavefunction Ψ.
- Microstructure density follows a log-normal distribution to model gravitational attraction.
- Geodesics are the paths that minimize the spectral encoding length (L) while preserving observer continuity.
- Adaptive step length arises naturally from the local gradient of the spectral cost, 
  simulating emergent motion without external forces or parameters.

The observer's initial velocity represents its wavefunction, setting the initial direction
and "orbital" behavior. No external time, execution, or dynamics exist—motion emerges
from the abstract information ensemble.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, List

class LogNormalManifold:
    """Represents the configuration ensemble with log-normal microstructure density."""
    
    def __init__(self, size: float = 1.0, resolution: int = 200) -> None:
        """
        Initialize the configuration space.

        Args:
            size: Physical size of the 2D space (unit square).
            resolution: Number of grid points along each axis.
        """
        self.size: float = size
        self.res: int = resolution
        self.coords: np.ndarray = np.linspace(0, size, resolution)
        self.X, self.Y = np.meshgrid(self.coords, self.coords)
        self.density: np.ndarray = np.zeros_like(self.X)

    def add_mass(self, pos: Tuple[float, float], sigma: float = 0.6, mu: float = -1.8) -> None:
        """
        Add a "mass" by defining a log-normal density distribution around a point.

        Args:
            pos: Center of the mass (x, y).
            sigma: Shape parameter of the log-normal distribution.
            mu: Mean of the log-normal distribution in log space.
        """
        dist = np.sqrt((self.X - pos[0])**2 + (self.Y - pos[1])**2)
        d = np.clip(dist, 1e-9, None)  # Avoid log(0)
        term1 = 1.0 / (d * sigma * np.sqrt(2 * np.pi))
        term2 = np.exp(-((np.log(d) - mu)**2) / (2 * sigma**2))
        self.density = term1 * term2
        # Normalize peak to 1 for stability
        self.density /= np.max(self.density)


class SpectralObserver:
    """
    An observer whose trajectory is determined by minimal spectral encoding (geodesic path).
    """
    def __init__(self, start_pos: np.ndarray, initial_vel: np.ndarray, memory_depth: int = 15) -> None:
        """
        Initialize the observer.

        Args:
            start_pos: Initial 2D position.
            initial_vel: Initial velocity vector (represents the initial wavefunction).
            memory_depth: Number of previous steps to maintain inertial history.
        """
        self.history: List[np.ndarray] = [start_pos - initial_vel * i for i in range(memory_depth, 0, -1)]
        self.history.append(start_pos)
        self.memory_depth: int = memory_depth

    def get_next_step(self, manifold: LogNormalManifold, gradient_weight: float = 1.0) -> None:
        """
        Compute the next step along the minimal spectral path.

        Args:
            manifold: The LogNormalManifold defining the spectral cost landscape.
            gradient_weight: Scaling factor for the probability gradient (gravity strength).
        """
        curr = self.history[-1]

        # Inertia: average velocity over history
        velocities = [self.history[i] - self.history[i-1] for i in range(1, len(self.history))]
        avg_vel = np.mean(velocities[-self.memory_depth:], axis=0)

        # Local spectral cost gradient
        ix = np.searchsorted(manifold.coords, curr[0]) - 1
        iy = np.searchsorted(manifold.coords, curr[1]) - 1
        ix = np.clip(ix, 1, manifold.res - 2)
        iy = np.clip(iy, 1, manifold.res - 2)
        cost_grid = 1.0 / (manifold.density + 1e-7)
        dy, dx = np.gradient(cost_grid)
        grad = np.array([dx[iy, ix], dy[iy, ix]])

        # Adaptive step direction and length
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 0:
            step_dir = -grad / grad_norm  # move downhill in spectral cost
            step_len = 0.005 + 0.05 * grad_norm
        else:
            step_dir = avg_vel / (np.linalg.norm(avg_vel) + 1e-7)
            step_len = 0.005

        # Candidate next position
        predicted_step = curr + avg_vel
        candidate_step = curr + step_dir * step_len

        # Minimal spectral cost evaluation (inertia + probability gradient)
        innovation_cost = np.linalg.norm(candidate_step - predicted_step)**2
        ix_c = np.searchsorted(manifold.coords, candidate_step[0]) - 1
        iy_c = np.searchsorted(manifold.coords, candidate_step[1]) - 1
        ix_c = np.clip(ix_c, 0, manifold.res-1)
        iy_c = np.clip(iy_c, 0, manifold.res-1)
        density_val = manifold.density[iy_c, ix_c]
        spectral_cost = gradient_weight * (1.0 / (density_val + 1e-7))
        total_cost = innovation_cost + spectral_cost

        # Append step to history
        self.history.append(candidate_step)


def run_simulation() -> None:
    """
    Simulation entry point for spectral geodesic in log-normal manifold.

    Startup Argsuments:
    --steps	Finite rendering
    --gravity Emergent	Numerical scaling
    --memory Observer continuity
    --sigma Microstructure statistics
    --mu Mass / entropy scale
    --init_speed Wavefunction magnitude
    --init_angle Wavefunction phase
    """
    parser = argparse.ArgumentParser(description="Spectral Geodesic Simulation")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--gravity", type=float, default=1.0, help="Strength of probability gradient")
    parser.add_argument("--init_angle", type=float, default=math.pi/2.0, help="Initial angle deviation from radial fall (radians)")
    parser.add_argument("--init_speed", type=float, default=0.01, help="Initial velocity magnitude")
    parser.add_argument("--memory", type=int, default=15)
    parser.add_argument("--sigma", type=float, default=0.6)
    parser.add_argument("--mu", type=float, default=-1.8)
    args = parser.parse_args()

    # Setup manifold
    world = LogNormalManifold()
    center = (0.5, 0.5)
    world.add_mass(pos=center, sigma=args.sigma, mu=args.mu)

    # Setup initial velocity from angle
    radial_dir = np.array([center[0]-0.1, center[1]-0.85])
    radial_dir /= np.linalg.norm(radial_dir)
    angle_rot = np.array([[np.cos(args.init_angle), -np.sin(args.init_angle)],
                          [np.sin(args.init_angle),  np.cos(args.init_angle)]])
    initial_vel = radial_dir @ angle_rot * args.init_speed

    # Setup observer
    obs = SpectralObserver(
        start_pos=np.array([0.1, 0.85]),
        initial_vel=initial_vel,
        memory_depth=args.memory
    )

    # Run simulation
    for _ in range(args.steps):
        obs.get_next_step(world, gradient_weight=args.gravity)

    # Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(world.density, extent=[0, 1, 0, 1], origin='lower', cmap='magma')
    plt.colorbar(label='Spectral Probability Density')
    path = np.array(obs.history)
    plt.plot(path[:, 0], path[:, 1], 'w-', linewidth=2, label='Observer Geodesic')
    plt.scatter(path[-1, 0], path[-1, 1], color='white', s=30)
    peak_dist = np.exp(args.mu - args.sigma**2)
    circle = plt.Circle(center, peak_dist, color='cyan', fill=False, linestyle='--', label='Peak Density Ring')
    plt.gca().add_patch(circle)
    plt.scatter(center[0], center[1], color='cyan', s=100, marker='*', label='Singularity (d=0)')

    # Gradient arrows (subsampled)
    cost = 1.0 / (world.density + 1e-7)
    dy, dx = np.gradient(cost)
    skip = 10
    plt.quiver(world.X[::skip, ::skip], world.Y[::skip, ::skip],
               -dx[::skip, ::skip], -dy[::skip, ::skip],
               color='white', alpha=0.3, scale=50, width=0.002)

    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.title(f"Spectral Geodesic Simulation\nGravity={args.gravity}, Sigma={args.sigma}")
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    run_simulation()
