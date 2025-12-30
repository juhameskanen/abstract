import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, List

"""
Emergent Gravitation via Log-Normal Probability Pressure

- Density (rho): Log-normal (starts at 0, peaks, has a long tail).
- Cost (K): 1 / rho (becomes infinite as d -> 0).
- Gravity: Gradient of K (Probability pressure).
- Inertia: Resistance to deviation from the historical trace (Memory).
"""

class LogNormalManifold:
    """The configuration ensemble where microstructure density follows a Log-normal distribution."""
    def __init__(self, size: float = 1.0, resolution: int = 200):
        self.size = size
        self.res = resolution
        self.coords = np.linspace(0, size, resolution)
        self.X, self.Y = np.meshgrid(self.coords, self.coords)
        self.density = np.zeros_like(self.X)

    def add_mass(self, pos: Tuple[float, float], sigma: float = 0.6, mu: float = -1.8):
        """
        Calculates density based on Log-normal distribution of distance 'd'.
        The 'Singularity' is at d=0 (Amplitude 0).
        """
        dist = np.sqrt((self.X - pos[0])**2 + (self.Y - pos[1])**2)
        d = np.clip(dist, 1e-9, None) # Avoid log(0)
        
        # Log-normal PDF formula
        term1 = 1.0 / (d * sigma * np.sqrt(2 * np.pi))
        term2 = np.exp(-((np.log(d) - mu)**2) / (2 * sigma**2))
        
        self.density = term1 * term2
        # Normalize so the peak is 1.0 for computational stability
        self.density /= np.max(self.density)

class InertialObserver:
    """An observer defined by its historical execution trace."""
    def __init__(self, start_pos: np.ndarray, direction_vec: np.ndarray, 
                 step_len: float, inertia_weight: float, memory_depth: int):
        
        # Normalize initial velocity to ensure history starts within the world
        norm = np.linalg.norm(direction_vec)
        unit_vel = (direction_vec / norm) * step_len if norm > 0 else np.array([step_len, 0])
        
        self.history: List[np.ndarray] = [start_pos - unit_vel * i for i in range(memory_depth, 0, -1)]
        self.history.append(start_pos)
        self.inertia_weight = inertia_weight
        self.memory_depth = memory_depth

    def get_next_step(self, manifold: LogNormalManifold, gravity_weight: float, step_len: float):
        curr = self.history[-1]
        
        # 1. Inertial Prediction (Conservation of Trace)
        velocities = [self.history[i] - self.history[i-1] for i in range(1, len(self.history))]
        avg_vel = np.mean(velocities[-self.memory_depth:], axis=0)
        v_norm = np.linalg.norm(avg_vel)
        if v_norm > 0:
            avg_vel = (avg_vel / v_norm) * step_len
        predicted_step = curr + avg_vel
        
        # 2. MDL Selection (Finding the most probable next configuration)
        best_cand = None
        min_cost = float('inf')
        
        angles = np.linspace(0, 2 * np.pi, 120)
        candidates = curr + np.column_stack([np.cos(angles), np.sin(angles)]) * step_len
        
        for cand in candidates:
            if not (0 <= cand[0] <= manifold.size and 0 <= cand[1] <= manifold.size):
                continue
            
            # Innovation Cost: Effort to change the path
            innovation_cost = np.linalg.norm(cand - predicted_step)**2
            
            # Structural Cost: 1 / Probability (Log-normal density)
            ix = np.searchsorted(manifold.coords, cand[0]) - 1
            iy = np.searchsorted(manifold.coords, cand[1]) - 1
            density_val = manifold.density[max(0, iy), max(0, ix)]
            structural_cost = 1.0 / (density_val + 1e-7)
            
            total_cost = (self.inertia_weight * innovation_cost) + (gravity_weight * structural_cost)
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_cand = cand
        
        if best_cand is not None:
            self.history.append(best_cand)

def run_simulation():
    parser = argparse.ArgumentParser(description="Pillar 7: Log-Normal Geodesics")
    parser.add_argument("--steps", type=int, default=350)
    parser.add_argument("--inertia", type=float, default=450.0)
    parser.add_argument("--gravity", type=float, default=1.8)
    parser.add_argument("--sigma", type=float, default=0.6, help="Skew of the gravity well")
    parser.add_argument("--mu", type=float, default=-1.8, help="Shifts the peak density ring")
    parser.add_argument("--memory", type=int, default=15)
    parser.add_argument("--vel", type=float, default=0.01)
    args = parser.parse_args()

    # 1. Setup the Manifold
    world = LogNormalManifold()
    center = (0.5, 0.5)
    world.add_mass(pos=center, sigma=args.sigma, mu=args.mu)

    # 2. Setup the Observer
    obs = InertialObserver(
        start_pos=np.array([0.1, 0.85]), 
        direction_vec=np.array([1.0, -0.15]), 
        step_len=args.vel,
        inertia_weight=args.inertia, 
        memory_depth=args.memory
    )

    # 3. Run Simulation
    for _ in range(args.steps):
        obs.get_next_step(world, args.gravity, step_len=args.vel)

    # 4. Visualization
    plt.figure(figsize=(10, 10))
    
    # Heatmap of the Log-normal Density
    plt.imshow(world.density, extent=[0, 1, 0, 1], origin='lower', cmap='magma')
    plt.colorbar(label='Informational Density (Structural Probability)')

    # Plot the Geodesic Trace
    path = np.array(obs.history)
    plt.plot(path[:, 0], path[:, 1], 'w-', linewidth=2, label='Observer Geodesic')
    plt.scatter(path[-1, 0], path[-1, 1], color='white', s=30) # Current position
    
    # Peak Density Circle (Analytical Mode)
    peak_dist = np.exp(args.mu - args.sigma**2)
    circle = plt.Circle(center, peak_dist, color='cyan', fill=False, linestyle='--', label='Peak Density Ring')
    plt.gca().add_patch(circle)
    
    # The Singularity
    plt.scatter(center[0], center[1], color='cyan', s=100, marker='*', label='Singularity (d=0)')

    # Add Gradient 'Force' Arrows (Subsampled)
    cost = 1.0 / (world.density + 1e-7)
    dy, dx = np.gradient(cost)
    skip = 10
    plt.quiver(world.X[::skip, ::skip], world.Y[::skip, ::skip], 
               -dx[::skip, ::skip], -dy[::skip, ::skip], 
               color='white', alpha=0.3, scale=50, width=0.002)

    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.title(f"Geodesic from Log-Normal Probability Distribution \nInertia={args.inertia}, Sigma={args.sigma}")
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    run_simulation()