import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, List

class LogNormalManifold:
    """
    Configuration space where microstructure density follows a Log-normal distribution.
    This represents the 'Probability Pressure' of the Abstract Universe.
    """
    def __init__(self, size: float = 1.0, resolution: int = 200):
        self.size = size
        self.res = resolution
        self.coords = np.linspace(0, size, resolution)
        self.X, self.Y = np.meshgrid(self.coords, self.coords)
        self.density = np.zeros_like(self.X)

    def add_mass(self, pos: Tuple[float, float], sigma: float = 0.5, mu: float = -1.5):
        """
        Calculates density based on distance 'd' from center.
        mu: shifts the peak (Event Horizon location)
        sigma: controls the 'spread' or skew of the gravity well
        """
        # Distance from the 'Singularity'
        dist = np.sqrt((self.X - pos[0])**2 + (self.Y - pos[1])**2)
        # Avoid log(0)
        d = np.clip(dist, 1e-9, None)
        
        # Standard Log-normal PDF formula
        term1 = 1.0 / (d * sigma * np.sqrt(2 * np.pi))
        term2 = np.exp(-((np.log(d) - mu)**2) / (2 * sigma**2))
        
        self.density = term1 * term2
        # Normalize for visualization
        self.density /= np.max(self.density)

class LogNormalObserver:
    def __init__(self, start_pos: np.ndarray, direction_vec: np.ndarray, 
                 step_len: float, inertia_weight: float, memory_depth: int):
        norm = np.linalg.norm(direction_vec)
        unit_vel = (direction_vec / norm) * step_len if norm > 0 else np.array([step_len, 0])
        
        self.history: List[np.ndarray] = [start_pos - unit_vel * i for i in range(memory_depth, 0, -1)]
        self.history.append(start_pos)
        self.inertia_weight = inertia_weight
        self.memory_depth = memory_depth

    def get_next_step(self, manifold: LogNormalManifold, gravity_weight: float, step_len: float):
        curr = self.history[-1]
        
        # 1. Inertial Trend (Memory)
        velocities = [self.history[i] - self.history[i-1] for i in range(1, len(self.history))]
        avg_vel = np.mean(velocities[-self.memory_depth:], axis=0)
        v_norm = np.linalg.norm(avg_vel)
        if v_norm > 0:
            avg_vel = (avg_vel / v_norm) * step_len
        predicted_step = curr + avg_vel
        
        # 2. MDL Selection over the Log-normal Field
        best_cand = None
        min_cost = float('inf')
        
        angles = np.linspace(0, 2 * np.pi, 120)
        candidates = curr + np.column_stack([np.cos(angles), np.sin(angles)]) * step_len
        
        for cand in candidates:
            if not (0 <= cand[0] <= manifold.size and 0 <= cand[1] <= manifold.size):
                continue
                
            innovation_cost = np.linalg.norm(cand - predicted_step)**2
            
            # Lookup density in our Log-normal field
            ix = np.searchsorted(manifold.coords, cand[0]) - 1
            iy = np.searchsorted(manifold.coords, cand[1]) - 1
            density_val = manifold.density[max(0, iy), max(0, ix)]
            
            # MDL Selection: Fall toward higher density
            structural_cost = 1.0 / (density_val + 1e-7)
            
            total_cost = (self.inertia_weight * innovation_cost) + (gravity_weight * structural_cost)
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_cand = cand
        
        if best_cand is not None:
            self.history.append(best_cand)

def main():
    parser = argparse.ArgumentParser(description="Pillar 7: Log-Normal Gravity Emergence")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--inertia", type=float, default=400.0)
    parser.add_argument("--gravity", type=float, default=2.0)
    parser.add_argument("--sigma", type=float, default=0.6, help="Skew of the log-normal well")
    parser.add_argument("--mu", type=float, default=-1.8, help="Peak location (Horizon)")
    parser.add_argument("--vel", type=float, default=0.01)
    args = parser.parse_args()

    world = LogNormalManifold()
    world.add_mass(pos=(0.5, 0.5), sigma=args.sigma, mu=args.mu)

    # Start slightly outside the peak to see the 'Long Tail' attraction
    obs = LogNormalObserver(
        start_pos=np.array([0.1, 0.8]), 
        direction_vec=np.array([1.0, -0.1]), 
        step_len=args.vel,
        inertia_weight=args.inertia, 
        memory_depth=15
    )

    for _ in range(args.steps):
        obs.get_next_step(world, args.gravity, step_len=args.vel)

    # Visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(world.density, extent=[0, 1, 0, 1], origin='lower', cmap='magma')
    plt.colorbar(label='Microstructure Probability (Log-Normal)')
    
    path = np.array(obs.history)
    plt.plot(path[:,0], path[:,1], 'w-', linewidth=1.5, alpha=0.8, label='Observer Geodesic')
    plt.scatter(0.5, 0.5, color='cyan', s=50, marker='o', label='Singularity (d=0)')
    
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.title("Gravity as Log-Normal Probability Pressure")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()