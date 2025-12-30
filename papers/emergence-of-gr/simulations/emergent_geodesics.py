import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Tuple, List

class InformationalManifold:
    """The configuration ensemble with a strictly monotonic density gradient."""
    def __init__(self, size: float = 1.0, resolution: int = 150):
        self.size = size
        self.res = resolution
        self.coords = np.linspace(0, size, resolution)
        self.X, self.Y = np.meshgrid(self.coords, self.coords)
        self.density = np.zeros_like(self.X)

    def add_mass(self, pos: Tuple[float, float], sigma: float = 0.2):
        # Gaussian-like attractor: Ensures the center is the absolute minimum MDL cost
        dist_sq = (self.X - pos[0])**2 + (self.Y - pos[1])**2
        self.density = np.exp(-dist_sq / (2 * sigma**2))

class InertialObserver:
    """An observer defined by its historical execution trace (Memory)."""
    def __init__(self, start_pos: np.ndarray, direction_vec: np.ndarray, 
                 step_len: float, inertia_weight: float, memory_depth: int):
        
        # FIX: Normalize the direction vector to the actual step length
        # so the history doesn't 'jump' outside the manifold.
        norm = np.linalg.norm(direction_vec)
        unit_vel = (direction_vec / norm) * step_len if norm > 0 else np.array([step_len, 0])
        
        # Pre-populate history as a smooth, consistent trace
        self.history: List[np.ndarray] = [start_pos - unit_vel * i for i in range(memory_depth, 0, -1)]
        self.history.append(start_pos)
        
        self.inertia_weight = inertia_weight
        self.memory_depth = memory_depth

    def get_next_step(self, manifold: InformationalManifold, gravity_weight: float, step_len: float):
        curr = self.history[-1]
        
        # 1. Inertial Prediction (Extrapolate from average history)
        velocities = [self.history[i] - self.history[i-1] for i in range(1, len(self.history))]
        avg_vel = np.mean(velocities[-self.memory_depth:], axis=0)
        
        v_norm = np.linalg.norm(avg_vel)
        if v_norm > 0:
            avg_vel = (avg_vel / v_norm) * step_len
        
        predicted_step = curr + avg_vel
        
        # 2. MDL Selection
        best_cand = None
        min_cost = float('inf')
        
        angles = np.linspace(0, 2 * np.pi, 120)
        candidates = curr + np.column_stack([np.cos(angles), np.sin(angles)]) * step_len
        
        for cand in candidates:
            # Keep observer within the observable manifold
            if not (0 <= cand[0] <= manifold.size and 0 <= cand[1] <= manifold.size):
                continue
                
            # Cost A: Innovation (Breaking the trace/Inertia)
            innovation_cost = np.linalg.norm(cand - predicted_step)**2
            
            # Cost B: Structure (Density/Gravity)
            ix = np.searchsorted(manifold.coords, cand[0]) - 1
            iy = np.searchsorted(manifold.coords, cand[1]) - 1
            density_val = manifold.density[max(0, iy), max(0, ix)]
            structural_cost = 1.0 / (density_val + 1e-6)
            
            total_cost = (self.inertia_weight * innovation_cost) + (gravity_weight * structural_cost)
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_cand = cand
        
        if best_cand is not None:
            self.history.append(best_cand)

def main():
    parser = argparse.ArgumentParser(description="ASP Geodesic Simulation - Bug Fixed Version")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--inertia", type=float, default=300.0)
    parser.add_argument("--gravity", type=float, default=1.5)
    parser.add_argument("--sigma", type=float, default=0.2)
    parser.add_argument("--memory", type=int, default=15)
    parser.add_argument("--vel", type=float, default=0.012)
    args = parser.parse_args()

    world = InformationalManifold()
    world.add_mass(pos=(0.5, 0.5), sigma=args.sigma)

    # Initial Position: 0.1, 0.9 (Top Left)
    # Direction: [1, -0.4] (Heading towards the center-right)
    obs = InertialObserver(
        start_pos=np.array([0.1, 0.9]), 
        direction_vec=np.array([1.0, -0.4]), 
        step_len=args.vel,
        inertia_weight=args.inertia, 
        memory_depth=args.memory
    )

    for _ in range(args.steps):
        obs.get_next_step(world, args.gravity, step_len=args.vel)

    # Rendering
    plt.figure(figsize=(10, 8))
    # background is strictly 0 to 1
    plt.imshow(world.density, extent=[0, 1, 0, 1], origin='lower', cmap='inferno')
    
    path = np.array(obs.history)
    plt.plot(path[:,0], path[:,1], 'w-', linewidth=2, label='Observer Path')
    plt.scatter(0.5, 0.5, color='cyan', s=150, marker='*', label='Singularity')
    
    # Strictly lock the view to the manifold bounds
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"Clean ASP Geodesic (Steps={args.steps}, Mem={args.memory})")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()