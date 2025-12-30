import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class InformationalManifold:
    """The static configuration space with a log-normal density gradient."""
    def __init__(self, size: float = 1.0, resolution: int = 100):
        self.size = size
        self.res = resolution
        x = np.linspace(0, size, resolution)
        self.X, self.Y = np.meshgrid(x, x)
        self.density = np.zeros_like(self.X)

    def add_mass(self, pos: Tuple[float, float], sigma: float = 0.15):
        dist = np.sqrt((self.X - pos[0])**2 + (self.Y - pos[1])**2)
        # The 'Mass' provides high density (low MDL cost)
        # Using the log-normal logic from your Pillar 7
        self.density += np.exp(-(np.log(dist + 0.05) - 0)**2 / (2 * sigma**2))

class InertialObserver:
    """An observer that maintains a history (memory) to minimize transition MDL."""
    def __init__(self, start_pos: np.ndarray, initial_vel: np.ndarray):
        self.history = [start_pos - initial_vel, start_pos] # Needs 2 points for inertia
        self.inertia_weight = 15.0 # How much the observer 'resists' turns
        self.gravity_weight = 1.0  # Attraction to high density

    def get_next_step(self, manifold: InformationalManifold, n_candidates: int = 50):
        curr = self.history[-1]
        prev = self.history[-2]
        
        # 1. Calculate the 'Predictable Path' (Inertia)
        predicted_step = curr + (curr - prev)
        
        # 2. Generate candidates in a small radius around current position
        angles = np.linspace(0, 2*np.pi, n_candidates)
        step_size = np.linalg.norm(curr - prev)
        candidates = curr + np.column_stack([np.cos(angles), np.sin(angles)]) * step_size
        
        best_cand = None
        min_cost = float('inf')
        
        for cand in candidates:
            # Check bounds
            if not (0 <= cand[0] <= manifold.size and 0 <= cand[1] <= manifold.size):
                continue
                
            # MDL cost calculation:
            # A) Innovation Cost (extra bits needed): Deviation from the trace-extension (History)
            innovation_cost = np.linalg.norm(cand - predicted_step)**2
            
            # B) Structural Cost: Inverse density at candidate location (Gravity)
            # Map candidate to grid indices, clamp to avoid out-of-bounds
            ix = int((cand[0] / manifold.size) * (manifold.res - 1))
            iy = int((cand[1] / manifold.size) * (manifold.res - 1))
            density_val = manifold.density[iy, ix]
            structural_cost = 1.0 / (density_val + 1e-6)
            
            total_cost = (self.inertia_weight * innovation_cost) + (self.gravity_weight * structural_cost)
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_cand = cand
        
        if best_cand is not None:
            self.history.append(best_cand)

def run_simulation():
    manifold = InformationalManifold(size=1.0)
    manifold.add_mass(pos=(0.5, 0.5), sigma=0.2) # Central Mass

    # Two observers with different 'initial velocities' (historical traces)
    obs1 = InertialObserver(start_pos=np.array([0.1, 0.7]), initial_vel=np.array([0.02, -0.005]))
    obs2 = InertialObserver(start_pos=np.array([0.1, 0.3]), initial_vel=np.array([0.02, 0.005]))

    for _ in range(60):
        obs1.get_next_step(manifold)
        obs2.get_next_step(manifold)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.imshow(manifold.density, extent=[0, 1, 0, 1], origin='lower', cmap='magma', alpha=0.8)
    
    h1 = np.array(obs1.history)
    h2 = np.array(obs2.history)
    
    plt.plot(h1[:,0], h1[:,1], 'c-o', markersize=3, label='Observer A (Bending)')
    plt.plot(h2[:,0], h2[:,1], 'm-o', markersize=3, label='Observer B (Bending)')
    
    plt.title("Emergent Geodesics: MDL Minimization with Memory (Inertia)")
    plt.xlabel("Config Space X")
    plt.ylabel("Config Space Y")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    run_simulation()