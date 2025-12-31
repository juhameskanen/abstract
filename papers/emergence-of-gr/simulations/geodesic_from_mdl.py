import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class AbstractEnsemble:
    """
    Represents the static configuration space (The 'Static Universe').
    The density of microstructures follows a log-normal distribution.
    """
    def __init__(self, size: int = 100, scale: float = 5.0):
        """
        Initialize the ensemble grid.
        :param size: Number of points per axis.
        :param scale: Coordinate range (from -scale to scale).
        """
        self.size = size
        self.scale = scale
        self.grid_coords = np.linspace(-scale, scale, size)
        self.X, self.Y = np.meshgrid(self.grid_coords, self.grid_coords)
        self.density_map = np.zeros((size, size))

    def inject_mass_motif(self, center: Tuple[float, float] = (0, 0), sigma: float = 1.0):
        """
        Creates a density gradient representing a central 'mass' or complex microstructure.
        Uses a log-normal distribution to represent the frequency of motifs.
        """
        dist = np.sqrt((self.X - center[0])**2 + (self.Y - center[1])**2)
        # Log-normal distribution of microstructures
        # We add a small epsilon to avoid log(0)
        self.density_map = np.exp(-(np.log(dist + 0.5) - 0)**2 / (2 * sigma**2))

    def get_cost(self, x_idx: int, y_idx: int) -> float:
        """
        The 'Description Length' cost of a state. 
        Higher density means the observer is easier to describe (Lower MDL).
        """
        # Cost is inversely proportional to density
        return 1.0 / (self.density_map[y_idx, x_idx] + 1e-6)

class MDLAgent:
    """
    An observer motif that 'falls' through the ensemble by minimizing 
    total description length (MDL).
    """
    def __init__(self, ensemble: AbstractEnsemble):
        self.ensemble = ensemble
        self.trajectory: List[Tuple[int, int]] = []

    def find_geodesic(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Calculates the path between start and end that minimizes the aggregate cost.
        This is a discrete approximation of a Riemannian geodesic.
        Uses a simplified A* algorithm where 'distance' is 'informational cost'.
        """
        import heapq
        
        queue = [(0, start, [])]
        visited = set()
        min_costs = {start: 0}

        while queue:
            (cost, current, path) = heapq.heappop(queue)

            if current in visited:
                continue

            visited.add(current)
            path = path + [current]

            if current == end:
                self.trajectory = path
                return path

            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (0 <= neighbor[0] < self.ensemble.size and 
                    0 <= neighbor[1] < self.ensemble.size):
                    
                    # Movement cost is the average MDL cost of the transition
                    step_cost = self.ensemble.get_cost(*neighbor)
                    new_cost = cost + step_cost

                    if neighbor not in min_costs or new_cost < min_costs[neighbor]:
                        min_costs[neighbor] = new_cost
                        # Priority = cost + heuristic (Euclidean distance to end)
                        priority = new_cost + np.sqrt((neighbor[0]-end[0])**2 + (neighbor[1]-end[1])**2)
                        heapq.heappush(queue, (priority, neighbor, path))
        
        return []

class Visualizer:
    """Handles the rendering of the informational manifold and agent path."""
    @staticmethod
    def plot_results(ensemble: AbstractEnsemble, agent: MDLAgent):
        plt.figure(figsize=(10, 8))
        
        # Plot the Density Map (The 'Curvature')
        plt.contourf(ensemble.X, ensemble.Y, ensemble.density_map, levels=20, cmap='viridis')
        plt.colorbar(label='Microstructure Density (1/MDL)')

        # Plot the Path
        if agent.trajectory:
            path_coords = np.array([(ensemble.grid_coords[x], ensemble.grid_coords[y]) 
                                    for x, y in agent.trajectory])
            plt.plot(path_coords[:, 0], path_coords[:, 1], 'r-', linewidth=2, label='MDL Geodesic')
            plt.scatter(path_coords[0,0], path_coords[0,1], color='white', label='Start')
            plt.scatter(path_coords[-1,0], path_coords[-1,1], color='black', label='End')

        plt.title("Emergent Gravity: Geodesic Path via Minimal Description Length")
        plt.xlabel("Configuration Dimension X")
        plt.ylabel("Configuration Dimension Y")
        plt.legend()
        plt.show()

# --- Execution ---
if __name__ == "__main__":
    # 1. Create the static universe
    world = AbstractEnsemble(size=60, scale=10)
    world.inject_mass_motif(center=(0, 0), sigma=1.2)

    # 2. Initialize observer and find the 'cheapest' path around the center
    observer = MDLAgent(world)
    # Move from top-left to bottom-right, but the 'mass' is in the way
    start_node = (5, 5)
    end_node = (55, 55)
    
    print("Calculating MDL Geodesic...")
    observer.find_geodesic(start_node, end_node)

    # 3. Visualize
    Visualizer.plot_results(world, observer)
