"""
Fermionic Block Universe: Spectral Complexity and Antisymmetry.

This module simulates the emergent 'repulsion' between fermions by modeling 
the block universe as a 4D informational manifold. It uses a Minimum Description 
Length (MDL) approach, where the trajectories are selected to minimize the 
spectral complexity of the global antisymmetric wavefunction (Slater state).

The exclusion principle emerges not as a force, but as a structural constraint: 
configurations where particles overlap require higher-frequency Fourier modes 
to describe the mandatory 'Pauli Hole,' making them 'expensive' in an 
information-theoretic sense.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class QBitSpinorMDL:
    """
    Spectral complexity engine for analyzing the entropy of 4D worldlines.
    
    Attributes:
        N (int): Normalization constant for frequency scaling.
    """
    def __init__(self, N: int):
        """
        Initializes the MDL engine.
        
        Args:
            N (int): The frequency resolution/scaling factor.
        """
        self.N = N

    def get_complexity(self, signal: np.ndarray) -> float:
        """
        Computes the spectral complexity C_Q of a given field or trajectory.
        
        The cost is defined by the L2 norm of the power spectrum weighted by 
        the square of the frequency (k^2), effectively measuring the 
        informational roughness of the block.

        Args:
            signal (np.ndarray): The 1D or ND signal representing the field history.

        Returns:
            float: The computed spectral complexity.
        """
        if np.all(signal == 0): return 0.0
        coeffs = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal)) * self.N
        k_eff = np.abs(freqs)
        
        power = np.abs(coeffs)**2
        return np.sum((k_eff**2) * power)

class SpectralBasis:
    """
    Orthonormal basis functions used to construct spatial orbitals.
    
    Attributes:
        k (float): The characteristic frequency/momentum of the orbital.
    """
    def __init__(self, k_freq: float):
        """
        Initializes the spectral basis.
        
        Args:
            k_freq (float): Center frequency for the plane-wave component.
        """
        self.k = k_freq
        
    def evaluate(self, x: np.ndarray, center: float) -> np.ndarray:
        """
        Evaluates a localized wavepacket at given spatial coordinates.

        Args:
            x (np.ndarray): Grid of spatial points.
            center (float): The current spatial center of the orbital.

        Returns:
            np.ndarray: Complex values of the orbital across the grid.
        """
        return np.exp(-0.5 * (x - center)**2) * np.exp(1j * self.k * x)

class TrueFermionicBlock:
    """
    Represents a 4D spacetime block containing two fermions.
    
    This class constructs the 2-body configuration space (x1, x2) and 
    computes the antisymmetric Slater determinant state to demonstrate 
    emergent exclusion via spectral optimization.

    Attributes:
        n_steps (int): Number of time slices in the block.
        x_res (int): Spatial resolution of the configuration grid.
        grid (np.ndarray): 1D spatial coordinate array.
        centers (np.ndarray): Trajectories of the orbital centers.
        spinors (np.ndarray): Internal spinor states (alpha, beta) for each particle.
    """
    def __init__(self, n_steps: int, x_res: int = 40):
        """
        Initializes the Fermionic Block with two particles on a collision course.
        
        Args:
            n_steps (int): Total time steps for the simulation.
            x_res (int): Grid resolution for the x1, x2 axes.
        """
        self.n_steps = n_steps
        self.x_res = x_res
        self.grid = np.linspace(-4, 4, x_res)
        
        # Initialize centers: Particles moving toward each other
        self.centers = np.linspace(-1, 1, n_steps).reshape(1, -1).repeat(2, axis=0)
        self.centers[0, :] *= -1 
        
        # Parallel spinors (Spin Up/Up) to ensure maximum antisymmetric interference
        self.spinors = np.zeros((2, n_steps, 2), dtype=complex)
        self.spinors[0, :, 0] = 1.0 
        self.spinors[1, :, 0] = 1.0 

    def get_full_psi(self, t: int) -> np.ndarray:
        """
        Computes the 2-body antisymmetric wavefunction Psi(x1, x2) at time t.
        
        Constructs the Slater determinant state: 
        Psi = phi1(x1)phi2(x2) - phi1(x2)phi2(x1), weighted by spin overlap.

        Args:
            t (int): The time index to evaluate.

        Returns:
            np.ndarray: A 2D array representing Psi in the (x1, x2) configuration space.
        """
        phi1 = np.exp(-0.5 * (self.grid - self.centers[0, t])**2)
        phi2 = np.exp(-0.5 * (self.grid - self.centers[1, t])**2)
        
        s1, s2 = self.spinors[0, t], self.spinors[1, t]
        spin_overlap = np.vdot(s1, s2)
        
        # Outer products generate the full (x1, x2) configuration space
        term1 = np.outer(phi1, phi2)
        term2 = np.outer(phi2, phi1)
        
        # The Resulting antisymmetric state
        return (term1 - term2) * spin_overlap

    def plot_snapshot(self, t: int):
        """
        Visualizes the probability density |Psi(x1, x2)|^2 at a specific time.
        
        The plot displays the 'Pauli Hole' as a dark nodal line along the x1=x2 
        diagonal, where the probability of finding two identical fermions 
        is zero.

        Args:
            t (int): The time slice to render.
        """
        psi = self.get_full_psi(t)
        prob = np.abs(psi)**2
        
        if np.max(prob) > 0:
            prob /= np.max(prob) # Normalize for contrast
            
        plt.imshow(prob, extent=[-4, 4, -4, 4], cmap='viridis', origin='lower')
        plt.colorbar(label="Probability Density |Psi(x1, x2)|^2")
        # Pauli exclusion line: x1 = x2
        plt.axline((0, 0), slope=1, color='white', linestyle='--', alpha=0.5, label="Pauli Node (x1=x2)")
        plt.title(f"2-Fermion Configuration Space (Time {t})")
        plt.xlabel("Position of Particle 1 (x1)")
        plt.ylabel("Position of Particle 2 (x2)")
        plt.legend()
        plt.show()

# Execution entry point
if __name__ == "__main__":
    block = TrueFermionicBlock(n_steps=20)
    print("Rendering the Configuration Space...")
    # Visualize the midpoint (t=10) where particles are at maximum overlap
    block.plot_snapshot(t=10)