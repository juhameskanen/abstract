"""
Emergent atom simulation

This script imports QBitwave to replace parametric wavefunctions with 
bit-substrate emergent wavefunctions.
"""

import numpy as np
import argparse
from typing import List, Tuple, Optional, Any
from qbitwave import QBitwave
from simulation_engine import GravitySim

# --- The Bridge: Mapping Bits to Space ---

class QBitwaveAdapter:
    """
    Adapts the discrete QBitwave bit-substrate to a 2D spatial interface.
    This allows the simulation to treat a bitstring as a physical wave.
    """
    def __init__(self, center: np.ndarray, sigma: float, bit_length: int = 512):
        self.center = np.array(center, dtype=float)
        self.sigma = sigma
        # Initialize QBitwave with a random bitstring
        initial_bits = [np.random.randint(0, 2) for _ in range(bit_length)]
        self.qbit = QBitwave(bitstring=initial_bits, fixed_basis_size=8)
        
    def evaluate(self, points: np.ndarray, t: float) -> np.ndarray:
        """Maps spatial coordinates to bit-indices to return complex amplitudes."""
        # 1. Calculate radial distance from observer center
        r = np.linalg.norm(points - self.center, axis=1)
        
        # 2. Map space to bit-blocks
        # We sample the bitstring based on distance, mapping 0 -> 3*sigma 
        # to the full range of the bitstring amplitudes.
        amps = self.qbit.get_amplitudes()
        n_amps = len(amps)
        
        indices = (r / (3 * self.sigma) * (n_amps - 1)).astype(int)
        indices = np.clip(indices, 0, n_amps - 1)
        
        # 3. Spatial Localization
        # Even an emergent wave needs a 'window' of observation (The Gaussian Filter)
        envelope = np.exp(-(r**2) / (2 * self.sigma**2))
        
        # 4. Return complex amplitude from the bit-substrate
        return amps[indices] * envelope * np.exp(-1j * t)

# --- The Emergent Simulation ---

class EmergentGravitySim(GravitySim):
    def __init__(
        self, 
        n_particles: int = 4096, 
        n_steps: int = 200, 
        blob_sigma: float = 0.1
    ) -> None:
        super().__init__(n_particles=n_particles, n_steps=n_steps, blob_sigma=blob_sigma)
        self.t = 0.0
        
        # Observers now 'carry' a bitstring-based wavefunction
        self.wavefunctions: List[QBitwaveAdapter] = [
            QBitwaveAdapter(pos, blob_sigma) for pos in self.positions
        ]

    def compute_pdf(self, grid_points: np.ndarray) -> np.ndarray:
        """Interference pattern emerging from the bit-substrate."""
        psi_total = np.zeros(grid_points.shape[0], dtype=np.complex128)
        for wf in self.wavefunctions:
            psi_total += wf.evaluate(grid_points, self.t)
        
        pdf = np.abs(psi_total) ** 2
        return pdf / (pdf.sum() + 1e-12)

    def update_positions(self, new_particles: np.ndarray) -> None:
        """
        Logic: 
        1. Bits mutate (Micro-evolution).
        2. Centers drift toward particle density (Macro-evolution).
        """
        for i, wf in enumerate(self.wavefunctions):
            # Mutate internal bits to simulate quantum fluctuations
            wf.qbit.flip(n_flips=2) 
            
            # Standard center-of-mass update for the 'observer'
            assigned = self.assign_particles_to_blob(i)
            if len(assigned) > 0:
                target_pos = assigned.mean(axis=0)
                # Apply informational inertia (0.1 learning rate)
                wf.center = wf.center * 0.9 + target_pos * 0.1
        
        # Sync positions for the base-class plotter
        self.positions = np.array([wf.center for wf in self.wavefunctions])
        self.t += 0.05

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="emergent_atom", help="Filename for the video to be created")
    parser.add_argument("--sigma", type=float, default=0.15, help="Blob sigma")
    parser.add_argument("--particles", type=int, default=2048, help="Number of particles")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--res", type=int, default=120, help="Resolution of the output video")
    parser.add_argument('--format', choices=['gif', 'mp4'], default='gif', help="Output format: gif for README, mp4 for high-res")
    args = parser.parse_args()

    if args.format == 'gif':
        # Optimized for GitHub README (small file size)
        sim_res = min(args.res, 64) 
        sim_fps = 12
        output_name = f"{args.file}.gif"
        writer_type = 'pillow'
    else:
        # Optimized for serious review (high fidelity)
        sim_res = args.res
        sim_fps = 24
        output_name = f"{args.file}.mp4"
        writer_type = 'ffmpeg'


    sim = EmergentGravitySim(
        n_particles=args.particles, 
        n_steps=args.steps, 
        blob_sigma=args.sigma
    )
    sim.run(output_name, res=sim_res, fps=sim_fps, writer=writer_type)

if __name__ == "__main__":
    main()