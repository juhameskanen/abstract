import numpy as np
import argparse
from typing import List, Optional
from qbitwave import QBitwave
from visualization_engine import GravitySim  


class ObserverWavefunction:
    """
    Represents a single observer as a wavefunction with memory.

    Attributes
    ----------
    center : np.ndarray
        Current 2D position of the observer.
    sigma : float
        Spatial localization width of the Gaussian envelope.
    trajectory : List[np.ndarray]
        Historical positions for memory / spectral complexity.
    qbit : QBitwave
        Underlying bitstring representation.
    initial_velocity : np.ndarray
        Initial velocity to seed starting spectral components.
    """

    def __init__(self, center: np.ndarray, sigma: float, bit_length: int = 512,
                 initial_velocity: Optional[np.ndarray] = None):
        self.center = np.array(center, dtype=float)
        self.sigma = sigma
        self.trajectory: List[np.ndarray] = [self.center.copy()]
        self.initial_velocity = np.array(initial_velocity, dtype=float) if initial_velocity is not None else np.zeros_like(self.center)

        initial_bits = [np.random.randint(0, 2) for _ in range(bit_length)]
        self.qbit = QBitwave(bitstring=initial_bits, fixed_basis_size=8)

    def record_position(self, pos: np.ndarray):
        self.trajectory.append(np.array(pos, dtype=float))
        self.center = np.array(pos, dtype=float)

    def evaluate(self, points: np.ndarray, t: float) -> np.ndarray:
        """
        Compute complex amplitude at given spatial points.
        """
        r = np.linalg.norm(points - self.center, axis=1)
        amps = self.qbit.get_amplitudes()
        n_amps = len(amps)
        indices = (r / (3 * self.sigma) * (n_amps - 1)).astype(int)
        indices = np.clip(indices, 0, n_amps - 1)
        envelope = np.exp(-(r ** 2) / (2 * self.sigma ** 2))

        traj_array = np.array(self.trajectory)
        complex_traj = traj_array[:, 0] + 1j * traj_array[:, 1]
        if np.any(self.initial_velocity):
            complex_traj[0] += self.initial_velocity[0] + 1j * self.initial_velocity[1]

        spectral_factor = np.abs(np.fft.fft(complex_traj))
        spectral_factor = spectral_factor[0] / (np.max(spectral_factor) + 1e-12)

        return amps[indices] * envelope * np.exp(-1j * t) * spectral_factor


# --- Emergent ψ–G Physics Simulation ---
class PsiEmergentSim(GravitySim):
    """
    Subclass implementing emergent physics via observer wavefunctions.

    Parameters
    ----------
    alpha_psi : float
        Spectral complexity weight (ℏ analog).
    alpha_g : float
        Geometric overlap weight (G analog).
    initial_velocity : np.ndarray
        Optional initial velocity for all observers.
    """

    def __init__(self,
                alpha_psi: float = 0.01,
                alpha_g: float = 2.0,
                initial_velocity: Optional[np.ndarray] = None,
                span: float = 0,
                n_particles: int = 4096,
                n_steps: int = 100,
                blob_sigma: float = 0.1,
                n_candidates: int = 5):
        super().__init__(n_particles, n_steps, blob_sigma, n_candidates)
        self.t = 0.0
        self.alpha_psi = alpha_psi
        self.alpha_g = alpha_g
        self.initial_velocity = initial_velocity if initial_velocity is not None else np.zeros(2)

        # Create wavefunctions, one per blob
        self.wavefunctions: List[ObserverWavefunction] = [
            ObserverWavefunction(pos, self.blob_sigma, initial_velocity=self.initial_velocity)
            for pos in self.positions
        ]

        # Birth schedule and activity flags
        self.observer_span = span
        self.birth_frame = np.linspace(0, self.n_steps - 1, num=self.n_blobs, dtype=int) if self.observer_span > 0 else np.zeros(self.n_blobs, dtype=int)
        self.active_observers = [False] * self.n_blobs

    def compute_pdf(self, grid_points: np.ndarray) -> np.ndarray:
        """
        Compute the normalized probability density from all observer wavefunctions.
        """
        psi_total = np.zeros(grid_points.shape[0], dtype=np.complex128)
        for wf in self.wavefunctions:
            psi_total += wf.evaluate(grid_points, self.t)
        pdf = np.abs(psi_total) ** 2
        return pdf / (pdf.sum() + 1e-12)

    def calculate_spectral_complexity(self, pos: np.ndarray) -> float:
        """
        Estimate spectral entropy at candidate position.
        """
        grid_res = 8
        x = np.linspace(pos[0] - self.blob_sigma, pos[0] + self.blob_sigma, grid_res)
        y = np.linspace(pos[1] - self.blob_sigma, pos[1] + self.blob_sigma, grid_res)
        patch_density, _, _ = np.histogram2d(self.particles[:, 0], self.particles[:, 1], bins=[x, y])
        f_coeffs = np.abs(np.fft.fft2(patch_density))
        p_freq = f_coeffs.flatten() / (np.sum(f_coeffs) + 1e-10)
        return -np.sum(p_freq * np.log2(p_freq + 1e-10))


    def generate_candidates_emergent(self, blob_idx: int) -> np.ndarray:
        """
        Generate candidate positions for a blob using fully emergent displacement.
        
        Displacement is limited by:
        1. Local PDF variance (dense regions -> smaller moves)
        2. Trajectory MDL cost (high deviation -> penalized)
        
        Returns
        -------
        np.ndarray
            Candidate positions [n_candidates*2, 2].
        """
        current_pos = self.positions[blob_idx]

        # --- 1. Local PDF variance ---
        # Evaluate total PDF on a fine grid around current position
        grid_res = 12
        x = np.linspace(current_pos[0] - self.blob_sigma, current_pos[0] + self.blob_sigma, grid_res)
        y = np.linspace(current_pos[1] - self.blob_sigma, current_pos[1] + self.blob_sigma, grid_res)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack([X.ravel(), Y.ravel()])
        pdf_vals = self.compute_pdf(points)

        # Weighted variance gives local "spread"
        mu = np.average(points, axis=0, weights=pdf_vals)
        sigma_local = np.sqrt(np.average(np.sum((points - mu)**2, axis=1), weights=pdf_vals))
        
        # --- 2. Trajectory-based inertia ---
        traj = self.wavefunctions[blob_idx].trajectory
        if len(traj) < 2:
            inertia_vec = np.zeros(2)
        else:
            inertia_vec = traj[-1] - traj[-2]  # MDL prefers continuation of recent motion

        # --- 3. Generate candidates ---
        n_cand = self.n_candidates * 2
        r = sigma_local * np.sqrt(np.random.rand(n_cand))           # magnitude proportional to local PDF spread
        theta = np.random.rand(n_cand) * 2 * np.pi
        candidates = np.column_stack([
            current_pos[0] + r * np.cos(theta) + 0.5 * inertia_vec[0],
            current_pos[1] + r * np.sin(theta) + 0.5 * inertia_vec[1]
        ])

        # Keep candidates in [0,1]^2
        candidates = np.clip(candidates, 0, 1)
        return candidates



    def update_positions(self, new_particles: np.ndarray):
        """
        Move observers by evaluating ψ-G weighted candidate positions.

        Key changes:
        - Geometric cost is derived from total ψ-field at candidate location.
        - No local particle assignment.
        - Spectral cost remains as before.
        - Particles are only resampled from PDF for visualization.
        """
        self.t += 1
        self.particles = new_particles  # visualization only

        # Precompute grid and PDF once for efficiency
        grid_res = 64
        x = np.linspace(0, 1, grid_res)
        y = np.linspace(0, 1, grid_res)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        pdf_total = self.compute_pdf(grid_points)

        for i in range(self.n_blobs):
            if self.birth_frame[i] > self.t:
                continue
            if not self.active_observers[i]:
                self.active_observers[i] = True

            candidates = self.generate_candidates_emergent(i)
            costs = []

            for cand in candidates:
                # --- Geometric cost based on total ψ-field ---
                idx = np.argmin(np.linalg.norm(grid_points - cand, axis=1))
                c_g = 1.0 - pdf_total[idx]  # high local PDF → low cost

                # --- Spectral cost based on observer trajectory memory ---
                c_psi = self.calculate_spectral_complexity(cand)

                costs.append(self.alpha_g * c_g + self.alpha_psi * c_psi)

            costs = np.array(costs)
            probs = np.exp(-5.0 * (costs - np.min(costs)))
            probs /= probs.sum()

            chosen_idx = np.random.choice(len(candidates), p=probs)
            self.positions[i] = candidates[chosen_idx]
            self.wavefunctions[i].record_position(self.positions[i].copy())
            self.trajs[i].append(self.positions[i].copy())




# --- CLI Interface ---
def main():
    parser = argparse.ArgumentParser(description="Emergent ψ-G Simulation")
    parser.add_argument("--hbar", type=float, default=10, help="Spectral complexity weight ℏ")
    parser.add_argument("--G", type=float, default=20.0, help="Geometric overlap weight G")
    parser.add_argument("--vx", type=float, default=0)
    parser.add_argument("--vy", type=float, default=0)
    parser.add_argument("--sigma", type=float, default=0.10)
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--span", type=float, default=0.0)
    parser.add_argument("--file", type=str, default="emergent_sim")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--resolution", type=int, default=640)

    args = parser.parse_args()
    initial_velocity = np.array([args.vx, args.vy])

    sim = PsiEmergentSim(
        n_particles=args.particles,
        n_steps=args.steps,
        blob_sigma=args.sigma,
        alpha_psi=args.hbar,
        alpha_g=args.G,
        initial_velocity=initial_velocity,
        span=int(args.steps * args.span)
    )

    sim.run(f"{args.file}.{args.format}", res=args.resolution, fps=15)


if __name__ == "__main__":
    main()
