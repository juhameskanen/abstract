import numpy as np
import argparse
from typing import List, Optional
from qbitwave import QBitwaveMDL as QBitwave
from visualization_engine import GravitySim  


class ObserverWavefunction:
    """
    Represents a single observer as a wavefunction with memory in 2D space.

    The observer's motion emerges from minimizing the spectral complexity of
    its wavefunction, ensuring smooth trajectories and inertial behavior
    without hard-coded geometric forces.

    Attributes
    ----------
    center : np.ndarray
        Current 2D position of the observer.
    sigma : float
        Spatial localization width of the Gaussian envelope.
    trajectory : List[np.ndarray]
        Historical positions for memory and spectral complexity computation.
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
        """Append a new position to the trajectory and update center."""
        self.trajectory.append(np.array(pos, dtype=float))
        self.center = np.array(pos, dtype=float)

    def evaluate(self, points: np.ndarray, t: float) -> np.ndarray:
        """
        Compute complex amplitude of the wavefunction at given spatial points.

        Parameters
        ----------
        points : np.ndarray
            Array of 2D coordinates to evaluate.
        t : float
            Current simulation time step.

        Returns
        -------
        np.ndarray
            Complex amplitudes at each point.
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
            # Inject initial velocity into the first position
            complex_traj[0] += self.initial_velocity[0] + 1j * self.initial_velocity[1]

        spectral_factor = np.abs(np.fft.fft(complex_traj))
        spectral_factor = spectral_factor[0] / (np.max(spectral_factor) + 1e-12)

        return amps[indices] * envelope * np.exp(-1j * t) * spectral_factor


# --- Emergent ψ–G Physics Simulation ---
class PsiEmergentSim(GravitySim):
    """
    Emergent observer dynamics driven purely by wavefunction spectral complexity.

    Observers evolve in 2D space without any geometric forces. Inertia emerges
    naturally: rapid changes require high-frequency components, which increases
    wavefunction complexity and is thus probabilistically suppressed.

    Parameters
    ----------
    initial_velocity : np.ndarray, optional
        Initial 2D velocity for all observers to seed motion.
    span : float
        Fraction of total steps over which observers are born.
    n_particles : int
        Number of particles for visualization.
    n_steps : int
        Number of simulation steps.
    blob_sigma : float
        Spatial size of each observer blob.
    n_candidates : int
        Number of candidate positions evaluated per step.
    """

    def __init__(self,
                 initial_velocity: Optional[np.ndarray] = None,
                 span: float = 0,
                 n_particles: int = 4096,
                 n_steps: int = 100,
                 blob_sigma: float = 0.1,
                 n_candidates: int = 5):
        super().__init__(n_particles, n_steps, blob_sigma, n_candidates)
        self.t = 0.0
        self.initial_velocity = initial_velocity if initial_velocity is not None else np.zeros(2)

        # Initialize wavefunctions for each observer
        self.wavefunctions: List[ObserverWavefunction] = [
            ObserverWavefunction(pos, self.blob_sigma, initial_velocity=self.initial_velocity)
            for pos in self.positions
        ]

        # Birth schedule for observers
        self.observer_span = span
        self.birth_frame = np.linspace(0, self.n_steps - 1, num=self.n_blobs, dtype=int) if self.observer_span > 0 else np.zeros(self.n_blobs, dtype=int)
        self.active_observers = [False] * self.n_blobs

    def compute_pdf(self, grid_points: np.ndarray) -> np.ndarray:
        """
        Compute normalized probability density from all observer wavefunctions.

        Parameters
        ----------
        grid_points : np.ndarray
            Array of 2D points to evaluate PDF.

        Returns
        -------
        np.ndarray
            Normalized PDF across points.
        """
        psi_total = np.zeros(grid_points.shape[0], dtype=np.complex128)
        for wf in self.wavefunctions:
            psi_total += wf.evaluate(grid_points, self.t)
        pdf = np.abs(psi_total) ** 2
        return pdf / (pdf.sum() + 1e-12)

    def calculate_spectral_complexity(self, pos: np.ndarray) -> float:
        """
        Estimate local spectral entropy (complexity) from particle distribution.

        Parameters
        ----------
        pos : np.ndarray
            2D candidate position.

        Returns
        -------
        float
            Spectral complexity (Shannon entropy in bits).
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
        Generate candidate positions for an observer based on local PDF spread
        and trajectory inertia.

        Parameters
        ----------
        blob_idx : int
            Index of the observer.

        Returns
        -------
        np.ndarray
            Candidate positions [n_candidates*2, 2].
        """
        current_pos = self.positions[blob_idx]

        # --- Local PDF spread ---
        grid_res = 12
        x = np.linspace(current_pos[0] - self.blob_sigma, current_pos[0] + self.blob_sigma, grid_res)
        y = np.linspace(current_pos[1] - self.blob_sigma, current_pos[1] + self.blob_sigma, grid_res)
        X, Y = np.meshgrid(x, y)
        points = np.column_stack([X.ravel(), Y.ravel()])
        pdf_vals = self.compute_pdf(points)

        mu = np.average(points, axis=0, weights=pdf_vals)
        sigma_local = np.sqrt(np.average(np.sum((points - mu) ** 2, axis=1), weights=pdf_vals))

        # --- Trajectory-based inertia ---
        traj = self.wavefunctions[blob_idx].trajectory
        inertia_vec = np.zeros(2) if len(traj) < 2 else traj[-1] - traj[-2]

        # --- Generate candidates ---
        n_cand = self.n_candidates * 2
        r = sigma_local * np.sqrt(np.random.rand(n_cand))
        theta = np.random.rand(n_cand) * 2 * np.pi
        candidates = np.column_stack([
            current_pos[0] + r * np.cos(theta) + 0.5 * inertia_vec[0],
            current_pos[1] + r * np.sin(theta) + 0.5 * inertia_vec[1]
        ])
        candidates = np.clip(candidates, 0, 1)
        return candidates

    def update_positions(self, new_particles: np.ndarray):
        """
        Update observer positions by minimizing wavefunction spectral complexity.

        Principles
        ----------
        1. Small candidate displacements are generated around current positions.
        2. Wavefunction complexity is evaluated for each candidate.
        3. Probabilistic selection favors candidates that minimally increase complexity.
        4. Inertia and smooth motion emerge naturally; no geometric forces applied.
        """
        self.t += 1
        self.particles = new_particles  # visualization only

        for i, wf in enumerate(self.wavefunctions):
            if self.birth_frame[i] > self.t:
                continue
            if not self.active_observers[i]:
                self.active_observers[i] = True

            current_pos = self.positions[i]
            candidates = self.generate_candidates_emergent(i)

            complexities = []
            for cand in candidates:
                wf_center_backup = wf.center.copy()
                wf.record_position(cand)
                complexities.append(wf.qbit.wave_complexity())
                wf.trajectory.pop()
                wf.center = wf_center_backup

            complexities = np.array(complexities)
            delta_c = np.max(complexities) - np.min(complexities) + 1e-12
            k = 3.0 / delta_c
            probs = np.exp(-k * (complexities - np.min(complexities)))
            probs /= probs.sum()

            chosen_idx = np.random.choice(len(candidates), p=probs)
            chosen_pos = candidates[chosen_idx]

            self.positions[i] = chosen_pos
            wf.record_position(chosen_pos)
            self.trajs[i].append(chosen_pos.copy())


# --- IaMe demonstrator ---
def main():
    parser = argparse.ArgumentParser(description="Emergent ψ Simulation")
    parser.add_argument("--vx", type=float, default=10, help="Initial velocity x-component")
    parser.add_argument("--vy", type=float, default=7, help="Initial velocity y-component")
    parser.add_argument("--sigma", type=float, default=0.10, help="Blob size (σ)")
    parser.add_argument("--particles", type=int, default=4096, help="Number of particles for visualization")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps")
    parser.add_argument("--span", type=float, default=0.0, help="Fraction of total steps over which observers are born")
    parser.add_argument("--file", type=str, default="emergent_sim")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--resolution", type=int, default=640, help="Output video resolution (width in pixels)")

    args = parser.parse_args()
    initial_velocity = np.array([args.vx, args.vy])

    sim = PsiEmergentSim(
        n_particles=args.particles,
        n_steps=args.steps,
        blob_sigma=args.sigma,
        initial_velocity=initial_velocity,
        span=int(args.steps * args.span)
    )

    sim.run(f"{args.file}.{args.format}", res=args.resolution, fps=15)


if __name__ == "__main__":
    main()
