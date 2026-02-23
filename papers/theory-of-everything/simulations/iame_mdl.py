import numpy as np
import argparse
from typing import List, Optional
from qbitwave import QBitwaveMDL as QBitwave
from visualization_engine import GravitySim


class ObserverWavefunction:
    """
    Observer represented as integer spectral encoding of its full trajectory.

    The trajectory is encoded via FFT into discrete spectral modes.
    Spectral complexity acts as an informational cost functional.

    Parameters
    ----------
    center : np.ndarray
        Initial 2D position.
    sigma : float
        Spatial width parameter (size of the observer).
    oscillations : float
        Number of oscillations in the wavepacket.
    initial_velocity : np.ndarray, optional
        Initial velocity vector.
    """

    def __init__(self,
                 center: np.ndarray,
                 sigma: float,
                 oscillations: float,
                 initial_velocity: Optional[np.ndarray] = None):

        self.center = np.array(center, dtype=float)
        self.sigma = sigma
        self.oscillations = oscillations
        self.trajectory: List[np.ndarray] = [self.center.copy()]
        self.initial_velocity = (
            np.array(initial_velocity, dtype=float)
            if initial_velocity is not None
            else np.zeros_like(self.center)
        )

        self.qbit = QBitwave(N=256)

    # ------------------------------------------------------------

    def record_position(self, pos: np.ndarray):
        """Append a new position to the trajectory."""
        pos = np.array(pos, dtype=float)
        self.trajectory.append(pos)
        self.center = pos



    def rebuild_spectral_modes(self, trajectory: Optional[np.ndarray] = None):
        """
        Rebuild spectral representation from trajectory.

        Parameters
        ----------
        trajectory : np.ndarray, optional
            Alternative trajectory to encode (used for candidate testing).
        """
        self.qbit.clear_modes()

        traj = np.array(self.trajectory if trajectory is None else trajectory)
        if len(traj) < 2:
            return

        # Interpret 2D trajectory as complex signal
        complex_traj = traj[:, 0] + 1j * traj[:, 1]

        fft_vals = np.fft.fft(complex_traj)

        for k, coeff in enumerate(fft_vals):
            A = np.abs(coeff)
            if A < 1e-10:  # numerical threshold instead of integer zero
                continue

            phi = np.angle(coeff)

            self.qbit.add_mode(k, A, phi)


    def spectral_complexity(self,
                            trajectory: Optional[np.ndarray] = None) -> float:
        """
        Compute informational complexity of trajectory.

        Complexity = weighted spectral energy.

        Parameters
        ----------
        trajectory : np.ndarray, optional
            Alternative trajectory for hypothetical evaluation.

        Returns
        -------
        float
            Spectral complexity.
        """
        self.rebuild_spectral_modes(trajectory)

        complexity = 0.0
        for k, amp, _ in self.qbit.modes:
            complexity += (k ** 2) * (amp ** 2)

        return complexity

    def evaluate(self, points: np.ndarray, t: float) -> np.ndarray:
        """
        Complex Gaussian wavepacket with momentum phase.
        """
        diff = points - self.center
        r2 = np.sum(diff ** 2, axis=1)

        # infer local velocity (momentum)
        if len(self.trajectory) >= 2:
            v = self.trajectory[-1] - self.trajectory[-2]
        else:
            v = np.zeros(2)

        # plane-wave phase term
        k_micro = 2 * np.pi * self.oscillations / self.sigma

        v = self.trajectory[-1] - self.trajectory[-2] if len(self.trajectory) >= 2 else np.zeros(2)

        norm = np.linalg.norm(v)
        direction = v / (norm + 1e-12)

        k_micro = 2 * np.pi * self.oscillations / self.sigma
        phase = points @ direction * k_micro


        envelope = np.exp(-r2 / (2 * self.sigma ** 2))
        return envelope * np.exp(1j * phase)




class PsiEmergentSim(GravitySim):
    """
    Emergent observer dynamics driven by spectral informational complexity.

    Observers evolve by minimizing the increase in spectral encoding cost
    of their full trajectory.

    No geometric forces are applied — inertia emerges from compression cost.
    """

    def __init__(self,
                 initial_velocity: Optional[np.ndarray] = None,
                 span: float = 0.0,
                 n_particles: int = 4096,
                 n_steps: int = 100,
                 oscillations_per_radius: float = 5.0 ,
                 blob_sigma: float = 0.1,
                 n_candidates: int = 5):
        """
        Initialize emergent simulation.

        Parameters
        ----------
        initial_velocity : np.ndarray, optional
            Initial velocity vector.
        span : float
            Fraction of total steps during which observers are born.
        n_particles : int
            Number of visualization particles.
        n_steps : int
            Number of simulation steps.
        blob_sigma : float
            Gaussian width of each observer.
        oscillations_per_radius : float
            Number of oscillations in the observer wavepacket per σ radius.
        n_candidates : int
            Base number of candidate positions (internally doubled).
        """
        super().__init__(n_particles, n_steps, blob_sigma, n_candidates)

        self.t = 0.0
        self.oscillations_per_radius = oscillations_per_radius
        self.initial_velocity = (
            initial_velocity if initial_velocity is not None
            else np.zeros(2)
        )

        self.wavefunctions: List[ObserverWavefunction] = [
            ObserverWavefunction(pos, self.blob_sigma, 
                                 oscillations = self.oscillations_per_radius,
                                 initial_velocity=self.initial_velocity)
            for pos in self.positions
        ]

        # Interpret span as fraction of total simulation
        if span > 0:
            birth_window = int(self.n_steps * span)
            self.birth_frame = np.linspace(
                0, birth_window, num=self.n_blobs, dtype=int
            )
        else:
            self.birth_frame = np.zeros(self.n_blobs, dtype=int)

        self.active_observers = [False] * self.n_blobs


    def compute_pdf(self, grid_points: np.ndarray) -> np.ndarray:
        """
        Compute Born probability from total coherent wavefunction.
        """
        psi_total = np.zeros(len(grid_points), dtype=complex)
        for wf in self.wavefunctions:
            psi_total += wf.evaluate(grid_points, self.t)
        prob = np.abs(psi_total) ** 2
        s = prob.sum()
        if s <= 0:
            prob[:] = 1.0 / len(prob)
        else:
            prob /= s
        return prob


    def compute_pdf_new(self, grid_points: np.ndarray) -> np.ndarray:
        """
        Compute visualization PDF from all active observer wavefunctions.
        """

        rho = np.zeros(len(grid_points))

        for i, wf in enumerate(self.wavefunctions):

            # skip unborn observers
            if not self.active_observers[i]:
                continue

            psi = wf.evaluate(grid_points, self.t)
            rho += np.abs(psi) ** 2

        total = rho.sum()
        if total > 0:
            rho /= total
        else:
            rho = np.ones_like(rho) / len(rho)

        return rho

    def get_spectral_data(self):
        """
        Uses the same definition as total_spectral_complexity,
        but returns the per-k weighted spectrum.
        """
        complex_signal = []

        for wf in self.wavefunctions:
            traj = np.array(wf.trajectory)
            if len(traj) < 2:
                continue
            complex_signal.extend(traj[:, 0] + 1j * traj[:, 1])

        if len(complex_signal) < 2:
            return None, None

        z = np.array(complex_signal)
        fft_vals = np.fft.fft(z)

        N = len(fft_vals)
        k = np.arange(N)
        k_eff = np.minimum(k, N - k)

        power = np.abs(fft_vals) ** 2
        weighted = (k_eff ** 2) * power

        return k_eff[:N//2], weighted[:N//2]

    def generate_candidates_emergent(self, blob_idx: int) -> np.ndarray:
        """
        Generate candidate positions based on:

        1. Local PDF spread
        2. Trajectory-based inertia

        Parameters
        ----------
        blob_idx : int
            Observer index.

        Returns
        -------
        np.ndarray
            Candidate positions of shape (2*n_candidates, 2).
        """
        current_pos = self.positions[blob_idx]

        # --- Local PDF sampling grid ---
        grid_res = 12
        x = np.linspace(current_pos[0] - self.blob_sigma,
                        current_pos[0] + self.blob_sigma,
                        grid_res)
        y = np.linspace(current_pos[1] - self.blob_sigma,
                        current_pos[1] + self.blob_sigma,
                        grid_res)

        X, Y = np.meshgrid(x, y)
        points = np.column_stack([X.ravel(), Y.ravel()])
        pdf_vals = self.compute_pdf(points)

        # Weighted local mean and spread
        mu = np.average(points, axis=0, weights=pdf_vals)
        sigma_local = np.sqrt(
            np.average(np.sum((points - mu) ** 2, axis=1),
                       weights=pdf_vals)
        )

        # --- Inertia from trajectory ---
        traj = self.wavefunctions[blob_idx].trajectory
        if len(traj) < 2:
            inertia_vec = np.zeros(2)
        else:
            inertia_vec = traj[-1] - traj[-2]

        # --- Candidate sampling ---
        n_cand = self.n_candidates * 2
        r = sigma_local * np.sqrt(np.random.rand(n_cand))
        theta = 2 * np.pi * np.random.rand(n_cand)

        candidates = np.column_stack([
            current_pos[0] + r * np.cos(theta) + 0.5 * inertia_vec[0],
            current_pos[1] + r * np.sin(theta) + 0.5 * inertia_vec[1]
        ])

        return np.clip(candidates, 0.0, 1.0)

    # ------------------------------------------------------------

    def update_positions(self, new_particles: np.ndarray):
        """
        Update observer positions by minimizing joint universal
        spectral complexity.
        """
        self.t += 1
        self.particles = new_particles

        # Compute baseline complexity
        base_complexity = self.total_spectral_complexity()

        for i, wf in enumerate(self.wavefunctions):

            if self.birth_frame[i] > self.t:
                continue

            if not self.active_observers[i]:
                self.active_observers[i] = True

            candidates = self.generate_candidates_emergent(i)

            complexities = []

            for cand in candidates:
                # --- Temporarily modify trajectory ---
                wf.trajectory.append(cand)

                # Compute new joint complexity
                c_total = self.total_spectral_complexity()
                complexities.append(c_total)

                # Restore state
                wf.trajectory.pop()

            complexities = np.array(complexities)

            delta_c = max(
                np.max(complexities) - np.min(complexities),
                1e-6
            )

            k = 3.0 / delta_c
            probs = np.exp(-k * (complexities - np.min(complexities)))
            probs /= probs.sum()

            chosen_idx = np.random.choice(len(candidates), p=probs)
            chosen_pos = candidates[chosen_idx]

            # Apply permanently
            self.positions[i] = chosen_pos
            wf.record_position(chosen_pos)
            self.trajs[i].append(chosen_pos.copy())


    def get_complexity_spectrum(self):
        """
        Returns (k_eff, weighted_power, raw_power)
        for the joint trajectory signal.
        """
        complex_signal = []

        for wf in self.wavefunctions:
            traj = np.array(wf.trajectory)
            if len(traj) < 2:
                continue
            complex_signal.extend(traj[:, 0] + 1j * traj[:, 1])

        if len(complex_signal) < 2:
            return None, None, None

        z = np.array(complex_signal)
        fft_vals = np.fft.fft(z)

        N = len(fft_vals)
        k = np.arange(N)
        k_eff = np.minimum(k, N - k)

        power = np.abs(fft_vals) ** 2
        weighted = (k_eff ** 2) * power

        half = N // 2
        return k_eff[:half], weighted[:half], power[:half]

    def total_spectral_complexity(self) -> float:
        """
        Compute spectral complexity of the joint universal wavefunction.

        All observer trajectories are concatenated into a single complex
        signal and encoded spectrally.

        Returns
        -------
        float
            Total informational complexity of the universe.
        """
        total = 0.0

        for wf in self.wavefunctions:
            traj = np.array(wf.trajectory)
            if len(traj) < 2:
                continue

            complex_traj = traj[:, 0] + 1j * traj[:, 1]
            fft_vals = np.fft.fft(complex_traj)

            N = len(fft_vals)
            for k, coeff in enumerate(fft_vals):
                amp = np.abs(coeff)
                k_eff = min(k, N - k)
                total += (k_eff ** 2) * (amp ** 2)

        return total

# --- IaMe demonstrator ---
def main():
    parser = argparse.ArgumentParser(description="Emergent ψ Simulation")
    parser.add_argument("--vx", type=float, default=0, help="Initial velocity x-component")
    parser.add_argument("--vy", type=float, default=0, help="Initial velocity y-component")
    parser.add_argument("--sigma", type=float, default=0.20, help="Blob size (σ)")
    parser.add_argument("--osc", type=float, default=10, help="Oscillations per observer radius")
    parser.add_argument("--particles", type=int, default=1024, help="Number of particles for visualization")
    parser.add_argument("--steps", type=int, default=100, help="Number of simulation steps")
    parser.add_argument("--span", type=float, default=0.0, help="Fraction of total steps over which observers are born")
    parser.add_argument("--file", type=str, default="emergent_sim")
    parser.add_argument("--format", choices=["mp4", "gif"], default="mp4")
    parser.add_argument("--resolution", type=int, default=128, help="Output video resolution (width in pixels)")

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
