import numpy as np
import argparse
from typing import List, Optional
from qbitwave import QBitwave
from gbitwave import GBitwave
from visualization_engine import GravitySim  
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import scipy.ndimage

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
         # Track entropy history (bits) and effective energy (Joules)
        self.entropy_history: List[float] = []
        self.energy_history: List[float] = []

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


    def update_entropy_energy(self, kB: float = 1.380649e-23, T: float = 300.0):
        """
        Compute current entropy and effective energy based on bitstring changes.
        Energy uses Landauer's principle: E = k_B T ΔS ln 2.
        """
        # Compute Shannon entropy of current qbit
        current_entropy = self.qbit.entropy()  # in bits
        self.entropy_history.append(current_entropy)

        # Compute stepwise energy
        if len(self.entropy_history) > 1:
            delta_S = self.entropy_history[-1] - self.entropy_history[-2]
        else:
            delta_S = current_entropy  # first step

        energy = kB * T * delta_S * np.log(2)
        self.energy_history.append(energy)


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

    def __init__(
        self,
        initial_velocity: Optional[np.ndarray] = None,
        span: float = 0.0,
        n_particles: int = 4096,
        n_steps: int = 100,
        blob_sigma: float = 0.1,
        n_candidates: int = 5,
        hilbert_grid_size: int = 64,
        pdf_grid_size: int = 100
    ):
        super().__init__(n_particles, n_steps, blob_sigma, n_candidates)
        self.distance_history: List[float] = []
        self.velocity_history: List[float] = []
        self.acceleration_history: List[float] = []
        self.prev_positions: np.ndarray = self.positions.copy()

        self.t = 0.0
        self.initial_velocity = initial_velocity if initial_velocity is not None else np.zeros(2)
        self.hilbert_grid_size = hilbert_grid_size
        self.gbit = GBitwave(
            grid_size=self.hilbert_grid_size,
            smoothing=4
        )

        # Wavefunctions
        self.wavefunctions: List[ObserverWavefunction] = [
            ObserverWavefunction(pos, self.blob_sigma, initial_velocity=self.initial_velocity)
            for pos in self.positions
        ]

        # Birth schedule and activity flags
        self.observer_span = span
        self.birth_frame = (
            np.linspace(0, self.n_steps - 1, num=self.n_blobs, dtype=int)
            if self.observer_span > 0 else np.zeros(self.n_blobs, dtype=int)
        )
        self.active_observers = [False] * self.n_blobs

        # Visualization
        self.current_curvature: Optional[np.ndarray] = None

        # --- FIXED GRID FOR PDF ---
        x = np.linspace(0, 1, pdf_grid_size)
        y = np.linspace(0, 1, pdf_grid_size)
        X, Y = np.meshgrid(x, y)
        self.grid_points = np.column_stack([X.ravel(), Y.ravel()])  # Nx2 mesh

    def hilbert_to_curvature(self, hilbert_grid: np.ndarray, patch_size: int = 4, use_log: bool = True) -> np.ndarray:
        """
        Naive mapping from Hilbert-space amplitudes to a curvature-like 2D field.

        Parameters
        ----------
        hilbert_grid : np.ndarray
            NxN complex Hilbert grid of wavefunction amplitudes.
        patch_size : int
            Size of smoothing patch.
        use_log : bool
            Whether to log-scale the output to mimic curvature variation.

        Returns
        -------
        np.ndarray
            NxN real curvature grid.
        """
        # --- Compute local magnitude ---
        mag_grid = np.abs(hilbert_grid)

        # --- Smooth over small patches to reduce noise ---
        smoothed = scipy.ndimage.uniform_filter(mag_grid, size=patch_size)

        # --- Laplacian to get "curvature-like" signal ---
        laplacian = scipy.ndimage.laplace(smoothed)

        # --- Optional log scaling to mimic field nonlinearity ---
        if use_log:
            laplacian = np.sign(laplacian) * np.log1p(np.abs(laplacian))

        return laplacian


    # --- Naive Hilbert -> Geometry Projection ---
    def compute_curvature(self) -> np.ndarray:
        N = self.hilbert_grid_size
        hilbert_grid = np.zeros((N, N), dtype=np.complex128)

        for wf in self.wavefunctions:
            x_idx = int(np.clip(wf.center[0] * N, 0, N - 1))
            y_idx = int(np.clip(wf.center[1] * N, 0, N - 1))
            hilbert_grid[x_idx, y_idx] += np.sum(wf.qbit.get_amplitudes())

        return self.hilbert_to_curvature(hilbert_grid, patch_size=4, use_log=True)

    # --- Override update_positions ---
    def update_positions(self, new_particles: np.ndarray) -> None:
        """
        Update observers using wave_complexity-driven MDL inertia.
        new_particles: array from resample_particles()
        """
        for i, wf in enumerate(self.wavefunctions):
            if self.birth_frame[i] > self.t:
                continue
            if not self.active_observers[i]:
                self.active_observers[i] = True

            current_pos = self.positions[i]
            candidates_with_vel = self.generate_candidates_for_blob(i, current_pos, velocity=self.initial_velocity)
            candidates = [c[0] for c in candidates_with_vel]  # drop velocities for evaluation

            # Evaluate wavefunction complexity for each candidate
            complexities = []
            for cand in candidates:
                wf_center_backup = wf.center.copy()
                wf.record_position(cand)
                
                L_psi = wf.qbit.wave_complexity()
                L_G = self.gbit.geometric_complexity(wf.trajectory + [cand])
                joint_cost = L_psi +  L_G

                complexities.append(joint_cost)
                wf.trajectory.pop()
                wf.center = wf_center_backup

            complexities = np.array(complexities)
            delta_c = np.max(complexities) - np.min(complexities) + 1e-12
            k = 3.0 / delta_c
            probs = np.exp(-k * (complexities - np.min(complexities)))
            probs /= probs.sum()

            chosen_idx = np.random.choice(len(candidates), p=probs)
            chosen_pos = candidates[chosen_idx]

            # Commit new position
            self.positions[i] = chosen_pos
            wf.record_position(chosen_pos)
            self.trajs[i].append(chosen_pos.copy())

    def step(self) -> None:
        """Perform a single simulation step."""
        # --- 1. Compute PDF from wavefunctions ---
        psi_total = np.zeros(self.grid_points.shape[0], dtype=np.complex128)
        for wf in self.wavefunctions:
            psi_total += wf.evaluate(self.grid_points, self.t)
            wf.update_entropy_energy(kB=1.380649e-23, T=300.0)  # room temp


        # --- 3. Update observer positions using wavefunction MDL ---
        self.update_positions(self.particles)

        # --- 4. Compute naive curvature from Hilbert grid ---
        self.current_curvature = self.compute_curvature()

        # --- 5. Advance simulation time ---
        self.t += 1

        # --- 6. Store previous positions for velocity/acceleration ---
        pos_array = np.array(self.positions)
        prev_array = np.array(getattr(self, "prev_positions", pos_array))
        delta = pos_array - prev_array

        # Distance: sum of all observers' displacement magnitudes
        total_dist = np.linalg.norm(delta, axis=1).sum()
        if not hasattr(self, "distance_history"):
            self.distance_history = []
            self.velocity_history = []
            self.acceleration_history = []
        self.distance_history.append(total_dist)

        # Velocity: mean speed
        mean_vel = np.mean(np.linalg.norm(delta, axis=1))
        self.velocity_history.append(mean_vel)

        # Acceleration: change in velocity
        if len(self.velocity_history) > 1:
            accel = mean_vel - self.velocity_history[-2]
        else:
            accel = 0.0
        self.acceleration_history.append(accel)

        # Update previous positions
        self.prev_positions = pos_array.copy()

    @property
    def total_energy(self) -> float:
        return sum(sum(wf.energy_history) for wf in self.wavefunctions)

    def run_with_curvature(self, save_video: str, res: int, fps: int = 15) -> None:
        plt.style.use("dark_background")
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(4, 2, width_ratios=[2.5, 1.0])

        ax_geom = fig.add_subplot(gs[:, 0])
        ax_dist = fig.add_subplot(gs[0, 1])
        ax_vel  = fig.add_subplot(gs[1, 1])
        ax_acc  = fig.add_subplot(gs[2, 1])
        ax_energy = fig.add_subplot(gs[3, 1])  # replace or move as needed

        def update_frame(step_idx: int):
            self.step()  # wavefunction PDF → particles, update curvature

            # --- Clear axes ---
            ax_geom.clear()
            ax_dist.clear()
            ax_vel.clear()
            ax_acc.clear()
            ax_energy.clear()

            # --- Plot curvature ---
            if self.current_curvature is not None:
                norm = plt.Normalize(np.min(self.current_curvature), np.max(self.current_curvature))
                ax_geom.imshow(
                    self.current_curvature.T,
                    origin="lower",
                    cmap="coolwarm",
                    norm=norm,
                    extent=[0, 1, 0, 1],
                    alpha=0.7
                )

            # --- Overlay observer particles from wavefunction ---
            ax_geom.scatter(
                self.particles[:, 0],
                self.particles[:, 1],
                c="cyan",
                s=30,
                alpha=0.8,
                edgecolors="white"
            )

            ax_geom.set_xlim(0, 1)
            ax_geom.set_ylim(0, 1)
            ax_geom.set_title(f"Emergent Geometry (Step {step_idx+1})")

            # --- Diagnostics panels ---
            t = np.arange(len(self.distance_history))
            ax_dist.plot(t, self.distance_history, color="white")
            ax_dist.axvline(step_idx, color="gray", alpha=0.4)
            ax_dist.set_title("Distance")

            ax_vel.plot(t, self.velocity_history, color="orange")
            ax_vel.axvline(step_idx, color="gray", alpha=0.4)
            ax_vel.set_title("Velocity")

            ax_acc.plot(t, self.acceleration_history, color="red")
            ax_acc.axvline(step_idx, color="gray", alpha=0.4)
            ax_acc.set_title("Acceleration")

            # --- Plot total energy ---
            total_energy_steps = [sum(wf.energy_history[:step_idx+1]) for wf in self.wavefunctions]
            total_energy_array = np.sum(total_energy_steps, axis=0)
            ax_energy.plot(total_energy_array, color="yellow")
            ax_energy.set_title("Cumulative Energy (J)")
            ax_energy.set_xlim(0, self.n_steps)

            for ax in (ax_dist, ax_vel, ax_acc):
                ax.set_xlim(0, self.n_steps)

        # --- Writer ---
        ext = save_video.split(".")[-1].lower()
        if ext in ["mp4", "avi"]:
            writer = FFMpegWriter(fps=fps, metadata=dict(artist="IaMe"), bitrate=1800)
        elif ext in ["gif"]:
            writer = PillowWriter(fps=fps)
        else:
            raise ValueError(f"Unsupported video format: {ext}")

        ani = FuncAnimation(fig, update_frame, frames=self.n_steps, blit=False)
        ani.save(save_video, writer=writer)
        plt.close(fig)
        print(f"Simulation saved as {save_video}")


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

    #sim.run(f"{args.file}.{args.format}", res=args.resolution, fps=15)
    sim.run_with_curvature(f"{args.file}.{args.format}", res=args.resolution, fps=15)


if __name__ == "__main__":
    main()
