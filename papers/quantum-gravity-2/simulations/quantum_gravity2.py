#!/usr/bin/env python3
"""
Wavefunction-based Gravity Simulation 
Integrated with Triple-Trace Diagnostic Plots and QBitwave Logic.
"""
import argparse
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eigh
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

class Wavefunction:
    def __init__(
        self,
        center: np.ndarray,
        covariance: np.ndarray,
        amplitude: float = 1.0,
        omega: float = 0.0,
        phase: float = 0.0,
    ):
        self.center = np.array(center, dtype=float)
        self.covariance = np.array(covariance, dtype=float)
        self.amplitude = float(amplitude)
        self.omega = float(omega)
        self.phase = float(phase)

    def evaluate(self, points: np.ndarray, t: float) -> np.ndarray:
        diff = points - self.center
        cov = self.covariance + 1e-9 * np.eye(2)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(2))
        exponent = -0.5 * np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
        env = self.amplitude * np.exp(exponent)
        phase_factor = np.exp(-1j * (self.omega * t - self.phase))
        return env * phase_factor

class WavefunctionGravitySim:
    def __init__(self, n_particles=4096, n_steps=200, blob_sigma_init=0.12, 
                 n_candidates=8, grid_size=100, size=1.0, jitter_mean=0.05, 
                 jitter_cov=0.01, seed=1):
        np.random.seed(seed)
        # --- Restored Original Args ---
        self.n_particles = int(n_particles)
        self.n_steps = int(n_steps)
        self.n_candidates = int(n_candidates)
        self.grid_size = int(grid_size)
        self.jitter_mean = float(jitter_mean)
        self.jitter_cov = float(jitter_cov)
        self.size = size
        self.t, self.dt = 0.0, 1.0

        # Observer Setup
        self.positions = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]])
        self.n_blobs = len(self.positions)
        self.colors = ['#1f77b4', '#d62728', '#2ca02c']
        
        var0 = (blob_sigma_init ** 2) * size
        self.covariances = [np.eye(2) * var0 for _ in range(self.n_blobs)]
        self.target_area = np.array([np.linalg.det(cov) for cov in self.covariances])

        self.wavefunctions = [
            Wavefunction(center=self.positions[i].copy(),
                         covariance=self.covariances[i].copy(),
                         omega=np.random.normal(scale=0.05),
                         phase=np.random.uniform(0, 2*np.pi))
            for i in range(self.n_blobs)
        ]

        # Initialization
        grid_pts = self._make_grid_points()
        pdf_init = self._compute_pdf_on_grid(grid_pts)
        self.particles = self._resample_particles_from_grid(pdf_init, grid_pts)

        # Histories for MDL and Plots
        self.trajs_mean = [[wf.center.copy()] for wf in self.wavefunctions]
        self.history_dist = [[] for _ in range(self.n_blobs)]
        self.history_vel = [[] for _ in range(self.n_blobs)]
        self.history_acc = [[] for _ in range(self.n_blobs)]


    def _make_grid_points(self):
        x = np.linspace(0, 1, self.grid_size)
        y = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, y)
        return np.column_stack([X.ravel(), Y.ravel()])

    def _compute_pdf_on_grid(self, grid_points: np.ndarray) -> np.ndarray:
        psi_total = np.zeros(grid_points.shape[0], dtype=np.complex128)
        for wf in self.wavefunctions:
            psi_total += wf.evaluate(grid_points, self.t)
        pdf = np.abs(psi_total) ** 2 + 1e-20
        return pdf / pdf.sum()

    def _resample_particles_from_grid(self, pdf: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
        idx = np.random.choice(grid_points.shape[0], size=self.n_particles, p=pdf)
        return grid_points[idx]

    def assign_particles_to_blob(self, blob_index: int) -> np.ndarray:
        mu = self.wavefunctions[blob_index].center
        Sigma = self.wavefunctions[blob_index].covariance + 1e-9 * np.eye(2)
        invS = np.linalg.inv(Sigma)
        diff = self.particles - mu
        d2 = np.einsum('ij,jk,ik->i', diff, invS, diff)
        return self.particles[d2 < 9.0]

    def generate_candidates_for_blob(self, blob_index: int, assigned: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
        wf = self.wavefunctions[blob_index]
        if assigned.size < 2:
            return [(wf.center, wf.covariance, wf.phase, wf.omega)]
        
        mu_mle = assigned.mean(axis=0)
        raw_cov = np.cov(assigned.T) + 1e-9 * np.eye(2)
        candidates = []
        for _ in range(self.n_candidates):
            jitter = np.random.normal(scale=self.jitter_cov, size=(2,2))
            c_cov = raw_cov + (jitter @ jitter.T)
            scale = np.sqrt(self.target_area[blob_index] / np.linalg.det(c_cov))
            candidates.append((mu_mle, c_cov * scale, wf.phase, wf.omega))
        return candidates

    def compute_spectral_entropy(self, trajectory_segment: np.ndarray) -> float:
        """
        Calculates H: the complexity of the description. 
        Smoother paths = lower spectral entropy.
        """
        if len(trajectory_segment) < 2:
            return 0.0
            
        # Compute FFT across the time axis (axis 0) for both x and y
        fft_vals = np.fft.rfft(trajectory_segment, axis=0)
        
        # Power Spectral Density
        psd = np.abs(fft_vals)**2
        
        # Normalize to get a probability distribution across modes
        psd_sum = np.sum(psd)
        if psd_sum < 1e-12: 
            return 0.0
            
        p = psd / psd_sum
        
        # Shannon entropy of the power spectrum (the "Complexity" of the path)
        return -np.sum(p * np.log2(p + 1e-12))

    def description_length_full(self, blob_index: int, candidate: Tuple) -> float:
        """
        The Informational Action: S_I = H + alpha * D + Body_Integrity.
        """
        mean_c, cov_c, phase_c, _ = candidate
        window = 8
        history = self.trajs_mean[blob_index]
        
        # Construct trajectory segment for Spectral Entropy (H)
        segment = np.array(history[-window:] + [mean_c])
        
        # 1. Potential/Complexity (Spectral Entropy)
        H = self.compute_spectral_entropy(segment)
        
        # 2. Kinetic/Inertia (Quantum Infidelity)
        # Using the Euclidean distance between centers as a phase-coherence proxy
        mu_old = history[-1] if history else mean_c
        D = np.linalg.norm(mean_c - mu_old)**2
        
        # 3. THE FIX: Body Integrity (Anisotropy penalty)
        vals, _ = eigh(cov_c)
        # Use np.maximum for element-wise floor to avoid log(0) or division by zero
        vals = np.maximum(vals, 1e-12) 
        
        # L_shape ensures the observer maintains a 2D/3D 'body' 
        # instead of collapsing into a string or point.
        L_shape = (np.log(np.max(vals) / np.min(vals)))**2
        
        # 4. Scale Persistence (Information Volume)
        curr_det = np.linalg.det(cov_c)
        target_det = self.target_area[blob_index]
        L_scale = (np.log(curr_det / target_det))**2
        
        # Weights for the 'Observer Filter'
        return float(H + (5.0 * D) + (10.0 * L_shape) + (50.0 * L_scale))

    def select_best_candidates_all(self) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
        chosen = []
        for i in range(self.n_blobs):
            assigned = self.assign_particles_to_blob(i)
            cands = self.generate_candidates_for_blob(i, assigned)
            costs = [self.description_length_full(i, c) for c in cands]
            chosen.append(cands[np.argmin(costs)])
        return chosen

    def apply_chosen_candidates(self, chosen: List[Tuple[np.ndarray, np.ndarray, float, float]]):
        for i, (m, c, p, o) in enumerate(chosen):
            wf = self.wavefunctions[i]
            wf.center, wf.covariance, wf.phase, wf.omega = m, c, p, o
            self.trajs_mean[i].append(m.copy())

    def update_stats(self):
        # Calculate Barycenter of all blobs
        centers = np.array([wf.center for wf in self.wavefunctions])
        barycenter = centers.mean(axis=0)
        
        for i in range(self.n_blobs):
            d = np.linalg.norm(self.wavefunctions[i].center - barycenter)
            self.history_dist[i].append(d)
            if len(self.history_dist[i]) > 1:
                v = self.history_dist[i][-1] - self.history_dist[i][-2]
                self.history_vel[i].append(v)
                if len(self.history_vel[i]) > 1:
                    self.history_acc[i].append(self.history_vel[i][-1] - self.history_vel[i][-2])
                else: self.history_acc[i].append(0.0)
            else:
                self.history_vel[i].append(0.0)
                self.history_acc[i].append(0.0)

    def run(self, save_gif: str = "wavefunction_gravity.gif", fps: int = 15):
        grid_points = self._make_grid_points()
        self.fig = plt.figure(figsize=(14, 8))
        
        # Left Panel
        self.ax_sim = self.fig.add_subplot(1, 2, 1)
        
        # Right Panel: host_subplot for triple y-axes
        self.ax_stats = host_subplot(111, axes_class=AA.Axes, figure=self.fig)
        self.ax_stats.set_position([0.55, 0.1, 0.35, 0.8])
        
        par1 = self.ax_stats.twinx()
        par2 = self.ax_stats.twinx()
        par2.axis['right'] = par2.new_fixed_axis(loc='right', offset=(60,0))
        
        self.ax_stats.set_ylabel("Distance to Barycenter")
        par1.set_ylabel("Velocity")
        par2.set_ylabel("Acceleration")
        
        # Initialize lines - 3 for each attribute (9 lines total)
        lines_dist = [self.ax_stats.plot([], [], color=self.colors[i], ls='-', label=f'B{i} Dist')[0] for i in range(self.n_blobs)]
        lines_vel  = [par1.plot([], [], color=self.colors[i], ls='--', alpha=0.6)[0] for i in range(self.n_blobs)]
        lines_acc  = [par2.plot([], [], color=self.colors[i], ls=':', alpha=0.4)[0] for i in range(self.n_blobs)]
        
        self.ax_stats.legend(loc='upper left', fontsize='x-small')

        def update_frame(step):
            # 1. Physics/MDL Update
            pdf = self._compute_pdf_on_grid(grid_points)
            self.particles = self._resample_particles_from_grid(pdf, grid_points)
            chosen = self.select_best_candidates_all()
            self.apply_chosen_candidates(chosen)
            self.t += self.dt
            self.update_stats()

            # 2. Render Simulation (Left)
            self.ax_sim.clear()
            self.ax_sim.imshow(pdf.reshape(self.grid_size, self.grid_size), 
                               origin='lower', cmap='inferno', extent=[0,1,0,1])
            self.ax_sim.scatter(self.particles[:,0], self.particles[:,1], s=1, c='cyan', alpha=0.1)
            
            for i, wf in enumerate(self.wavefunctions):
                vals, vecs = eigh(wf.covariance)
                angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
                e = plt.matplotlib.patches.Ellipse(wf.center, 2*np.sqrt(vals[1]), 2*np.sqrt(vals[0]), 
                                                 angle=angle, edgecolor=self.colors[i], facecolor='none', lw=2)
                self.ax_sim.add_patch(e)
            self.ax_sim.set_title(f"Informational Geodesics - Step {step}")

            # 3. Render Stats (Right)
            t_axis = np.arange(len(self.history_dist[0]))
            for i in range(self.n_blobs):
                lines_dist[i].set_data(t_axis, self.history_dist[i])
                lines_vel[i].set_data(t_axis, self.history_vel[i])
                lines_acc[i].set_data(t_axis, self.history_acc[i])

            # Rescale the complex triple-axis setup
            self.ax_stats.relim(); self.ax_stats.autoscale_view()
            par1.relim(); par1.autoscale_view()
            par2.relim(); par2.autoscale_view()
            self.ax_stats.set_xlim(0, max(1, len(t_axis)))

            # Return all artists to satisfy the animation back-end
            return [self.ax_sim] + lines_dist + lines_vel + lines_acc

        # Important: interval and frames should be enough for the first frame to stabilize
        ani = animation.FuncAnimation(self.fig, update_frame, frames=self.n_steps, blit=False, repeat=False)
        
        # Using pillow writer - if this still fails, try 'ffmpeg'
        try:
            ani.save(save_gif, writer='pillow', fps=fps)
        except Exception as e:
            print(f"Error saving animation: {e}")
        
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--grid", type=int, default=120)
    parser.add_argument("--sigma", type=float, default=0.09)
    parser.add_argument("--size", type=float, default=1.6)
    parser.add_argument("--jitter_mean", type=float, default=0.01)
    parser.add_argument("--jitter_cov", type=float, default=0.001)
    parser.add_argument("--out", type=str, default="wavefunction_gravity.gif")
    args = parser.parse_args()
    sim = WavefunctionGravitySim(
        n_particles=args.particles, n_steps=args.steps, blob_sigma_init=args.sigma,
        n_candidates=args.candidates, grid_size=args.grid, size=args.size,
        jitter_mean=args.jitter_mean, jitter_cov=args.jitter_cov
    )
    sim.run()