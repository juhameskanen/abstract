#!/usr/bin/env python3
"""
Wavefunction-based Gravity Simulation 

- Blobs are wavefunctions with full state: center (mu), covariance (Sigma),
  amplitude, angular frequency (omega), and phase.
- Each blob keeps full memory histories of (mu, Sigma, phase, omega).
- At each step:
    * compute total psi = sum_i psi_i(grid), pdf = |psi|^2
    * resample particles from pdf (back-reaction)
    * for each blob: gather assigned particles, generate N candidates (mean,cov,phase,omega)
    * evaluate **full-information MDL** cost for each candidate using entire blob memory
    * select candidate with minimal description length (no hand-tuned smoothing)
    * after selecting best candidates for all blobs, update all blobs simultaneously
- Visualization: heatmap of pdf + particle scatter + 1-sigma ellipses for blobs
- Saves GIF using matplotlib.animation + Pillow writer

Usage:
    python quantum_gravity2.py --particles 8192 --steps 300 --candidates 8


"""
import argparse
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eigh
from scipy.stats import multivariate_normal
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA


class Wavefunction:
    """
    Complex wavefunction modeled as a Gaussian (center + covariance) times a phase factor.
    psi(x, t) = amplitude * exp(-0.5 * (x-mu)^T Sigma^{-1} (x-mu)) * exp(-i*(omega * t - phase))
    """
    def __init__(
        self,
        center: np.ndarray,
        covariance: np.ndarray,
        amplitude: float = 1.0,
        omega: float = 0.0,
        phase: float = 0.0,
    ):
        self.center = np.array(center, dtype=float)          # mu
        self.covariance = np.array(covariance, dtype=float)  # Sigma (2x2)
        self.amplitude = float(amplitude)
        self.omega = float(omega)
        self.phase = float(phase)

    def evaluate(self, points: np.ndarray, t: float) -> np.ndarray:
        """
        Evaluate complex psi at points (N x 2).
        Returns complex array length N.
        """
        diff = points - self.center  # (N,2)
        # ensure PD covariance
        cov = self.covariance + 1e-9 * np.eye(2)
        try:
            inv_cov = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # fallback small regularizer
            inv_cov = np.linalg.inv(cov + 1e-6 * np.eye(2))
        exponent = -0.5 * np.einsum('ij,jk,ik->i', diff, inv_cov, diff)  # (N,)
        env = self.amplitude * np.exp(exponent)
        phase_factor = np.exp(-1j * (self.omega * t - self.phase))
        return env * phase_factor

    def copy_state(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        return self.center.copy(), self.covariance.copy(), self.phase, self.omega

class WavefunctionGravitySim:
    def __init__(
        self,
        n_particles: int = 4096,
        n_steps: int = 200,
        blob_sigma_init: float = 0.12,
        n_candidates: int = 8,
        grid_size: int = 100,
        size: float = 1.0,
        jitter_mean: float = 0.05,
        jitter_cov: float = 0.01,
        seed: int = 1,
    ):
        np.random.seed(seed)
        self.n_particles = int(n_particles)
        self.n_steps = int(n_steps)
        self.n_candidates = int(n_candidates)
        self.grid_size = int(grid_size)
        self.jitter_mean = float(jitter_mean)
        self.jitter_cov = float(jitter_cov) 
        # initial blob centers (same as your original triple)
        self.positions = np.array([[0.2, 0.2], [0.8, 0.2], [0.5, 0.8]])
        self.n_blobs = len(self.positions)

        # initial covariance scale (use variance)
        var0 = (blob_sigma_init ** 2) * size
        self.covariances = [np.eye(2) * var0 for _ in range(self.n_blobs)]
        self.target_area = np.array([np.linalg.det(cov) for cov in self.covariances])

        # create wavefunctions (with small random phases/omegas)
        self.wavefunctions: List[Wavefunction] = [
            Wavefunction(center=self.positions[i].copy(),
                         covariance=self.covariances[i].copy(),
                         amplitude=1.0,
                         omega=np.random.normal(scale=0.05),
                         phase=np.random.uniform(0, 2*np.pi))
            for i in range(self.n_blobs)
        ]

        # particle initialization: sample from the initial pdf (mixture of |psi|^2)
        self.t = 0.0
        self.dt = 1.0
        grid_pts = self._make_grid_points()
        pdf_init = self._compute_pdf_on_grid(grid_pts)
        self.particles = self._resample_particles_from_grid(pdf_init, grid_pts)

        # memory histories for each blob: lists of states
        # each history stores sequences of: center (2,), covariance (2x2), phase (float), omega (float)
        self.trajs_mean: List[List[np.ndarray]] = [[wf.center.copy()] for wf in self.wavefunctions]
        self.trajs_cov:  List[List[np.ndarray]] = [[wf.covariance.copy()] for wf in self.wavefunctions]
        self.trajs_phase: List[List[float]] = [[wf.phase] for wf in self.wavefunctions]
        self.trajs_omega: List[List[float]] = [[wf.omega] for wf in self.wavefunctions]

        # diagnostics
        self.distances: List[float] = []
        self.velocities: List[float] = []

        # plotting fields
        self.fig = None
        self.ax_sim = None
        self.ax_stats = None

    def _make_grid_points(self):
        x = np.linspace(0, 1, self.grid_size)
        y = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        return grid_points

    def _compute_pdf_on_grid(self, grid_points: np.ndarray) -> np.ndarray:
        """Compute pdf = |sum_i psi_i|^2 on supplied grid_points."""
        psi_total = np.zeros(grid_points.shape[0], dtype=np.complex128)
        for wf in self.wavefunctions:
            psi_total += wf.evaluate(grid_points, self.t)
        pdf = np.abs(psi_total) ** 2
        pdf += 1e-20
        pdf /= pdf.sum()
        return pdf

    def _resample_particles_from_grid(self, pdf: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
        idx = np.random.choice(grid_points.shape[0], size=self.n_particles, p=pdf)
        return grid_points[idx]

    def assign_particles_to_blob(self, blob_index: int) -> np.ndarray:
        """Return particles within ~3-sigma Mahalanobis ellipsoid of blob (using its covariance)."""
        mu = self.wavefunctions[blob_index].center
        Sigma = self.wavefunctions[blob_index].covariance + 1e-9 * np.eye(2)
        invS = np.linalg.inv(Sigma)
        diff = self.particles - mu
        d2 = np.einsum('ij,jk,ik->i', diff, invS, diff)
        assigned = self.particles[d2 < 9.0]  # 3-sigma threshold
        return assigned

    def generate_candidates_for_blob(self, blob_index: int, assigned: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
        """
        Produce candidates as (mean, cov, phase, omega).
        Candidates are centered on MLE of assigned particles, with small jitter on mean/cov/phase/omega.
        """
        candidates = []
        wf = self.wavefunctions[blob_index]

        if assigned.size == 0:
            # no local particles -> small random perturbations around current state
            for _ in range(self.n_candidates):
                mean_c = wf.center + np.random.normal(scale=1e-3, size=2)
                A = np.random.normal(scale=1e-3, size=(2,2))
                cov_c = wf.covariance + 0.5 * (A @ A.T)
                phase_c = wf.phase + np.random.normal(scale=1e-2)
                omega_c = wf.omega + np.random.normal(scale=1e-3)
                candidates.append((mean_c, cov_c, phase_c, omega_c))
            return candidates

        # MLE mean/cov from assigned particles
        mu_mle = assigned.mean(axis=0)

        # Robust weighted/sample covariance for small samples
        n_assigned = assigned.shape[0]
        if n_assigned >= 2:
            cov_mle = np.cov(assigned.T)
            # numerical regularizer proportional to scale
            scale = max(1e-6, np.trace(cov_mle) / 2.0)
            eps = max(1e-9, scale * 1e-4)
            cov_mle = cov_mle + eps * np.eye(2)
        else:
            # Not enough samples — fall back to current blob covariance, slightly perturbed
            wf = self.wavefunctions[blob_index]
            cov_mle = wf.covariance.copy()
            # small regularizer to avoid singularity
            eps = max(1e-9, np.trace(cov_mle) * 1e-6)
            cov_mle = cov_mle + eps * np.eye(2)

        # jitter scales relative to blob size
        scale_mean = max(1e-4, np.sqrt(np.trace(cov_mle)) * self.jitter_mean)
        scale_cov = max(1e-6, np.sqrt(np.trace(cov_mle)) * self.jitter_cov)

        # inside generate_candidates_for_blob()
        for _ in range(self.n_candidates):
            mean_j = mu_mle + np.random.normal(scale=scale_mean, size=2)
            
            # generate SPD matrix with controlled determinant
            A = np.random.normal(scale=scale_cov, size=(2,2))
            cov_j = cov_mle + 0.5 * (A @ A.T)
            
            # rescale covariance to match target determinant
            det_j = np.linalg.det(cov_j)
            if det_j > 0:
                target_det = self.target_area[blob_index]
                scale_factor = np.sqrt(target_det / det_j)
                cov_j *= scale_factor
            
            phase_j = wf.phase + np.random.normal(scale=1e-2)
            omega_j = wf.omega + np.random.normal(scale=1e-3)
            candidates.append((mean_j, cov_j, phase_j, omega_j))
        return candidates


    def description_length_full(
        self,
        blob_index: int,
        candidate: Tuple[np.ndarray, np.ndarray, float, float],
        alpha: float = 1.0,
        beta: float = 0.05
    ) -> float:
        """
        Compute description-length cost of appending candidate to the full memory histories.
        Uses:
         - mean linear-extrapolation deviation summed over existing history plus candidate incremental,
         - covariance Frobenius-squared deviations summed over history plus candidate incremental,
         - phase+omega incremental deviations summed over history (squared),
         - optional small entropy penalty (log det of covariance) to prefer well-formed covariances.
        """
        mean_c, cov_c, phase_c, omega_c = candidate

        # histories
        Hm = self.trajs_mean[blob_index]
        Hc = self.trajs_cov[blob_index]
        Hp = self.trajs_phase[blob_index]
        Ho = self.trajs_omega[blob_index]

        # Mean: sum squared "second-difference" (deviation from linear extrapolation) over history + candidate
        L_mean = 0.0
        # existing history contributions
        if len(Hm) >= 3:
            for i in range(2, len(Hm)):
                pred = Hm[i-1] + (Hm[i-1] - Hm[i-2])
                dev = np.linalg.norm(Hm[i] - pred)
                L_mean += dev**2
        # candidate incremental (predict next from last two if available)
        if len(Hm) >= 2:
            pred = Hm[-1] + (Hm[-1] - Hm[-2])
            dev = np.linalg.norm(mean_c - pred)
            L_mean += dev**2

        # Covariance: sum Frobenius-sq of differences over history + candidate incremental
        L_cov = 0.0
        if len(Hc) >= 2:
            for i in range(1, len(Hc)):
                dF = Hc[i] - Hc[i-1]
                L_cov += np.linalg.norm(dF, ord='fro')**2
        # candidate incremental
        if len(Hc) >= 1:
            dF = cov_c - Hc[-1]
            L_cov += np.linalg.norm(dF, ord='fro')**2

        # Phase & Omega: penalize jumps relative to last value (squared)
        L_phase = 0.0
        if len(Hp) >= 1:
            dphi = phase_c - Hp[-1]
            L_phase += (dphi)**2
        if len(Ho) >= 1:
            domega = omega_c - Ho[-1]
            L_phase += (domega)**2

        # Entropy-ish penalty to avoid degenerate huge covariances (small weight)
        try:
            logdet = np.log(np.linalg.det(cov_c + 1e-12 * np.eye(2)))
        except np.linalg.LinAlgError:
            logdet = 1e6
        L_entropy = logdet

        
        # Current determinant (area) of candidate covariance
        area_c = np.linalg.det(cov_c + 1e-12 * np.eye(2))
        # deviation from target
        target = self.target_area[blob_index]
        L_size = (area_c - target)**2

        # add weighted penalty to total cost
        total = L_mean + alpha * L_cov + L_phase + beta * L_entropy + 1000.0 * L_size

        return float(total)

    def select_best_candidates_all(self) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
        """
        For each blob, generate candidates based on current particles and select best by MDL.
        Return list of chosen states; do NOT modify wavefunctions until all chosen.
        """
        chosen = []
        # Use current particle set for all blobs (simultaneous selection)
        for i in range(self.n_blobs):
            assigned = self.assign_particles_to_blob(i)
            cands = self.generate_candidates_for_blob(i, assigned)
            best_cost = float('inf')
            best_cand = None
            for cand in cands:
                cost = self.description_length_full(i, cand)
                if cost < best_cost:
                    best_cost = cost
                    best_cand = cand
            # Fallback: if none, keep current
            if best_cand is None:
                wf = self.wavefunctions[i]
                best_cand = (wf.center.copy(), wf.covariance.copy(), wf.phase, wf.omega)
            chosen.append(best_cand)
        return chosen

    def apply_chosen_candidates(self, chosen: List[Tuple[np.ndarray, np.ndarray, float, float]]):
        """Apply chosen states to wavefunctions and append to history."""
        for i, (mean_c, cov_c, phase_c, omega_c) in enumerate(chosen):
            # enforce SPD & small regularizer
            cov_c = (cov_c + cov_c.T) * 0.5
            # regularize tiny eigenvalues
            vals, vecs = eigh(cov_c)
            vals_clipped = np.clip(vals, 1e-9, 1.0)
            cov_c = (vecs * vals_clipped) @ vecs.T
            # apply to wavefunction
            wf = self.wavefunctions[i]
            wf.center = mean_c
            wf.covariance = cov_c
            wf.phase = float(phase_c)
            wf.omega = float(omega_c)
            # append to histories
            self.trajs_mean[i].append(mean_c.copy())
            self.trajs_cov[i].append(cov_c.copy())
            self.trajs_phase[i].append(float(phase_c))
            self.trajs_omega[i].append(float(omega_c))

   
    def update_stats(self):
        d = np.linalg.norm(self.wavefunctions[0].center - self.wavefunctions[1].center)
        self.distances.append(d)
        if len(self.distances) > 1:
            self.velocities.append(self.distances[-1] - self.distances[-2])
        else:
            self.velocities.append(0.0)

    def _plot_ellipse(self, ax, mu, cov, **kwargs):
        # 1-sigma ellipse
        vals, vecs = eigh(cov)
        vals = np.maximum(vals, 1e-12)
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        width, height = 2*np.sqrt(vals)  # 1-sigma -> 2*sqrt(eig)
        from matplotlib.patches import Ellipse
        e = Ellipse(mu, width, height, angle=angle, edgecolor='white', facecolor='none', lw=1)
        ax.add_patch(e)

    def run(self, save_gif: str = "wavefunction_gravity.gif", fps: int = 15):
        grid_points = self._make_grid_points()

        # Figure / axes
        self.fig = plt.figure(figsize=(12,6))
        self.ax_sim = self.fig.add_subplot(1,2,1)
        self.ax_stats = host_subplot(111, axes_class=AA.Axes, figure=self.fig)
        self.ax_stats.set_position([0.55, 0.1, 0.4, 0.8])
        par1 = self.ax_stats.twinx()
        par2 = self.ax_stats.twinx()
        par2.axis['right'] = par2.new_fixed_axis(loc='right', offset=(60,0))
        self.ax_stats.set_xlabel("Step")
        self.ax_stats.set_ylabel("Distance")
        par1.set_ylabel("Velocity")
        par2.set_ylabel("Acceleration")
        self.ax_stats.set_title("Distance, Velocity & Acceleration")
        self.ax_stats.grid(True)
        line_dist, = self.ax_stats.plot([], [], 'b-', label='Distance')
        line_vel, = par1.plot([], [], 'g-', label='Velocity')
        line_acc, = par2.plot([], [], 'r--', label='Acceleration')

        def update_frame(step):
            # 1) compute pdf from current wavefunctions
            pdf = self._compute_pdf_on_grid(grid_points)

            # 2) resample particles (back-reaction)
            self.particles = self._resample_particles_from_grid(pdf, grid_points)

            # 3) select best candidates for all blobs using current particle set
            chosen = self.select_best_candidates_all()

            # 4) apply chosen states synchronously
            self.apply_chosen_candidates(chosen)

            # advance time
            self.t += self.dt

            # 5) update diagnostics
            self.update_stats()

            # --- render left panel: heatmap & particles & ellipses
            self.ax_sim.clear()
            self.ax_sim.imshow(pdf.reshape(self.grid_size, self.grid_size),
                               origin='lower', cmap='inferno', extent=[0,1,0,1])
            self.ax_sim.scatter(self.particles[:,0], self.particles[:,1], s=1, c='cyan', alpha=0.35)
            # draw ellipses for each wavefunction
            for wf in self.wavefunctions:
                self._plot_ellipse(self.ax_sim, wf.center, wf.covariance)
            self.ax_sim.set_xlim(0,1); self.ax_sim.set_ylim(0,1)
            self.ax_sim.set_title(f"Step {step+1}")

            # --- right panel stats
            t = np.arange(len(self.distances))
            line_dist.set_data(t, self.distances)
            line_vel.set_data(t, self.velocities)
            if len(self.velocities) > 1:
                acc = np.diff(self.velocities)
                t_acc = np.arange(1, len(self.velocities))
                line_acc.set_data(t_acc, acc)
                par2.set_xlim(0, max(len(t), len(t_acc)))
                par2.relim()
                par2.autoscale_view()
            self.ax_stats.set_xlim(0, max(1, len(t)))
            self.ax_stats.relim()
            self.ax_stats.autoscale_view()
            par1.relim(); par1.autoscale_view()

            return self.ax_sim, self.ax_stats, line_dist, line_vel, line_acc

        ani = animation.FuncAnimation(self.fig, update_frame, frames=self.n_steps, blit=False)
        ani.save(save_gif, writer='pillow', fps=fps)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Wavefunction Gravity Simulation (full memory + covariance + MDL)")
    parser.add_argument("--particles", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--grid", type=int, default=120)
    parser.add_argument("--sigma", type=float, default=0.09)
    parser.add_argument("--size", type=float, default=1.6)
    parser.add_argument("--jitter_mean", type=float, default=0.01)
    parser.add_argument("--jitter_cov", type=float, default=0.001)
    parser.add_argument("--out", type=str, default="wavefunction_gravity.gif")
    args = parser.parse_args()

    sim = WavefunctionGravitySim(
        n_particles=args.particles,
        n_steps=args.steps,
        blob_sigma_init=args.sigma,
        n_candidates=args.candidates,
        grid_size=args.grid,
        size=args.size,
        jitter_mean=args.jitter_mean,
        jitter_cov=args.jitter_cov, 
        seed=1,
    )
    sim.run(save_gif=args.out, fps=15)

if __name__ == "__main__":
    main()
