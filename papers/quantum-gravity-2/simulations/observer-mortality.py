"""
Emergent Gravity from Compression and Observer Filters.

This module implements a proof-of-concept simulation in which gravity,
inertia, and smooth dynamics emerge from two principles only:

1. Compression → Probability:
   Physical evolution is selected by minimizing description length.
   Trajectories that are more compressible (lower informational action)
   dominate the probability measure.

2. Observer Filter:
   Only configurations that preserve the geometric and informational
   integrity of an object are considered viable. Here, objects are
   Gaussian blobs whose identity is defined by:
     - a coherent geometric shape (approximately circular),
     - continuity of motion,
     - and compressible history.

No forces, masses, or spacetime geometry are postulated.
Instead, apparent gravitational attraction and geodesic motion emerge
from iterative selection of the most compressible observer-consistent
histories.

This simulation uses QBitwave to evaluate emergent informational
complexity at the bit level.
"""

#!/usr/bin/env python3
import argparse
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eigh
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

from qbitwave import QBitwave

class Wavefunction:
    """Complex-valued Gaussian wavefunction.

    Represents an emergent informational object whose spatial footprint
    acts as an observer filter. The wavefunction defines how particles
    are probabilistically generated in space.
    """
    def __init__(self, center, covariance, amplitude=1.0, omega=0.0, phase=0.0):
        self.center = np.array(center, dtype=float)
        self.covariance = np.array(covariance, dtype=float)
        self.amplitude = float(amplitude)
        self.omega = float(omega)
        self.phase = float(phase)

    def evaluate(self, points: np.ndarray, t: float) -> np.ndarray:
        """Evaluates the wavefunction at given points and time.

        Args:
            points: Array of shape (N, 2) of spatial points.
            t: Simulation time.

        Returns:
            Complex-valued wavefunction evaluated at each point.
        """
        diff = points - self.center
        cov = self.covariance + 1e-9 * np.eye(2)
        inv_cov = np.linalg.inv(cov)
        exponent = -0.5 * np.einsum('ij,jk,ik->i', diff, inv_cov, diff)
        env = self.amplitude * np.exp(exponent)
        phase_factor = np.exp(-1j * (self.omega * t - self.phase))
        return env * phase_factor


class WavefunctionGravitySim:
    """Emergent gravity simulation driven by informational selection.

    Gaussian blobs are treated as observer-defined objects whose evolution
    is determined by minimizing an informational action composed of:
      - emergent bit-level complexity (compression),
      - inertial continuity,
      - and geometric integrity.

    Apparent gravitational attraction arises from probabilistic dominance
    of compressible, observer-consistent histories.
    """
    def __init__(self, n_particles=4096, n_steps=200, blob_sigma_init=0.12, 
                 n_candidates=8, grid_size=100, size=1.0, jitter_mean=0.05, 
                 jitter_cov=0.01, seed=1):
        """Initializes the simulation.

        Args:
            n_particles: Number of particles sampled per step.
            n_steps: Number of animation steps.
            blob_sigma_init: Initial blob width.
            n_candidates: Candidate updates per blob per step.
            grid_size: Resolution of spatial grid.
            size: Global spatial scaling.
            jitter_mean: Unused (reserved for future extensions).
            jitter_cov: Covariance noise scale for candidates.
            seed: Random seed.
        """
        np.random.seed(seed)
        self.n_particles, self.n_steps = n_particles, n_steps
        self.n_candidates, self.grid_size = n_candidates, grid_size
        self.jitter_mean, self.jitter_cov = jitter_mean, jitter_cov
        self.size = size
        self.t, self.dt = 0.0, 1.0

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

        grid_pts = self._make_grid_points()
        pdf_init = self._compute_pdf_on_grid(grid_pts)
        self.particles = self._resample_particles_from_grid(pdf_init, grid_pts)

        # History and Diagnostic Storage
        self.trajs_mean = [[wf.center.copy()] for wf in self.wavefunctions]
        self.history_dist = [[] for _ in range(self.n_blobs)]
        self.history_vel = [[] for _ in range(self.n_blobs)]
        self.history_acc = [[] for _ in range(self.n_blobs)]

        # --- Cumulative MDL state for full-history inertia ---
        self.history_bits = [[] for _ in range(self.n_blobs)]
        self.history_mdl = [0.0 for _ in range(self.n_blobs)]


    def _make_grid_points(self):
        """Creates a uniform 2D grid of points.

        Returns:
            Array of shape (grid_size^2, 2).
        """
        x = np.linspace(0, 1, self.grid_size)
        y = np.linspace(0, 1, self.grid_size)
        X, Y = np.meshgrid(x, y, indexing='xy') 
        return np.column_stack([X.ravel(), Y.ravel()])

    def _compute_pdf_on_grid(self, grid_points):
        """Computes particle probability density from all wavefunctions.

        Args:
            grid_points: Spatial grid.

        Returns:
            Normalized probability density over the grid.
        """
        psi_total = np.zeros(grid_points.shape[0], dtype=np.complex128)
        for wf in self.wavefunctions:
            psi_total += wf.evaluate(grid_points, self.t)
        pdf = np.abs(psi_total) ** 2 + 1e-20
        return pdf / pdf.sum()

    def _resample_particles_from_grid(self, pdf, grid_points):
        """Samples particles according to the probability density.

        Args:
            pdf: Probability density.
            grid_points: Corresponding grid locations.

        Returns:
            Sampled particle positions.
        """
        idx = np.random.choice(grid_points.shape[0], size=self.n_particles, p=pdf)
        return grid_points[idx]

    def encode_point_to_bits(self, point: np.ndarray) -> List[int]:
        """Encodes a single 2D point into a fixed-length bitstring.

        Args:
            point: 2D coordinates (x, y) in [0,1].

        Returns:
            List of bits.
        """
        bits = []
        for coord in point:
            val = int(np.clip(coord, 0, 1) * 255)
            bits.extend([(val >> i) & 1 for i in reversed(range(8))])
        return bits

    def trajectory_to_bitstring(self, trajectory: np.ndarray) -> List[int]:
        """Converts an entire sequence of 2D points into a single bitstring.

        Args:
            trajectory: Array of shape (N, 2) containing sequence of positions.

        Returns:
            A flat list of bits representing the full history.
        """
        full_bits = []
        for point in trajectory:
            # Reuse your existing point encoder
            full_bits.extend(self.encode_point_to_bits(point))
        return full_bits

    def description_length_full(self, blob_index, candidate):
        """Computes the full-history informational action (MDL) for a blob.

        This implements the core principle: the probability of an observer
        following a trajectory is determined by the compressibility of the
        entire history plus the candidate step. No arbitrary weights are used;
        the wavefunction itself defines the cost.

        Args:
            blob_index: Index of the blob.
            candidate: Proposed (mean, covariance, phase, omega) tuple.

        Returns:
            Scalar MDL cost of the blob’s history extended with the candidate.
        """
        mean_c, cov_c, _, _ = candidate
        # Include the full history + candidate step
        history = self.trajs_mean[blob_index]  # full previous trajectory
        full_trajectory = np.array(history + [mean_c])

        # Encode entire trajectory into bits for QBitwave
        bits = self.trajectory_to_bitstring(full_trajectory)

        # Compute wavefunction complexity (MDL) of full trajectory
        qb = QBitwave(bitstring=bits, fixed_basis_size=8)
        H_total = qb.wave_complexity()  # informational cost over full history

        return H_total


    def assign_particles_to_blob(self, idx):
        """Assigns nearby particles to a given blob using Mahalanobis distance.

        Args:
            idx: Index of the blob.

        Returns:
            Array of particle positions assigned to the blob.
        """
        wf = self.wavefunctions[idx]
        invS = np.linalg.inv(wf.covariance + 1e-9 * np.eye(2))
        diff = self.particles - wf.center
        d2 = np.einsum('ij,jk,ik->i', diff, invS, diff)
        return self.particles[d2 < 9.0]

    def generate_candidates_for_blob(self, idx, assigned):
        """Generates candidate updates for a blob from assigned particles.

        Args:
            idx: Index of the blob.
            assigned: Particles currently associated with the blob.

        Returns:
            List of candidate (center, covariance, phase, omega) tuples.
        """
        wf = self.wavefunctions[idx]
        if assigned.size < 2: return [(wf.center, wf.covariance, wf.phase, wf.omega)]
        mu_mle = assigned.mean(axis=0)
        raw_cov = np.cov(assigned.T) + 1e-9 * np.eye(2)
        cands = []
        for _ in range(self.n_candidates):
            jitter = np.random.normal(scale=self.jitter_cov, size=(2,2))
            c_cov = raw_cov + (jitter @ jitter.T)
            scale = np.sqrt(self.target_area[idx] / np.linalg.det(c_cov))
            cands.append((mu_mle, c_cov * scale, wf.phase, wf.omega))
        return cands

    def update_stats(self):
        """Updates diagnostic statistics for visualization."""
        centers = np.array([wf.center for wf in self.wavefunctions])
        barycenter = centers.mean(axis=0)
        for i in range(self.n_blobs):
            d = np.linalg.norm(self.wavefunctions[i].center - barycenter)
            self.history_dist[i].append(d)
            v = self.history_dist[i][-1] - self.history_dist[i][-2] if len(self.history_dist[i]) > 1 else 0.0
            self.history_vel[i].append(v)
            a = self.history_vel[i][-1] - self.history_vel[i][-2] if len(self.history_vel[i]) > 1 else 0.0
            self.history_acc[i].append(a)

    def run(self, save_gif="observer-mortality.gif", fps=15):
        """Runs the simulation and renders the animation."""
        grid_pts = self._make_grid_points()
        self.fig = plt.figure(figsize=(14, 8))
        self.ax_sim = self.fig.add_subplot(1, 2, 1)
        self.ax_stats = host_subplot(111, axes_class=AA.Axes, figure=self.fig)
        self.ax_stats.set_position([0.55, 0.1, 0.35, 0.8])
        par1, par2 = self.ax_stats.twinx(), self.ax_stats.twinx()
        par2.axis['right'] = par2.new_fixed_axis(loc='right', offset=(60,0))

        lines_d = [self.ax_stats.plot([], [], color=self.colors[i], ls='-', label=f'B{i}')[0] for i in range(self.n_blobs)]
        lines_v = [par1.plot([], [], color=self.colors[i], ls='--', alpha=0.6)[0] for i in range(self.n_blobs)]
        lines_a = [par2.plot([], [], color=self.colors[i], ls=':', alpha=0.4)[0] for i in range(self.n_blobs)]
        self.ax_stats.legend(loc='upper left', fontsize='x-small')

        def animate(step):
            pdf = self._compute_pdf_on_grid(grid_pts)
            self.particles = self._resample_particles_from_grid(pdf, grid_pts)
            for i in range(self.n_blobs):
                assigned = self.assign_particles_to_blob(i)
                cands = self.generate_candidates_for_blob(i, assigned)
                costs = [self.description_length_full(i, c) for c in cands]
                best = cands[np.argmin(costs)]

                # Commit chosen candidate
                self.wavefunctions[i].center, self.wavefunctions[i].covariance = best[0], best[1]
                self.trajs_mean[i].append(best[0].copy())

                # Update cumulative MDL and history bits
                new_bits = self.encode_point_to_bits(best[0])
                self.history_bits[i].extend(new_bits)
                qb = QBitwave(bitstring=self.history_bits[i], fixed_basis_size=8)
                self.history_mdl[i] = qb.wave_complexity()

            self.t += self.dt
            self.update_stats()
            self.ax_sim.clear()

            # Background PDF Heatmap
            img_data = pdf.reshape(self.grid_size, self.grid_size)
            self.ax_sim.imshow(img_data, origin='lower', extent=[0,1,0,1], cmap='inferno', aspect='equal')

            # Particle points
            self.ax_sim.scatter(self.particles[:,0], self.particles[:,1], s=0.5, c='cyan', alpha=0.1)

            # Ellipse Rendering
            for i, wf in enumerate(self.wavefunctions):
                vals, vecs = eigh(wf.covariance)
                angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
                width = 2 * np.sqrt(vals[0])
                height = 2 * np.sqrt(vals[1])
                e = plt.matplotlib.patches.Ellipse(
                    xy=wf.center,
                    width=width,
                    height=height,
                    angle=angle,
                    edgecolor=self.colors[i],
                    facecolor='none',
                    lw=2
                )
                self.ax_sim.add_patch(e)

            t_ax = np.arange(len(self.history_dist[0]))
            for i in range(self.n_blobs):
                lines_d[i].set_data(t_ax, self.history_dist[i])
                lines_v[i].set_data(t_ax, self.history_vel[i])
                lines_a[i].set_data(t_ax, self.history_acc[i])
            for ax in [self.ax_stats, par1, par2]: ax.relim(); ax.autoscale_view()
            return [self.ax_sim] + lines_d + lines_v + lines_a

        ani = animation.FuncAnimation(self.fig, animate, frames=self.n_steps, blit=False)
        ani.save(save_gif, writer='pillow', fps=fps)
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--candidates", type=int, default=100)
    parser.add_argument("--grid", type=int, default=120)
    parser.add_argument("--sigma", type=float, default=0.09)
    parser.add_argument("--size", type=float, default=1.6)
    parser.add_argument("--jitter_mean", type=float, default=0.01)
    parser.add_argument("--jitter_cov", type=float, default=0.001)
    parser.add_argument("--out", type=str, default="quantum-gravity-3.gif")
    args = parser.parse_args()

    sim = WavefunctionGravitySim(
        n_particles=args.particles, n_steps=args.steps, blob_sigma_init=args.sigma,
        n_candidates=args.candidates, grid_size=args.grid, size=args.size,
        jitter_mean=args.jitter_mean, jitter_cov=args.jitter_cov
    )
    sim.run(save_gif=args.out)


if __name__ == "__main__":
    main()
