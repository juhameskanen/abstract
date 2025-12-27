"""
Base class for Theory of Everything. 

This simulation demonstrates how *inertia* can emerge from 
an informational principle: **Minimum Description Length (MDL)**.

Concept:
--------
- Gaussian blobs represent *observers*. Each blob defines a 
  probability distribution (a Gaussian filter) that selects 
  which particle configurations are "observable".
- Particles are not moved via forces. Instead, they are resampled 
  according to the joint probability distribution of all blobs.
- Blob positions are updated from assigned particles, but their 
  trajectories are chosen using an MDL criterion:
  
    The "best" next position is the one that minimizes the 
    description length of the trajectory, i.e., the deviation 
    from linear extrapolation, resulting best compression.
  
- This introduces *informational inertia*: blobs resist sudden 
  changes, leading to smoother, mass-like trajectories.

Result:
-------
- Overlapping observer filters produce attraction (gravity-like effect).
- MDL trajectory compression produces persistence (inertia-like effect).
- Distance, velocity, and acceleration of blobs are visualized 
  alongside the particle field.

Usage:
------
Run as a script:

    python mdl_gravity_sim.py --sigma 0.11 --particles 4096 --steps 400

Parameters:
-----------
--sigma       Standard deviation of Gaussian blobs (observer filter width).
--particles   Number of particles (bits) in the system.
--steps       Number of resampling iterations.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from typing import List, Tuple
import matplotlib.animation as animation

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA

class GravitySim:
    def __init__(
        self,
        n_particles: int = 4096,
        n_steps: int = 100,
        blob_sigma: float = 0.1,
        n_candidates: int = 5
    ) -> None:
        self.n_particles: int = n_particles
        self.n_steps: int = n_steps
        self.blob_sigma: float = blob_sigma
        self.n_candidates: int = n_candidates

        # Initial blob positions
        self.positions: np.ndarray = np.array([
            [0.2, 0.2],
            [0.8, 0.2],
            [0.5, 0.8]
        ])
        self.n_blobs: int = len(self.positions)

        # Particle initialization
        self.particles: np.ndarray = self.init_particles()

        # Trajectories for MDL inertia (list of lists of positions)
        self.trajs: List[List[np.ndarray]] = [[pos.copy()] for pos in self.positions]

        # Stats tracking
        self.distances: List[float] = []
        self.velocities: List[float] = []


    def assign_particles_to_blob(self, blob_index: int) -> np.ndarray:
        """
        Assigns particles to a specific blob based on proximity.

        Returns the array of particles assigned to this blob.
        """
        pos = self.positions[blob_index]
        distances = np.linalg.norm(self.particles - pos, axis=1)
        assigned = self.particles[distances < 3.0 * self.blob_sigma]
        return assigned


    def generate_candidates_for_blob(
        self,
        blob_index: int,
        target_pos: np.ndarray,
        velocity: np.ndarray = None,
        velocity_bias: float = 0.1,
        jitter_scale: float = 0.01
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate candidate positions around a statistical target for a blob.
        
        Args:
            blob_index: index of the blob
            target_pos: statistical target position (mean of assigned particles)
            velocity: optional current velocity to bias candidate moves
            velocity_bias: weight of the velocity bias
            jitter_scale: scale of random jitter added to each candidate
        """
        current_pos = self.positions[blob_index]
        candidates = []

        for _ in range(self.n_candidates * 2):
            move_vec = 0.1 * (target_pos - current_pos)
            if velocity is not None:
                move_vec += velocity_bias * velocity
            candidate = current_pos + move_vec + np.random.normal(scale=self.blob_sigma * jitter_scale, size=2)
            candidate_velocity = candidate - current_pos
            if np.linalg.norm(candidate_velocity) > 1e-8:
                candidates.append((candidate, candidate_velocity))

        if not candidates:
            v_candidate = np.random.normal(scale=1e-6, size=2)
            candidates.append((current_pos + v_candidate, v_candidate))

        return candidates


    def init_particles(self) -> np.ndarray:
        """Initialize particles around blob positions using Gaussian sampling."""
        particles = []
        for pos in self.positions:
            cov = [[self.blob_sigma**2, 0], [0, self.blob_sigma**2]]
            particles.append(
                np.random.multivariate_normal(pos, cov, self.n_particles // self.n_blobs)
            )
        return np.vstack(particles)

    def compute_pdf(self, grid_points: np.ndarray) -> np.ndarray:
        """Compute the combined PDF of all blobs at given grid points."""
        pdf_total = np.zeros(grid_points.shape[0])
        for pos in self.positions:
            cov = [[self.blob_sigma**2, 0], [0, self.blob_sigma**2]]
            pdf_total += multivariate_normal.pdf(grid_points, mean=pos, cov=cov)
        pdf_total /= pdf_total.sum()
        return pdf_total

    def resample_particles(self, pdf_total: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
        """Resample particles according to the probability distribution."""
        idx = np.random.choice(grid_points.shape[0], size=self.n_particles, p=pdf_total)
        return grid_points[idx]

    @staticmethod
    def description_length(traj: List[np.ndarray]) -> float:
        """MDL-inspired cost: sum squared deviations from linear extrapolation."""
        if len(traj) < 3:
            return 0.0
        cost = 0.0
        for i in range(2, len(traj)):
            pred = traj[i-1] + (traj[i-1] - traj[i-2])
            dev = np.linalg.norm(traj[i] - pred)
            cost += dev**2
        return cost

    def update_positions(self, new_particles: np.ndarray) -> None:
        """Update blob positions with MDL inertia (center of gravity + compression)."""
        

    def update_stats(self) -> None:
        """Update distance and velocity stats for blobs 0–1."""
        d = np.linalg.norm(self.positions[0] - self.positions[1])
        self.distances.append(d)
        if len(self.distances) > 1:
            v = self.distances[-1] - self.distances[-2]
        else:
            v = 0.0
        self.velocities.append(v)


    def run(self, save_gif: str = "simulation.gif", fps: int = 15) -> None:
        """Run the simulation loop with density+scatter and 3-y-axis stats."""
        grid_size = 100
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        from mpl_toolkits.axes_grid1 import host_subplot
        import mpl_toolkits.axisartist as AA

        # --- Figure setup ---
        self.fig = plt.figure(figsize=(12, 6))

        # Left panel: simulation (scatter + density)
        self.ax_sim = self.fig.add_subplot(1, 2, 1)

        # Right panel: stats with 3 y-axes
        self.ax_stats = host_subplot(111, axes_class=AA.Axes, figure=self.fig)
        self.ax_stats.set_position([0.55, 0.1, 0.4, 0.8])  # right panel

        self.par1 = self.ax_stats.twinx()  # second y-axis
        self.par2 = self.ax_stats.twinx()  # third y-axis
        self.par2.axis['right'] = self.par2.new_fixed_axis(loc='right', offset=(60, 0))

        # Axis labels and title
        self.ax_stats.set_xlabel("Step")
        self.ax_stats.set_ylabel("Distance")
        self.par1.set_ylabel("Velocity")
        self.par2.set_ylabel("Acceleration")
        self.ax_stats.set_title("Distance, Velocity & Acceleration")
        self.ax_stats.grid(True)

        # Initialize lines
        line_dist, = self.ax_stats.plot([], [], 'b-', label='Distance')
        line_vel, = self.par1.plot([], [], 'g-', label='Velocity')
        line_acc, = self.par2.plot([], [], 'r--', label='Acceleration')

        # Combine legends
        lines = [line_dist, line_vel, line_acc]
        labels = [l.get_label() for l in lines]
        self.ax_stats.legend(lines, labels, loc='upper left')

        # --- Animation update function ---
        def update_frame(step: int):
            # Compute PDF and resample particles
            pdf_total = self.compute_pdf(grid_points)
            self.particles = self.resample_particles(pdf_total, grid_points)

            # Update blob positions (MDL inertia)
            self.update_positions(self.particles)

            # Update distance/velocity stats
            self.update_stats()

            # --- Left panel: scatter + density ---
            self.ax_sim.clear()
            self.ax_sim.imshow(pdf_total.reshape(grid_size, grid_size),
                            origin='lower', cmap='inferno', extent=[0,1,0,1])
            self.ax_sim.scatter(self.particles[:,0], self.particles[:,1],
                                s=1, c='cyan', alpha=0.4)
            self.ax_sim.set_xlim(0,1)
            self.ax_sim.set_ylim(0,1)
            self.ax_sim.set_title(f"Step {step+1}")

            # --- Right panel: stats ---
            t = np.arange(len(self.distances))
            line_dist.set_data(t, self.distances)
            line_vel.set_data(t, self.velocities)

            if len(self.velocities) > 1:
                acceleration = np.diff(self.velocities)
                t_acc = np.arange(1, len(self.velocities))
                line_acc.set_data(t_acc, acceleration)
                self.par2.set_xlim(0, max(len(t), len(t_acc)))
                self.par2.relim()
                self.par2.autoscale_view()

            # Rescale main axes
            self.ax_stats.set_xlim(0, len(t))
            self.ax_stats.relim()
            self.ax_stats.autoscale_view()
            self.par1.relim()
            self.par1.autoscale_view()

            return self.ax_sim, self.ax_stats, line_dist, line_vel, line_acc

        # --- Run animation ---
        #self.fig.set_size_inches(8, 6)   # 800x600 at 100 dpi
        ani = animation.FuncAnimation(
            self.fig, update_frame, frames=self.n_steps, blit=False
        )

        # Save GIF
        #ani.save(save_gif, writer="pillow", fps=fps, dpi=150)  # 1200x900
        ani.save(save_gif, writer="pillow", fps=fps)  # 1200x900
        plt.show()


    
    def setup_axes(self):
        """Set up a 3-y-axis plot for distance, velocity, and acceleration."""
        self.fig, self.ax_sim = plt.subplots(1, 1, figsize=(12, 6))
        
        # Host subplot for stats (right panel)
        self.ax_stats = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)  # leave space for third axis

        self.par1 = self.ax_stats.twinx()  # second y-axis
        self.par2 = self.ax_stats.twinx()  # third y-axis

        # Offset the third axis to the right
        self.par2.axis['right'] = self.par2.new_fixed_axis(loc='right', offset=(60,0))

        # Labels
        self.ax_stats.set_xlabel("Step")
        self.ax_stats.set_ylabel("Distance")
        self.par1.set_ylabel("Velocity")
        self.par2.set_ylabel("Acceleration")

        self.ax_stats.grid(True)
        self.ax_stats.set_title("Distance, Velocity & Acceleration")

        # Legends
        self.lines = {}
        self.lines['dist'], = self.ax_stats.plot([], [], 'b-', label='Distance')
        self.lines['vel'], = self.par1.plot([], [], 'g-', label='Velocity')
        self.lines['acc'], = self.par2.plot([], [], 'r--', label='Acceleration')

        # Combine legends
        lines = [self.lines['dist'], self.lines['vel'], self.lines['acc']]
        labels = [l.get_label() for l in lines]
        self.ax_stats.legend(lines, labels, loc='upper left')


    def setup_axes_(self, ax2):
        """Initial setup of statistics axes."""
        self.ax_stats.set_xlabel("Step")
        self.ax_stats.set_title("Distance, Velocity & Acceleration")
        self.ax_stats.grid(True)
        self.ax_stats.legend(loc="upper left")
        ax2.set_ylabel("Acceleration")
        ax2.legend(loc="upper right")

    def plot_density_particles(self, pdf_total, grid_size, step):
        """Plot density map and particle positions."""
        self.ax_sim.clear()
        self.ax_sim.imshow(
            pdf_total.reshape(grid_size, grid_size),
            origin="lower",
            cmap="inferno",
            extent=[0, 1, 0, 1]
        )
        self.ax_sim.scatter(
            self.particles[:, 0], self.particles[:, 1],
            s=1, c="cyan", alpha=0.4
        )
        self.ax_sim.set_title(f"Step {step+1}")
        self.ax_sim.set_xlim(0, 1)
        self.ax_sim.set_ylim(0, 1)

    def update_lines(self):
        """Update distance, velocity, and acceleration lines for animation."""
        t = np.arange(len(self.distances))
        self.lines['dist'].set_data(t, self.distances)
        self.lines['vel'].set_data(t, self.velocities)

        if len(self.velocities) > 1:
            acceleration = np.diff(self.velocities)
            t_acc = np.arange(1, len(self.velocities))
            self.lines['acc'].set_data(t_acc, acceleration)
            self.par2.set_xlim(0, max(len(t), len(t_acc)))
            self.par2.relim()
            self.par2.autoscale_view()
        
        # Rescale main axes
        self.ax_stats.set_xlim(0, len(t))
        self.ax_stats.relim()
        self.ax_stats.autoscale_view()

        self.par1.relim()
        self.par1.autoscale_view()
