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
from typing import List, Tuple, Optional, Dict, Any, Union
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

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
        
        # Plotting components
        self.fig: Optional[Figure] = None
        self.ax_sim: Optional[Axes] = None
        self.ax_stats: Optional[Any] = None
        self.par1: Optional[Any] = None
        self.par2: Optional[Any] = None
        self.lines: Dict[str, Line2D] = {}
        self.accumulated_pdf = None

    def assign_particles_to_blob(self, blob_index: int) -> np.ndarray:
        """Assigns particles to a specific blob based on proximity."""
        pos: np.ndarray = self.positions[blob_index]
        distances: np.ndarray = np.linalg.norm(self.particles - pos, axis=1)
        assigned: np.ndarray = self.particles[distances < 3.0 * self.blob_sigma]
        return assigned

    def generate_candidates_for_blob(
        self,
        blob_index: int,
        target_pos: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        velocity_bias: float = 0.1,
        jitter_scale: float = 0.01
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate candidate positions around a statistical target for a blob."""
        current_pos: np.ndarray = self.positions[blob_index]
        candidates: List[Tuple[np.ndarray, np.ndarray]] = []

        for _ in range(self.n_candidates * 2):
            move_vec: np.ndarray = 0.1 * (target_pos - current_pos)
            if velocity is not None:
                move_vec += velocity_bias * velocity
            
            candidate: np.ndarray = current_pos + move_vec + np.random.normal(scale=self.blob_sigma * jitter_scale, size=2)
            candidate_velocity: np.ndarray = candidate - current_pos
            if np.linalg.norm(candidate_velocity) > 1e-8:
                candidates.append((candidate, candidate_velocity))

        if not candidates:
            v_candidate: np.ndarray = np.random.normal(scale=1e-6, size=2)
            candidates.append((current_pos + v_candidate, v_candidate))

        return candidates

    def init_particles(self) -> np.ndarray:
        """Initialize particles around blob positions using Gaussian sampling."""
        particles: List[np.ndarray] = []
        for pos in self.positions:
            cov: List[List[float]] = [[self.blob_sigma**2, 0], [0, self.blob_sigma**2]]
            particles.append(
                np.random.multivariate_normal(pos, cov, self.n_particles // self.n_blobs)
            )
        return np.vstack(particles)

    def compute_pdf(self, grid_points: np.ndarray) -> np.ndarray:
        """Compute the combined PDF of all blobs at given grid points."""
        pdf_total: np.ndarray = np.zeros(grid_points.shape[0])
        for pos in self.positions:
            cov: List[List[float]] = [[self.blob_sigma**2, 0], [0, self.blob_sigma**2]]
            pdf_total += multivariate_normal.pdf(grid_points, mean=pos, cov=cov)
        pdf_total /= pdf_total.sum()
        return pdf_total

    def resample_particles(self, pdf_total: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
        """Resample particles according to the probability distribution."""
        idx: np.ndarray = np.random.choice(grid_points.shape[0], size=self.n_particles, p=pdf_total)
        return grid_points[idx]

    @staticmethod
    def description_length(traj: List[np.ndarray]) -> float:
        """MDL-inspired cost: sum squared deviations from linear extrapolation."""
        if len(traj) < 3:
            return 0.0
        cost: float = 0.0
        for i in range(2, len(traj)):
            pred: np.ndarray = traj[i-1] + (traj[i-1] - traj[i-2])
            dev: float = float(np.linalg.norm(traj[i] - pred))
            cost += dev**2
        return cost

    def update_positions(self, new_particles: np.ndarray) -> None:
        """Update blob positions with MDL inertia. Implemented in subclasses."""
        pass

    def smooth(self, data: Union[np.ndarray, List[float]], window: int = 5) -> np.ndarray:
        """Applies a moving average filter to smooth data."""
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def update_stats(self) -> None:
        """Update distance and velocity stats for blobs 0–1."""
        d: float = float(np.linalg.norm(self.positions[0] - self.positions[1]))
        self.distances.append(d)
        if len(self.distances) > 1:
            v: float = abs(self.distances[-1] - self.distances[-2])
        else:
            v = 0.0
        self.velocities.append(v)

    def run(self, save_gif: str = "simulation.mp4", fps: int = 15) -> None:
        """Run the simulation loop with density+scatter and 3-y-axis stats."""
        grid_size: int = 100
        x: np.ndarray = np.linspace(0, 1, grid_size)
        y: np.ndarray = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points: np.ndarray = np.column_stack([X.ravel(), Y.ravel()])

        self.fig = plt.figure(figsize=(12, 6))
        self.ax_sim = self.fig.add_subplot(1, 2, 1)

        self.ax_stats = host_subplot(111, axes_class=AA.Axes, figure=self.fig)
        self.ax_stats.set_position([0.55, 0.1, 0.4, 0.8])

        self.par1 = self.ax_stats.twinx()
        self.par2 = self.ax_stats.twinx()
        self.par2.axis['right'] = self.par2.new_fixed_axis(loc='right', offset=(60, 0))

        self.ax_stats.set_xlabel("Step")
        self.ax_stats.set_ylabel("Distance")
        self.par1.set_ylabel("Velocity")
        self.par2.set_ylabel("Acceleration")
        self.ax_stats.set_title("Distance, Velocity & Acceleration")
        self.ax_stats.grid(True)

        line_dist, = self.ax_stats.plot([], [], 'b-', label='Distance')
        line_vel, = self.par1.plot([], [], 'g-', label='Velocity')
        line_acc, = self.par2.plot([], [], 'r--', label='Acceleration')

        lines: List[Line2D] = [line_dist, line_vel, line_acc]
        labels: List[str] = [l.get_label() for l in lines]
        self.ax_stats.legend(lines, labels, loc='upper left')

        def update_frame(step: int) -> Tuple[Axes, Any, Line2D, Line2D, Line2D]:
            pdf_total: np.ndarray = self.compute_pdf(grid_points)
            self.particles = self.resample_particles(pdf_total, grid_points)
            self.update_positions(self.particles)
            self.update_stats()

            # --- SMOOTHING LOGIC ---
            # Blend current PDF with previous PDF (Temporal Smoothing)
            # 0.8 weight on the past makes the "field" look rock-solid
            if self.accumulated_pdf is None:
                self.accumulated_pdf = pdf_total
            else:
                self.accumulated_pdf = self.accumulated_pdf * 0.8 + pdf_total * 0.2

            # SIMULATION PLOT
            self.ax_sim.clear()
            
            # Use the accumulated (smoothed) PDF for the heatmap
            self.ax_sim.imshow(
                self.accumulated_pdf.reshape(grid_size, grid_size),
                origin='lower',
                cmap='magma', 
                extent=[0, 1, 0, 1],
                interpolation='bicubic', 
                alpha=1.0 # The field is now opaque and solid
            )

            # Draw particles as faint "quantum mist"
            # Using very small s and low alpha makes them look like sub-structure
            self.ax_sim.scatter(
                self.particles[:,0], 
                self.particles[:,1],
                s=0.2,             # Tiny particles
                c='cyan', 
                alpha=0.15,         # Faint presence
                edgecolors='none'
            )
            
            self.ax_sim.set_xlim(0,1)
            self.ax_sim.set_ylim(0,1)
            self.ax_sim.set_facecolor('black') # Black background for contrast
            self.ax_sim.set_title(f"Step {step+1} | Informational Field")

            # STATS PLOT
            t: np.ndarray = np.arange(len(self.distances))
            line_dist.set_data(t, self.distances)
            line_vel.set_data(t, self.velocities)

            if len(self.velocities) > 5:
                raw_acc: np.ndarray = np.diff(self.velocities)
                smoothed_acc: np.ndarray = self.smooth(raw_acc, window=5)
                t_acc: np.ndarray = np.arange(len(raw_acc) - len(smoothed_acc) + 1, len(self.velocities))
                line_acc.set_data(t_acc, smoothed_acc)
                
                self.par2.set_xlim(0, max(len(t), len(t_acc)))
                self.par2.relim()
                
                a_min: float; a_max: float
                a_min, a_max = self.par2.get_ybound()
                if (a_max - a_min) < 0.6:
                    self.par2.set_ylim(-0.01, 0.01)
                else:
                    self.par2.autoscale_view()                

            self.ax_stats.set_xlim(0, len(t))
            self.ax_stats.relim()
            self.ax_stats.autoscale_view()
            
            self.par1.relim()
            v_min: float; v_max: float
            v_min, v_max = self.par1.get_ybound()
            if (v_max - v_min) < 0.6:
                mid: float = (v_max + v_min) / 2
                self.par1.set_ylim(mid - 0.025, mid + 0.025)
            else:
                self.par1.autoscale_view()

            return self.ax_sim, self.ax_stats, line_dist, line_vel, line_acc

        # Define the writer with your desired FPS
        try:
            # metadata is optional but good for file headers
            writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Gemini'), bitrate=1800)
            
            # Change the filename extension to .mp4
            save_video = save_gif.replace(".gif", ".mp4")
            
            ani: animation.FuncAnimation = animation.FuncAnimation(
                self.fig, update_frame, frames=self.n_steps, blit=False
            )
            
            # Using the ffmpeg writer instead of pillow
            ani.save(save_video, writer=writer)
            print(f"Simulation saved as {save_video}")
            
        except Exception as e:
            print(f"FFMpeg not found, falling back to GIF. Error: {e}")
            ani.save(save_gif, writer="pillow", fps=fps)
        plt.show()

    def setup_axes(self) -> None:
        """Set up a 3-y-axis plot for distance, velocity, and acceleration."""
        self.fig, self.ax_sim = plt.subplots(1, 1, figsize=(12, 6))
        self.ax_stats = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.75)

        self.par1 = self.ax_stats.twinx()
        self.par2 = self.ax_stats.twinx()
        self.par2.axis['right'] = self.par2.new_fixed_axis(loc='right', offset=(60,0))

        self.ax_stats.set_xlabel("Step")
        self.ax_stats.set_ylabel("Distance")
        self.par1.set_ylabel("Velocity")
        self.par2.set_ylabel("Acceleration")

        self.ax_stats.grid(True)
        self.ax_stats.set_title("Distance, Velocity & Acceleration")

        self.lines['dist'], = self.ax_stats.plot([], [], 'b-', label='Distance')
        self.lines['vel'], = self.par1.plot([], [], 'g-', label='Velocity')
        self.lines['acc'], = self.par2.plot([], [], 'r--', label='Acceleration')

        lines: List[Line2D] = [self.lines['dist'], self.lines['vel'], self.lines['acc']]
        labels: List[str] = [l.get_label() for l in lines]
        self.ax_stats.legend(lines, labels, loc='upper left')

    def setup_axes_(self, ax2: Axes) -> None:
        """Initial setup of statistics axes."""
        if self.ax_stats:
            self.ax_stats.set_xlabel("Step")
            self.ax_stats.set_title("Distance, Velocity & Acceleration")
            self.ax_stats.grid(True)
            self.ax_stats.legend(loc="upper left")
        ax2.set_ylabel("Acceleration")
        ax2.legend(loc="upper right")

    def plot_density_particles(self, pdf_total: np.ndarray, grid_size: int, step: int) -> None:
        """Plot density map and particle positions."""
        if self.ax_sim:
            self.ax_sim.clear()
            #self.ax_sim.imshow(
            #    pdf_total.reshape(grid_size, grid_size),
            #    origin="lower",
            #    cmap="inferno",
            #    extent=[0, 1, 0, 1]
            #)

            self.ax_sim.imshow(
                pdf_total.reshape(grid_size, grid_size),
                origin='lower',
                cmap='magma', 
                extent=[0, 1, 0, 1],
                interpolation='bicubic',  # This "smooths" the pixelated raster
                alpha=0.9                 # Allows for slight blending
            )


            self.ax_sim.scatter(
                self.particles[:, 0], self.particles[:, 1],
                s=1, c="cyan", alpha=0.4
            )
            self.ax_sim.set_title(f"Step {step+1}")
            self.ax_sim.set_xlim(0, 1)
            self.ax_sim.set_ylim(0, 1)

    def update_lines(self) -> None:
        """Update lines with a pinned-base smart scale."""
        if not self.ax_stats or not self.par1 or not self.par2:
            return

        t: np.ndarray = np.arange(len(self.distances))
        self.lines['dist'].set_data(t, self.distances)
        self.lines['vel'].set_data(t, self.velocities)

        # DISTANCE (Lock bottom to 0, ensure world view)
        self.ax_stats.relim()
        d_max: float = float(self.ax_stats.get_ybound()[1])
        self.ax_stats.set_ylim(0, max(1.0, d_max))
        self.ax_stats.set_xlim(0, len(t))

        # VELOCITY (Lock bottom to 0, minimum scale 0.6)
        self.par1.relim()
        v_max: float = float(self.par1.get_ybound()[1])
        self.par1.set_ylim(0, max(0.6, v_max))

        # ACCELERATION (Symmetric around 0)
        if len(self.velocities) > 1:
            acceleration: np.ndarray = np.diff(self.velocities)
            t_acc: np.ndarray = np.arange(1, len(self.velocities))
            self.lines['acc'].set_data(t_acc, acceleration)
            
            self.par2.relim()
            a_min: float; a_max: float
            a_min, a_max = self.par2.get_ybound()
            abs_max: float = max(abs(a_min), abs(a_max), 0.01)
            self.par2.set_ylim(-abs_max, abs_max)
            self.par2.set_xlim(0, max(len(t), len(t_acc)))