"""
Physics agnostic base class for IaMe simulations.

Concept:
--------
- Gaussian blobs represent observers.
- Particles are resampled according to the joint probability distribution.
- Visualization shows:
    1) Geometric density
    2) Spectral representation
    3) Distance / velocity / acceleration
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple, Optional, Dict, Any, Union
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


class GravitySim:

    def __init__(
        self,
        n_particles: int = 4096,
        n_steps: int = 100,
        blob_sigma: float = 0.1,
        n_candidates: int = 5
    ) -> None:

        self.n_particles = n_particles
        self.n_steps = n_steps
        self.blob_sigma = blob_sigma
        self.n_candidates = n_candidates
        self._spec_ymin = 1e-12
        self._spec_ymax = 1e-12
        self.positions = np.array([
            [0.2, 0.2],
            [0.8, 0.2],
            [0.5, 0.8]
        ])

        self.n_blobs = len(self.positions)
        self.particles = self.init_particles()

        self.trajs: List[List[np.ndarray]] = [[pos.copy()] for pos in self.positions]

        self.distances: List[float] = []
        self.velocities: List[float] = []

        self.accumulated_pdf = None

        # plotting objects
        self.fig: Optional[Figure] = None
        self.ax_sim: Optional[Axes] = None
        self.ax_spec: Optional[Axes] = None
        self.ax_stats: Optional[Axes] = None

    # ============================================================
    # CORE
    # ============================================================

    def init_particles(self) -> np.ndarray:
        particles = []
        for pos in self.positions:
            cov = [[self.blob_sigma**2, 0], [0, self.blob_sigma**2]]
            particles.append(
                np.random.multivariate_normal(pos, cov, self.n_particles // self.n_blobs)
            )
        return np.vstack(particles)

    def compute_pdf(self, grid_points: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement compute_pdf.")

    def resample_particles(self, pdf_total: np.ndarray, grid_points: np.ndarray) -> np.ndarray:
        idx = np.random.choice(
            grid_points.shape[0],
            size=self.n_particles,
            p=pdf_total
        )
        return grid_points[idx]

    def update_positions(self, new_particles: np.ndarray) -> None:
        pass

    def update_stats(self) -> None:
        d = float(np.linalg.norm(self.positions[0] - self.positions[1]))
        self.distances.append(d)

        if len(self.distances) > 1:
            v = abs(self.distances[-1] - self.distances[-2])
        else:
            v = 0.0

        self.velocities.append(v)

    def smooth(self, data: Union[np.ndarray, List[float]], window: int = 5) -> np.ndarray:
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def get_spectral_data(self):
        return None, None

    # ============================================================
    # VISUALIZATION
    # ============================================================

    def run(self, save_video: str, res: int, fps: int = 15):

        grid_size = res
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])

        plt.style.use('dark_background')

        self.fig = plt.figure(figsize=(14, 7))
        self.fig.patch.set_facecolor("black")

        gs = self.fig.add_gridspec(
            2, 2,
            width_ratios=[1, 1.4],
            height_ratios=[1, 1],
            wspace=0.3,
            hspace=0.35
        )

        # Left: heatmap spans both rows
        self.ax_sim = self.fig.add_subplot(gs[:, 0])

        # Right top: spectral
        self.ax_spec = self.fig.add_subplot(gs[0, 1])

        # Right bottom: stats
        self.ax_stats = self.fig.add_subplot(gs[1, 1])

        # ======================
        # SIM PANEL
        # ======================

        self.ax_sim.set_xlim(0, 1)
        self.ax_sim.set_ylim(0, 1)
        self.ax_sim.set_xlabel("x")
        self.ax_sim.set_ylabel("y")
        self.ax_sim.set_aspect("equal")

        self.sim_image = self.ax_sim.imshow(
            np.zeros((grid_size, grid_size)),
            origin="lower",
            cmap="magma",
            extent=[0, 1, 0, 1],
            interpolation="bicubic"
        )

        self.sim_scatter = self.ax_sim.scatter(
            [], [],
            s=1,
            c="cyan",
            alpha=0.2,
            edgecolors="none"
        )

        # ======================
        # SPECTRAL PANEL
        # ======================

        self.ax_spec.set_title("Spectral Complexity")
        self.ax_spec.set_xlabel("k")
        self.ax_spec.set_ylabel("k² |Aₖ|²")
        self.ax_spec.set_yscale("log")
        self.ax_spec.grid(True, alpha=0.2)

        self.spec_line, = self.ax_spec.plot([], [], lw=1.2)

        # ======================
        # STATS PANEL
        # ======================

        self.ax_stats.set_title("Distance / Velocity / Acceleration")
        self.ax_stats.set_xlabel("Step")
        self.ax_stats.set_ylabel("Distance")

        self.par1 = self.ax_stats.twinx()
        self.par2 = self.ax_stats.twinx()
        self.par2.spines["right"].set_position(("outward", 60))

        self.par1.set_ylabel("Velocity")
        self.par2.set_ylabel("Acceleration")

        self.line_dist, = self.ax_stats.plot([], [], color="cyan", label="Distance")
        self.ax_stats.set_ylabel("Distance", color="cyan")
        self.ax_stats.tick_params(axis="y", colors="cyan")

        self.line_vel, = self.par1.plot([], [], color="orange", label="Velocity")
        self.par1.set_ylabel("Velocity", color="orange")
        self.par1.tick_params(axis="y", colors="orange")

        self.line_acc, = self.par2.plot([], [], color="magenta",
                                        linestyle="--", label="Acceleration")
        self.par2.set_ylabel("Acceleration", color="magenta")
        self.par2.tick_params(axis="y", colors="magenta")

        self.ax_stats.legend(loc="upper left")

        # ============================================================
        # FRAME UPDATE
        # ============================================================

        def update_frame(step):

            pdf_total = self.compute_pdf(grid_points)
            self.particles = self.resample_particles(pdf_total, grid_points)

            self.update_positions(self.particles)
            self.update_stats()


            # ---- SIM UPDATE ----
            self.sim_image.set_data(pdf_total.reshape(grid_size, grid_size))
            self.sim_image.set_clim(
                pdf_total.min(),
                pdf_total.max()
            )
            self.sim_scatter.set_offsets(self.particles)
            self.ax_sim.set_title(f"Step {step+1}")

            # ---- SPECTRAL UPDATE ----
            k_vals, spec_vals = self.get_spectral_data()
            if k_vals is not None:
                self.spec_line.set_data(k_vals, spec_vals)
                self.ax_spec.set_xlim(np.min(k_vals), np.max(k_vals))
                current_max = np.nanmax(spec_vals)

                if current_max > self._spec_ymax:
                    self._spec_ymax = current_max * 1.5

                self.ax_spec.set_ylim(self._spec_ymin, self._spec_ymax)

            # ---- STATS UPDATE ----
            t = np.arange(len(self.distances))
            self.line_dist.set_data(t, self.distances)
            self.line_vel.set_data(t, self.velocities)

            if len(self.velocities) > 5:
                raw_acc = np.diff(self.velocities)
                smoothed_acc = self.smooth(raw_acc, 5)
                t_acc = np.arange(len(self.velocities) - len(smoothed_acc), len(self.velocities))
                self.line_acc.set_data(t_acc, smoothed_acc)

            self.ax_stats.set_xlim(0, len(t))
            self.ax_stats.relim()
            self.ax_stats.autoscale_view()
            self.par1.relim()
            self.par1.autoscale_view()
            self.par2.relim()
            self.par2.autoscale_view()

            return (
                self.sim_image,
                self.sim_scatter,
                self.spec_line,
                self.line_dist,
                self.line_vel,
                self.line_acc
            )

        ani = animation.FuncAnimation(
            self.fig,
            update_frame,
            frames=self.n_steps,
            blit=False
        )

        ani.save(save_video, writer="ffmpeg", fps=fps)
        plt.show()

        