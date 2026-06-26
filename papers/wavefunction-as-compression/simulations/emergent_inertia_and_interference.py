"""
emergent_inertia_gravity.py
===========================
Minimal POC demonstrating the G-ψ-D trinity.

  G — Gravity-like attraction emerges from overlapping Gaussian
      probability distributions. No force law is postulated.
      Observers cluster because the joint PDF |Σψ_i|² has higher
      density between observers, so Born-rule sampled particles
      accumulate there, making smooth trajectories toward neighbours
      naturally lower-cost than trajectories away.

  ψ — Each observer is a complex-valued wavefunction. Its Gaussian
      envelope defines the spatial probability density. Particles
      are Born-rule samples from the joint |ψ_total|². The observer
      phase is the natural frequency ω₀ = 2π/T of the trajectory
      wavefunction — the minimum-complexity closed worldline frequency
      from Theorem 1. No free parameter.

  D — Inertia from spectral complexity. The observer's trajectory
      is encoded as a complex worldline ψ_traj(t) = x(t) + i·y(t).
      At each step, N candidates are drawn from the Born-rule distributed 
      local particles. The candidate minimising C_s of the extended 
      trajectory is chosen. Smooth paths win purely because they are 
      more probable, they have lower C_s and therefore higher Solomonoff
      weight 2^{-C_s}.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from typing import List, Optional
import sys
sys.path.insert(0, '/home/claude')
from wavefunction import Wavefunction


# ── Observer ──────────────────────────────────────────────────────────────────

class Observer:
    """
    A single observer: a Gaussian blob with a complex-valued spatial
    wavefunction and a trajectory wavefunction for inertia.
    """

    TRAJ_LEN = 16   # number of past positions kept for C_s computation

    def __init__(self, pos: np.ndarray, sigma: float, phase: float = 0.0):
        self.pos    = np.array(pos, dtype=float)
        self.sigma  = sigma
        self.phase  = phase
        self.traj: List[complex] = [complex(pos[0], pos[1])]

    def spatial_amplitude(self, points: np.ndarray) -> np.ndarray:
        r2 = np.sum((points - self.pos) ** 2, axis=1)
        return np.exp(-r2 / (2.0 * self.sigma ** 2)) * np.exp(1j * self.phase)

    def trajectory_cs(self) -> float:
        if len(self.traj) < 4:
            return 0.0
        psi = np.array(self.traj, dtype=complex)
        try:
            wf = Wavefunction(psi, dx=1.0)
            return wf.spectral_complexity()
        except Exception:
            return 0.0

    def solomonoff_weight(self) -> float:
        return 2.0 ** (-self.trajectory_cs())

    def candidate_cs(self, candidate: np.ndarray) -> float:
        extended = self.traj + [complex(candidate[0], candidate[1])]
        if len(extended) < 4:
            return 0.0
        psi = np.array(extended[-self.TRAJ_LEN:], dtype=complex)
        try:
            wf = Wavefunction(psi, dx=1.0)
            return wf.spectral_complexity()
        except Exception:
            return float('inf')

    def move_to(self, new_pos: np.ndarray) -> None:
        self.pos = np.array(new_pos, dtype=float)
        self.traj.append(complex(new_pos[0], new_pos[1]))
        if len(self.traj) > self.TRAJ_LEN:
            self.traj.pop(0)
        T = len(self.traj)
        self.phase = 2.0 * np.pi / T


# ── Simulation ────────────────────────────────────────────────────────────────

class EmergentGravitySim:
    """
    The G-ψ-D trinity simulation with particle-density-weighted proposals.
    """

    N_CANDIDATES = 64     # Part 2: Increased candidate pool to find tiny gradients
    GRID_SIZE    = 80     # spatial grid resolution

    def __init__(
        self,
        n_blobs:     int   = 3,
        n_particles: int   = 4096,
        n_steps:     int   = 200,
        sigma:       float = 0.12,
    ) -> None:
        self.n_particles = n_particles
        self.n_steps     = n_steps
        self.sigma       = sigma

        angles = np.linspace(0, 2 * np.pi, n_blobs, endpoint=False)
        r      = 0.28
        cx, cy = 0.5, 0.5
        self.observers: List[Observer] = [
            Observer(
                pos   = np.array([cx + r * np.cos(a), cy + r * np.sin(a)]),
                sigma = sigma,
                phase = float(i) * 2 * np.pi / n_blobs,
            )
            for i, a in enumerate(angles)
        ]

        x = np.linspace(0, 1, self.GRID_SIZE)
        y = np.linspace(0, 1, self.GRID_SIZE)
        X, Y = np.meshgrid(x, y)
        self.grid_points = np.column_stack([X.ravel(), Y.ravel()])
        self.particles = self._resample()

        self.step_idx:    List[int]   = []
        self.distances:   List[float] = []
        self.cs_vals:     List[float] = []
        self.sw_vals:     List[float] = []
        self._acc_pdf: Optional[np.ndarray] = None

    def _joint_pdf(self) -> np.ndarray:
        psi_total = np.zeros(self.grid_points.shape[0], dtype=complex)
        for obs in self.observers:
            psi_total += obs.spatial_amplitude(self.grid_points)
        pdf = np.abs(psi_total) ** 2
        s   = pdf.sum()
        if s < 1e-30:
            return np.ones(len(pdf)) / len(pdf)
        return pdf / s

    def _resample(self) -> np.ndarray:
        pdf = self._joint_pdf()
        idx = np.random.choice(len(self.grid_points), size=self.n_particles, p=pdf)
        return self.grid_points[idx].copy()

    def _assigned_particles(self, obs: Observer) -> np.ndarray:
        dist = np.linalg.norm(self.particles - obs.pos, axis=1)
        return self.particles[dist < 3.0 * obs.sigma]

    def _best_next_pos(self, obs: Observer) -> np.ndarray:
        """
        Choose the next position by generating candidates along directions matching 
        local Born-rule sampled particle positions, utilizing a tight step size ε.
        """
        # Part 1: Tiny step size relative to sigma (ε ~ σ / 100)
        epsilon = self.sigma / 100.0  

        # Grab particle distribution assigned to this observer
        assigned = self._assigned_particles(obs)
        candidates = []

        for _ in range(self.N_CANDIDATES):
            if len(assigned) > 0:
                # Draw candidate directions directly via Born-rule sampling
                p_k = assigned[np.random.choice(len(assigned))]
                vec = p_k - obs.pos
                dist = np.linalg.norm(vec)
                
                if dist > 1e-8:
                    unit_vec = vec / dist
                else:
                    angle = np.random.uniform(0, 2 * np.pi)
                    unit_vec = np.array([np.cos(angle), np.sin(angle)])
                
                candidate = obs.pos + epsilon * unit_vec
            else:
                # Fallback to a uniform direction step if no particles fall within 3σ
                angle = np.random.uniform(0, 2 * np.pi)
                candidate = obs.pos + epsilon * np.array([np.cos(angle), np.sin(angle)])

            candidate = np.clip(candidate, 0.01, 0.99)
            candidates.append(candidate)

        # Include staying stationary
        candidates.append(obs.pos.copy())

        # Select candidate that minimizes the trajectory spectral complexity
        scores = [(obs.candidate_cs(c), c) for c in candidates]
        scores.sort(key=lambda x: x[0])
        _, best_pos = scores[0]

        return best_pos

    def _update_observers(self) -> None:
        new_positions = [self._best_next_pos(obs) for obs in self.observers]
        for obs, new_pos in zip(self.observers, new_positions):
            obs.move_to(new_pos)

    def _record_stats(self, step: int) -> None:
        self.step_idx.append(step)
        positions = np.array([o.pos for o in self.observers])
        n = len(positions)
        if n >= 2:
            dists = [np.linalg.norm(positions[i] - positions[j])
                     for i in range(n) for j in range(i+1, n)]
            self.distances.append(float(np.mean(dists)))
        else:
            self.distances.append(0.0)
        cs_list = [o.trajectory_cs() for o in self.observers]
        sw_list = [o.solomonoff_weight() for o in self.observers]
        self.cs_vals.append(float(np.mean(cs_list)))
        self.sw_vals.append(float(np.mean(sw_list)))

    def run(self, output: str = "emergent_gravity.gif", fps: int = 12, writer: str = "pillow") -> None:
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(13, 6), facecolor='#080810')
        gs  = GridSpec(2, 2, figure=fig, left=0.04, right=0.98, top=0.93, bottom=0.09, hspace=0.45, wspace=0.35)

        ax_sim  = fig.add_subplot(gs[:, 0])
        ax_dist = fig.add_subplot(gs[0, 1])
        ax_cs   = fig.add_subplot(gs[1, 1])

        ax_sim.set_facecolor('#000008')
        ax_sim.set_xlim(0, 1); ax_sim.set_ylim(0, 1)
        ax_sim.set_xticks([]); ax_sim.set_yticks([])

        colours = ['#7dd3fc', '#fbbf24', '#a78bfa', '#34d399', '#f87171', '#fb923c'][:len(self.observers)]

        img = ax_sim.imshow(np.zeros((self.GRID_SIZE, self.GRID_SIZE)), origin='lower', extent=[0,1,0,1],
                            cmap='magma', vmin=0, vmax=1, interpolation='bicubic', aspect='auto')
        scat = ax_sim.scatter(self.particles[:,0], self.particles[:,1], s=0.3, c='cyan', alpha=0.12, edgecolors='none')
        blob_dots = [ax_sim.plot([], [], 'o', color=c, ms=8, zorder=5)[0] for c in colours]
        blob_trails = [ax_sim.plot([], [], '-', color=c, alpha=0.4, lw=1)[0] for c in colours]
        title = ax_sim.set_title('', color='#94a3b8', fontsize=10)

        line_dist, = ax_dist.plot([], [], color='#7dd3fc', lw=1.5)
        ax_dist.set_ylabel('Mean distance', color='#7dd3fc', fontsize=8)
        ax_dist.set_xlabel('Step', fontsize=8)
        ax_dist.tick_params(colors='#475569', labelsize=7)
        ax_dist.set_facecolor('#0a0a18')
        ax_dist.grid(True, color='#1e1e36', lw=0.5)

        line_cs, = ax_cs.plot([], [], color='#a78bfa', lw=1.5, label='C_s')
        line_sw, = ax_cs.plot([], [], color='#34d399', lw=1.5, linestyle='--', label='2^{-C_s}')
        ax_cs.set_ylabel('C_s  /  Solomonoff weight', color='#94a3b8', fontsize=8)
        ax_cs.set_xlabel('Step', fontsize=8)
        ax_cs.tick_params(colors='#475569', labelsize=7)
        ax_cs.set_facecolor('#0a0a18')
        ax_cs.grid(True, color='#1e1e36', lw=0.5)
        ax_cs.legend(fontsize=7, loc='upper right', facecolor='#0a0a18', edgecolor='#1e1e36')

        fig.suptitle('G-ψ-D Trinity  ·  Emergent Gravity from Compression', color='#dde3ee', fontsize=11, y=0.99)

        def update(step: int):
            pdf = self._joint_pdf()
            self.particles = self._resample()
            self._update_observers()
            self._record_stats(step)

            if self._acc_pdf is None:
                self._acc_pdf = pdf.copy()
            else:
                self._acc_pdf = 0.8 * self._acc_pdf + 0.2 * pdf

            field = self._acc_pdf.reshape(self.GRID_SIZE, self.GRID_SIZE)
            fmax  = field.max()
            if fmax > 1e-30:
                field = field / fmax
            img.set_data(field)
            scat.set_offsets(self.particles)

            for i, (obs, dot, trail) in enumerate(zip(self.observers, blob_dots, blob_trails)):
                dot.set_data([obs.pos[0]], [obs.pos[1]])
                traj_arr = np.array([[z.real, z.imag] for z in obs.traj], dtype=float)
                trail.set_data(traj_arr[:,0], traj_arr[:,1])

            title.set_text(f'Step {step+1}  ·  C_s = {self.cs_vals[-1]:.1f}  ·  2^{{-C_s}} = {self.sw_vals[-1]:.4f}')

            t = np.array(self.step_idx)
            line_dist.set_data(t, self.distances)
            ax_dist.relim(); ax_dist.autoscale_view()

            line_cs.set_data(t, self.cs_vals)
            line_sw.set_data(t, self.sw_vals)
            ax_cs.relim(); ax_cs.autoscale_view()

            return [img, scat] + blob_dots + blob_trails + [line_dist, line_cs, line_sw, title]

        print(f"Running {self.n_steps} steps …")
        ani = animation.FuncAnimation(fig, update, frames=self.n_steps, blit=False, interval=80)

        if writer == 'pillow':
            ani.save(output, writer='pillow', fps=fps)
        else:
            w = animation.FFMpegWriter(fps=fps, bitrate=1800)
            ani.save(output, writer=w)

        print(f"Saved → {output}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='G-ψ-D Trinity: emergent gravity from compression.')
    parser.add_argument('--blobs',     type=int,   default=3, help='Number of observers (default 3)')
    parser.add_argument('--particles', type=int,   default=4096, help='Number of particles (default 4096)')
    parser.add_argument('--steps',     type=int,   default=200, help='Simulation steps (default 200)')
    parser.add_argument('--sigma',     type=float, default=0.12, help='Observer Gaussian width (default 0.12)')
    parser.add_argument('--k_mod',     type=float, default=0.0,
                        help='Spatial phase modulation frequency for visible interference (default 0.0)')
    parser.add_argument('--file',      type=str,   default='emergent_gravity', help='Output filename without extension')
    parser.add_argument('--format',    choices=['gif', 'mp4'], default='gif', help='Output format (default gif)')
    parser.add_argument('--fps',       type=int,   default=12, help='Frames per second (default 12)')
    args = parser.parse_args()

    output = f"{args.file}.{args.format}"
    writer = 'pillow' if args.format == 'gif' else 'ffmpeg'

    sim = EmergentGravitySim(
        n_blobs     = args.blobs,
        n_particles = args.particles,
        n_steps     = args.steps,
        sigma       = args.sigma,
    )
    sim.run(output=output, fps=args.fps, writer=writer)


if __name__ == '__main__':
    main()
