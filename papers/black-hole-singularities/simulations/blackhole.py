"""
blackhole.py 
Base classes for black hole simulations with multi-metric complexity tracking.
"""

from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Self
import math

# Force Matplotlib into non-interactive mode immediately to bypass headless environment stalls
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
from numba import njit


@njit
def sph_from_cart(pos: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = pos[0], pos[1], pos[2]
    r = math.sqrt(x*x + y*y + z*z)
    theta = 0.0 if r == 0.0 else math.acos(z / (r + 1e-16))
    phi = math.atan2(y, x)
    return r, theta, phi

@njit
def cart_from_sph(r: float, theta: float, phi: float) -> np.ndarray:
    s = math.sin(theta)
    c = math.cos(theta)
    return np.array([r * s * math.cos(phi), r * s * math.sin(phi), r * c])


@dataclass
class BlackHole:
    """Central black hole."""
    mass: float
    radius: float | None = None
    spin: float = 0.0  

    def __post_init__(self) -> None:
        if self.radius is None:
            self.radius = 2.0 * self.mass


@dataclass
class Particle:
    """Particle in Cartesian phase space."""
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz], dtype=np.float64)

    def set_state(self, pos: np.ndarray, vel: np.ndarray) -> None:
        self.x, self.y, self.z = float(pos[0]), float(pos[1]), float(pos[2])
        self.vx, self.vy, self.vz = float(vel[0]), float(vel[1]), float(vel[2])


def rk4_step(pos: np.ndarray, vel: np.ndarray, accel_fn, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """RK4 single step for system."""
    def deriv(state_pos, state_vel):
        return state_vel, accel_fn(state_pos, state_vel)

    k1_p, k1_v = deriv(pos, vel)
    k2_p, k2_v = deriv(pos + 0.5 * dt * k1_p, vel + 0.5 * dt * k1_v)
    k3_p, k3_v = deriv(pos + 0.5 * dt * k2_p, vel + 0.5 * dt * k2_v)
    k4_p, k4_v = deriv(pos + dt * k3_p, vel + dt * k3_v)

    pos_new = pos + (dt / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
    vel_new = vel + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)

    return pos_new, vel_new


class DustCloud:
    """Dust cloud base framing."""

    def __init__(self, n: int, r0: float, spacing: float, bh: BlackHole,
                 tangential_fraction: float = 0.8, radial_fraction: float = 0.15,
                 rng_seed: int = 42) -> None:
        self.particles = self.make_particles(n, r0, spacing, bh, tangential_fraction, radial_fraction)
        self.bh = bh
        self.radius = r0 + spacing * n * 1.2

    def acceleration(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def evolve(self, dt: float, max_t: float, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @staticmethod
    def make_particles(n: int, r0: float, spacing: float, bh: BlackHole,
                       tangential_fraction: float = 0.8, radial_fraction: float = 0.15,
                       rng_seed: int = 42) -> List[Particle]:
        rng = np.random.default_rng(rng_seed)
        rs = [r0 + i * spacing for i in range(n)]
        thetas = np.arccos(1 - 2 * rng.random(n))
        phis = 2 * np.pi * rng.random(n)

        particles: List[Particle] = []
        for r, theta, phi in zip(rs, thetas, phis):
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            pos = np.array([x, y, z], dtype=np.float64)

            rhat = pos / (np.linalg.norm(pos) + 1e-12)
            rand_vec = rng.normal(size=3)
            tang = np.cross(rhat, rand_vec)
            tang /= (np.linalg.norm(tang) + 1e-12)

            v_circ = np.sqrt(bh.mass / r)
            v_radial = np.sqrt(2 * bh.mass / r) * radial_fraction

            vel_vec = (-rhat * v_radial) + (tang * v_circ * tangential_fraction)

            particles.append(Particle(x=float(x), y=float(y), z=float(z),
                                      vx=float(vel_vec[0]), vy=float(vel_vec[1]), vz=float(vel_vec[2])))
        return particles

    @staticmethod
    @njit
    def bit_entropy(positions_1d: np.ndarray, scale: float = 1000.0) -> float:
        size = positions_1d.size
        if size == 0:
            return 0.0
        
        total_bits = float(size * 32)
        count_ones = 0
        
        for i in range(size):
            val = np.uint32(abs(positions_1d[i]) * scale)
            c = 0
            temp_val = val
            while temp_val > 0:
                if temp_val & np.uint32(1):
                    c += 1
                temp_val >>= np.uint32(1)
            count_ones += c
        
        p1 = count_ones / total_bits
        p0 = 1.0 - p1
        
        ent = 0.0
        if 0.0 < p1 < 1.0:
            ent -= p1 * math.log2(p1)
            ent -= p0 * math.log2(p0)
            
        return float(ent)

    @staticmethod
    @njit
    def block_entropy(positions_1d: np.ndarray, block_size: int = 4, scale: float = 1000.0) -> float:
        size = positions_1d.size
        if size == 0 or block_size < 1 or block_size > 16:
            return 0.0
        
        total_bits = size * 32
        bit_stream = np.zeros(total_bits, dtype=np.uint8)
        
        bit_idx = 0
        for i in range(size):
            val = np.uint32(abs(positions_1d[i]) * scale)
            for b in range(32):
                bit_stream[bit_idx] = np.uint8((val >> np.uint32(b)) & np.uint32(1))
                bit_idx += 1
                
        num_blocks = total_bits // block_size
        if num_blocks == 0:
            return 0.0
        
        num_possible_patterns = 1 << block_size
        counts = np.zeros(num_possible_patterns, dtype=np.uint32)
        
        for b in range(num_blocks):
            pattern_val = 0
            start_bit = b * block_size
            for step in range(block_size):
                if bit_stream[start_bit + step] == 1:
                    pattern_val |= (1 << step)
            counts[pattern_val] += 1
            
        ent = 0.0
        for p in range(num_possible_patterns):
            if counts[p] > 0:
                p_pattern = counts[p] / num_blocks
                ent -= p_pattern * math.log2(p_pattern)
                
        return float(ent / block_size)



    @staticmethod
    @njit
    def spectral_complexity(positions_1d: np.ndarray, delta_omega: float = 0.05) -> float:
        """Normalized spatial rendering complexity tracking raw spectral power weights."""
        size = positions_1d.size
        if size <= 1:
            return 0.0
        
        # Calculate mean to isolate structural AC variance from bulk scale
        mean_val = 0.0
        for n in range(size):
            mean_val += positions_1d[n]
        mean_val /= size
        
        fft_len = size // 2 + 1
        c_s = 0.0
        cost_phi = 1.0  
        
        # Capture spatial dispersion limit
        max_amplitude = 0.0
        amplitudes = np.zeros(fft_len)
        
        for k in range(1, fft_len):
            real_part = 0.0
            imag_part = 0.0
            for n in range(size):
                # Analyze variations around the shifting average profile
                v = positions_1d[n] - mean_val
                angle = (2.0 * math.pi * float(k) * float(n)) / float(size)
                real_part += v * math.cos(angle)
                imag_part -= v * math.sin(angle)
                
            amp = math.sqrt(real_part * real_part + imag_part * imag_part)
            amplitudes[k] = amp
            if amp > max_amplitude:
                max_amplitude = amp
                
        # Dynamic noise floor thresholding (1/10000th of peak structural mode)
        threshold = max_amplitude * 1e-4 if max_amplitude > 1e-8 else 1e-12
            
        for k in range(1, fft_len):
            if amplitudes[k] > threshold:
                omega_k = float(k)
                # Weights grow with frequency and relative spectral mode prominence
                weight = amplitudes[k] / (max_amplitude + 1e-12)
                c_s += (cost_phi + (omega_k / delta_omega)) * weight
                
        return float(c_s)




class DustCloudSimulation:
    """DustCloud simulation executing and recording geometric collapse profiles."""

    def __init__(self, cloud: DustCloud, dt: float, max_t: float, tolerance: float = 1e-8, label: str | None = None) -> None:
        self.cloud = cloud
        self.dt = float(dt)
        self.max_t = float(max_t)
        self.tolerance = float(tolerance)
        self.label = label

        self.times: np.ndarray | None = None
        self.positions: np.ndarray | None = None  
        self.radii: np.ndarray | None = None      
        
        self.entropies: np.ndarray | None = None            
        self.block_entropies: np.ndarray | None = None      
        self.spectral_complexities: np.ndarray | None = None 

    def run(self) -> Self:
        """Evolve and safely parse metrics, catching and dropping numerical overflows."""
        times, positions = self.cloud.evolve(dt=self.dt, max_t=self.max_t, tolerance=self.tolerance)
        
        self.times = times
        
        # Guard rails: Convert broken integration anomalies (NaNs/Infs) into a clean bounded state
        positions_cleaned = np.where(np.isnan(positions) | np.isinf(positions), 0.0, positions)
        # Prevent wild coordinate spikes from triggering matrix multiply overflows
        self.positions = np.clip(positions_cleaned, -1e5, 1e5)

        # Safely capture norms without overflow hazard
        radii_steps_n = np.linalg.norm(self.positions, axis=2)
        self.radii = radii_steps_n.T  

        steps = radii_steps_n.shape[0]
        self.entropies = np.zeros(steps, dtype=np.float64)
        self.block_entropies = np.zeros(steps, dtype=np.float64)
        self.spectral_complexities = np.zeros(steps, dtype=np.float64)
        
        for i in range(steps):
            spatial_state = radii_steps_n[i, :]
            self.entropies[i] = self.cloud.bit_entropy(spatial_state)
            self.block_entropies[i] = self.cloud.block_entropy(spatial_state, block_size=4)
            self.spectral_complexities[i] = self.cloud.spectral_complexity(spatial_state, delta_omega=0.05)

        return self

    def visualize_3d(self, every_n: int = 1, save_path: str | None = None, title: str | None = None) -> None:
        if self.times is None or self.positions is None or self.entropies is None:
            raise RuntimeError("Call run() before visualize_3d().")

        times = self.times
        positions = self.positions  
        ent = self.entropies
        _, N, _ = positions.shape
        title = title if title is not None else self.label

        fig = plt.figure(figsize=(12, 8))
        traj_ax = fig.add_subplot(211, projection='3d')
        for j in range(0, N, max(1, int(every_n))):
            traj = positions[:, j, :]  
            traj_ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=0.8, alpha=0.8)
        
        limit = self.cloud.radius
        traj_ax.set_xlim(-limit, limit)
        traj_ax.set_ylim(-limit, limit)
        traj_ax.set_zlim(-limit, limit)

        traj_ax.set_xlabel('X')
        traj_ax.set_ylabel('Y')
        traj_ax.set_zlabel('Z')
        traj_ax.set_title((title + " — " if title else "") + "3D trajectories")
        traj_ax.grid(True, alpha=0.3)

        ent_ax = fig.add_subplot(212)
        ent_ax.plot(times, ent)
        ent_ax.set_xlabel("t")
        ent_ax.set_ylabel("Entropy (bits)")
        ent_ax.set_title((title + " — " if title else "") + "Entropy over time")
        ent_ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            # Safely handle headless systems hitting standard visual triggers
            plt.savefig(f"temp_render_3d_{self.label.replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    def visualize(self, every_n: int = 1, save_path: str | None = None, title: str | None = None) -> None:
        if self.times is None or self.radii is None or self.entropies is None:
            raise RuntimeError("Call run() before visualize().")

        limit = self.cloud.radius
        times = self.times
        radii = self.radii
        title = title if title is not None else self.label

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax = axes[0, 0]
        for j in range(0, radii.shape[0], max(1, int(every_n))):
            ax.plot(times, radii[j], lw=0.8, alpha=0.9)
        ax.axhline(y=self.cloud.bh.radius, linestyle="--", color="r", label=f"r_h = {self.cloud.bh.radius:.3g}")
        ax.set_ylim(0, limit)
        ax.set_xlabel("t")
        ax.set_ylabel("r")
        ax.set_title("Geometric Particle Convergence")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[0, 1]
        ax.plot(times, self.entropies, color="tab:blue", lw=1.5)
        ax.set_xlabel("t")
        ax.set_ylabel("Entropy (bits)")
        ax.set_title("Single-Bit Shannon Entropy")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(times, self.block_entropies, color="tab:green", lw=1.5)
        ax.set_xlabel("t")
        ax.set_ylabel("Normalized Block Entropy")
        ax.set_title("Spatial Block Pattern Entropy (Word Size = 4)")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(times, self.spectral_complexities, color="tab:orange", lw=1.5)
        ax.set_xlabel("t")
        ax.set_ylabel("Complexity $\mathcal{C}_s$")
        ax.set_title("IAME Spectral Complexity $\mathcal{C}_s$")
        ax.grid(True, alpha=0.3)

        plt.suptitle(title if title else "Gravitational Collapse Complexity Analysis", fontsize=14)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.savefig(f"temp_render_metrics_{self.label.replace(' ', '_')}.png", dpi=150)
        plt.close(fig)

    def animate(self, every_n: int = 1, save_path: str = "blackhole.mp4",
                fps: int = 30, elev: float = 25, azim: float = 45) -> None:
        if self.times is None or self.positions is None or self.spectral_complexities is None:
            raise RuntimeError("Call run() before animate().")

        times = self.times
        positions = self.positions  
        steps, N, _ = positions.shape
        limit = self.cloud.radius

        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1])
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        axmetrics = fig.add_subplot(gs[0, 1])

        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.view_init(elev=elev, azim=azim)
        ax3d.set_title("Cartesian Geometric Domain")

        lines = []
        for j in range(0, N, max(1, int(every_n))):
            line, = ax3d.plot([], [], [], lw=0.8, alpha=0.8)
            lines.append(line)

        axmetrics.set_xlim(times[0], times[-1])
        max_y = max(max(self.entropies), max(self.block_entropies), 1.0)
        axmetrics.set_ylim(0, max_y * 1.1)
        axmetrics.set_xlabel("t")
        axmetrics.set_ylabel("Metric Profiles")
        axmetrics.set_title("Informational Spectrum Collapse")
        axmetrics.grid(True, alpha=0.3)
        
        line_shannon, = axmetrics.plot([], [], color="tab:blue", lw=1.5, label="Shannon Entropy")
        line_block, = axmetrics.plot([], [], color="tab:green", lw=1.5, label="Block Entropy")
        
        ax_spec = axmetrics.twinx()
        ax_spec.set_ylim(0, max(self.spectral_complexities) * 1.1)
        ax_spec.set_ylabel("Spectral Complexity $\mathcal{C}_s$", color="tab:orange")
        line_spectral, = ax_spec.plot([], [], color="tab:orange", lw=2.0, label="Spectral Complexity ($\mathcal{C}_s$)")
        ax_spec.tick_params(axis='y', labelcolor="tab:orange")

        lns = [line_shannon, line_block, line_spectral]
        labs = [l.get_label() for l in lns]
        axmetrics.legend(lns, labs, loc="upper right")

        time_marker = axmetrics.axvline(x=times[0], color="r", ls="--", lw=1.0)

        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            line_shannon.set_data([], [])
            line_block.set_data([], [])
            line_spectral.set_data([], [])
            time_marker.set_xdata([times[0]])
            return lines + [line_shannon, line_block, line_spectral, time_marker]

        def update(frame):
            for idx, line in enumerate(lines):
                j = idx * every_n
                traj = positions[:frame, j, :]
                line.set_data(traj[:, 0], traj[:, 1])
                line.set_3d_properties(traj[:, 2])
            
            line_shannon.set_data(times[:frame], self.entropies[:frame])
            line_block.set_data(times[:frame], self.block_entropies[:frame])
            line_spectral.set_data(times[:frame], self.spectral_complexities[:frame])
            time_marker.set_xdata([times[frame]])
            ax3d.view_init(elev=elev, azim=azim + 0.2 * frame)  
            return lines + [line_shannon, line_block, line_spectral, time_marker]

        ani = animation.FuncAnimation(fig, update, init_func=init, frames=steps, interval=1000 / fps, blit=False)

        print(f"Saving animation to {save_path} ...")
        ani.save(save_path, fps=fps, dpi=180, writer="ffmpeg")
        print("Done.")
        plt.close(fig)

    def animate_density(self, every_n: int = 1, save_path="collapse_density.mp4", fps=30, grid_size=128, sigma=1.2):
        if self.positions is None or self.entropies is None:
            raise RuntimeError("Call run() before animate_density().")
        pos = self.positions
        ent = self.entropies
        steps, N, _ = pos.shape
        limit = self.cloud.radius
        X, Y = np.meshgrid(np.linspace(-limit, limit, grid_size), np.linspace(-limit, limit, grid_size))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        heat = ax1.imshow(np.zeros((grid_size, grid_size)), cmap="inferno", origin="lower", extent=[-limit, limit, -limit, limit], vmin=0, vmax=1)
        ax1.set_title("Dust density collapse")
        ax1.axis("off")
        ax2.set_xlim(0, steps)
        ax2.set_ylim(0, max(ent) * 1.1)
        ax2.set_xlabel("t")
        ax2.set_ylabel("Entropy")
        ax2.set_title("Entropy over time")
        ax2.grid(True, alpha=0.3)
        line_ent, = ax2.plot([], [], lw=2)
        marker = ax2.axvline(0, color="red", ls="--")
        def density_frame(frame):
            grid = np.zeros((grid_size, grid_size))
            for j in range(N):
                x, y = pos[frame, j, 0], pos[frame, j, 1]
                g = np.exp(-((X-x)**2 + (Y-y)**2)/(2*sigma**2))
                grid += g
            grid /= grid.max() + 1e-12
            return grid
        def update(frame):
            heat.set_data(density_frame(frame))
            line_ent.set_data(np.arange(frame), ent[:frame])
            marker.set_xdata([frame])
            return heat, line_ent, marker
        ani = animation.FuncAnimation(fig, update, frames=steps, blit=False)
        print(f"Saving density animation -> {save_path}")
        ani.save(save_path, fps=fps, dpi=200, writer="ffmpeg")
        plt.close(fig)

    def animate_geodesics(self, every_n: int = 1, save_path: str="geodesics.mp4", fps: int=30):
        if self.positions is None:
            raise RuntimeError("Call run() before animate_geodesics().")
        pos = self.positions
        steps, N, _ = pos.shape
        limit = self.cloud.radius
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_title("Geodesic convergence")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, alpha=0.3)
        lines = []
        for j in range(0, N, max(1, int(every_n))):
            line, = ax.plot([], [], lw=1)
            lines.append(line)
        def update(frame):
            for idx, line in enumerate(lines):
                j = idx * every_n
                traj = pos[:frame, j, :]
                line.set_data(traj[:, 0], traj[:, 1])
            return lines
        ani = animation.FuncAnimation(fig, update, frames=steps, blit=False)
        print(f"Saving geodesic animation -> {save_path}")
        ani.save(save_path, fps=fps, dpi=200, writer="ffmpeg")
        plt.close(fig)
