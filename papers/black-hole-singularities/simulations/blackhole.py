"""
blackhole.py 
Base classes for black hole simulations
"""

from __future__ import annotations
from abc import  abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Self
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation


# Helpers: spherical <-> cartesian (for output plotting)
def sph_from_cart(pos: np.ndarray) -> tuple:
    x, y, z = pos
    r = np.linalg.norm(pos)
    theta = 0.0 if r == 0.0 else np.arccos(z / (r + 1e-16))
    phi = np.arctan2(y, x)
    return r, theta, phi

def cart_from_sph(r: float, theta: float, phi: float) -> np.ndarray:
    s, c = np.sin(theta), np.cos(theta)
    return np.array([r * s * np.cos(phi), r * s * np.sin(phi), r * c], dtype=np.float64)



@dataclass
class BlackHole:
    """Central black hole.

    Attributes:
        mass: Mass parameter M (simulation units, G=c=1).
        radius: Event horizon radius. Defaults to 2M (Schwarzschild radius).
        spin: Kerr spin parameter a = J/M (geometric units).
    """
    mass: float
    radius: float | None = None
    spin: float = 0.0  # specific angular momentum

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
    """RK4 single step for system:
       dpos/dt = vel
       dvel/dt = accel_fn(pos, vel)
    """
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
    """Dust cloud that evolves particles in Cartesian coordinates.

    Subclasses must implement `acceleration(pos, vel) -> np.ndarray`.
    """

    def __init__(self, n: int, r0: float, spacing: float, bh: BlackHole,
                    tangential_fraction: float = 0.8, radial_fraction: float = 0.15,
                    rng_seed: int = 42) -> None:
        self.particles = self.make_particles(n, r0, spacing, bh, tangential_fraction, radial_fraction)
        self.radius = r0 + spacing * n * 1.2

    def acceleration(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """Return acceleration vector for a given particle state.
        Override in subclasses."""
        raise NotImplementedError


    @abstractmethod
    def evolve(self, dt: float, max_t: float, tolerance: float):
        """
        Abstract method for evolving the particle cloud.
        Must return time array and per-particle trajectory data.
        """
        pass

    @staticmethod
    def make_particles(n: int, r0: float, spacing: float, bh: BlackHole,
                    tangential_fraction: float = 0.8, radial_fraction: float = 0.15,
                    rng_seed: int = 42) -> List[Particle]:
        """Generate particles with initial positions and velocities around a black hole.

        Particles are initialized in approximately spherical shells with some
        tangential (circular) and radial velocity components. The distribution of
        positions is randomized over solid angle, and the velocities are chosen to
        reflect orbital and infall dynamics near the black hole.

        Args:
            n (int): Number of particles to generate.
            r0 (float): Initial radial distance of the first particle from the black hole.
            spacing (float): Radial spacing between consecutive particles.
            bh (BlackHole): Black hole object providing the gravitational mass.
            tangential_fraction (float, optional): Fraction of the local circular
                orbital speed assigned to the tangential velocity component.
                Defaults to 0.8.
            radial_fraction (float, optional): Fraction of the local escape speed
                assigned to the inward radial velocity component. Defaults to 0.15.
            rng_seed (int, optional): Random seed for reproducibility. Defaults to 42.

        Returns:
            List[Particle]: A list of initialized Particle objects with positions
            and velocities in Cartesian coordinates.

        Notes:
            - Positions are distributed randomly over the sphere at increasing radii.
            - Velocities are set by combining an inward radial infall component
            and a tangential orbital component.
            - The system uses Newtonian approximations (not full GR).
        """
        rng = np.random.default_rng(rng_seed)
        rs = [r0 + i * spacing for i in range(n)]
        thetas = np.arccos(1 - 2 * rng.random(n))
        phis = 2 * np.pi * rng.random(n)

        particles: List[Particle] = []
        for r, theta, phi in zip(rs, thetas, phis):
            # Cartesian position
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            pos = np.array([x, y, z], dtype=np.float64)

            # radial and tangential unit vectors
            rhat = pos / (np.linalg.norm(pos) + 1e-12)
            # choose a perpendicular direction for tangential
            rand_vec = rng.normal(size=3)
            tang = np.cross(rhat, rand_vec)
            tang /= (np.linalg.norm(tang) + 1e-12)

            # speeds
            v_circ = np.sqrt(bh.mass / r)      # circular speed magnitude
            v_radial = np.sqrt(2 * bh.mass / r) * radial_fraction

            vel_vec = (-rhat * v_radial) + (tang * v_circ * tangential_fraction)

            particles.append(Particle(x=float(x), y=float(y), z=float(z),
                                    vx=float(vel_vec[0]), vy=float(vel_vec[1]), vz=float(vel_vec[2])))
        return particles


    @staticmethod
    def bit_entropy(positions_1d: np.ndarray, scale: float = 1000.0) -> float:
        """Bitwise Shannon entropy of particle positions (1D array of scalars).
        Quantize to nonnegative integers and concantenate to bistring to avoid 
        entropy emerging from numeric representation e.g. floating points"""
        if positions_1d.size == 0:
            return 0.0
        # map to nonnegative ints
        ints = np.floor(np.abs(positions_1d) * scale).astype(np.uint32)
        bits = np.unpackbits(ints.view(np.uint8)).reshape(-1)
        p0 = np.count_nonzero(bits == 0) / bits.size
        p1 = 1.0 - p0
        ent = 0.0
        if p0 > 0:
            ent -= p0 * np.log2(p0)
        if p1 > 0:
            ent -= p1 * np.log2(p1)
        return float(ent)


class DustCloudSimulation:
    """DustCloud simulation (cartesian) and visualizations."""

    def __init__(self, cloud: DustCloud, dt: float, max_t: float, tolerance: float = 1e-8, label: str | None = None) -> None:
        self.cloud = cloud
        self.dt = float(dt)
        self.max_t = float(max_t)
        self.tolerance = float(tolerance)
        self.label = label

        self.times: np.ndarray | None = None
        self.positions: np.ndarray | None = None  # (steps, N, 3)
        self.radii: np.ndarray | None = None      # (N, steps)
        self.entropies: np.ndarray | None = None  # (steps,)

    def run(self) -> Self:
        """Evolve and compute radii + entropies.

        Returns:
            times (steps,), radii (N, steps)
        """
        times, positions = self.cloud.evolve(dt=self.dt, max_t=self.max_t, tolerance=self.tolerance)
        # positions shape (steps, N, 3)
        self.times = times
        self.positions = positions

        # radii: (steps, N) -> transpose to (N, steps) for compatibility with previous plotting
        radii_steps_n = np.linalg.norm(positions, axis=2)  # (steps, N)
        self.radii = radii_steps_n.T  # (N, steps)

        # entropy time series (one scalar per time step)
        ent : float = []
        for i in range(radii_steps_n.shape[0]):
            ent.append(self.cloud.bit_entropy(radii_steps_n[i, :]))
        self.entropies = np.array(ent)

        return self

    def visualize_3d(self, every_n: int = 1, save_path: str | None = None, title: str | None = None) -> None:
        """3D trajectories + entropy subplot."""
        if self.times is None or self.positions is None or self.entropies is None:
            raise RuntimeError("Call run() before visualize_3d().")

        times = self.times
        positions = self.positions  # (steps, N, 3)
        ent = self.entropies
        steps, N, _ = positions.shape
        title = title if title is not None else self.label

        # Plot
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(12, 8))
        traj_ax = fig.add_subplot(211, projection='3d')
        for j in range(0, N, max(1, int(every_n))):
            traj = positions[:, j, :]  # (steps, 3)
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

        # entropy
        ent_ax = fig.add_subplot(212)
        ent_ax.plot(times, ent)
        ent_ax.set_xlabel("t")
        ent_ax.set_ylabel("Entropy (bits)")
        ent_ax.set_title((title + " — " if title else "") + "Entropy over time")
        ent_ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def visualize(self, every_n: int = 1, save_path: str | None = None, title: str | None = None) -> None:
        """2D radial trajectories + entropy plot (backwards-compatible)."""
        if self.times is None or self.radii is None or self.entropies is None:
            raise RuntimeError("Call run() before visualize().")

        limit = self.cloud.radius
        times = self.times
        radii = self.radii
        ent = self.entropies
        title = title if title is not None else self.label

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        for j in range(0, radii.shape[0], max(1, int(every_n))):
            ax.plot(times, radii[j], lw=0.8, alpha=0.9)
        ax.axhline(y=self.cloud.bh.radius, linestyle="--", label=f"r_h = {self.cloud.bh.radius:.3g}")
        ax.set_ylim(0, limit)
        ax.set_xlabel("t")
        ax.set_ylabel("r")
        ax.set_title((title + " — " if title else "") + "Trajectories")
        ax.grid(True, alpha=0.3)
        ax.legend()

        ax = axes[1]
        ax.plot(times, ent)
        ax.set_xlabel("t")
        ax.set_ylabel("Entropy (bits)")
        ax.set_title((title + " — " if title else "") + "Entropy over time")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()


    def animate(self, every_n: int = 1, save_path: str = "blackhole.mp4",
                fps: int = 30, elev: float = 25, azim: float = 45) -> None:
        """Render an MP4 animation of particle trajectories and entropy evolution."""
        if self.times is None or self.positions is None or self.entropies is None:
            raise RuntimeError("Call run() before animate().")

        times = self.times
        positions = self.positions  # shape (steps, N, 3)
        ent = self.entropies
        steps, N, _ = positions.shape

        limit = self.cloud.radius
        title = self.label or "Dust Cloud Evolution"

        # --- Figure layout ---
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])
        ax3d = fig.add_subplot(gs[0, 0], projection="3d")
        axent = fig.add_subplot(gs[0, 1])

        # --- 3D setup ---
        ax3d.set_xlim(-limit, limit)
        ax3d.set_ylim(-limit, limit)
        ax3d.set_zlim(-limit, limit)
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.view_init(elev=elev, azim=azim)
        ax3d.set_title("Particle Trajectories")

        # Initialize lines for each particle
        lines = []
        for j in range(0, N, max(1, int(every_n))):
            line, = ax3d.plot([], [], [], lw=0.8, alpha=0.8)
            lines.append(line)

        # --- Entropy plot setup ---
        axent.set_xlim(times[0], times[-1])
        axent.set_ylim(0, max(ent) * 1.1)
        axent.set_xlabel("t")
        axent.set_ylabel("Entropy (bits)")
        axent.set_title("Entropy over Time")
        axent.grid(True, alpha=0.3)
        line_ent, = axent.plot([], [], color="tab:blue", lw=1.8)
        time_marker = axent.axvline(x=times[0], color="r", ls="--", lw=1.0)

        # --- Animation update ---
        def init():
            for line in lines:
                line.set_data([], [])
                line.set_3d_properties([])
            line_ent.set_data([], [])
            time_marker.set_xdata(times[0])
            return lines + [line_ent, time_marker]

        def update(frame):
            # trajectories
            for idx, line in enumerate(lines):
                j = idx * every_n
                traj = positions[:frame, j, :]
                line.set_data(traj[:, 0], traj[:, 1])
                line.set_3d_properties(traj[:, 2])
            # entropy
            line_ent.set_data(times[:frame], ent[:frame])
            time_marker.set_xdata(times[frame])
            ax3d.view_init(elev=elev, azim=azim + 0.2 * frame)  # slow rotation
            return lines + [line_ent, time_marker]

        # --- Build and save animation ---
        ani = animation.FuncAnimation(
            fig, update, init_func=init, frames=steps, interval=1000 / fps, blit=False
        )

        print(f"Saving animation to {save_path} ...")
        ani.save(save_path, fps=fps, dpi=180, writer="ffmpeg")
        print("Done.")
        plt.close(fig)
