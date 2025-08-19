from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from kerr import KerrEquatorialGeodesicCloud
from schwarzschild import SchwarzschildGeodesicCloud
from blackhole import BlackHole, Particle, DustCloudSimulation

# =========================
# Example usage (Cartesian initializer)
# =========================

if __name__ == "__main__":
    def make_particles(n: int, r0: float, spacing: float, bh: BlackHole,
                       tangential_fraction: float = 0.8, radial_fraction: float = 0.15,
                       rng_seed: int = 42) -> List[Particle]:
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

    bh = BlackHole(mass=1.0, spin=0.9)

    sch_cloud = SchwarzschildGeodesicCloud(bh, make_particles(50, r0=6.0, spacing=0.02, bh=bh, tangential_fraction=0.4, radial_fraction=0.6))
    sim_sch = DustCloudSimulation(sch_cloud, dt=1e-3, max_t=12.0, tolerance=1e-6)
    times2, radii2 = sim_sch.run()
    sim_sch.visualize(every_n=2, title="Schwarzschild (a=0.0)")
    sim_sch.visualize_3d(every_n=4, title="Schwarzschild (a=0.0)")


    kerr_cloud = KerrEquatorialGeodesicCloud(bh, make_particles(50, r0=6.0, spacing=0.02, bh=bh, tangential_fraction=0.4, radial_fraction=0.6))
    sim_kerr = DustCloudSimulation(kerr_cloud, dt=1e-3, max_t=12.0, tolerance=1e-6)
    times, radii = sim_kerr.run()
    sim_kerr.visualize(every_n=2, title="Kerr (a=0.9)")
    sim_kerr.visualize_3d(every_n=4, title="Kerr (a=0.9)")



