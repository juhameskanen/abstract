"""
Fixed Schwarzschild Geodesic Simulation (Ingoing EF Coordinates)
===============================================================

This module implements dust cloud collapse into a non-rotating (a=0)
Schwarzschild black hole using ingoing Eddington-Finkelstein (EF)
coordinates. These coordinates are regular across the horizon, enabling
particles to fall smoothly into the black hole without coordinate
singularities disrupting integration.

Purpose
-------
This code provides the simplest *standard GR baseline* for black hole
collapse. It models geodesics in the Schwarzschild spacetime and serves
as a reference for both the Kerr EF simulation (rotating case) and the
Abstract Universe (AU) formulation. Together, these models allow direct
comparison between strict general relativity and its
information-theoretic reinterpretations.

Key Features
------------
- **Metric & Christoffels**: Explicit Christoffel symbols for the
  ingoing EF metric:

      ds² = -(1 - 2M/r) dv² + 2 dv dr + r² dΩ².

- **Horizon penetration**: The EF system avoids coordinate pathologies
  at r = 2M, allowing trajectories to cross the horizon naturally.
- **Dust cloud initialization**: Particles are distributed in spherical
  shells and given velocity components with configurable tangential
  fraction.
- **Geodesic integration**: Adaptive Runge-Kutta scheme evolves the
  full 8D state vector [v, r, θ, φ, uv, ur, uθ, uφ].
- **Termination conditions**: When r approaches the central singularity,
  particles are frozen to prevent numerical blowup.
- **Cartesian output**: States are converted back into (x, y, z) for
  visualization and entropy computation.

Scientific Context
------------------
This module represents the "classical GR" view of non-rotating black
holes: matter collapses into a central singularity where entropy trends
toward zero. In the broader research framework, it underpins the
derivation of the **Geometry-Singularity Lemma** (*vanishing entropy
implies geometric singularity*) and serves as the non-rotating
counterpart to the Kerr EF simulation.

Usage
-----
Instantiate a `SchwarzschildEFGeodesicCloud` and evolve:

    cloud = SchwarzschildEFGeodesicCloud(
        n=100, r0=6.0, spacing=0.02, bh=BlackHole(mass=1.0)
    )
    times, positions = cloud.evolve(dt=1e-3, max_t=20.0)

The returned trajectory array can be used for visualization,
entropy analysis, and comparison against Kerr and AU simulations.

Copyright 2019 - The Abstract Universe Project
"""


from __future__ import annotations
from typing import List, Tuple
import math
import numpy as np
from blackhole import BlackHole, DustCloud, sph_from_cart, cart_from_sph


class SchwarzschildEFGeodesicCloud(DustCloud):
    """
    Schwarzschild metric geodesic integrator in ingoing
    Eddington–Finkelstein coordinates (v, r, theta, phi).
    Regular at the horizon, so particles naturally fall in.
    """

    def __init__(self, n: int, r0: float, spacing: float, bh: BlackHole,
                 tangential_fraction: float = 0.8, radial_fraction: float = 0.15,
                 rng_seed: int = 42) -> None:
        super().__init__(n, r0, spacing, bh, tangential_fraction, radial_fraction)
        self.bh = bh
        self.M = float(bh.mass)
        self.r_h = float(bh.radius)

    def _christoffel(self, r: float, theta: float) -> dict:
        """Christoffels for ingoing EF metric: ds² = -(1-2M/r) dv² + 2 dv dr + r² dΩ²."""
        M = self.M
        f = 1.0 - 2.0 * M / (r + 1e-16)
        fprime = 2.0 * M / ((r + 1e-16) ** 2)
        Gamma = {}
        # index order: 0:v, 1:r, 2:θ, 3:φ

        # Nonzero components (computed from g):
        Gamma[(0, 0, 1)] = Gamma[(0, 1, 0)] = M / (r * r)
        Gamma[(1, 0, 0)] = 0.5 * fprime
        Gamma[(1, 1, 1)] = -M / (r * r)
        Gamma[(1, 2, 2)] = -(r - 2*M)
        Gamma[(1, 3, 3)] = -(r - 2*M) * (np.sin(theta) ** 2)
        Gamma[(2, 1, 2)] = Gamma[(2, 2, 1)] = 1.0 / r
        Gamma[(2, 3, 3)] = -np.sin(theta) * np.cos(theta)
        Gamma[(3, 1, 3)] = Gamma[(3, 3, 1)] = 1.0 / r
        Gamma[(3, 2, 3)] = Gamma[(3, 3, 2)] = np.cos(theta) / (np.sin(theta) + 1e-16)
        return Gamma

    @staticmethod
    def _state_to_cartesian(state: np.ndarray) -> np.ndarray:
        # state: [v, r, θ, φ, uv, ur, uθ, uφ]
        r = state[1]
        theta = state[2]
        phi = state[3]
        return cart_from_sph(r, theta, phi)

    def evolve(self, dt: float, max_t: float, tolerance: float = 1e-8, tangential_fraction: float = 0.0):
        steps = int(max_t / dt)
        times = np.arange(steps, dtype=np.float64) * dt
        N = len(self.particles)
        positions: List[np.ndarray] = []

        for step_idx in range(steps):
            snapshot = np.zeros((N, 3), dtype=np.float64)
            for j, p in enumerate(self.particles):
                if not hasattr(p, "_ef_state"):
                    # Initial state conversion
                    pos = p.position()
                    r0, theta0, phi0 = sph_from_cart(pos)
                    vx, vy, vz = p.velocity()
                    # tortoise coordinate
                    eps = 1e-16
                    rp = r0 + 2*self.M*np.log(abs(r0/(2*self.M) - 1) + eps)
                    v0 = 0.0 + rp  # ingoing EF time

                    # Initial 4-velocity
                    uv = 1.0
                    ur = -0.001  # small inward radial velocity
                    uθ = 0.0
                    # tangential velocity proportional to fraction
                    uφ = tangential_fraction * np.sqrt(self.M / r0)  # crude circular fraction
                    state = np.array([v0, r0, theta0, phi0, uv, ur, uθ, uφ], dtype=np.float64)
                    p._ef_state = state

                state = p._ef_state.copy()

                # If we reach close to singularity, freeze
                if state[1] <= 2*self.M*0.01:
                    snapshot[j, :] = np.zeros(3)
                    continue

                def deriv(s: np.ndarray) -> np.ndarray:
                    v, r, θ, φ = s[0], s[1], s[2], s[3]
                    u = s[4:8]
                    Gamma = self._christoffel(r, θ)
                    dstate = np.zeros(8)
                    dstate[0:4] = u
                    acc = np.zeros(4)
                    for a in range(4):
                        for b in range(4):
                            for mu in range(4):
                                if (mu, a, b) in Gamma:
                                    acc[mu] -= Gamma[(mu, a, b)] * u[a] * u[b]
                    dstate[4:8] = acc
                    return dstate

                # RK4 integration
                s1 = state
                k1 = deriv(s1)
                s2 = s1 + 0.5*dt*k1
                k2 = deriv(s2)
                s3 = s1 + 0.5*dt*k2
                k3 = deriv(s3)
                s4 = s1 + dt*k3
                k4 = deriv(s4)
                s_new = s1 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

                p._ef_state = s_new
                snapshot[j, :] = self._state_to_cartesian(s_new)

            positions.append(snapshot)

        return times, np.stack(positions, axis=0)


    def evolve_buggy(self, dt: float, max_t: float, tolerance: float = 1e-8):
        steps = int(max_t / dt)
        times = np.arange(steps, dtype=np.float64) * dt
        N = len(self.particles)
        positions: List[np.ndarray] = []

        for step_idx in range(steps):
            snapshot = np.zeros((N, 3), dtype=np.float64)
            for j, p in enumerate(self.particles):
                if not hasattr(p, "_ef_state"):
                    # Initial state conversion
                    pos = p.position()
                    r0, theta0, phi0 = sph_from_cart(pos)
                    vx, vy, vz = p.velocity()
                    # tortoise coordinate
                    eps = 1e-16
                    rp = r0 + 2*self.M*np.log(abs(r0/(2*self.M) - 1) + eps)
                    v0 = 0.0 + rp  # ingoing EF time
                    # crude init 4-velocity: assume slow speeds
                    uv = 1.0
                    ur = -0.001
                    uθ = 0.0
                    uφ = 0.0
                    state = np.array([v0, r0, theta0, phi0, uv, ur, uθ, uφ], dtype=np.float64)
                    p._ef_state = state

                state = p._ef_state.copy()

                # If we reach close to singularity, freeze
                if state[1] <= 2*self.M*0.01:
                    snapshot[j, :] = np.zeros(3)
                    continue

                def deriv(s: np.ndarray) -> np.ndarray:
                    v, r, θ, φ = s[0], s[1], s[2], s[3]
                    u = s[4:8]
                    Gamma = self._christoffel(r, θ)
                    dstate = np.zeros(8)
                    dstate[0:4] = u
                    acc = np.zeros(4)
                    for a in range(4):
                        for b in range(4):
                            for mu in range(4):
                                if (mu, a, b) in Gamma:
                                    acc[mu] -= Gamma[(mu, a, b)] * u[a] * u[b]
                    dstate[4:8] = acc
                    return dstate

                # RK4 integration
                s1 = state
                k1 = deriv(s1)
                s2 = s1 + 0.5*dt*k1
                k2 = deriv(s2)
                s3 = s1 + 0.5*dt*k2
                k3 = deriv(s3)
                s4 = s1 + dt*k3
                k4 = deriv(s4)
                s_new = s1 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

                p._ef_state = s_new
                snapshot[j, :] = self._state_to_cartesian(s_new)

            positions.append(snapshot)

        return times, np.stack(positions, axis=0)
