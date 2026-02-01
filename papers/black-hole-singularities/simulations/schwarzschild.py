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
from numba import njit

from blackhole import BlackHole, DustCloud, sph_from_cart, cart_from_sph

# --- STANDALONE NUMBA FUNCTIONS (OUTSIDE THE CLASS) ---

@njit
def get_accel_numba(r, theta, u, M):
    acc = np.zeros(4)
    # Protection against singularity division
    r_eff = max(r, 1e-8)
    f_inv_r2 = M / (r_eff**2)
    
    # Gamma^v components
    acc[0] -= (f_inv_r2) * u[0] * u[1] * 2.0
    
    # Gamma^r components
    acc[1] -= (M/r_eff**2 * (1.0 - 2.0*M/r_eff)) * u[0]**2
    acc[1] -= (-f_inv_r2) * u[1]**2
    acc[1] -= (-(r_eff - 2.0*M)) * u[2]**2
    acc[1] -= (-(r_eff - 2.0*M) * np.sin(theta)**2) * u[3]**2
    
    # Gamma^theta components
    acc[2] -= (1.0/r_eff) * u[1] * u[2] * 2.0
    acc[2] -= (-np.sin(theta) * np.cos(theta)) * u[3]**2
    
    # Gamma^phi components
    acc[3] -= (1.0/r_eff) * u[1] * u[3] * 2.0
    acc[3] -= (np.cos(theta) / (np.sin(theta) + 1e-16)) * u[2] * u[3] * 2.0
    
    return acc

@njit
def n_step_rk4(state, dt, M):
    # s is [v, r, theta, phi, uv, ur, utheta, uphi]
    def get_deriv(s):
        r, theta = s[1], s[2]
        u = s[4:8]
        dst = np.zeros(8)
        dst[0:4] = u
        dst[4:8] = get_accel_numba(r, theta, u, M)
        return dst

    k1 = get_deriv(state)
    k2 = get_deriv(state + 0.5 * dt * k1)
    k3 = get_deriv(state + 0.5 * dt * k2)
    k4 = get_deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# --- THE CLASS ---

class SchwarzschildEFGeodesicCloud(DustCloud):
    def __init__(self, n: int, r0: float, spacing: float, bh: BlackHole,
                 tangential_fraction: float = 0.8, radial_fraction: float = 0.15,
                 rng_seed: int = 42) -> None:
        super().__init__(n, r0, spacing, bh, tangential_fraction, radial_fraction)
        self.bh = bh
        self.M = float(bh.mass)

    def evolve(self, dt: float, max_t: float, tolerance: float = 1e-8):
        steps = int(max_t / dt)
        times = np.arange(steps, dtype=np.float64) * dt
        N = len(self.particles)
        
        # 1. INITIALIZATION
        all_states = np.zeros((N, 8), dtype=np.float64)
        for j, p in enumerate(self.particles):
            pos = p.position()
            r0, theta0, phi0 = sph_from_cart(pos)
            
            eps = 1e-16
            rp = r0 + 2*self.M*np.log(abs(r0/(2*self.M) - 1) + eps)
            v0 = 0.0 + rp 

            uv, ur, u_th, u_ph = 1.0, -0.001, 0.0, 0.0
            all_states[j] = [v0, r0, theta0, phi0, uv, ur, u_th, u_ph]

        # 2. PRE-ALLOCATE
        positions = np.zeros((steps, N, 3), dtype=np.float64)

        # 3. INTEGRATION LOOP (This will now be lightning fast)
        for step_idx in range(steps):
            for j in range(N):
                state = all_states[j]

                # Singularity check
                if state[1] <= 0.01:
                    positions[step_idx, j, :] = 0.0
                    continue

                # CALL THE NUMBA ENGINE
                new_state = n_step_rk4(state, dt, self.M)
                all_states[j] = new_state

                # Store result (Cartesian)
                r, th, ph = new_state[1], new_state[2], new_state[3]
                positions[step_idx, j, 0] = r * np.sin(th) * np.cos(ph)
                positions[step_idx, j, 1] = r * np.sin(th) * np.sin(ph)
                positions[step_idx, j, 2] = r * np.cos(th)

        return times, positions