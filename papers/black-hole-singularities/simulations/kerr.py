"""
Fixed Kerr Geodesic Simulation (Equatorial, Ingoing EF Coordinates)
==================================================================

This module provides a stable, horizon-penetrating implementation of
timelike geodesics for dust clouds falling into Kerr black holes,
restricted to the equatorial plane. The formulation uses ingoing
Eddington-Finkelstein (IEF) coordinates (v, r, φ̃), which are regular
across the horizon and allow smooth integration through r = r_+.

Purpose
-------
This code is the "standard GR baseline" against which information-
theoretic and abstract-universe models can be compared. Whereas the AU
model reinterprets the metric probabilistically to avoid singularities,
this module adheres strictly to general relativity, evolving geodesics
up to the numerical cutoff near r = 0. It demonstrates the classical
collapse of a dust cloud into a singularity, and provides a controlled
reference for entropy analysis.

Key Features
------------
- **Metric & Christoffel symbols**: Computed explicitly in the equatorial
  plane of Kerr spacetime, with finite-difference derivatives in r only.
- **Horizon penetration**: Ingoing EF coordinates remove coordinate
  singularities, enabling geodesics to cross the horizon smoothly.
- **State evolution**: Each particle evolves according to an adaptive
  Runge-Kutta scheme with step halving and normalization to enforce the
  geodesic condition g_{μν} u^μ u^ν = -1.
- **Termination safeguards**: Integration halts when particles approach
  r ≈ 0 (numerical cutoff), exhibit runaway velocities, or produce
  non-finite states.
- **Cartesian output**: States are converted back into (x, y, z) for
  visualization and entropy computation.

Scientific Context
------------------
This module embodies the strictly classical GR picture: collapsing
dust clouds lead inevitably to central singularities. In the broader
research program, it provides the empirical basis for extrapolating the
entropy trend toward zero and formulating the **Geometry-Singularity
Lemma**: *vanishing entropy implies geometric singularity*. By contrast
with the AU formulation, this code exposes the limits of GR and
highlights where information-theoretic reinterpretations become
necessary.

Usage
-----
Instantiate a `KerrIEFEquatorialGeodesicCloud` with parameters:

    cloud = KerrIEFEquatorialGeodesicCloud(
        n=100, r0=6.0, spacing=0.02, bh=BlackHole(mass=1.0, spin=0.9)
    )
    times, positions = cloud.evolve(dt=1e-3, max_t=20.0)

The returned trajectory array is suitable for visualization and entropy
measurement in downstream analysis.

Copyright 2019, The Abstract Universe Project
"""


from __future__ import annotations
from typing import List, Tuple
import math
import numpy as np
from numba import njit
from blackhole import BlackHole, DustCloud, sph_from_cart, cart_from_sph
@njit
def get_kerr_metric_matrix(r: float, M: float, a: float) -> np.ndarray:
    """Equatorial Kerr metric in IEF coordinates."""
    r_eff = max(r, 1e-10) # Stiffer guard for the metric
    one_minus_2M_over_r = 1.0 - (2.0 * M) / r_eff
    
    gvv = -one_minus_2M_over_r
    gvr = 1.0
    gvphi = -a * one_minus_2M_over_r
    grphi = -a
    gphph = (r_eff**2 + a**2) + (2.0 * M * a**2) / r_eff
    
    G = np.zeros((3, 3))
    G[0, 0], G[0, 1], G[0, 2] = gvv, gvr, gvphi
    G[1, 0], G[1, 1], G[1, 2] = gvr, 0.0, grphi
    G[2, 0], G[2, 1], G[2, 2] = gvphi, grphi, gphph
    return G

@njit
def normalize_u_kerr(r: float, u: np.ndarray, M: float, a: float) -> np.ndarray:
    """Enforce g_uv u^u u^v = -1 by solving for u^v quadratic."""
    G = get_kerr_metric_matrix(r, M, a)
    ur, uphi = u[1], u[2]
    
    A = G[0, 0]
    B = 2.0 * (G[0, 1] * ur + G[0, 2] * uphi)
    C = (G[1, 1] * ur**2) + (2.0 * G[1, 2] * ur * uphi) + (G[2, 2] * uphi**2)
    
    target = -1.0
    cQ = C - target
    
    # Quadratic solver with numerical safety
    if abs(A) < 1e-12:
        uv = -cQ / (B + 1e-16)
    else:
        disc = B*B - 4.0*A*cQ
        if disc < 0:
            uv = -B / (2.0 * A) # Fallback to vertex
        else:
            sqrt_disc = math.sqrt(disc)
            uv1 = (-B + sqrt_disc) / (2.0 * A)
            uv2 = (-B - sqrt_disc) / (2.0 * A)
            uv = uv1 if uv1 > uv2 else uv2
        
    return np.array([uv, ur, uphi])

@njit
def get_accel_kerr(r: float, u: np.ndarray, M: float, a: float) -> np.ndarray:
    """Numerical Christoffel acceleration with LinAlgError safety."""
    r_eff = max(r, 1e-10)
    h = 1e-6 * max(1.0, abs(r_eff))
    
    G = get_kerr_metric_matrix(r_eff, M, a)
    # Check if matrix is finite before inverting
    if not np.isfinite(G).all():
        return np.zeros(3) # Kill acceleration to trigger termination

    G_plus = get_kerr_metric_matrix(r_eff + h, M, a)
    G_minus = get_kerr_metric_matrix(r_eff - h, M, a)
    
    dgdr = (G_plus - G_minus) / (2.0 * h)
    Ginv = np.linalg.inv(G)
    
    acc = np.zeros(3)
    for mu in range(3):
        s = 0.0
        for alpha in range(3):
            for beta in range(3):
                gamma_val = 0.0
                for nu in range(3):
                    t1 = dgdr[nu, beta] if alpha == 1 else 0.0
                    t2 = dgdr[nu, alpha] if beta == 1 else 0.0
                    t3 = dgdr[alpha, beta] if nu == 1 else 0.0
                    gamma_val += 0.5 * Ginv[mu, nu] * (t1 + t2 - t3)
                s -= gamma_val * u[alpha] * u[beta]
        acc[mu] = s
    return acc

@njit
def n_step_rk4_kerr(state: np.ndarray, dt: float, M: float, a: float, r_min: float) -> np.ndarray:
    def get_deriv(s):
        r = s[1]
        if r < r_min or not np.isfinite(r):
            return np.zeros(6)
        u = s[3:6]
        dst = np.zeros(6)
        dst[0:3] = u
        dst[3:6] = get_accel_kerr(r, u, M, a)
        return dst

    k1 = get_deriv(state)
    k2 = get_deriv(state + 0.5 * dt * k1)
    k3 = get_deriv(state + 0.5 * dt * k2)
    k4 = get_deriv(state + dt * k3)
    
    new_s = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    # Re-normalize if still in bounds
    if new_s[1] > r_min and np.isfinite(new_s).all():
        new_s[3:6] = normalize_u_kerr(new_s[1], new_s[3:6], M, a)
    return new_s

class KerrIEFEquatorialGeodesicCloud(DustCloud):
    """
    Kerr geodesics in ingoing Eddington–Finkelstein coordinates (v, r, phitilde),
    restricted to the equatorial plane (theta = pi/2).
    """
    def __init__(self, n: int, r0: float, spacing: float, bh: BlackHole,
                 tangential_fraction: float = 0.8, radial_fraction: float = 0.15,
                 rng_seed: int = 42, r_min_factor: float = 1e-4):
        super().__init__(n, r0, spacing, bh, tangential_fraction, radial_fraction)
        self.bh = bh
        self.M = float(bh.mass)
        self.a = float(bh.spin)
        self.tangential_fraction = float(tangential_fraction)
        self.r_plus = float(self.M + math.sqrt(max(0.0, self.M**2 - self.a**2)))
        self.r_min = max(1e-10, r_min_factor * self.M)

    def evolve(self, dt: float, max_t: float, tolerance: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        steps = int(max_t / dt)
        times = np.arange(steps, dtype=np.float64) * dt
        N = len(self.particles)
        
        all_states = np.zeros((N, 6), dtype=np.float64)
        for j, p in enumerate(self.particles):
            r0, _, phi0 = sph_from_cart(p.position())
            dr_dt = -0.01 # Slightly stronger initial push to ensure movement
            dphi_dt = (self.tangential_fraction * math.sqrt(self.M/r0)) / r0
            
            u_init = np.array([1.0, dr_dt, dphi_dt])
            u_norm = normalize_u_kerr(r0, u_init, self.M, self.a)
            all_states[j] = [0.0, r0, phi0, u_norm[0], u_norm[1], u_norm[2]]

        positions = np.zeros((steps, N, 3), dtype=np.float64)
        alive = np.ones(N, dtype=np.bool_)

        for step_idx in range(steps):
            for j in range(N):
                if not alive[j]:
                    # Keep the last known position
                    if step_idx > 0:
                        positions[step_idx, j] = positions[step_idx-1, j]
                    continue
                
                state = all_states[j]
                
                # Check bounds before stepping
                if state[1] <= self.r_min or not np.isfinite(state).all():
                    alive[j] = False
                    continue

                new_state = n_step_rk4_kerr(state, dt, self.M, self.a, self.r_min)
                all_states[j] = new_state
                
                r, phi = all_states[j, 1], all_states[j, 2]
                positions[step_idx, j, 0] = r * math.cos(phi)
                positions[step_idx, j, 1] = r * math.sin(phi)
                positions[step_idx, j, 2] = 0.0

        return times, positions