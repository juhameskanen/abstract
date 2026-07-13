"""
Schwarzschild Geodesic Simulation based on Counting
===================================================

This module implements dust cloud collapse into a non-rotating (a=0)
Schwarzschild black hole using ingoing Eddington-Finkelstein (EF)
coordinates and the frameworks counting equation. 


Purpose
-------
This code provides the simplest black hole
collapse based on the framworks informational model. 


Copyright 2001 ... 2026 - Juha Meskanen
The Abstract Universe Project
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


class SchwarzschildCounting(DustCloud):
    """
    A pure realization of the pixel conservation equation: rho_fabric = n - m(r)
    Zero artificial coupling or scaling parameters. All boundaries and forces 
    emerge strictly from the Black Hole's mass (M), spin (a), and charge (q).
    """
    def __init__(self, n: int, r0: float, spacing: float, bh: BlackHole,
                 tangential_fraction: float = 0.8, radial_fraction: float = 0.15, 
                 rng_seed: int = 42) -> None:
        
        # Pull the only physical properties that exist
        self.M = float(bh.mass)
        self.a = float(getattr(bh, 'spin', 0.0))    # Angular momentum per unit mass
        self.q = float(getattr(bh, 'charge', 0.0))  # Electric charge
        
        # Calculate the real physical outer horizon radius from the metric equations
        # r_h = M + sqrt(M^2 - a^2 - q^2)
        discriminant = self.M**2 - self.a**2 - self.q**2
        if discriminant >= 0:
            self.r_h = self.M + np.sqrt(discriminant)
        else:
            # Naked singularity scenario (no horizon exists physically)
            self.r_h = 0.0
            
        # Background pixel capacity density (Normalized to Planck baseline)
        self.n = 1.0  # don't confuse to n parameter, which is the number of dust particles
        
        super().__init__(n, r0, spacing, bh, tangential_fraction, radial_fraction, rng_seed)

    def calculate_rho_fabric(self, r: float) -> Tuple[float, float]:
        """
        Calculates rho_fabric = n - m(r) where m(r) is strictly governed 
        by the intrinsic physical dimensions of the black hole.
        """
        if self.r_h == 0.0 or r > self.r_h:
            # Outside the physical horizon boundary, matter structure count matches 
            # the classic external mass footprint
            m_r = self.M / r
            dm_dr = -self.M / (r**2 + 1e-16)
        else:
            # Inside the horizon, the sin profile dictates how the mass M 
            # distributes its pixel consumption across its own natural horizon radius r_h.
            m_r = self.M * (np.sin(np.pi * r / self.r_h))**2
            dm_dr = self.M * (np.pi / self.r_h) * np.sin(2.0 * np.pi * r / self.r_h)

        # Fundamental counting equation
        rho_fabric = self.n - m_r
        d_rho_dr = -dm_dr
        
        return rho_fabric, d_rho_dr

    def acceleration(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """
        Acceleration is the direct, unscaled manifestation of the spatial bit gradient.
        No alpha coupling variable used.
        """
        x, y, z = pos[0], pos[1], pos[2]
        r = np.sqrt(x*x + y*y + z*z) + 1e-16
        r_hat = pos / r
        
        # Extract the fundamental canvas density change
        rho_fabric, d_rho_dr = self.calculate_rho_fabric(r)
        
        # Acceleration is purely the spatial gradient vector of the fabric density
        # a = -grad(rho) -> a_mag = -d_rho_dr
        return -d_rho_dr * r_hat

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

        # 3. INTEGRATION LOOP
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
