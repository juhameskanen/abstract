"""
Schwarzschild geodesics with lognormal entropy-driven smoothing. The 
spacetime metric inside the event horizon is determined by the 
lognormal probability distribution of emergent structures (particles) 
emerging from information.

The degree of emergence is governed by entropy:

- At zero entropy, no physical structures exist. Energy, curvature, and geometry 
  are absent.
- As entropy increases, structures emerge explosively—particles, fields, and 
  gravitational curvature arise as collective phenomena.
- The probability of emergence follows a lognormal distribution: emergence is 
  suppressed at very low entropy, maximized at a characteristic entropy scale, 
  and then diminishes at higher scales.

Accordingly, the Schwarzschild metric inside the horizon is modified by a 
lognormal smoothing factor applied to the g_vv component:

    g_vv(r) → g_vv(r) * exp( -½ [ ln(r / r_min) / σ ]² ),

where r_min defines the characteristic scale of maximum emergence and σ sets 
the width of the lognormal profile.

This construction has two consequences:
1. Physical interpretation — Curvature is defined as probability density of emergent micro structures as entropy rises, 
   consistent with the entropy-driven emergence of spacetime. No information, no particles, 
   no gravity -> the metric at the singularity is smooth and well defined. 
2. Mathematical regularization — The lognormal factor suppresses the singularity 
   at r → 0 smoothly, ensuring that geodesic integration remains deterministic 
   and well-behaved across the horizon and down to the center.

The lognormal profile is not arbitrary but directly reflects the 
information-theoretic origin of spacetime: curvature and geodesic structure 
emerge only where entropy has grown sufficiently to support them.

Copyright 2019 - The Abstract Universe Project
"""

from __future__ import annotations
from typing import List
import numpy as np
from numba import njit
from blackhole import BlackHole, DustCloud, sph_from_cart, cart_from_sph

@njit
def get_accel_au(r: float, theta: float, u: np.ndarray, M: float, r_h: float, r_min: float, sigma: float) -> np.ndarray:
    acc = np.zeros(4)
    r_eps = max(r, 1e-12)
    sinth = np.sin(theta)
    
    if r_eps > r_h:
        factor = 1.0
        fprime_scaled = 2.0 * M / (r_eps * r_eps)
        f_scaled = 1.0 - 2.0 * M / r_eps
    else:
        # Lognormal smoothing factor
        rmin_eps = max(r_min, 1e-12)
        x = np.log(r_eps / rmin_eps)
        factor = np.exp(-0.5 * (x / sigma)**2)
        factor_prime = - factor * x / (sigma**2 * r_eps)
        
        f = 1.0 - 2.0 * M / r_eps
        fprime = 2.0 * M / (r_eps * r_eps)
        
        f_scaled = f * factor
        fprime_scaled = factor * fprime + f * factor_prime

    # Gamma^v components
    acc[0] -= 0.5 * fprime_scaled * u[0] * u[1] * 2.0
    
    # Gamma^r components
    acc[1] -= 0.5 * fprime_scaled * u[0]**2
    acc[1] -= (-M / (r_eps * r_eps)) * u[1]**2
    acc[1] -= (-(r_eps * f_scaled)) * u[2]**2
    acc[1] -= (-(r_eps * f_scaled) * sinth**2) * u[3]**2
    
    # Gamma^theta components
    acc[2] -= (1.0 / r_eps) * u[1] * u[2] * 2.0
    acc[2] -= (-sinth * np.cos(theta)) * u[3]**2
    
    # Gamma^phi components
    acc[3] -= (1.0 / r_eps) * u[1] * u[3] * 2.0
    term_ph = 0.0 if abs(sinth) < 1e-12 else np.cos(theta) / sinth
    acc[3] -= term_ph * u[2] * u[3] * 2.0
    
    return acc

@njit
def n_step_rk4_au(state: np.ndarray, dt: float, M: float, r_h: float, r_min: float, sigma: float) -> np.ndarray:
    def get_deriv(s):
        r, theta = s[1], s[2]
        u = s[4:8]
        dst = np.zeros(8)
        dst[0:4] = u
        dst[4:8] = get_accel_au(r, theta, u, M, r_h, r_min, sigma)
        return dst

    k1 = get_deriv(state)
    k2 = get_deriv(state + 0.5 * dt * k1)
    k3 = get_deriv(state + 0.5 * dt * k2)
    k4 = get_deriv(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

class SchwarzschildAUGeodesicCloud(DustCloud):
    """
    Schwarzschild geodesics with a lognormal-probability metric inside the horizon.

    - Outside horizon (r > r_h): standard Schwarzschild EF metric
    - Inside horizon (r <= r_h): g_vv component smoothed with lognormal
    - Fully deterministic; trajectories integrate smoothly to r -> 0
    """

    def __init__(self, n: int, r0: float, spacing: float, bh: BlackHole,
                 tangential_fraction: float = 0.8, radial_fraction: float = 0.15,
                 r_min: float = 1e-4, sigma: float = 0.2):
        super().__init__(n=n, r0=r0, spacing=spacing, bh=bh,
                         tangential_fraction=tangential_fraction,
                         radial_fraction=radial_fraction)
        self.bh = bh
        self.M = float(bh.mass)
        self.r_h = float(bh.radius)
        self.r_min = float(r_min)
        self.sigma = float(sigma)

    def evolve(self, dt: float, max_t: float, tolerance: float = 1e-8):
        steps = int(np.ceil(max_t / dt)) + 1
        times = np.linspace(0.0, max_t, steps, dtype=np.float64)
        N = len(self.particles)
        
        all_states = np.zeros((N, 8), dtype=np.float64)
        for j, p in enumerate(self.particles):
            r0, theta0, phi0 = sph_from_cart(p.position())
            v0 = r0 + 2*self.M*np.log(abs(r0/(2*self.M)-1)+1e-16)
            all_states[j] = [v0, r0, theta0, phi0, 1.0, -0.001, 0.0, 0.0]

        positions = np.zeros((steps, N, 3), dtype=np.float64)

        for step_idx in range(steps):
            for j in range(N):
                state = all_states[j]
                
                if state[1] <= 1e-12:
                    positions[step_idx, j, :] = 0.0
                    continue

                new_state = n_step_rk4_au(state, dt, self.M, self.r_h, self.r_min, self.sigma)
                all_states[j] = new_state
                
                r, th, ph = new_state[1], new_state[2], new_state[3]
                s, c = np.sin(th), np.cos(th)
                positions[step_idx, j, 0] = r * s * np.cos(ph)
                positions[step_idx, j, 1] = r * s * np.sin(ph)
                positions[step_idx, j, 2] = r * c

        return times, positions