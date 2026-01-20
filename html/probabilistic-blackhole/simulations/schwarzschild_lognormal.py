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
from blackhole import BlackHole, DustCloud, sph_from_cart, cart_from_sph

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

        # smoothing parameters
        self.r_min = float(r_min)   # radius near singularity where smoothing peaks
        self.sigma = float(sigma)   # lognormal width of smooth hill

    def _smooth_factor(self, r: float) -> float:
        """Deterministic lognormal smoothing factor for g_vv inside horizon.

        Uses a Gaussian in x = ln(r / r_min). Protect tiny r/r_min values to avoid
        extreme logs / division by zero.
        """
        # numerical guards
        r_eps = max(r, 1e-12)
        rmin_eps = max(self.r_min, 1e-12)
        x = np.log(r_eps / rmin_eps)
        return float(np.exp(-0.5 * (x / self.sigma) ** 2))


    def _christoffel(self, r: float, theta: float) -> dict:
        """
        Christoffel symbols for ingoing EF metric (v,r,theta,phi):
        - Outside horizon: standard Schwarzschild EF (factor == 1)
        - Inside horizon: g_vv smoothly adjusted using _smooth_factor

        This routine:
        - computes factor = factor(r) and factor_prime = dfactor/dr
        - forms f = 1 - 2M/r and f_scaled = f * factor
        - forms fprime_scaled = d(f_scaled)/dr = factor * f' + f * factor'
        - uses f_scaled where metric's g_vv (and quantities derived from it) appear
        """
        M = self.M
        # guards
        r_eps = max(r, 1e-12)
        sinth = np.sin(theta)

        # choose factor and its radial derivative
        if r > self.r_h:
            factor = 1.0
            factor_prime = 0.0
        else:
            factor = self._smooth_factor(r_eps)
            # derivative of factor: factor' = factor * (- x / sigma^2) * (1/r)
            x = np.log(r_eps / max(self.r_min, 1e-12))
            factor_prime = - factor * x / (self.sigma**2 * r_eps)

        # base EF metric function and its derivative
        f = 1.0 - 2.0 * M / r_eps               # f(r)
        fprime = 2.0 * M / (r_eps * r_eps)      # df/dr (note sign: df/dr = 2M/r^2)

        # scaled (smoothed) g_vv term and its r-derivative
        f_scaled = f * factor
        fprime_scaled = factor * fprime + f * factor_prime

        Gamma = {}
        # Index order: 0:v, 1:r, 2:theta, 3:phi
        # --- v,r components ---
        # The (0,0,1) and (0,1,0) components involve derivatives of g_vv.
        # For ingoing EF, the symmetric connection components including g_vv should
        # reflect fprime_scaled where appropriate. Keep  previous convention for
        # Gamma[(0,0,1)] but use factor and fprime_scaled consistently.
        Gamma[(0,0,1)] = Gamma[(0,1,0)] = 0.5 * fprime_scaled

        # (1,0,0) uses the same derivative (raising/lowering choices differ by sign),
        # keep your previous style but use fprime_scaled:
        Gamma[(1,0,0)] = 0.5 * fprime_scaled

        # Radial self-term (unchanged; geometric source M/r^2)
        Gamma[(1,1,1)] = -M / (r_eps * r_eps)

        # --- Angular components ---
        # For Schwarzschild, r - 2M = r * f. Replace with r * f_scaled so angular
        # Christoffel inside the horizon respects smoothed g_vv (via f_scaled).
        Gamma[(1,2,2)] = - (r_eps * f_scaled)
        Gamma[(1,3,3)] = - (r_eps * f_scaled) * (sinth ** 2)

        Gamma[(2,1,2)] = Gamma[(2,2,1)] = 1.0 / r_eps
        Gamma[(2,3,3)] = -np.sin(theta) * np.cos(theta)

        Gamma[(3,1,3)] = Gamma[(3,3,1)] = 1.0 / r_eps
        # protect poles explicitly
        if abs(sinth) < 1e-12:
            Gamma[(3,2,3)] = Gamma[(3,3,2)] = 0.0
        else:
            Gamma[(3,2,3)] = Gamma[(3,3,2)] = np.cos(theta) / sinth

        return Gamma

    @staticmethod
    def _state_to_cartesian(state: np.ndarray) -> np.ndarray:
        # state: [v, r, θ, φ, uv, ur, uθ, uφ]
        r, theta, phi = state[1], state[2], state[3]
        return cart_from_sph(r, theta, phi)

    def evolve(self, dt: float, max_t: float, tolerance: float = 1e-8):
        steps = int(np.ceil(max_t / dt)) + 1
        times = np.linspace(0.0, max_t, steps, dtype=np.float64)
        N = len(self.particles)
        positions = []

        for j, p in enumerate(self.particles):
            # initialize EF state
            pos0 = p.position()
            r0, theta0, phi0 = sph_from_cart(pos0)
            uv0, ur0, utheta0, uphi0 = 1.0, -0.001, 0.0, 0.0
            v0 = r0 + 2*self.M*np.log(abs(r0/(2*self.M)-1)+1e-16)
            p._ef_state = np.array([v0, r0, theta0, phi0, uv0, ur0, utheta0, uphi0], dtype=np.float64)

        for step in range(steps):
            snapshot = np.zeros((N, 3), dtype=np.float64)
            for j, p in enumerate(self.particles):
                state = p._ef_state.copy()
                if state[1] <= 1e-16:
                    snapshot[j,:] = np.zeros(3)
                    continue

                # RK4 integration
                def deriv(s):
                    v, r, theta, phi = s[0], s[1], s[2], s[3]
                    u = s[4:8]
                    Gamma = self._christoffel(r, theta)
                    ds = np.zeros(8)
                    ds[0:4] = u
                    acc = np.zeros(4)
                    for a in range(4):
                        for b in range(4):
                            for mu in range(4):
                                if (mu,a,b) in Gamma:
                                    acc[mu] -= Gamma[(mu,a,b)]*u[a]*u[b]
                    ds[4:8] = acc
                    return ds

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
                snapshot[j,:] = self._state_to_cartesian(s_new)

            positions.append(snapshot)

        return times, np.stack(positions, axis=0)
