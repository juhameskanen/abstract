# fixed_simulation.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import math
import matplotlib.pyplot as plt
import numpy as np
from blackhole import BaseDustCloud, sph_from_cart, cart_from_sph

# ---------------------
# Schwarzschild geodesics (full 4D)
# ---------------------
class SchwarzschildGeodesicCloud(BaseDustCloud):
    """
    Schwarzschild metric geodesic integrator in Schwarzschild coordinates
    (t, r, theta, phi). Integrates d^2 x^mu / d\tau^2 = -Gamma^mu_ab u^a u^b,
    with u^mu = dx^mu/d\tau.

    Accepts initial Particle objects (Cartesian pos & Cartesian vel). Uses a helper
    to convert those into coordinate initial conditions (t0=0, r0,theta0,phi0,
    and the 4-velocity u^mu normalized).
    """

    def __init__(self, bh: BlackHole, particles: List[Particle]) -> None:
        super().__init__(particles)
        self.bh = bh
        self.M = float(bh.mass)
        self.r_h = float(bh.radius)

    # Christoffel components (nonzero) for Schwarzschild metric:
    # Using standard Schwarzschild metric: f = 1 - 2M/r
    def _christoffel(self, r: float, theta: float) -> dict:
        M = self.M
        f = 1.0 - 2.0 * M / (r + 1e-16)
        fprime = 2.0 * M / ((r + 1e-16) ** 2)
        # Precompute some factors
        Gamma = {}
        # indices: 0:t, 1:r, 2:theta, 3:phi
        # Γ^t_{tr} = f'/(2f)
        Gamma[(0, 0, 1)] = Gamma[(0, 1, 0)] = 0.5 * fprime / (f + 1e-16)
        # Γ^r_{tt} = f * f'/2
        Gamma[(1, 0, 0)] = 0.5 * f * fprime
        # Γ^r_{rr} = -f'/(2f)
        Gamma[(1, 1, 1)] = -0.5 * fprime / (f + 1e-16)
        # Γ^r_{θθ} = -r f
        Gamma[(1, 2, 2)] = -r * f
        # Γ^r_{φφ} = -r f sin^2θ
        Gamma[(1, 3, 3)] = -r * f * (math.sin(theta) ** 2)
        # Γ^θ_{rθ} = Γ^θ_{θr} = 1/r
        Gamma[(2, 1, 2)] = Gamma[(2, 2, 1)] = 1.0 / (r + 1e-16)
        # Γ^θ_{φφ} = -sinθ cosθ
        Gamma[(2, 3, 3)] = -math.sin(theta) * math.cos(theta)
        # Γ^φ_{rφ} = Γ^φ_{φr} = 1/r
        Gamma[(3, 1, 3)] = Gamma[(3, 3, 1)] = 1.0 / (r + 1e-16)
        # Γ^φ_{θφ} = Γ^φ_{φθ} = cotθ
        Gamma[(3, 2, 3)] = Gamma[(3, 3, 2)] = math.cos(theta) / (math.sin(theta) + 1e-16)
        return Gamma

    @staticmethod
    def _state_to_cartesian(state: np.ndarray) -> np.ndarray:
        # state: [t, r, theta, phi, ut, ur, utheta, uphi]
        r = state[1]
        theta = state[2]
        phi = state[3]
        return cart_from_sph(r, theta, phi)

    def evolve(self, dt: float, max_t: float, tolerance: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate each particle geodesically in proper time using fixed-step RK4.
        Returns (times (steps,), positions (steps, N, 3)).
        """
        steps = int(max_t / dt)
        times = np.arange(steps, dtype=np.float64) * dt
        N = len(self.particles)
        positions: List[np.ndarray] = []

        for step_idx in range(steps):
            snapshot = np.zeros((N, 3), dtype=np.float64)
            for j, p in enumerate(self.particles):
                # We'll store per-particle state on the particle instance (using an attribute)
                # If not present, initialize from Cartesian particle data
                if not hasattr(p, "_sch_state"):
                    # build initial coordinate state from cartesian pos+vel
                    t0 = 0.0
                    pos = p.position()
                    r0, theta0, phi0 = sph_from_cart(pos)
                    # compute coordinate velocities dx^i/dt from cartesian velocities
                    x, y, z = pos
                    vx, vy, vz = p.velocity()
                    eps = 1e-16
                    denom = x * x + y * y + eps
                    dr_dt = (x * vx + y * vy + z * vz) / (r0 + eps)
                    dtheta_dt = 0.0
                    if r0 > eps:
                        dtheta_dt = ((z * (x * vx + y * vy + z * vz)) - (r0 * r0) * vz) / ((r0 * r0) * math.sqrt(x * x + y * y) + eps)
                    dphi_dt = (x * vy - y * vx) / (denom + eps)

                    # Now form v_coord = (dr/dt, dtheta/dt, dphi/dt)
                    vcoord = np.array([dr_dt, dtheta_dt, dphi_dt], dtype=np.float64)

                    # Metric components at (r0,theta0)
                    M = self.M
                    f = 1.0 - 2.0 * M / (r0 + eps)
                    g_tt = -f
                    g_rr = 1.0 / (f + eps)
                    g_thth = r0 * r0
                    g_phph = (r0 * r0) * (math.sin(theta0) ** 2)

                    # compute spatial metric contraction g_ij v^i v^j (with v^i = dx^i/dt)
                    gij_vv = g_rr * (vcoord[0] ** 2) + g_thth * (vcoord[1] ** 2) + g_phph * (vcoord[2] ** 2)

                    denom_ut = g_tt + gij_vv
                    if denom_ut >= 0.0:
                        # degenerate/very large speed artifact: fall back to low-velocity approx
                        ut = 1.0 / math.sqrt(-g_tt + 1e-8)
                    else:
                        ut = math.sqrt(-1.0 / (denom_ut + 1e-16))
                    # u^i = v^i * u^t (convert to d()/d\tau)
                    ur = vcoord[0] * ut
                    utheta = vcoord[1] * ut
                    uphi = vcoord[2] * ut

                    # store state vector as numpy array: [t, r, theta, phi, ut, ur, utheta, uphi]
                    state = np.array([t0, r0, theta0, phi0, ut, ur, utheta, uphi], dtype=np.float64)
                    p._sch_state = state

                state = p._sch_state.copy()

                # If already inside horizon, keep at horizon and zero 4-vel
                if state[1] <= self.r_h + tolerance:
                    # freeze at horizon (one-way capture)
                    r_h = self.r_h
                    # convert to Cartesian for plotting; keep same angles
                    theta = state[2]
                    phi = state[3]
                    pos_cart = cart_from_sph(r_h, theta, phi)
                    snapshot[j, :] = pos_cart
                    # zero velocity to avoid movement
                    p._sch_state = np.array([state[0], r_h, theta, phi, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
                    continue

                # define derivative function for RK4
                def deriv(s: np.ndarray) -> np.ndarray:
                    # s: [t, r, theta, phi, ut, ur, utheta, uphi]
                    t, r, theta, phi = s[0], s[1], s[2], s[3]
                    u = s[4:8]
                    # compute Christoffel at (r,theta)
                    Gamma = self._christoffel(r, theta)
                    # dx^mu/dtau = u^mu
                    dstatedtau = np.zeros(8, dtype=np.float64)
                    dstatedtau[0:4] = u  # tdot, rdot, thetadot, phidot

                    # acceleration: du^mu/dtau = -Gamma^mu_ab u^a u^b
                    acc = np.zeros(4, dtype=np.float64)
                    # sum over a,b from 0..3
                    for a in range(4):
                        for b in range(4):
                            key = (0, a, b)
                            if key in Gamma:
                                acc[0] -= Gamma[key] * u[a] * u[b]
                            key = (1, a, b)
                            if key in Gamma:
                                acc[1] -= Gamma[key] * u[a] * u[b]
                            key = (2, a, b)
                            if key in Gamma:
                                acc[2] -= Gamma[key] * u[a] * u[b]
                            key = (3, a, b)
                            if key in Gamma:
                                acc[3] -= Gamma[key] * u[a] * u[b]
                    dstatedtau[4:8] = acc
                    return dstatedtau

                # RK4 step in proper time
                s1 = state
                k1 = deriv(s1)
                s2 = s1 + 0.5 * dt * k1
                k2 = deriv(s2)
                s3 = s1 + 0.5 * dt * k2
                k3 = deriv(s3)
                s4 = s1 + dt * k3
                k4 = deriv(s4)
                s_new = s1 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

                # If particle crosses horizon in this step, clamp to horizon and kill velocity
                if s_new[1] <= self.r_h + tolerance:
                    t, r, theta, phi = s_new[0], s_new[1], s_new[2], s_new[3]
                    # project to horizon along radial direction using angles
                    r_h = self.r_h
                    pos_cart = cart_from_sph(r_h, theta, phi)
                    snapshot[j, :] = pos_cart
                    p._sch_state = np.array([s_new[0], r_h, s_new[2], s_new[3], 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
                    continue

                # accept the step
                p._sch_state = s_new
                snapshot[j, :] = self._state_to_cartesian(s_new)

            positions.append(snapshot)

        return np.arange(steps, dtype=np.float64) * dt, np.stack(positions, axis=0)


