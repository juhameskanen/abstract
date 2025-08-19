# fixed_simulation.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import math
import matplotlib.pyplot as plt
import numpy as np
from blackhole import BaseDustCloud, sph_from_cart, cart_from_sph


class KerrEquatorialGeodesicCloud(BaseDustCloud):
    """
    Kerr metric geodesics restricted to the equatorial plane (theta = pi/2).
    Coordinates: (t, r, phi). We integrate those plus their derivatives u^t,u^r,u^phi.
    """

    def __init__(self, bh: BlackHole, particles: List[Particle]) -> None:
        super().__init__(particles)
        self.bh = bh
        self.M = float(bh.mass)
        self.a = float(bh.spin)
        # outer horizon r_+ = M + sqrt(M^2 - a^2)
        self.r_plus = float(self.M + math.sqrt(max(0.0, self.M * self.M - self.a * self.a)))

    def _metric_components_equatorial(self, r: float) -> dict:
        M = self.M
        a = self.a
        Sigma = r * r  # at theta = pi/2, cos^2θ=0 => Σ=r^2
        Delta = r * r - 2.0 * M * r + a * a
        sin2 = 1.0
        gtt = -(1.0 - 2.0 * M * r / (Sigma + 1e-16))
        gtphi = -2.0 * M * a * r / (Sigma + 1e-16) * sin2
        grr = Sigma / (Delta + 1e-16)
        gphph = ( (r * r + a * a) + (2.0 * M * a * a * r) / (Sigma + 1e-16) ) * sin2
        return {"gtt": gtt, "gtphi": gtphi, "grr": grr, "gphph": gphph, "Sigma": Sigma, "Delta": Delta}

    def _christoffel_equatorial(self, r: float) -> dict:
        # We'll compute Γ^μ_{αβ} for μ,α,β in (t=0,r=1,phi=2) by computing
        # needed derivatives ∂_r g_{μν} analytically and inverting the 3x3 metric.
        M = self.M
        a = self.a
        Sigma = r * r
        Delta = r * r - 2.0 * M * r + a * a
        # metric components (3x3: t, r, phi)
        m = self._metric_components_equatorial(r)
        gtt = m["gtt"]; gtphi = m["gtphi"]; grr = m["grr"]; gphph = m["gphph"]

        # metric as matrix: indices 0=t,1=r,2=phi
        G = np.array([[gtt, 0.0, gtphi],
                      [0.0, grr, 0.0],
                      [gtphi, 0.0, gphph]], dtype=np.float64)
        Ginv = np.linalg.inv(G)

        # derivatives wrt r (analytical)
        eps = 1e-16
        dSigma_dr = 2.0 * r
        dDelta_dr = 2.0 * r - 2.0 * M
        # ∂_r g_tt = -∂_r (1 - 2Mr/Σ) = -(-2M*Σ - (-2Mr)*dΣ_dr)/Σ^2  ... easier: compute numeric derivative
        # For simplicity and safety, use small-step numeric derivative on metric components
        def metric_r_deriv(rr):
            pm = self._metric_components_equatorial(rr)
            return np.array([pm["gtt"], pm["gtphi"], pm["grr"], pm["gphph"]], dtype=np.float64)
        h = 1e-6 * max(1.0, r)
        g_plus = metric_r_deriv(r + h)
        g_minus = metric_r_deriv(r - h)
        dg_dr = (g_plus - g_minus) / (2.0 * h)
        dg = {"gtt": dg_dr[0], "gtphi": dg_dr[1], "grr": dg_dr[2], "gphph": dg_dr[3]}

        # compute Christoffel using Γ^μ_{αβ} = 0.5 * g^{μν} ( ∂_α g_{νβ} + ∂_β g_{να} - ∂_ν g_{αβ} )
        # here only derivative wrt r is nonzero; so only combinations where α==r or β==r contribute.
        Gamma = {}
        # loop over mu, alpha, beta in {0,1,2}
        for mu in range(3):
            for alpha in range(3):
                for beta in range(3):
                    # only if alpha==1 or beta==1 or nu==1 etc -> do full sum over nu
                    val = 0.0
                    for nu in range(3):
                        # ∂_alpha g_{nu beta} term: nonzero only if alpha==1 (r)
                        term1 = dg[list(dg.keys())[nu]] if alpha == 1 else 0.0
                        # but mapping of dg keys to g_{nu beta} isn't direct; easier: build g_{nu beta} index map
                        # To avoid mistakes, compute using index access on small matrices and numerical derivative:
                        # We'll numerically compute ∂_alpha g_{nu beta} by finite diff in r, but only for alpha==r.
                        # Using that above dg array mapping: [gtt, gtphi, grr, gphph]
                        pass
        # The equatorial Christoffel calculation is a bit heavy to write out fully; instead
        # we'll compute Gamma numerically by finite-differencing the metric and using the
        # general formula (works fine for equatorial and is robust).
        # Build full 3x3 metric field as a function of r and numerically compute partials
        def metric_matrix(rr):
            pm = self._metric_components_equatorial(rr)
            return np.array([[pm["gtt"], 0.0, pm["gtphi"]],
                             [0.0, pm["grr"], 0.0],
                             [pm["gtphi"], 0.0, pm["gphph"]]], dtype=np.float64)
        # numeric partial wrt r
        h = 1e-6 * max(1.0, r)
        g_r_plus = metric_matrix(r + h)
        g_r_minus = metric_matrix(r - h)
        dgdr_mat = (g_r_plus - g_r_minus) / (2.0 * h)

        # now compute Gamma^mu_{alpha beta}
        for mu in range(3):
            for alpha in range(3):
                for beta in range(3):
                    s = 0.0
                    for nu in range(3):
                        s += Ginv[mu, nu] * ( (alpha == 1) * dgdr_mat[nu, beta] + (beta == 1) * dgdr_mat[nu, alpha] - dgdr_mat[alpha, beta] if isinstance(dgdr_mat, np.ndarray) else 0.0 )
                    Gamma[(mu, alpha, beta)] = 0.5 * s
        return Gamma

    def _state_to_cartesian(self, state: np.ndarray) -> np.ndarray:
        # state: [t, r, phi, ut, ur, uphi]
        r = state[1]
        phi = state[2]
        theta = math.pi / 2.0
        return cart_from_sph(r, theta, phi)

    def evolve(self, dt: float, max_t: float, tolerance: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        steps = int(max_t / dt)
        times = np.arange(steps, dtype=np.float64) * dt
        N = len(self.particles)
        positions: List[np.ndarray] = []

        for step_idx in range(steps):
            snapshot = np.zeros((N, 3), dtype=np.float64)
            for j, p in enumerate(self.particles):
                if not hasattr(p, "_kerr_state"):
                    # initialize from cartesian pos+vel (restricted to equatorial plane)
                    pos = p.position()
                    x, y, z = pos
                    r0, theta0, phi0 = sph_from_cart(pos)
                    theta0 = math.pi / 2.0  # force equatorial
                    # compute coordinate velocities dx^i/dt from cartesian velocities
                    vx, vy, vz = p.velocity()
                    # we project the cartesian velocity to equatorial coordinate velocities
                    denom = x * x + y * y + 1e-16
                    dr_dt = (x * vx + y * vy + z * vz) / (r0 + 1e-16)
                    dphi_dt = (x * vy - y * vx) / (denom + 1e-16)

                    vcoord = np.array([dr_dt, dphi_dt], dtype=np.float64)
                    # metric components (t,r,phi) at equatorial plane
                    m = self._metric_components_equatorial(r0)
                    gtt = m["gtt"]; gtphi = m["gtphi"]; grr = m["grr"]; gphph = m["gphph"]

                    # spatial metric contraction g_ij v^i v^j  where i,j in (r,phi)
                    gij_vv = grr * (vcoord[0] ** 2) + gphph * (vcoord[1] ** 2) + 2.0 * gtphi * (0.0 * vcoord[0] * vcoord[1])  # gtphi mixes t and phi only
                    denom_ut = gtt + 2.0 * gtphi * vcoord[1] + gij_vv
                    if denom_ut >= 0.0:
                        ut = 1.0 / math.sqrt(-gtt + 1e-8)
                    else:
                        ut = math.sqrt(-1.0 / (denom_ut + 1e-16))
                    ur = vcoord[0] * ut
                    uphi = vcoord[1] * ut
                    state = np.array([0.0, r0, phi0, ut, ur, uphi], dtype=np.float64)
                    p._kerr_state = state

                state = p._kerr_state.copy()

                # if already inside outer horizon, freeze at horizon
                if state[1] <= self.r_plus + tolerance:
                    rfreeze = self.r_plus
                    pos_cart = cart_from_sph(rfreeze, math.pi / 2.0, state[2])
                    snapshot[j, :] = pos_cart
                    p._kerr_state = np.array([state[0], rfreeze, state[2], 0.0, 0.0, 0.0], dtype=np.float64)
                    continue

                def deriv(s: np.ndarray) -> np.ndarray:
                    # s: [t, r, phi, ut, ur, uphi]
                    t, r, phi = s[0], s[1], s[2]
                    ut, ur, uphi = s[3], s[4], s[5]
                    u = np.array([ut, ur, uphi], dtype=np.float64)
                    # compute Gamma numerically for equatorial
                    Gamma = self._christoffel_equatorial(r)
                    dstatedtau = np.zeros(6, dtype=np.float64)
                    dstatedtau[0:3] = u  # tdot, rdot, phidot
                    # acceleration
                    acc = np.zeros(3, dtype=np.float64)
                    for a in range(3):
                        for b in range(3):
                            acc[0] -= Gamma.get((0, a, b), 0.0) * u[a] * u[b]
                            acc[1] -= Gamma.get((1, a, b), 0.0) * u[a] * u[b]
                            acc[2] -= Gamma.get((2, a, b), 0.0) * u[a] * u[b]
                    dstatedtau[3:6] = acc
                    return dstatedtau

                s1 = state
                k1 = deriv(s1)
                s2 = s1 + 0.5 * dt * k1
                k2 = deriv(s2)
                s3 = s1 + 0.5 * dt * k2
                k3 = deriv(s3)
                s4 = s1 + dt * k3
                k4 = deriv(s4)
                s_new = s1 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

                if s_new[1] <= self.r_plus + tolerance:
                    rfreeze = self.r_plus
                    pos_cart = cart_from_sph(rfreeze, math.pi / 2.0, s_new[2])
                    snapshot[j, :] = pos_cart
                    p._kerr_state = np.array([s_new[0], rfreeze, s_new[2], 0.0, 0.0, 0.0], dtype=np.float64)
                    continue

                p._kerr_state = s_new
                snapshot[j, :] = self._state_to_cartesian(s_new)

            positions.append(snapshot)

        return np.arange(steps, dtype=np.float64) * dt, np.stack(positions, axis=0)

