# fixed_simulation.py
from __future__ import annotations
from typing import List, Tuple
import math
import numpy as np
from blackhole import BlackHole, DustCloud, sph_from_cart, cart_from_sph


class KerrIEFEquatorialGeodesicCloud(DustCloud):
    """
    Kerr geodesics in ingoing Eddington–Finkelstein coordinates (v, r, phitilde),
    restricted to the equatorial plane (theta = pi/2).
    State: [v, r, phitilde, uv, ur, uphi].
    This is regular at the horizon and integrates smoothly through r = r_+.
    """
    def __init__(self, n: int, r0: float, spacing: float, bh: BlackHole,
                 tangential_fraction: float = 0.8, radial_fraction: float = 0.15,
                 rng_seed: int = 42, r_min_factor: float = 1e-4):
        super().__init__(n, r0, spacing, bh, tangential_fraction, radial_fraction)
        self.bh = bh
        self.M = float(bh.mass)
        self.a = float(bh.spin)
        self.r_plus = float(self.M + math.sqrt(max(0.0, self.M * self.M - self.a * self.a)))
        # Stop before r=0 to avoid division-by-zero; e.g., r_min ~ 1e-4 M
        self.r_min = max(1e-12, r_min_factor * self.M)

    # ---------- Metric (IEF, equatorial) ----------
    def _metric_components_equatorial(self, r: float) -> dict:
        M = self.M
        a = self.a
        # Equatorial: Sigma = r^2, sin^2 = 1
        Sigma = r * r
        one_minus_2M_over_r = 1.0 - (2.0 * M) / (r + 1e-16)

        gvv   = -one_minus_2M_over_r
        gvr   = 1.0
        gvphi = -a * one_minus_2M_over_r
        grphi = -a
        gphph = (r * r + a * a) + (2.0 * M * a * a) / (r + 1e-16)

        # 3x3 metric in (v, r, phi_tilde)
        return {
            "gvv": gvv, "gvr": gvr, "gvphi": gvphi,
            "grphi": grphi, "gphph": gphph
        }

    def _metric_matrix(self, r: float) -> np.ndarray:
        m = self._metric_components_equatorial(r)
        # indices 0=v,1=r,2=phi
        G = np.array([
            [m["gvv"],  m["gvr"],   m["gvphi"]],
            [m["gvr"],  0.0,        m["grphi"]],
            [m["gvphi"], m["grphi"], m["gphph"]],
        ], dtype=np.float64)
        return G

    def _christoffel_equatorial(self, r: float) -> dict:
        # Finite-difference only wrt r since equatorial metric depends on r alone
        G = self._metric_matrix(r)
        Ginv = np.linalg.inv(G)
        h = 1e-6 * max(1.0, abs(r))
        G_plus = self._metric_matrix(r + h)
        G_minus = self._metric_matrix(r - h)
        dgdr = (G_plus - G_minus) / (2.0 * h)  # ∂_r g_{μν}

        # Γ^μ_{αβ} = 0.5 g^{μν} (∂_α g_{νβ} + ∂_β g_{να} - ∂_ν g_{αβ})
        # Only ∂_r is nonzero ⇒ ∂_α nonzero iff α==1
        Gamma = {}
        for mu in range(3):
            for alpha in range(3):
                for beta in range(3):
                    s = 0.0
                    for nu in range(3):
                        t1 = dgdr[nu, beta] if alpha == 1 else 0.0
                        t2 = dgdr[nu, alpha] if beta == 1 else 0.0
                        t3 = dgdr[alpha, beta] if nu == 1 else 0.0
                        s += Ginv[mu, nu] * (t1 + t2 - t3)
                    Gamma[(mu, alpha, beta)] = 0.5 * s
        return Gamma

    def _state_to_cartesian(self, state: np.ndarray) -> np.ndarray:
        # state: [v, r, phi, uv, ur, uphi]
        r = max(state[1], self.r_min)
        phi = state[2]
        theta = math.pi / 2.0
        return cart_from_sph(r, theta, phi)

    # ---------- Initialization helpers ----------
    def _normalize_u(self, r: float, u: np.ndarray, make_null: bool = False) -> np.ndarray:
        """
        Enforce g_{μν} u^μ u^ν = -1 for timelike (or 0 for null if make_null=True)
        by solving for u^v from a quadratic. Keep u^r and u^phi as provided.
        """
        G = self._metric_matrix(r)
        uv, ur, uphi = u
        # We solve for uv to satisfy norm = target
        target = 0.0 if make_null else -1.0

        # Quadratic in uv: A uv^2 + B uv + C = target
        # A = g_vv
        A = G[0,0]
        # B = 2 (g_vr ur + g_vφ uphi)
        B = 2.0 * (G[0,1] * ur + G[0,2] * uphi)
        # C = g_rr ur^2 + 2 g_rφ ur uphi + g_φφ uphi^2
        C = (G[1,1] * ur * ur) + (2.0 * G[1,2] * ur * uphi) + (G[2,2] * uphi * uphi)

        # Solve A uv^2 + B uv + (C - target)=0
        aQ = A
        bQ = B
        cQ = C - target

        # Handle near-zero A gracefully
        if abs(aQ) < 1e-14:
            # Linear: b uv + c = 0
            uv = -cQ / (bQ + 1e-16)
            return np.array([uv, ur, uphi], dtype=np.float64)

        disc = bQ*bQ - 4.0*aQ*cQ
        if disc < 0.0:
            # Clamp to zero to keep real; this happens only under extreme rounding
            disc = 0.0
        sqrt_disc = math.sqrt(disc)
        # Choose the physically forward-in-time root (uv positive for ingoing EF)
        uv1 = (-bQ + sqrt_disc) / (2.0 * aQ)
        uv2 = (-bQ - sqrt_disc) / (2.0 * aQ)
        uv = uv1 if uv1 > uv2 else uv2
        return np.array([uv, ur, uphi], dtype=np.float64)


    def evolve(self, dt: float, max_t: float, tolerance: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Horizon-penetrating integration with local adaptivity and runaway termination.
        Returns (times, positions[steps, N, 3]) in Cartesian coords.
        """
        steps = int(max_t / dt)
        times = np.arange(steps, dtype=np.float64) * dt
        N = len(self.particles)

        # initialize states if needed (same as before, then normalized)
        for p in self.particles:
            if not hasattr(p, "_kerr_state"):
                x, y, z = p.position()
                r0, _, phi0 = sph_from_cart(np.array([x, y, z], dtype=np.float64))
                vx, vy, vz = p.velocity()
                denom = x * x + y * y + 1e-16
                dr_dt = (x * vx + y * vy + z * vz) / (r0 + 1e-16)
                dphi_dt = (x * vy - y * vx) / (denom + 1e-16)
                u = np.array([1.0, dr_dt, dphi_dt], dtype=np.float64)
                u = self._normalize_u(r0, u, make_null=False)
                p._kerr_state = np.array([0.0, r0, phi0, u[0], u[1], u[2]], dtype=np.float64)

        # helpers
        U_MAX = 1e6   # cap for |u| to declare runaway
        H_MAX_HALVE = 8

        def deriv(s: np.ndarray) -> np.ndarray:
            # s: [v, r, phi, uv, ur, uphi]
            r = max(s[1], self.r_min)
            u = s[3:6]
            ds = np.zeros(6, dtype=np.float64)
            ds[0:3] = u
            try:
                Gamma = self._christoffel_equatorial(r)
            except Exception:
                # if metric inversion/FD fails, return huge accel to force termination
                ds[3:6] = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                return ds
            acc = np.zeros(3, dtype=np.float64)
            # u[a]*u[b] can overflow; compute with checks
            for a in range(3):
                ua = u[a]
                if not np.isfinite(ua):
                    acc[:] = np.nan
                    break
                for b in range(3):
                    ub = u[b]
                    if not np.isfinite(ub):
                        acc[:] = np.nan
                        break
                    fac = ua * ub
                    if not np.isfinite(fac):
                        acc[:] = np.nan
                        break
                    acc[0] -= Gamma.get((0, a, b), 0.0) * fac
                    acc[1] -= Gamma.get((1, a, b), 0.0) * fac
                    acc[2] -= Gamma.get((2, a, b), 0.0) * fac
            ds[3:6] = acc
            return ds

        positions: list[np.ndarray] = []
        alive = np.ones(N, dtype=bool)
        last_pos = np.zeros((N, 3), dtype=np.float64)

        for step_idx in range(steps):
            snapshot = np.zeros((N, 3), dtype=np.float64)

            for j, p in enumerate(self.particles):
                s1 = p._kerr_state.copy()
                # already terminated?
                if not alive[j]:
                    snapshot[j, :] = last_pos[j]
                    continue

                # termination conditions
                r_now = s1[1]
                u_norm = np.linalg.norm(s1[3:6])
                if (r_now <= self.r_min) or (not np.isfinite(s1).all()) or (u_norm > U_MAX):
                    alive[j] = False
                    rc = max(r_now, self.r_min)
                    last_pos[j] = cart_from_sph(rc, math.pi / 2.0, s1[2])
                    snapshot[j, :] = last_pos[j]
                    continue

                # adaptive RK4 with step halving
                local_dt = dt
                accepted = False
                s_new = s1
                for _ in range(H_MAX_HALVE):
                    k1 = deriv(s1)
                    s2 = s1 + 0.5 * local_dt * k1
                    k2 = deriv(s2)
                    s3 = s1 + 0.5 * local_dt * k2
                    k3 = deriv(s3)
                    s4 = s1 + local_dt * k3
                    k4 = deriv(s4)
                    s_try = s1 + (local_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

                    # check viability
                    if np.isfinite(s_try).all() and (s_try[1] > 0.0):
                        accepted = True
                        s_new = s_try
                        break
                    local_dt *= 0.5

                if not accepted:
                    alive[j] = False
                    rc = max(s1[1], self.r_min)
                    last_pos[j] = cart_from_sph(rc, math.pi / 2.0, s1[2])
                    snapshot[j, :] = last_pos[j]
                    continue

                # re-normalize u to control drift
                s_new[3:6] = self._normalize_u(max(s_new[1], self.r_min), s_new[3:6], make_null=False)

                # clamp absurd growth
                if np.linalg.norm(s_new[3:6]) > U_MAX or not np.isfinite(s_new[3:6]).all():
                    alive[j] = False
                    rc = max(s_new[1], self.r_min)
                    last_pos[j] = cart_from_sph(rc, math.pi / 2.0, s_new[2])
                    snapshot[j, :] = last_pos[j]
                    continue

                # commit
                p._kerr_state = s_new
                pos_cart = self._state_to_cartesian(s_new)
                # sanitize just in case
                pos_cart = np.nan_to_num(pos_cart, nan=0.0, posinf=0.0, neginf=0.0)
                last_pos[j] = pos_cart
                snapshot[j, :] = pos_cart

            positions.append(snapshot)

        return times, np.stack(positions, axis=0)

