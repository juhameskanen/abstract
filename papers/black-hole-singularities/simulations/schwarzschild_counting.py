"""
Schwarzschild Counting-Equation Black Hole
============================================

r IS the entropy coordinate -- not a separate geometric variable layered
on top of a GR metric:

    r = 0      <->  tau = 0        zero-entropy state. Zero information
                                    cannot describe any microstructure, so
                                    there is nothing there to source a
                                    stress-energy tensor -- no singularity
                                    in the usual GR sense, just the
                                    complete absence of structure.
    r = r_h    <->  tau = tau_max  max-entropy state: full equilibrium,
                                    the event horizon. This boundary is
                                    fuzzy (the system saturates and then
                                    fluctuates), not sharp.

tau(r) = tau_max * (r / r_h) is a straight linear map, run through the
exact same Ehrenfest relaxation engine used for the cosmological model
(entropy_engine.py, factored out so both simulations share one source of
truth) -- no separately invented radial profile shape.

n_bits(M) is set by Bekenstein-Hawking horizon entropy (area law, a=q=0,
Planck units, G=c=1): r_h = 2M, A = 4*pi*r_h^2, n_bits = A / (4 ln 2).
The proportionality constant is exposed as `kappa` and should be treated
as a placeholder calibration -- same caveat as the cosmological n~184.

m(r) is the LOCAL matter/structure content at r -- the order-parameter-
weighted "bump" quantity (matter_content() in the cosmological script),
zero at r=0, peaking at some interior radius, small-but-nonzero
approaching r_h. Acceleration is sourced by d(m)/dr directly, which is
NOT monotonic:

    dm/dr > 0 near r=0      -> outward (repulsive core). A particle that
                                emerges on the singularity side of the
                                matter peak sees an entropy gradient
                                pointing outward -- structure has nowhere
                                to go but out.
    dm/dr < 0 past the peak -> inward (attractive), pulling back toward
                                the matter peak.

This gives every purely radial trajectory a genuine turning-point
structure with NO angular momentum required -- unlike vacuum Schwarzschild,
where a purely radial geodesic has no effective potential barrier at all
and always plunges to r=0. The barrier here comes from the informational
structure of the source itself, not from a centrifugal term.

INTEGRATOR NOTE (important, found the hard way): a fixed-timestep RK4
integrator can silently inject large amounts of spurious energy near the
steep part of the dm/dr curve close to r=0, making trajectories look like
they blow straight through the horizon when in fact, at proper resolution,
they turn around well inside it. This version uses scipy's adaptive
RK45 (`solve_ivp`), which automatically refines the internal step size in
steep regions regardless of the requested output spacing `dt` -- `dt` is
just the output cadence now, not the integration step, and `tolerance`
actually does something (it's the solver's error tolerance).

Open problem, not resolved here: exterior (r >> r_h) stitching to the
already-derived far-field 1/r^2 law (flux conservation over 4*pi*r^2
shells). tau(r) is simply extended linearly past r_h with the same slope
for now; matter keeps decaying there since eta(tau) keeps falling, but
this has not been checked against that result.


Copyright 2001 ... 2026 - Juha Meskanen
The Abstract Universe Project
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.integrate import solve_ivp

from blackhole import BlackHole, DustCloud
from entropy_engine import build_scale_hierarchy, matter_content, matter_diffusion_rate, default_k_rate


class SchwarzschildCounting(DustCloud):
    """
    Dust cloud collapse sourced entirely by the entropy-relaxation counting
    equation. r is the entropy coordinate; matter density m(r) is the
    reused structure-emergence curve from the cosmological model;
    acceleration is d(m)/dr, giving a repulsive core near r=0 for free --
    no separate "inside vs outside" branch, no invented interior shape
    function, no leftover GR geodesic integrator.

    Note: named/documented for the non-rotating (a=0) case. m(r) only
    depends on M and r_h, so spin/charge currently affect only where the
    horizon sits (via r_h), not the shape of the fabric profile or the
    dynamics. Extending the counting equation to genuinely rotating/
    charged fabric profiles is future work.
    """

    def __init__(
        self,
        n: int,
        r0: float,
        spacing: float,
        bh: BlackHole,
        tangential_fraction: float = 0.8,
        radial_fraction: float = 0.15,
        rng_seed: int = 42,
        scales: Tuple[float, ...] = (6.0, 12.0, 20.0),
        matter_power: float = 1.0,
        kappa: float = 4 * np.pi / np.log(2.0),  # Bekenstein-Hawking normalization, placeholder constant
        sat_fraction: float = 0.99,
        r_grid_max_factor: float = 3.0,
        r_grid_points: int = 4000,
    ) -> None:
        self.M = float(bh.mass)
        self.a = float(getattr(bh, "spin", 0.0))
        self.q = float(getattr(bh, "charge", 0.0))

        discriminant = self.M**2 - self.a**2 - self.q**2
        self.r_h = self.M + np.sqrt(discriminant) if discriminant >= 0 else 0.0

        self.n_bits = max(int(round(kappa * self.M**2)), 8)
        self.scales = list(scales)
        self.matter_power = matter_power

        # Same Ehrenfest mixing-time convention as the cosmological run
        self.t_bf_max = self.n_bits * np.log(self.n_bits)
        self.k_rate = default_k_rate(self.t_bf_max, sat_fraction)

        # Precompute m(r), dm/dr, D(r) on a grid once (hypergeometric
        # evaluation is too expensive to redo live inside the ODE RHS).
        r_max = r_grid_max_factor * max(self.r_h, 1.0)
        self._r_grid = np.linspace(1e-6, r_max, r_grid_points)
        tau_grid = (
            self.t_bf_max * (self._r_grid / self.r_h) if self.r_h > 0 else np.zeros_like(self._r_grid)
        )
        levels = build_scale_hierarchy(tau_grid, self.scales, self.n_bits, self.k_rate)
        total_matter_bits, *_ = matter_content(levels, tau_grid, self.k_rate, self.matter_power)
        self._m_grid = total_matter_bits
        self._dm_dr_grid = np.gradient(self._m_grid, self._r_grid)

        D_tau_grid = matter_diffusion_rate(levels, tau_grid, self.k_rate, self.matter_power, self.n_bits)
        dtau_dr = self.t_bf_max / self.r_h if self.r_h > 0 else 0.0
        self._D_grid = D_tau_grid * dtau_dr

        self.r_peak = float(self._r_grid[np.argmax(self._m_grid)])

        super().__init__(n, r0, spacing, bh, tangential_fraction, radial_fraction, rng_seed)

    # ------------------------------------------------------------------
    # Lookups (grid built once at construction, interpolated afterwards)
    # ------------------------------------------------------------------

    def m_of_r(self, r) -> np.ndarray:
        """Local matter/structure content at radius r (interpolated)."""
        return np.interp(r, self._r_grid, self._m_grid)

    def dm_dr_of_r(self, r) -> np.ndarray:
        """d(matter)/dr at radius r (interpolated from the precomputed grid)."""
        return np.interp(r, self._r_grid, self._dm_dr_grid)

    def D_of_r(self, r) -> np.ndarray:
        """Radial diffusion coefficient at r (interpolated), see matter_diffusion_rate()."""
        return np.interp(r, self._r_grid, self._D_grid)

    def calculate_rho_fabric(self, r) -> Tuple[np.ndarray, np.ndarray]:
        """rho_fabric = n_bits - m(r). Kept for interface/name continuity with earlier versions."""
        m_r = self.m_of_r(r)
        dm_dr = self.dm_dr_of_r(r)
        return self.n_bits - m_r, -dm_dr

    def acceleration(self, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """
        a = d(m)/dr * r_hat -- positive dm/dr (matter still emerging, near
        the zero-entropy core) pushes outward; negative dm/dr (past the
        matter peak, entropy saturating toward the horizon) pulls inward.
        vel is unused: no velocity-coupled term in this model yet.
        Single-particle form, kept for interface compatibility; evolve()
        uses the vectorized form internally for speed.
        """
        x, y, z = pos[0], pos[1], pos[2]
        r = np.sqrt(x * x + y * y + z * z) + 1e-9
        r_hat = pos / r
        return self.dm_dr_of_r(r) * r_hat

    def _acceleration_batch(self, pos: np.ndarray) -> np.ndarray:
        """Vectorized acceleration for an (N,3) position array -> (N,3)."""
        r = np.linalg.norm(pos, axis=1)
        r_safe = np.maximum(r, 1e-9)
        r_hat = pos / r_safe[:, None]
        return self.dm_dr_of_r(r_safe)[:, None] * r_hat

    # ------------------------------------------------------------------
    # Deterministic evolution -- adaptive RK45, no fixed-dt artifact risk
    # ------------------------------------------------------------------

    def evolve(self, dt: float, max_t: float, tolerance: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the dust cloud under the counting-equation acceleration
        using adaptive RK45 (scipy solve_ivp). `dt` is the OUTPUT time
        spacing (what you'll see in the returned trajectory), not the
        integration step -- the solver internally takes whatever step
        size `tolerance` (rtol/atol) demands, shrinking automatically
        near the steep part of dm/dr close to r=0. This is what actually
        fixes the earlier fixed-dt energy-injection artifact, rather than
        just picking a smaller dt by hand.
        """
        N = len(self.particles)
        y0 = np.concatenate([np.concatenate([p.position(), p.velocity()]) for p in self.particles])

        def rhs(t, y):
            y2 = y.reshape(N, 6)
            pos = y2[:, 0:3]
            vel = y2[:, 3:6]
            acc = self._acceleration_batch(pos)
            dydt = np.empty_like(y2)
            dydt[:, 0:3] = vel
            dydt[:, 3:6] = acc
            return dydt.reshape(-1)

        t_eval = np.arange(0.0, max_t, dt)
        sol = solve_ivp(
            rhs, (0.0, max_t), y0, method="RK45",
            t_eval=t_eval, rtol=tolerance, atol=tolerance * 1e-3,
            max_step=dt,
        )

        steps = len(t_eval)
        y_out = sol.y.reshape(N, 6, steps)
        positions = np.transpose(y_out[:, 0:3, :], (2, 0, 1))  # (steps, N, 3)
        return t_eval, positions

    # ------------------------------------------------------------------
    # Optional: stochastic (Langevin) evolution, radial noise from D(r).
    # Uses a fixed substep dt internally -- NOT yet given the same
    # adaptive-step treatment as evolve() above, so escape-rate /
    # M-scaling conclusions from this method should still be treated as
    # open questions, not settled results, until that's fixed too.
    # ------------------------------------------------------------------

    def evolve_langevin(
        self,
        n_particles: int,
        dt: float,
        max_t: float,
        escape_radius_factor: float = 1.5,
        start_r: float | None = None,
        seed: int = 0,
    ):
        rng = np.random.default_rng(seed)
        if start_r is None:
            start_r = self.r_peak

        steps = int(max_t / dt)
        times = np.arange(steps, dtype=np.float64) * dt

        r = np.full(n_particles, start_r, dtype=np.float64)
        v_r = np.zeros(n_particles, dtype=np.float64)

        r_traj = np.zeros((steps, n_particles), dtype=np.float64)
        escape_times = np.full(n_particles, np.nan, dtype=np.float64)
        escape_r = escape_radius_factor * self.r_h
        escaped_mask = np.zeros(n_particles, dtype=bool)

        for step_idx in range(steps):
            a_r = self.dm_dr_of_r(r)
            D_r = np.clip(self.D_of_r(r), 0.0, None)
            noise = np.sqrt(2.0 * D_r * dt) * rng.standard_normal(n_particles)

            v_r = v_r + a_r * dt + noise
            r = r + v_r * dt
            r = np.clip(r, 1e-6, None)

            r_traj[step_idx, :] = r

            newly_escaped = (~escaped_mask) & (r >= escape_r)
            escape_times[newly_escaped] = times[step_idx]
            escaped_mask |= newly_escaped

        return times, r_traj, escape_times
