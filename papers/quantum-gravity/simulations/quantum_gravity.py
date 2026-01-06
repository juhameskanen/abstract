"""
Wavefunction-Based Gravity Simulation with Inertia
==================================================

This module implements a 2D simulation described in *An Informational Theory of Quantum-Gravity*.

Core Idea:
----------
The central hypothesis is that physical laws are emergent artifacts of
informational compression. Observers are modeled as compressed informational
structures: the more compressible an observer's trajectory, the more copies
of that observer exist in the underlying informational substrate, and the
higher the probability that such an observer will be realized. Since smooth,
predictable motion compresses best, observers overwhelmingly find themselves
in worlds with smooth, law-like physics.

Simulation Mapping:
-------------------
- **Random Noise Substrate**: Represented by resampled particle positions,
  drawn each step from a probability density formed by interference of all
  observer wavefunctions.

- **Observers as Compressed Structures**: Each observer is modeled as a
  parametric Gaussian-shell wavefunction (`Wavefunction`) plus an associated
  Gaussian filter (`ObserverWindow`).

- **Filtering**: Observer windows weight local particles, producing smooth
  soft-assignment of particles to observers.

- **Compression Principle (MDL)**: Candidate trajectories for each observer
  are evaluated with a *compressibility cost function* in phase-frequency
  space. Small changes in velocity are cheaper (more compressible) than large,
  erratic deviations.

- **Emergent Inertia**: By always selecting the lowest-cost (most compressible)
  candidate, observers naturally follow smooth, continuous paths. Inertia thus
  arises as an informationally optimal bias, not a fundamental law.

- **Emergent Quantum Mechanics**: Particles follow determinstic wavefunction, 
  QM natively emerges reflecting the complex wavefunction compression.

Usage:
------
Run directly from the command line:

    python quantum_gravity.py --particles 8192 --steps 400 --sigma 0.12

This produces a dynamic visualization where initially random observers evolve
into smooth, inertial trajectories under the informational compression rule.

Relation to Theory:
-------------------
This code serves as a proof-of-concept for the paper's claim that inertia and
predictable physical laws emerge from the dominance of compressed observer
histories. The simulation operationalizes the **Compression-Existence
Principle**: compression → multiplicity → probability → predictability.
"""

import numpy as np
from typing import List, Tuple
import argparse
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from inertia_from_compression import GravitySim



class Wavefunction:
    """Parametric (2D) wavefunction: soft disk / Gaussian shell + phase/time dependence."""
    def __init__(
        self,
        center: np.ndarray,
        radius: float = 0.08,
        sigma: float = 0.03,
        amplitude: float = 1.0,
        omega: float = 0.0,
        phase: float = 0.0,
    ):
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        self.sigma = float(sigma)
        self.amplitude = float(amplitude)
        self.omega = float(omega)
        self.phase = float(phase)

    def evaluate(self, points: np.ndarray, t: float) -> np.ndarray:
        """
        Evaluate complex psi at given points (N x 2). Returns complex array length N.
        Default shape: Gaussian shell around radius.
        """
        # radial distance
        r = np.linalg.norm(points - self.center, axis=1)
        # gaussian shell amplitude envelope
        env = np.exp(-((r - self.radius) ** 2) / (2 * self.sigma ** 2))
        # optional smoothstep version commented; env is smooth enough
        psi = self.amplitude * env * np.exp(-1j * (self.omega * t - self.phase))
        return psi

    def copy(self):
        return Wavefunction(self.center.copy(), self.radius, self.sigma, self.amplitude, self.omega, self.phase)


class ObserverWindow:
    """A Gaussian 'observer' filter centered at same center as wavefunction (shape-preserving)."""
    def __init__(self, center: np.ndarray, sigma: float):
        self.center = np.array(center, dtype=float)
        self.sigma = float(sigma)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        r2 = np.sum((points - self.center) ** 2, axis=1)
        return np.exp(-0.5 * r2 / (self.sigma ** 2))

    def move_to(self, new_center: np.ndarray):
        self.center = np.array(new_center, dtype=float)


# --- Wavefunction-based GravitySim subclass ---

 
class WavefunctionGravitySim(GravitySim):

    def __init__(
        self,
        n_particles: int = 4096,
        n_steps: int = 200,
        blob_sigma: float = 0.09,
        n_candidates: int = 16,
        initial_speed: float = 0.01,
        eta_center: float = 0.15,
        keep_width: bool = True,
    ) -> None:
        """
        eta_center: small learning rate when updating centers from assigned particles
        keep_width: if True, radius/sigma are kept fixed to preserve shape
        """
        super().__init__(n_particles=n_particles, n_steps=n_steps, blob_sigma=blob_sigma)
        self.n_candidates = n_candidates
        self.initial_speed = initial_speed
        self.eta_center = eta_center
        self.keep_width = keep_width

        # Time for wavefunction time-evolution (keeps the same pace as steps)
        self.t = 0.0
        self.dt = 1.0  # step in "time" per simulation iteration (can be adjusted)

        # Create parametric wavefunctions initialized from base positions
        self.wavefunctions: List[Wavefunction] = []
        for pos in self.positions:
            # radius ~ blob_sigma scale (tweakable)
            wf = Wavefunction(center=pos.copy(), radius=0.06, sigma=max(1e-3, self.blob_sigma * 0.8),
                              amplitude=1.0, omega=0.0, phase=0.0)
            self.wavefunctions.append(wf)

        # Observer windows (Gaussian filters) used for soft assignment and shape preservation
        self.observers: List[ObserverWindow] = [ObserverWindow(wf.center.copy(), self.blob_sigma) for wf in self.wavefunctions]

        # Wavefunction memory: store list of (phase,freq) per blob (for inertia)
        self.wavefunction_memory: List[List[Tuple[float, float]]] = [[] for _ in range(len(self.wavefunctions))]
        # Initialize memory with tiny tangential velocities (phase/freq pairs)
        self.initialize_memory()

    # ----- memory helpers (phase/freq) -----

    def velocity_to_phasefreq(self, velocity: np.ndarray) -> Tuple[float, float]:
        angle = np.arctan2(velocity[1], velocity[0])
        magnitude = float(np.linalg.norm(velocity))
        return float(angle), float(magnitude)

    def phasefreq_to_velocity(self, phase: float, freq: float) -> np.ndarray:
        return np.array([freq * np.cos(phase), freq * np.sin(phase)], dtype=float)

    def record_velocity(self, blob_idx: int, velocity: np.ndarray) -> None:
        """Append (phase,freq) to memory (keeps memory growth)."""
        phase, freq = self.velocity_to_phasefreq(velocity)
        self.wavefunction_memory[blob_idx].append((phase, freq))

    def get_velocity(self, blob_idx: int) -> np.ndarray:
        mem = self.wavefunction_memory[blob_idx]
        if not mem:
            # default small tangential-like kick
            return np.random.normal(scale=self.initial_speed, size=2)
        phase, freq = mem[-1]
        return self.phasefreq_to_velocity(phase, freq)

    def initialize_memory(self):
        """Seed memory with small tangential velocities relative to COM."""
        com = self.positions.mean(axis=0)
        for i, wf in enumerate(self.wavefunctions):
            r_rel = wf.center - com
            if np.linalg.norm(r_rel) < 1e-8:
                v0 = np.random.normal(scale=self.initial_speed, size=2)
            else:
                v0 = self.initial_speed * np.array([-r_rel[1], r_rel[0]]) / np.linalg.norm(r_rel)
            self.record_velocity(i, v0)

    # ---- PDF / sampling ----

    def compute_pdf(self, grid_points: np.ndarray) -> np.ndarray:
        """
        Override base method: compute pdf from sum of complex wavefunctions.
        Returns normalized pdf (sums to 1).
        """
        psi_total = np.zeros(grid_points.shape[0], dtype=np.complex128)
        for wf in self.wavefunctions:
            psi_total += wf.evaluate(grid_points, self.t)
        pdf = np.abs(psi_total) ** 2
        # small floor then normalize to avoid degenerate zero arrays
        pdf += 1e-20
        pdf /= pdf.sum()
        return pdf

    # ---- soft assignment (observer windows) ----

    def soft_assign_weights(self, particle_positions: np.ndarray) -> np.ndarray:
        """
        Compute soft assignment weights W: shape (n_blobs, n_particles)
        w_i(x) ~ O_i(x) * |psi_i(x)|^2
        normalized per particle so sum_i w_i = 1.
        """
        n = particle_positions.shape[0]
        m = len(self.wavefunctions)
        W = np.zeros((m, n), dtype=float)

        # Evaluate |psi_i|^2 and observer at particle locations
        for i, (wf, obs) in enumerate(zip(self.wavefunctions, self.observers)):
            psi_i = wf.evaluate(particle_positions, self.t)
            obs_vals = obs.evaluate(particle_positions)
            # weight proportional to observer window times local |psi|^2
            W[i, :] = obs_vals * (np.abs(psi_i) ** 2)

        # normalize per particle
        s = W.sum(axis=0, keepdims=True)
        s[s <= 0] = 1.0  # avoid division by zero
        W /= s
        return W

    # ---- update / M-step (cheap, stable) ----

    def update_wavefunction_centers(self, particles: np.ndarray, W: np.ndarray) -> None:
        """
        Update each wavefunction's center with a small step toward the weighted mean
        of the particles assigned by soft weights W (shape: n_blobs x n_particles).
        """
        n_blobs = len(self.wavefunctions)
        # avoid moving too far in one iteration -> inertia-like behavior
        for i in range(n_blobs):
            weights = W[i]
            tot = weights.sum()
            if tot <= 0:
                continue
            mean = (weights[:, None] * particles).sum(axis=0) / tot
            # small step toward mean
            new_center = (1.0 - self.eta_center) * self.wavefunctions[i].center + self.eta_center * mean
            self.wavefunctions[i].center = new_center
            # also move observer window center to follow
            self.observers[i].move_to(new_center)

    # ---- candidate-selection & phase/velocity memory update (inertia) ----

    def wavefunction_cost(self, blob_idx: int, candidate_velocity: np.ndarray) -> float:
        """
        Simple compressibility cost in phase/freq space (small change preferred).
        Lower cost -> more compressible.
        """
        mem = self.wavefunction_memory[blob_idx]
        if not mem:
            return 0.0
        last_phase, last_freq = mem[-1]
        cand_phase, cand_freq = self.velocity_to_phasefreq(candidate_velocity)
        # angular wrap
        d_phase = np.mod(cand_phase - last_phase + np.pi, 2 * np.pi) - np.pi
        d_freq = cand_freq - last_freq
        return d_phase * d_phase + d_freq * d_freq

    def select_candidate_and_record(self, blob_idx: int, candidates: List[Tuple[np.ndarray, np.ndarray]], current_pos: np.ndarray) -> np.ndarray:
        """
        Given candidates (position, velocity), choose best by wavefunction_cost
        and record the chosen velocity into memory. Return chosen position.
        """
        best_cost = float("inf")
        best_c = None
        best_v = None
        for c, v in candidates:
            cost = self.wavefunction_cost(blob_idx, v)
            if cost < best_cost:
                best_cost, best_c, best_v = cost, c, v
        if best_v is None:
            best_v = np.random.normal(scale=1e-3, size=2)
            best_c = current_pos + best_v
        # record chosen velocity into memory
        self.record_velocity(blob_idx, best_v)
        return best_c

    # ---- high-level update used inside run() (keeps the candidate loop) ----

    def update_positions(self, new_particles: np.ndarray) -> None:
        """
        High-level update for each blob:
         - we use assign_particles_to_blob() to detect local particles (same as MDL).
         - compute soft-weights W on the whole particle set to update centers (EM-like).
         - then, for each blob, generate candidates around the local target and use
           wavefunction memory to pick the most compressible candidate; move center to it.
        """
        # Soft-assign using entire particle set (cheap)
        W = self.soft_assign_weights(new_particles)

        # Update centers with small step (shape-preserving)
        self.update_wavefunction_centers(new_particles, W)

        # Now for each blob, perform candidate selection around its local target (same logic as MDL)
        new_positions = []
        for i, wf in enumerate(self.wavefunctions):
            pos = wf.center.copy()
            # local assigned using same rule as before (for compatibility)
            assigned = self.assign_particles_to_blob(i)
            if len(assigned) > 0:
                target_pos = assigned.mean(axis=0)
                # use base-class generator if available; fallback to simple sampling
                try:
                    candidates = self.generate_candidates_for_blob(i, target_pos, velocity=self.get_velocity(i))
                except Exception:
                    # fallback: generate a small set of candidates around target
                    candidates = []
                    noise = max(1e-6, self.blob_sigma * 0.5)
                    for _ in range(self.n_candidates):
                        c = target_pos + np.random.normal(scale=noise, size=2)
                        v = c - pos
                        if np.linalg.norm(v) > 1e-9:
                            candidates.append((c, v))
                    if not candidates:
                        v = np.random.normal(scale=1e-3, size=2)
                        candidates.append((pos + v, v))

                choice = self.select_candidate_and_record(i, candidates, pos)

                # Move the parametric wavefunction center directly to chosen candidate
                self.wavefunctions[i].center = np.array(choice, dtype=float)
                self.observers[i].move_to(choice)
                new_positions.append(np.array(choice, dtype=float))
            else:
                # if no assigned particles, do a small inertial step from memory
                v_prev = self.get_velocity(i)
                new_center = pos + v_prev
                self.wavefunctions[i].center = new_center
                self.observers[i].move_to(new_center)
                self.record_velocity(i, v_prev)
                new_positions.append(new_center)

        # For compatibility with base class expectations:
        # update self.positions to the current centers (so plotting/statistics works)
        self.positions = np.array([wf.center.copy() for wf in self.wavefunctions])

        # advance internal time
        self.t += self.dt

    # We might optionally expose method to tweak wavefunction params, e.g. change radius or omega
    def tweak_wavefunction_params(self, blob_idx: int, **kwargs):
        wf = self.wavefunctions[blob_idx]
        for k, v in kwargs.items():
            if hasattr(wf, k):
                setattr(wf, k, v)

    
def main() -> None:
    parser = argparse.ArgumentParser(description="Probabilistic Gravity Simulation with MDL Inertia")
    parser.add_argument("--sigma", type=float, default=0.12, help="Blob sigma")
    parser.add_argument("--particles", type=int, default=8192, help="Number of particles")
    parser.add_argument("--steps", type=int, default=200, help="Number of steps")
    args = parser.parse_args()

    sim = WavefunctionGravitySim(
        n_particles=args.particles,
        n_steps=args.steps,
        blob_sigma=args.sigma
    )
    sim.run()


if __name__ == "__main__":
    main()
