import numpy as np
import matplotlib.pyplot as plt
import math
import time
from typing import List, Optional, Tuple
from scipy.integrate import solve_ivp

initial_radius: float = 6.0

class EddingtonFinkelsteinParticle:
    def __init__(
        self,
        r0: float,
        M: float,
        max_tau: float,
        max_step: float,
        tolerance: float,
    ) -> None:
        self.M = M
        self.r0 = r0
        self.max_tau = max_tau
        self.max_step = max_step
        self.tolerance = tolerance
        self.trajectory: List[Tuple[float, float]] = []  # (tau, r)
        self.integrate_geodesic()

    def geodesic_rhs(self, tau: float, y: List[float]) -> List[float]:
        r, dr_dtau = y
        d2r_dtau2 = -self.M / r**2  # correct radial acceleration in proper time
        return [dr_dtau, d2r_dtau2]
   

    def integrate_geodesic(self) -> None:
        """
        Integrates a single geodesic in the Schwarzschild spacetime.
        Particles freeze at r <= tolerance but the simulation continues.
        """

        def rhs(tau: float, y: np.ndarray) -> np.ndarray:
            r, dr_dtau = y
            if r <= self.tolerance:
                # Freeze particle at singularity
                return np.array([0.0, 0.0])
            return self.geodesic_rhs(tau, y)

        # Initial state: r0, radial velocity
        y0 = np.array([self.r0, -np.sqrt(2 * self.M / self.r0)])

        # Solve ODE
        result = solve_ivp(
            rhs,
            [0, self.max_tau],
            y0,
            method="RK45",
            max_step=self.max_step,
            rtol=1e-4,
            atol=1e-6
        )

        # Store trajectory
        self.trajectory = list(zip(result.t, result.y[0]))


    def integrate_geodesic_old(self) -> None:
        def stop_near_singularity(tau: float, y: List[float]) -> float:
            r, _ = y
            return r - self.tolerance

        stop_near_singularity.terminal = True
        stop_near_singularity.direction = -1


        result = solve_ivp(
            self.geodesic_rhs, 
            [0, self.max_tau],
            [self.r0, -np.sqrt(2 * self.M / self.r0)],
            method="RK45", # Runge-Kutta 4/5 solver (adaptive step size)
            max_step=self.max_step,
            rtol=1e-4,
            atol=1e-6,
            events=stop_near_singularity,
        )

        self.trajectory = list(zip(result.t, result.y[0]))

    def get_positions(self) -> List[float]:
        return [r for tau, r in self.trajectory]


class SchwarzschildDustCloud:
    def __init__(
        self,
        num_particles: int,
        r_start: float, # initial radius
        max_steps: int,
        mass: float,
        max_tau: float, # max proper time
        max_step: float,
        tolerance: float,
    ) -> None:
        self.num_particles: int = num_particles
        self.max_steps: int = max_steps
        self.mass: float = mass
        self.particles: List[EddingtonFinkelsteinParticle] = [
            EddingtonFinkelsteinParticle(
                r0=r_start + i * 10.0 / num_particles,  # spacing
                M=self.mass,
                max_tau=max_tau,
                max_step=max_step,
                tolerance=tolerance,
            )
            for i in range(num_particles)
        ]

        self.history_r: List[List[float]] = [[] for _ in self.particles]
        self.history_tau: List[List[float]] = [[] for _ in self.particles]
        self.entropies: List[float] = []
        self.entropy_taus: List[float] = []
        self.max_radius_bin: int = 15000

    def compute_entropy(self, positions: List[float], num_bins: int = 50) -> float:
        hist, _ = np.histogram(positions, bins=num_bins, density=False)
        probs = hist / np.sum(hist)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def run(self, num_steps: int = 500) -> None:
        max_tau: float = min(p.trajectory[-1][0] for p in self.particles)
        self.entropy_taus = np.linspace(0, max_tau, num_steps)
        self.history_r = [[] for _ in self.particles]
        self.history_tau = [[] for _ in self.particles]
        self.entropies = []

        trajectory_maps: List[Tuple[np.ndarray, np.ndarray]] = []
        tau_ranges: List[Tuple[float, float]] = []

        for p in self.particles:
            taus, rs = zip(*p.trajectory)
            taus = np.array(taus)
            rs = np.array(rs)
            trajectory_maps.append((taus, rs))
            tau_ranges.append((taus[0], taus[-1]))

        for step, current_tau in enumerate(self.entropy_taus):
            current_positions: List[float] = []
            for i, (taus, rs) in enumerate(trajectory_maps):
                min_tau, max_tau = tau_ranges[i]
                if current_tau < min_tau:
                    r = float(rs[0])
                elif current_tau > max_tau:
                    # Freeze particle at last position instead of nan
                    r = float(rs[-1])
                else:
                    r = float(np.interp(current_tau, taus, rs))
                current_positions.append(r)
                self.history_r[i].append(r)
                self.history_tau[i].append(current_tau)

            valid_rs: List[float] = [r for r in current_positions if not np.isnan(r)]
            entropy: float = self.compute_entropy(valid_rs) if valid_rs else 0.0
            self.entropies.append(entropy)
            if step % 50 == 0 or step == num_steps - 1:
                print(
                    f"Step {step}, τ = {current_tau:.5f}, mean r = {np.mean(valid_rs):.5f}, entropy = {entropy:.5f}"
                )

    def visualize(self) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(14, 20))

        max_particle_tau: float = (
            max(max(tau_list) for tau_list in self.history_tau if tau_list) * 1.02
        )
        for ax in axes:
            ax.set_xlim(0, max_particle_tau)

        axes[0].plot(self.entropy_taus, self.entropies, label="Shannon Entropy (Eddington-Finkelstein)")
        axes[0].set_ylabel("Entropy (bits)")
        axes[0].set_xlabel("Proper Time τ")
        axes[0].set_title("Total Shannon Entropy signature of with Eddington-Finkelstein Coordinates")
        axes[0].legend()
        axes[0].grid(True)

        for i, r_vals in enumerate(self.history_r):
            if i % 10 != 0:  # only every 10th particle
                continue
            axes[1].plot(self.history_tau[i], r_vals, alpha=0.5)

        axes[1].axhline(y=2.0, color="red", linestyle="--", label="Event Horizon (r=2)")
        axes[1].set_xlabel("Proper Time τ")
        axes[1].set_ylabel("Radius r")
        axes[1].set_title("Radial Infall of Particles")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig("schwarzschild_eddington_finkelstein.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    start = time.time()
    
    cloud = SchwarzschildDustCloud(
        num_particles=1000,
        r_start=initial_radius,
        max_steps=2500, 
        mass=1.0,
        max_tau=30.0, # maximum value of proper time ττ when integrating the geodesic equations.
        max_step=0.01,
        tolerance=1e-1
    )
    cloud.run(800)
    cloud.visualize()
    print(f"Simulation took {time.time() - start:.2f} seconds")
    