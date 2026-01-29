"""
Simulation of Emergent Observer-Filtered Histories

This module implements a minimal demonstration of the IaM^e information-theoretic
framework for emergent physical laws under observer conditioning.


Theory Background:

Given a history space Γ of all possible time-evolving configurations (here,
represented as a sequence of binary vectors), and an observer O that only
perceives coarse-grained features, the probability of a history is:

    P(γ | O) = (1 / Z_O) * exp(-λ C_O[γ]),   γ ∈ Γ_O

where:
- λ > 0 is a Gibbs parameter controlling preference for low-complexity
  histories,
- C_O[γ] is a complexity functional (here, sum of Hamming distances between
  consecutive time slices),
- Γ_O is the set of histories compatible with the observer O.


Simulation Approach:

- Histories are represented as binary matrices of shape (T, k), where T is
  the number of time steps and k is the number of bits per step.
- Metropolis-Hastings updates flip individual bits with probability
  proportional to exp(-λ ΔC), sampling from the observer-filtered Gibbs
  measure.
- Observer coarse-graining is performed by block averaging the bit density
  over non-overlapping windows.
- As λ increases, observer-projected histories concentrate onto smooth,
  predictable trajectories, demonstrating the emergence of apparent "laws"
  from a combinatorially huge set of micro-histories.

  
Visualization:

- Top subplot: micro-history bit density (chaotic underlying microstate)
- Bottom subplot: observer-averaged trajectory (emergent smooth law)
- Animation can be exported as GIF or MP4 via command-line arguments.


Usage:

python observer_gibbs_sim.py --lambda_ 1.0 --steps 300 --time 200 --bits 64 \
    --window 10 --file emergent_universe --format gif

Author:
2024 Juha Meskanen
"""


import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Tuple
from simulation_engine import SimulationEngine

class ObserverGibbs1D(SimulationEngine):
    """1D observer-Gibbs simulation with bitstring histories, demonstrating emergent laws
    via maximum compression -> maximum probability -> maximum smoothness,
    An observer is formally realized as a coarse-graining map that discards microstate 
    identity and fine temporal resolution, thereby defining observational equivalence classes over histories. 
    Physical regularities emerge as the most probable compressed representatives within these classes under a Gibbs complexity prior.
    """

    def __init__(self, time_steps: int, bits: int, lambda_: float, window: int = 1) -> None:
        super().__init__(time_steps, lambda_, window=window)
        self.k = bits
        self.initialize_history()

    def initialize_history(self) -> None:
        self.history = np.random.randint(0, 2, size=(self.T, self.k), dtype=np.int8)
        self.coarse_history = self.observer_projection()

    def complexity(self, t: int) -> float:
        return np.sum(np.abs(self.history[t] - self.history[t-1]))

    def metropolis_step(self) -> None:
        t = np.random.randint(1, self.T)
        i = np.random.randint(0, self.k)
        before = self.complexity(t)
        self.history[t, i] ^= 1
        after = self.complexity(t)
        delta = after - before
        if delta > 0 and np.random.rand() > np.exp(-self.lambda_ * delta):
            self.history[t, i] ^= 1  # reject

    def observer_projection(self) -> np.ndarray:
        blocks = self.T // self.window
        coarse = self.history[:blocks * self.window].reshape(blocks, self.window, self.k).mean(axis=(1,2))
        return coarse

    def _setup_plot(self, dpi: int):
        fig, ax = plt.subplots(2, 1, figsize=(6,6), dpi=dpi)
        self.micro_line, = ax[0].plot([], [], lw=1)
        self.coarse_line, = ax[1].plot([], [], lw=2)
        ax[0].set_xlim(0, self.T)
        ax[0].set_ylim(0,1)
        ax[0].set_title("Micro History Bit Density")
        ax[1].set_xlim(0, len(self.coarse_history))
        ax[1].set_ylim(0,1)
        ax[1].set_title("Observer Coarse Trajectory")
        return fig, ax

    def _update_plot(self, frame: int):
        for _ in range(50):
            self.metropolis_step()
        self.coarse_history[:] = self.observer_projection()
        micro = self.history.mean(axis=1)
        coarse = self.coarse_history
        self.micro_line.set_data(np.arange(len(micro)), micro)
        self.coarse_line.set_data(np.arange(len(coarse)), coarse)
        return self.micro_line, self.coarse_line
    


def main() -> None:
    parser = argparse.ArgumentParser(description="1D Observer-Filtered Gibbs Simulation")
    parser.add_argument("--file", type=str, default="observer_1d", help="Output filename")
    parser.add_argument("--steps", type=int, default=300, help="Animation frames")
    parser.add_argument("--time", type=int, default=200, help="History length T")
    parser.add_argument("--bits", type=int, default=64, help="Bits per time slice k")
    parser.add_argument("--lambda_", type=float, default=1.0, help="Gibbs parameter λ")
    parser.add_argument("--window", type=int, default=10, help="Observer coarse window")
    parser.add_argument("--res", type=int, default=120, help="Resolution")
    parser.add_argument('--format', choices=['gif', 'mp4'], default='gif', help="Output format")
    args = parser.parse_args()

    sim = ObserverGibbs1D(time_steps=args.time, bits=args.bits, lambda_=args.lambda_, window=args.window)
    sim.res = args.res
    sim.animate(steps=args.steps, filename=args.file, fmt=args.format)

if __name__ == "__main__":
    main()
