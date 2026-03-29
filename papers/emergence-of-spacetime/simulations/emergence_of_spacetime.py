"""
Emergent Spacetime and Particles
================================

Description:
-------------
Simulate an execution trace of increasing entropy and monitor the corresponding expansion of the emergent spacetime and particle populations.
Minimal and naive example of how complex structures and spacetime-like behavior can emerge from simple information-theoretic rules.


Emergent Spacetime and Particles: A Computational Ontology
==========================================================

1. Zero-Entropy Initialization:
   Represents the Pre-Geometric Vacuum. All mappings yield a single point: 
   the Initial Singularity. Structure is impossible as p(1) = 0.

2. Entropy Growth & The Quark Epoch:
   As mutations begin, the first '1' bits allow the simplest sub-patterns 
   to emerge. This represents the Quark Epoch: a state where only the 
   most fundamental, high-probability micro-structures can exist.

3. Recursive Hierarchy & Emergence:
   Higher-order structures (Atoms) are defined as 'Coupled 
   Patterns'. They require a specific density of lower-level 'hits' 
   to manifest, naturally creating the sequential epochs of the Big Bang.

4. Lognormal Structural Evolution:
   The population of micro-structures follows a log-normal distribution.
   - Inflation: The rapid explosion of Level 1 hits as entropy hits the 
     critical threshold.
   - Heat Death: The long, slow decay of complexity as the bitstring 
     saturates into maximum entropy (random noise).


Emergent Properties:
-------------------
- Entropy increases over time.
- Hierarchical structures of particles emerge from a minimal rule.
- Spatial extent grows, while particle sizes remain discrete, 
  reflecting the separation between spacetime expansion and particle scale.
- Particle populations follow log-normal-like distribution as a function of
  mutations.


Theoretical Implications: The Informational Singularity
-------------------------------------------------------

1. The Smooth Singularity (Zero-Gravity State):
In this model, the initial singularity is not a point of infinite density or curvature, but a state of Zero Entropy (H=0). 
The absence of detectable micro-structures at H=0 implies a perfectly smooth no-gravity state, similar to the gravitational null-point at the center of a planet.

2. Gravity as an Entropic Derivative:
Gravity is redefined as the statistical pressure of structural emergence. Mathematically, the gravitational 
force is the derivative of the log-normal distribution of particle populations (dN/dH).

    Inflation: Represents the steep positive gradient as the first particles crystallize from the bitstring.

    Heat Death: Represents the negative gradient as the probability of maintaining complex structures "dries out" against the background noise.

3. Non-Identity of Emergent Particles:
Particles in this simulation lack inherent "identity." A "particle" is simply a successful pattern-match at a specific coordinate. 
There is no underlying "object" moving through space; there is only the sequential re-occurrence of a pattern across the bitstring. 
"Motion" and "Time" are the observer's interpretation of these correlated hits, effectively resolving the Gibbs Paradox by
making indistinguishability a fundamental property of the information field.

Usage:
------
The simulation can be configured via command-line arguments for:
    --method         : bitstring mutation method (bitflip, hamming, entropy)
    --bits           : total bitstring size
    --steps          : number of simulation steps
    --bits_per_coord : width of each spacetime fragment
    --pattern        : bit pattern identifying elementary particles

Example:
--------
    python simulation.py --bits 1048576 --steps 80000 --bits_per_coord 16 --pattern 0b0010

This minimal model provides a concrete demonstration of how complex 
structures and spacetime-like behavior tend to  emerge from simple 
information-theoretic rules with remarkable similarity to observed cosmic phenomena.

Copyright 2019 - Juha Meskanen, The Abstract Universe Project
"""

import random
import numpy as np
import argparse
from numba import njit
import matplotlib.pyplot as plt

@njit
def entropy_bitstring(bitstring: np.ndarray) -> float:
    count0 = 0
    count1 = 0
    for i in range(bitstring.size):
        if bitstring[i] == 0:
            count0 += 1
        else:
            count1 += 1
    total = count0 + count1
    if total == 0:
        return 0.0
    p0 = count0 / total
    p1 = count1 / total
    entropy = 0.0
    if p0 > 0:
        entropy -= p0 * np.log2(p0)
    if p1 > 0:
        entropy -= p1 * np.log2(p1)
    return entropy

@njit
def space_size_jit(points: np.ndarray) -> int:
    if points.shape[0] == 0:
        return 0
    return int(points.max() - points.min())

class Simulation:
    def __init__(self, bits: int, steps: int, bits_per_coord : int, pattern : int):
        self.steps = steps
        self.bits = bits
        self.bits_per_coord = bits_per_coord
        self.pattern = pattern
        self.bitstring = np.zeros(bits, dtype=np.uint8)
        self.history = []

    def recursive_pattern_levels(self, bitstring: np.ndarray) -> list[np.ndarray]:
        if len(bitstring) < self.bits_per_coord:
            return []

        n_segments = len(bitstring) // self.bits_per_coord
        trimmed = bitstring[:n_segments * self.bits_per_coord]
        reshaped = trimmed.reshape((n_segments, self.bits_per_coord))

        if self.bits_per_coord <= 8:
            dtype = np.uint8
        elif self.bits_per_coord <= 16:
            dtype = np.uint16
        elif self.bits_per_coord <= 32:
            dtype = np.uint32
        else:
            raise ValueError("bits_per_coord too large, max 32 supported")

        weights = (1 << np.arange(self.bits_per_coord - 1, -1, -1, dtype=dtype))
        values = np.dot(reshaped.astype(dtype), weights)
        next_bitstring = (values == dtype(self.pattern)).astype(np.uint8)

        return [(values, next_bitstring)] + self.recursive_pattern_levels(next_bitstring)

    def run(self):
        for step in range(self.steps):
            entropy = entropy_bitstring(self.bitstring)
            levels = self.recursive_pattern_levels(self.bitstring)
            ones_count = [np.sum(next_bits) for (_, next_bits) in levels]

            space_size = 0
            if levels:
                coords0, _ = levels[0]
                space_size = space_size_jit(coords0)

            snapshot = [float(entropy), float(space_size)]
            snapshot.extend(float(c) for c in ones_count)
            self.history.append(snapshot)

            index = random.randint(0, self.bits - 1)
            self.bitstring[index] ^= 1

    def plot(self):
        history = np.array(self.history)
        steps = np.arange(len(history))

        # Create two subplots: Top for original data, Bottom for CMB Fluctuation Band
        fig, (ax1, ax_cmb) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})

        # --- TOP PLOT (Original Features) ---
        num_levels = history.shape[1] - 2
        colors = plt.cm.viridis(np.linspace(0, 1, num_levels))

        for i in range(num_levels):
            ax1.plot(steps, history[:, i+2], label=f"Level {i+1}", color=colors[i], alpha=0.7)

        ax1.set_ylabel("Structure Count", color="black")
        ax2 = ax1.twinx()
        ax2.plot(steps, history[:, 0], label="Entropy", color="red", linestyle="--")
        ax2.set_ylabel("Entropy (bits)", color="red")

        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.15))
        ax3.plot(steps, history[:, 1], label="Space Size", color="blue", linestyle=":")
        ax3.set_ylabel("Space Size", color="blue")

        # Combine legends for top plot
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        h3, l3 = ax3.get_legend_handles_labels()
        ax1.legend(h1 + h2 + h3, l1 + l2 + l3, loc="upper left")

        # --- BOTTOM PLOT (CMB Fluctuation Band) ---
        # We calculate the jitter of Level 1 (Protons/Initial Particles)
        l1_data = history[:, 2]
        window_size = max(10, self.steps // 50)
        
        # Calculate moving average and standard deviation (The Band)
        l1_mean = np.convolve(l1_data, np.ones(window_size)/window_size, mode='same')
        l1_std = np.zeros_like(l1_data)
        for i in range(len(l1_data)):
            start = max(0, i - window_size // 2)
            end = min(len(l1_data), i + window_size // 2)
            l1_std[i] = np.std(l1_data[start:end])

        ax_cmb.fill_between(steps, l1_mean - l1_std, l1_mean + l1_std, color='gray', alpha=0.3, label="CMB Thermal Band (±1σ)")
        ax_cmb.plot(steps, l1_mean, color='black', label="Smoothed L1 Trend", linewidth=1.5)
        ax_cmb.set_xlabel("Step")
        ax_cmb.set_ylabel("L1 Fluctuations")
        ax_cmb.legend(loc="upper left")
        ax_cmb.set_title("Thermal Fluctuation Band (Anisotropy Proxy)")

        fig.suptitle("Evolutionary Trace: Entropy, Spacetime, and CMB Fluctuations")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emerging spacetime")
    parser.add_argument("--bits", type=int, default=16384*5, help="Total number of bits in the universe")
    parser.add_argument("--steps", type=int, default=16384*5, help="Number of simulation steps")
    parser.add_argument("--bits_per_coord", type=int, default=16, help="Width of each coordinate in bits")
    parser.add_argument("--pattern", type=lambda x: int(x, 0), default=0b0010, help="Emergent pattern to be detected (supports 0b binary notation)")

    args = parser.parse_args()
    sim = Simulation(bits=args.bits, steps=args.steps, bits_per_coord=args.bits_per_coord, pattern=args.pattern)
    sim.run()
    sim.plot()
