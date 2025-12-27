"""
Emergent Spacetime and Particles
================================

Description:
-------------
Simulate an execution trace of increasing entropy and monitor the corresponding expansion of the emergent spacetime.


Key Concepts:
-------------
1. Zero-Entropy Initialization:
   The simulation begins with a bitstring of all zeros, representing 
   a minimal-information, perfectly ordered state.

2. Entropy Growth:
   Random bit flips incrementally increase the entropy of the system.

3. Spacetime Fabric:
   The bitstring is sliced into fixed-width segments ("coordinates") 
   representing elementary fragments of 1D spacetime.

4. Particle Detection:
   Specific patterns within spacetime fragments are identified as 
   "particles". A match against a given pattern produces a binary 
   bitstring indicating particle presence.

5. Recursive Structure Formation:
   Detected particles are further processed recursively to reveal 
   higher-level structures ("atoms", "molecules", etc.), illustrating 
   the hierarchical organization of matter.

Emergent Properties:
-------------------
- Entropy increases over time due to bitflip algorithm.
- Hierarchical structures of particles emerge from a minimal rule.
- Spatial extent grows, while particle sizes remain discrete, 
  reflecting the separation between spacetime expansion and particle scale.
- **Particle populations follow a log-normal distribution as a function of
  cumulative bit flips**.


Usage:
------
The simulation can be configured via command-line arguments for:
    --bits           : total bitstring size
    --steps          : number of simulation steps
    --bits_per_coord : width of each spacetime fragment
    --pattern        : bit pattern identifying elementary particles

Example:
--------
    python simulation.py --bits 1048576 --steps 80000 --bits_per_coord 8 --pattern 0b0010

This minimal model provides a concrete demonstration of how complex 
structures and spacetime-like behavior can emerge from simple 
information-theoretic rules.

Copyright 2019 - The Abstract Universe Project
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
        """
        Recursively slice bitstring into fixed-size segments and match against `pattern`.
        Works for any bits_per_coord (4..32) and patterns smaller than or equal to bits_per_coord.
        """
        if len(bitstring) < self.bits_per_coord:
            return []  # stop recursion

        # print(''.join(bitstring[:128].astype(str)))

        n_segments = len(bitstring) // self.bits_per_coord
        trimmed = bitstring[:n_segments * self.bits_per_coord]
        reshaped = trimmed.reshape((n_segments, self.bits_per_coord))

        # Choose integer type wide enough to hold the largest value
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

        #return [next_bitstring] + self.recursive_pattern_levels(next_bitstring)
        return [(values, next_bitstring)] + self.recursive_pattern_levels(next_bitstring)


    def run(self):
        for step in range(self.steps):
            entropy = entropy_bitstring(self.bitstring)

            # extract particles & coordinate values
            levels = self.recursive_pattern_levels(self.bitstring)

            # particle counts
            ones_count = [np.sum(next_bits) for (_, next_bits) in levels]

            # space size (only meaningful at level 0, i.e. first coords)
            space_size = 0
            if levels:
                coords0, _ = levels[0]
                space_size = space_size_jit(coords0)

            # snapshot = [entropy, space_size, level1count, level2count, ...]
            snapshot = [float(entropy), float(space_size)]
            snapshot.extend(float(c) for c in ones_count)

            self.history.append(snapshot)

            # Flip 1 random bit per step
            index = random.randint(0, self.bits - 1)
            self.bitstring[index] ^= 1


    def run2(self):
        for step in range(self.steps):

            # measure entropy
            entropy = entropy_bitstring(self.bitstring)
            
            # extract particles
            particles = self.recursive_pattern_levels(self.bitstring)

            # number of particles
            ones_count = [np.sum(particle) for particle in particles]

            # build up data array for plotting
            snapshot = []
            snapshot.append(float(entropy))
            for i, (lvl, count) in enumerate(zip(particles, ones_count)):
                snapshot.append(float(count))

            self.history.append(snapshot)
            
            # Flip k random bits per step
            index = random.randint(0, self.bits - 1)
            self.bitstring[index] ^= 1


    def plot(self):
        history = np.array(self.history)
        steps = np.arange(len(history))

        fig, ax1 = plt.subplots()

        # structure counts (left y-axis)
        num_levels = history.shape[1] - 2
        colors = plt.cm.viridis(np.linspace(0, 1, num_levels))

        for i in range(num_levels):
            ax1.plot(steps, history[:, i+2], label=f"Level {i+1}", color=colors[i])

        ax1.set_xlabel("Step")
        ax1.set_ylabel("Structure Count", color="black")
        ax1.tick_params(axis="y", labelcolor="black")

        # entropy axis (right side, position 1)
        ax2 = ax1.twinx()
        ax2.plot(steps, history[:, 0], label="Entropy", color="red", linestyle="--")
        ax2.set_ylabel("Entropy (bits)", color="red")
        ax2.tick_params(axis="y", labelcolor="red")

        # space size axis (second right side, shifted outward)
        ax3 = ax1.twinx()
        ax3.spines["right"].set_position(("axes", 1.15))  # shift outward
        ax3.plot(steps, history[:, 1], label="Space Size", color="blue", linestyle=":")
        ax3.set_ylabel("Space Size", color="blue")
        ax3.tick_params(axis="y", labelcolor="blue")

        # combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="upper left")

        fig.suptitle("Entropy, Structures, and Spatial Spread Over Time")
        plt.tight_layout()
        plt.show()




if __name__ == "__main__":

    smoothness : int = 128
    parser = argparse.ArgumentParser(description="Emerging spacetime")
    parser.add_argument("--bits", type=int, default=smoothness*1024, help="Total number of bits")
    parser.add_argument("--steps", type=int, default=smoothness*1024, help="Number of simulation steps")
    parser.add_argument("--bits_per_coord", type=int, default=16, help="Width of each coordinate in bits")
    parser.add_argument("--pattern", type=lambda x: int(x, 0), default=0b0010,
                        help="Bit pattern to match (supports 0b binary notation)")

    args = parser.parse_args()

    sim = Simulation(bits=args.bits, 
                     steps=args.steps, 
                     bits_per_coord=args.bits_per_coord, 
                     pattern=args.pattern)

    sim.run()
    sim.plot()

