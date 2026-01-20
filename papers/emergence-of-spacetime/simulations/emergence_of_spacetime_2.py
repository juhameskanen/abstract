import random
import numpy as np
import matplotlib.pyplot as plt

class PurePlanckSimulation:
    def __init__(self, bits=2**15, steps=100000):
        self.bits = bits
        self.steps = steps
        # Start at absolute zero (Order)
        self.bitstring = np.zeros(bits, dtype=np.uint8)
        
        # We search for a specific "Atom" (Level 1)
        self.pattern = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        self.history = []

    def run(self):
        # Tracking counts to make Entropy O(1)
        ones = 0
        
        for s in range(self.steps):
            # 1. Planck Time: Flip 1 Bit
            idx = random.randint(0, self.bits - 1)
            old_val = self.bitstring[idx]
            self.bitstring[idx] ^= 1
            ones += (1 if old_val == 0 else -1)
            
            # 2. Snapshot every 200 steps
            if s % 200 == 0:
                # Calculate Entropy
                p1 = ones / self.bits
                h = 0
                if 0 < p1 < 1:
                    h = -(p1 * np.log2(p1) + (1 - p1) * np.log2(1 - p1))
                
                # --- INSTANTANEOUS MEASUREMENT ---
                # We re-scan the string to see what exists RIGHT NOW.
                # No memory of previous steps.
                view = np.lib.stride_tricks.sliding_window_view(self.bitstring, 8)
                l1_matches = np.all(view == self.pattern, axis=1)
                l1_count = np.sum(l1_matches)
                
                # Level 2: Clusters (Matches within 12 bits of each other)
                match_indices = np.where(l1_matches)[0]
                l2_count = 0
                if len(match_indices) > 1:
                    l2_count = np.sum(np.diff(match_indices) < 12)
                
                self.history.append([h, l1_count, l2_count])

    def plot(self):
        data = np.array(self.history)
        steps = np.arange(len(data)) * 200
        
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        # Entropy
        ax2 = ax1.twinx()
        ax2.plot(steps, data[:, 0], color='black', lw=1.5, ls='--', label='Entropy')
        ax2.set_ylim(0, 1.1)
        
        # Populations
        ax1.plot(steps, data[:, 1], color='green', label='L1: Atoms', alpha=0.7)
        ax1.plot(steps, data[:, 2], color='red', label='L2: Molecules', lw=2)
        
        ax1.set_xlabel("Steps (Planck Time)")
        ax1.set_ylabel("Instantaneous Count")
        ax2.set_ylabel("Entropy")
        
        plt.title("Emergence vs Entropy: Reaching Statistical Equilibrium")
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.grid(alpha=0.3)
        plt.show()

if __name__ == "__main__":
    sim = PurePlanckSimulation()
    sim.run()
    sim.plot()