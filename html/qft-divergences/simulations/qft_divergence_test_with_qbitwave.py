"""
QFT Divergence Test using QBitwave
----------------------------------

1D scalar field on a lattice, testing entropy-weighted suppression of UV divergences.
High-frequency modes are expected to produce high raw bit entropy and low compressibility.
"""

import numpy as np
import matplotlib.pyplot as plt
from qbitwave import QBitwave  # your class

# -----------------------------
# Simulation parameters
# -----------------------------
L = 64                # number of lattice sites
n_realizations = 20   # number of random field configurations per k_max
k_max_values = np.arange(1, L//2 + 1, 4)  # UV cutoff (mode number)
bits_per_site = 8     # bits to encode each lattice site

# -----------------------------
# Helper: Discretize field to bitstring
# -----------------------------
def field_to_bitstring(phi_x, bits_per_site=8):
    """
    Map φ(x) ∈ [-1,1] → integer → bitstring
    """
    int_vals = ((phi_x + 1) / 2 * (2**bits_per_site - 1)).astype(int)
    bitstring = []
    for val in int_vals:
        bits = [(val >> i) & 1 for i in reversed(range(bits_per_site))]
        bitstring.extend(bits)
    return bitstring

# -----------------------------
# Storage for results
# -----------------------------
entropy_raw = []
compressibility = []
coherence_vals = []

# -----------------------------
# Main loop
# -----------------------------
for k_max in k_max_values:
    e_entropy = []
    e_compress = []
    e_coherence = []

    for _ in range(n_realizations):
        # Generate random scalar field with modes up to k_max
        x = np.arange(L)
        phi_x = np.zeros(L)
        for k in range(1, k_max + 1):
            A_k = np.random.randn()
            phi_x += A_k * np.sin(2 * np.pi * k * x / L)
        # Normalize to [-1,1]
        phi_x /= np.max(np.abs(phi_x)) + 1e-10

        # Convert field to bitstring
        bitstring = field_to_bitstring(phi_x, bits_per_site=bits_per_site)

        # Build QBitwave
        qbw = QBitwave(bitstring=bitstring)

        # Compute measures
        e_entropy.append(qbw.bit_entropy())
        e_compress.append(qbw.compressibility())
        e_coherence.append(qbw.coherence())

    # Store averages
    entropy_raw.append(np.mean(e_entropy))
    compressibility.append(np.mean(e_compress))
    coherence_vals.append(np.mean(e_coherence))

# -----------------------------
# Plot results
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(k_max_values, entropy_raw, label="Raw Bitstring Entropy")
plt.plot(k_max_values, compressibility, label="Wavefunction Compressibility")
plt.plot(k_max_values, coherence_vals, label="Bit-Wavefunction Coherence")
plt.xlabel("UV cutoff (k_max)")
plt.ylabel("Measure")
plt.title("Entropy-Weighted Suppression of Divergent Modes")
plt.legend()
plt.grid(True)
plt.show()
