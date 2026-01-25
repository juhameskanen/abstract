"""
PoC: Observer Probability vs Wave Complexity
============================================

Demonstrates that more compressible observer configurations (lower spectral complexity)
dominate the probability measure, consistent with:

    P(O) ∝ 2^{-H_spec} = exp(-H_spec * ln2)

where H_spec is the Shannon entropy of the wavefunction (Minimal Spectral Length, MSL).
"""

import numpy as np
import matplotlib.pyplot as plt
from qbitwave import QBitwave
from scipy.stats import binned_statistic

# ---------------------------
# Parameters
# ---------------------------
N = 64          # bitstring length
num_samples = 5000  # number of random samples

# ---------------------------
# Generate wavefunctions and compute MSL
# ---------------------------
msl_list = []
for _ in range(num_samples):
    bits = np.random.randint(0, 2, N).tolist()
    q = QBitwave(bitstring=bits)
    msl_list.append(q.wave_complexity())

msl_array = np.array(msl_list)

# ---------------------------
# Convert spectral complexity to observer probability
# ---------------------------
# Theoretical justification: Shannon entropy of amplitudes corresponds to effective
# bits required to encode the observer. Lower entropy = fewer bits = higher probability.
probabilities = 2.0 ** (-msl_array)  # P(O) ∝ 2^{-H_spec}

# ---------------------------
# Scatter plot: MSL vs P(O)
# ---------------------------
plt.figure(figsize=(8,5))
plt.scatter(msl_array, probabilities, alpha=0.3, s=5, color='blue')
plt.yscale("log")
plt.xlabel("Minimal Spectral Length (MSL)")
plt.ylabel("Observer Probability P(O)")
plt.title("Compression ⇒ Maximal Probability")
plt.grid(True)
plt.show()

# ---------------------------
# Smoothed trend using binned statistic
# ---------------------------
bin_means, bin_edges, _ = binned_statistic(msl_array, probabilities, statistic='mean', bins=50)
bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
plt.figure(figsize=(8,5))
plt.plot(bin_centers, bin_means, color='red', lw=2, label="Average P(O) per bin")
plt.xlabel("Minimal Spectral Length (MSL)")
plt.ylabel("Average Observer Probability P(O)")
plt.title("Smoothed Compression → Probability Trend")
plt.grid(True)
plt.legend()
plt.show()

# ---------------------------
# Histogram of probabilities
# ---------------------------
plt.figure(figsize=(8,5))
plt.hist(probabilities, bins=50, edgecolor='k', alpha=0.7)
plt.xlabel("Observer Probability P(O)")
plt.ylabel("Frequency")
plt.title("Distribution of Observer Probabilities")
plt.grid(True)
plt.show()
