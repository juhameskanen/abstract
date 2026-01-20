import numpy as np
import matplotlib.pyplot as plt


# Physical parameters

m = 1.0          # scalar field mass
C = 1e4          # configuration scale (informational, not physical cutoff)
alpha = 2.0      # collapse rate of distinguishable configurations


# Momentum domain

k_max = 1e4
num_points = 200_000
k = np.linspace(1e-6, k_max, num_points)


# QFT integrand (vacuum energy)

omega = np.sqrt(k**2 + m**2)
integrand = omega * k**2


# Entropy model

def configuration_count(k, C, alpha):
    return 1 + np.floor(C / (k**alpha))

N_k = configuration_count(k, C, alpha)

# Shannon entropy of uniform distribution over configurations
H_k = np.log(N_k)

# Entropy-weighted measure
w_k = np.exp(-H_k)


# Integrals

dk = k[1] - k[0]

raw_integral = np.sum(integrand) * dk
entropy_weighted_integral = np.sum(integrand * w_k) * dk


# Output

print("Raw divergent integral:", raw_integral)
print("Entropy-weighted integral:", entropy_weighted_integral)


# Plots

plt.figure(figsize=(10, 6))
plt.loglog(k, integrand, label="Raw integrand")
plt.loglog(k, integrand * w_k, label="Entropy-weighted integrand")
plt.xlabel("Momentum k")
plt.ylabel("Contribution")
plt.legend()
plt.title("Suppression of UV Divergence via Entropy Weighting")
plt.show()

plt.figure(figsize=(10, 6))
plt.semilogx(k, H_k)
plt.xlabel("Momentum k")
plt.ylabel("Entropy H(k)")
plt.title("Entropy Collapse of Field Configurations")
plt.show()
