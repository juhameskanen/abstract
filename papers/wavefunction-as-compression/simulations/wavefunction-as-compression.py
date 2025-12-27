"""
This module implements a computational simulation exploring the "Wave Function as Compression Algorithm" hypothesis.

Hypothesis Context:
------------------
- The universe is interpreted as a geometric manifestation of a vast random bitstring.
- Observers inhabit maximally compressed regions of information (the "observer filter").
- The quantum wave function ψ can be interpreted as a universal compression algorithm.
- PCA is used here as a linear compression analog to study coefficient dynamics,
  reconstruction accuracy, and emergent near-unitary evolution.

This script:
1. Generates a 1D tight-binding Hamiltonian to simulate linear-unitary quantum evolution.
2. Computes observed frames (real part and intensity of ψ).
3. Applies PCA to compress frames into lower-dimensional coefficient space.
4. Fits a linear propagator G between successive PCA coefficients.
5. Evaluates unitarity and reconstruction error.
6. Produces publication-ready plots:
   - PCA components
   - Coefficient dynamics
   - Singular values of G
   - Reconstruction MSE vs timestep
   - Reconstruction MSE vs number of PCA components

Author: Juha Meskanen
Date: 2024-12-23

TODO: add type annotations, object oriented

"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.linalg
import os

# Create output directory
output_dir = "simulation_plots"
os.makedirs(output_dir, exist_ok=True)

# Simulation Parameters
N = 64           # Number of lattice sites
T = 400          # Number of timesteps (increase for smoother plots)
pca_components = 8  # PCA compression dimension
max_components = 40  # Max PCA components for reconstruction error plot

# Tight-binding Hamiltonian

def generate_tight_binding_hamiltonian(N):
    H = np.zeros((N, N), dtype=complex)
    for i in range(N - 1):
        H[i, i+1] = H[i+1, i] = -1.0
    return H

# Time evolution
def time_evolve(H, psi0, steps):
    psi_t = np.zeros((steps, N), dtype=complex)
    psi_t[0] = psi0
    U = scipy.linalg.expm(-1j * H / 10.0)  # smaller timestep for smoother dynamics
    for t in range(1, steps):
        psi_t[t] = U @ psi_t[t-1]
    return psi_t

# Initial state
np.random.seed(42)  # reproducibility
psi0 = np.random.randn(N) + 1j * np.random.randn(N)
psi0 /= np.linalg.norm(psi0)

H = generate_tight_binding_hamiltonian(N)
psi_t = time_evolve(H, psi0, T)

# Observed frames
frames_real = psi_t.real
frames_intensity = np.abs(psi_t)**2


# PCA Compression

def apply_pca(frames, n_components):
    pca = PCA(n_components=n_components)
    coeffs = pca.fit_transform(frames)
    reconstructed = pca.inverse_transform(coeffs)
    return coeffs, reconstructed, pca

coeffs_real, recon_real, pca_real = apply_pca(frames_real, pca_components)
coeffs_intensity, recon_intensity, pca_intensity = apply_pca(frames_intensity, pca_components)


# Linear Propagator Fitting

def fit_linear_propagator(coeffs):
    G, _, _, _ = np.linalg.lstsq(coeffs[:-1], coeffs[1:], rcond=None)
    return G.T

G_real = fit_linear_propagator(coeffs_real)
G_intensity = fit_linear_propagator(coeffs_intensity)

def unitarity_error(G):
    return np.linalg.norm(G.conj().T @ G - np.eye(G.shape[0]))

unitarity_real = unitarity_error(G_real)
unitarity_intensity = unitarity_error(G_intensity)


# Reconstruction MSE

def reconstruction_mse(frames, recon):
    return np.mean((frames - recon)**2, axis=1)

mse_real = reconstruction_mse(frames_real, recon_real)
mse_intensity = reconstruction_mse(frames_intensity, recon_intensity)


# Plotting helper

def save_plot(fig, filename, dpi=300):
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {path}")


# 1. PCA Components (first 4)

fig, axs = plt.subplots(2,2,figsize=(12,8))
for i in range(4):
    axs[i//2, i%2].plot(pca_real.components_[i], lw=2)
    axs[i//2, i%2].set_title(f"PCA Component {i+1} (Real part)")
    axs[i//2, i%2].grid(True)
plt.tight_layout()
save_plot(fig, "pca_components_real.png")
plt.close(fig)


# 2. Coefficient dynamics (first 4)

fig, axs = plt.subplots(2,2,figsize=(12,8))
time = np.arange(T)
for i in range(4):
    axs[i//2, i%2].plot(time, coeffs_real[:,i], label='Real', lw=2)
    axs[i//2, i%2].plot(time, coeffs_intensity[:,i], label='Intensity', lw=2, linestyle='--')
    axs[i//2, i%2].set_title(f"Coefficient dynamics (component {i+1})")
    axs[i//2, i%2].grid(True)
    axs[i//2, i%2].legend()
plt.tight_layout()
save_plot(fig, "coefficient_dynamics.png")
plt.close(fig)


# 3. Singular values of G

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(np.linalg.svd(G_real, compute_uv=False), 'o-', label='Real PCA', lw=2)
ax.plot(np.linalg.svd(G_intensity, compute_uv=False), 'x-', label='Intensity PCA', lw=2)
ax.set_title("Singular values of fitted propagator G")
ax.set_xlabel("Index")
ax.set_ylabel("Singular value")
ax.grid(True)
ax.legend()
save_plot(fig, "singular_values_G.png")
plt.close(fig)


# 4. Reconstruction MSE vs timestep

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(time, mse_real, label='Real part', lw=2)
ax.plot(time, mse_intensity, label='Intensity', lw=2)
ax.set_title("Reconstruction MSE vs Timestep")
ax.set_xlabel("Timestep")
ax.set_ylabel("MSE")
ax.grid(True)
ax.legend()
save_plot(fig, "reconstruction_mse_timestep.png")
plt.close(fig)


# 5. Reconstruction MSE vs number of PCA components

mse_real_list = []
mse_intensity_list = []
for n in range(1, max_components+1):
    coeffs_r, recon_r, _ = apply_pca(frames_real, n)
    coeffs_i, recon_i, _ = apply_pca(frames_intensity, n)
    mse_real_list.append(np.mean((frames_real - recon_r)**2))
    mse_intensity_list.append(np.mean((frames_intensity - recon_i)**2))

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(range(1, max_components+1), mse_real_list, 'o-', label='Real part', lw=2)
ax.plot(range(1, max_components+1), mse_intensity_list, 'x-', label='Intensity', lw=2)
ax.set_title("Reconstruction MSE vs Number of PCA Components")
ax.set_xlabel("Number of PCA components")
ax.set_ylabel("MSE")
ax.grid(True)
ax.legend()
save_plot(fig, "reconstruction_mse_components.png")
plt.close(fig)


# Print unitarity results

print(f"Unitarity error (Real PCA)      : {unitarity_real:.4e}")
print(f"Unitarity error (Intensity PCA) : {unitarity_intensity:.4e}")
