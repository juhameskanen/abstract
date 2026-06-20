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
1. Generates a non-unitary, classical stochastic data field (no quantum equations hardcoded).
2. Computes observed frames (real projections and intensity densities).
3. Applies PCA/Karhunen-Loève expansions to compress frames into a lower-dimensional coefficient space.
4. Fits a linear propagator G between successive PCA coefficients using a regularized least-squares matrix solver.
5. Evaluates the emergent unitarity error and reconstruction mean-squared error (MSE).
6. Produces publication-ready plots matching the updated paper.

Author: Juha Meskanen & The IAME Collaboration
Year: 2019 ... 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.linalg
import os
from typing import Tuple, List

# Create output directory
output_dir: str = "../simulation_plots"
os.makedirs(output_dir, exist_ok=True)

# Simulation Parameters
N: int = 512             # Number of lattice sites
T: int = 3200            # Number of timesteps
pca_components: int = 8 # PCA compression subspace dimension (K)
max_components: int = 40 # Max PCA components for reconstruction error analysis

# Non-Unitary Data Generation (Replacing the circular Schrödinger step)
def generate_stochastic_classical_field(N: int, T: int) -> np.ndarray:
    """
    Generates a classical, strictly non-unitary statistical history grid.
    Uses an auto-regressive spatial-temporal smoothing filter over raw noise
    to simulate a correlated data substrate without hardcoding quantum mechanics.
    """
    np.random.seed(42)  # Strict reproducibility
    raw_noise: np.ndarray = np.random.randn(T, N) + 1j * np.random.randn(T, N)
    
    # Initialize a non-unitary propagation tensor using spatial smoothing (Gaussian filter)
    smoothed_field: np.ndarray = np.zeros((T, N), dtype=complex)
    smoothed_field[0] = raw_noise[0] / np.linalg.norm(raw_noise[0])
    
    spatial_kernel: np.ndarray = np.exp(-np.linspace(-2, 2, 5)**2)
    spatial_kernel /= np.sum(spatial_kernel)
    
    for t in range(1, T):
        # Strictly non-unitary update: dissipative decay mixed with random fluctuations
        dampened_history: np.ndarray = 0.95 * smoothed_field[t-1] + 0.05 * raw_noise[t]
        # Apply spatial correlation
        smoothed_field[t] = np.convolve(dampened_history, spatial_kernel, mode='same')
        
    return smoothed_field

# Generate raw non-unitary data field
data_substrate: np.ndarray = generate_stochastic_classical_field(N, T)

# Observed frames
frames_real: np.ndarray = data_substrate.real
frames_intensity: np.ndarray = np.abs(data_substrate)**2

# PCA / Karhunen-Loève Compression
def apply_pca(frames: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """
    Applies an orthogonal subspace projection to identify the optimal 
    low-complexity features inside the raw history field.
    """
    pca: PCA = PCA(n_components=n_components)
    coeffs: np.ndarray = pca.fit_transform(frames)
    reconstructed: np.ndarray = pca.inverse_transform(coeffs)
    return coeffs, reconstructed, pca

coeffs_real, recon_real, pca_real = apply_pca(frames_real, pca_components)
coeffs_intensity, recon_intensity, pca_intensity = apply_pca(frames_intensity, pca_components)

# Linear Propagator Fitting
def fit_linear_propagator(coeffs: np.ndarray) -> np.ndarray:
    """
    Finds the shift operator G that minimizes step-wise informational updates
    using a regularized least-squares optimization.
    """
    G, _, _, _ = np.linalg.lstsq(coeffs[:-1], coeffs[1:], rcond=None)
    return G.T

G_real: np.ndarray = fit_linear_propagator(coeffs_real)
G_intensity: np.ndarray = fit_linear_propagator(coeffs_intensity)

def unitarity_error(G: np.ndarray) -> float:
    """
    Measures the deviation from strict macroscopic unitarity via Frobenius norms.
    """
    identity_matrix: np.ndarray = np.eye(G.shape[0])
    return float(np.linalg.norm(G.conj().T @ G - identity_matrix) / np.linalg.norm(identity_matrix))

unitarity_real: float = unitarity_error(G_real)
unitarity_intensity: float = unitarity_error(G_intensity)

# Reconstruction Mean-Squared Error
def reconstruction_mse(frames: np.ndarray, recon: np.ndarray) -> np.ndarray:
    return np.mean((frames - recon)**2, axis=1)

mse_real: np.ndarray = reconstruction_mse(frames_real, recon_real)
mse_intensity: np.ndarray = reconstruction_mse(frames_intensity, recon_intensity)

# Plotting helper
def save_plot(fig: plt.Figure, filename: str, dpi: int = 300) -> None:
    path: str = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Saved: {path}")

# ==============================================================================
# Visualization Suite (Production-Ready)
# ==============================================================================

# 1. PCA Components (first 4 spatial empirical eigenfunctions)
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
for i in range(4):
    ax = axs[i//2, i%2]
    ax.plot(pca_real.components_[i], lw=2, color='navy')
    ax.set_title(f"Empirical Eigenfunction $\\phi_{i+1}$ (Real Substrate)")
    ax.set_xlabel("Spatial Site index (z)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, linestyle=':')
plt.tight_layout()
save_plot(fig, "pca_components_real.png")
plt.close(fig)

# 2. Coefficient dynamics (first 4 temporal tracking states)
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
time: np.ndarray = np.arange(T)
for i in range(4):
    ax = axs[i//2, i%2]
    ax.plot(time, coeffs_real[:, i], label='Real Phase ($c_n(t)$)', lw=2, color='crimson')
    ax.plot(time, coeffs_intensity[:, i], label='Intensity Density ($|\psi|^2$)', lw=2, color='darkorange', linestyle='--')
    ax.set_title(f"Emergent Trajectory Dynamics (Component {i+1})")
    ax.set_xlabel("Internal Timestep ($t$)")
    ax.set_ylabel("Subspace Coordinate Value")
    ax.grid(True, linestyle=':')
    ax.legend()
plt.tight_layout()
save_plot(fig, "coefficient_dynamics.png")
plt.close(fig)

# 3. Singular values of G (Testing the Unitary Ceiling Constraint)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(np.linalg.svd(G_real, compute_uv=False), 'o-', label='Real Wavefront Codec ($G$)', lw=2, color='purple')
ax.plot(np.linalg.svd(G_intensity, compute_uv=False), 'x-', label='Intensity Codec ($G$)', lw=2, color='teal')
ax.axhline(y=1.0, color='gray', linestyle=':', label='Strict Unitary Boundary')
ax.set_title("Singular Values of the Extracted Shift Operator $G$")
ax.set_xlabel("Eigenmode Index")
ax.set_ylabel("Singular Magnitude ($\sigma$)")
ax.grid(True, linestyle=':')
ax.set_ylim(0.0, 1.5)
ax.legend()
save_plot(fig, "singular_values_G.png")
plt.close(fig)

# 4. Reconstruction MSE vs timestep
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(time, mse_real, label='Real Projection Channel', lw=2, color='royalblue')
ax.plot(time, mse_intensity, label='Intensity Density Channel', lw=2, color='forestgreen')
ax.set_title("Reconstruction Mean Squared Error (MSE) Stability Over Time")
ax.set_xlabel("Internal Timestep ($t$)")
ax.set_ylabel("Reconstruction Variance ($\sigma^2$)")
ax.grid(True, linestyle=':')
ax.legend()
save_plot(fig, "reconstruction_mse_timestep.png")
plt.close(fig)

# 5. Reconstruction MSE vs number of PCA components (Scale Cutoff Convergence)
mse_real_list: List[float] = []
mse_intensity_list: List[float] = []
for n in range(1, max_components+1):
    _, recon_r, _ = apply_pca(frames_real, n)
    _, recon_i, _ = apply_pca(frames_intensity, n)
    mse_real_list.append(float(np.mean((frames_real - recon_r)**2)))
    mse_intensity_list.append(float(np.mean((frames_intensity - recon_i)**2)))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(range(1, max_components+1), mse_real_list, 'o-', label='Real Component Stream', lw=2, color='indigo')
ax.plot(range(1, max_components+1), mse_intensity_list, 'x-', label='Intensity Component Stream', lw=2, color='chocolate')
ax.set_title("Information Decay: MSE vs. Codec Complexity Budget ($K$)")
ax.set_xlabel("Number of Transmitted Base Modes ($K$)")
ax.set_ylabel("Global Mean Squared Residual Error")
ax.grid(True, linestyle=':')
ax.set_yscale('log') # Log scale helps emphasize smooth info-theoretic asymptotic decay boundaries
ax.legend()
save_plot(fig, "reconstruction_mse_components.png")
plt.close(fig)

# ==============================================================================
# Terminal Metrics Output
# ==============================================================================
print("\n" + "="*50)
print("IAME EXPERIMENTAL PROOF-OF-CONCEPT METRICS")
print("="*50)
print(f"Emergent Unitarity Error Matrix (Real PCA)      : {unitarity_real:.6e}")
print(f"Emergent Unitarity Error Matrix (Intensity PCA) : {unitarity_intensity:.6e}")
print("="*50 + "\n")

