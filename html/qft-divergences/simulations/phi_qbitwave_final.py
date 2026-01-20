#!/usr/bin/env python3
"""
phi4_qbitwave_final.py
======================

1D φ⁴ scalar field lattice simulation with full QBitwave compression-based
weighting for one-loop integrals.

Goal:
    Test whether entropy-amplitude weighting fully suppresses UV divergences
    in QFT integrals using emergent wavefunction perspective.

Author:
    Juha Meskanen (concept)
    ChatGPT (implementation)
"""

import numpy as np
import matplotlib.pyplot as plt
from qbitwave import QBitwave

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
N = 128               # lattice sites
bits_per_site = 8
m = 1.0
lambda4 = 1.0
beta = 10.0
n_realizations = 300

# ---------------------------------------------------------------------
# Lattice φ⁴ field generator
# ---------------------------------------------------------------------
def generate_phi4_field(k_cut: float) -> np.ndarray:
    """Generate a lattice φ⁴ field with momentum cutoff."""
    field_k = np.zeros(N, dtype=np.complex128)

    for n in range(N):
        k = 2 * np.pi * n / N
        if k > k_cut and (2*np.pi - k) > k_cut:
            continue

        omega = np.sqrt(4*np.sin(k/2)**2 + m**2)
        sigma = 1.0 / np.sqrt(2*omega)

        re = np.random.normal(scale=sigma)
        im = np.random.normal(scale=sigma)
        field_k[n] = re + 1j * im

    field_x = np.fft.ifft(field_k).real

    # φ⁴ interaction term
    field_x += lambda4 * field_x**3 / 6.0 / N

    return field_x

def field_to_bitstring(field: np.ndarray) -> list:
    """Encode a real lattice field into a bitstring."""
    fmin, fmax = field.min(), field.max()
    norm = (field - fmin) / (fmax - fmin + 1e-12)
    integers = np.floor(norm * (2**bits_per_site - 1)).astype(int)

    bits = []
    for val in integers:
        for i in reversed(range(bits_per_site)):
            bits.append((val >> i) & 1)
    return bits

# ---------------------------------------------------------------------
# One-loop integral computation with QBitwave amplitude weighting
# ---------------------------------------------------------------------
def compute_one_loop_qbitwave():
    k_vals = np.linspace(0.1, np.pi, 200)

    I_std = np.zeros_like(k_vals)
    I_info = np.zeros_like(k_vals)
    H_vals = np.zeros_like(k_vals)

    for i, k_cut in enumerate(k_vals):
        integrand_sum = 0.0
        info_sum = 0.0
        entropies = []

        for _ in range(n_realizations):
            field = generate_phi4_field(k_cut)
            bits = field_to_bitstring(field)
            qb = QBitwave(bitstring=bits)

            # Compute effective amplitude weight from minimal program
            weight = np.linalg.norm(qb.get_amplitudes())**2

            H_vals[i] += qb.bit_entropy()  # optional diagnostics

            integrand = 1.0 / (2 * np.sqrt(4*np.sin(k_cut/2)**2 + m**2))
            integrand_sum += integrand
            info_sum += weight * integrand

        I_std[i] = integrand_sum / n_realizations
        I_info[i] = info_sum / n_realizations
        H_vals[i] /= n_realizations

    # cumulative integral (discrete sum)
    dk = k_vals[1] - k_vals[0]
    I_std_cum = np.cumsum(I_std) * dk
    I_info_cum = np.cumsum(I_info) * dk

    return k_vals, I_std_cum, I_info_cum, H_vals

# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_results(k, I_std, I_info, H):
    fig, axs = plt.subplots(2,1,figsize=(9,8), sharex=True)

    axs[0].plot(k, I_std, label="Standard one-loop integral")
    axs[0].plot(k, I_info, label="QBitwave amplitude-weighted")
    axs[0].set_ylabel("Integral value")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(k, H)
    axs[1].set_xlabel("Momentum cutoff k")
    axs[1].set_ylabel("Bit entropy H(k)")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    k, I_std, I_info, H = compute_one_loop_qbitwave()
    plot_results(k, I_std, I_info, H)
