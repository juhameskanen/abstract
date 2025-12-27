#!/usr/bin/env python3
"""
entropy_weighted_qft_lattice.py
================================

Entropy-Regularized QFT Test Using a Lattice Klein–Gordon Field
---------------------------------------------------------------

This script replaces synthetic field generators with a genuine
lattice discretization of a free scalar (Klein–Gordon) field.

Goal:
    Test whether Shannon entropy of field configurations collapses
    in the ultraviolet (UV), leading to convergence of otherwise
    divergent QFT integrals.

No hand-tuned noise. No entropy engineering.
If this passes, the effect is structural.

Author:
    Juha Meskanen 
"""

import numpy as np
import matplotlib.pyplot as plt

from qbitwave import QBitwave


# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------

N = 128               # Lattice sites
bits_per_site = 8     # Encoding resolution
m = 1.0               # Scalar mass
beta = 10.0           # Entropy suppression strength

k_max = np.pi         # Max lattice momentum
n_k = 200             # Momentum samples
n_realizations = 300  # Monte Carlo samples per k


# ---------------------------------------------------------------------
# Lattice Klein–Gordon field generator
# ---------------------------------------------------------------------

def generate_lattice_field(k_cut: float) -> np.ndarray:
    """
    Generate a lattice Klein–Gordon field configuration
    with modes up to momentum k_cut.

    Args:
        k_cut (float): UV momentum cutoff

    Returns:
        np.ndarray: real-space field configuration
    """
    field_k = np.zeros(N, dtype=np.complex128)

    for n in range(N):
        k = 2 * np.pi * n / N
        if k > k_cut and (2*np.pi - k) > k_cut:
            continue

        omega = np.sqrt(4 * np.sin(k / 2)**2 + m**2)
        sigma = 1.0 / np.sqrt(2 * omega)

        re = np.random.normal(scale=sigma)
        im = np.random.normal(scale=sigma)

        field_k[n] = re + 1j * im

    field_x = np.fft.ifft(field_k).real
    return field_x


def field_to_bitstring(field: np.ndarray) -> list:
    """
    Encode a real scalar field into a bitstring.

    Args:
        field (np.ndarray): field values

    Returns:
        list[int]: bitstring
    """
    fmin, fmax = field.min(), field.max()
    norm = (field - fmin) / (fmax - fmin + 1e-12)

    integers = np.floor(norm * (2**bits_per_site - 1)).astype(int)

    bits = []
    for val in integers:
        for i in reversed(range(bits_per_site)):
            bits.append((val >> i) & 1)

    return bits


def entropy_weight(k_cut: float) -> float:
    """
    Monte Carlo estimate of entropy-based weight.

    Args:
        k_cut (float): momentum cutoff

    Returns:
        float: exp(-beta * <H>)
    """
    entropies = []

    for _ in range(n_realizations):
        field = generate_lattice_field(k_cut)
        bits = field_to_bitstring(field)
        qb = QBitwave(bitstring=bits)
        entropies.append(qb.bit_entropy())

    H = np.mean(entropies)
    return np.exp(-beta * H)


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def run():
    k_vals = np.linspace(0.1, k_max, n_k)

    I_std = np.zeros(n_k)
    I_info = np.zeros(n_k)
    H_vals = np.zeros(n_k)

    for i, k in enumerate(k_vals):
        integrand = k**2 / np.sqrt(k**2 + m**2)

        entropies = []
        for _ in range(n_realizations):
            field = generate_lattice_field(k)
            bits = field_to_bitstring(field)
            qb = QBitwave(bitstring=bits)
            entropies.append(qb.bit_entropy())

        H = np.mean(entropies)
        weight = np.exp(-beta * H)

        I_std[i] = integrand
        I_info[i] = weight * integrand
        H_vals[i] = H

        print(f"k={k:.3f}  H={H:.4f}  weight={weight:.6f}")

    dk = k_vals[1] - k_vals[0]
    return (
        k_vals,
        np.cumsum(I_std) * dk,
        np.cumsum(I_info) * dk,
        H_vals
    )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot(k, I_std, I_info, H):
    fig, axs = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axs[0].plot(k, I_std, label="Standard integral")
    axs[0].plot(k, I_info, label="Entropy-weighted integral")
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
    k, I_std, I_info, H = run()
    plot(k, I_std, I_info, H)
