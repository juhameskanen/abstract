#!/usr/bin/env python3
"""
entropy_weighted_qft_integral.py
================================

Entropy-Regularized Quantum Field Theory Integral
-------------------------------------------------

This script tests whether ultraviolet (UV) divergences in quantum field
theory (QFT) can be suppressed by weighting momentum modes using
information-theoretic entropy.

The core idea is simple and falsifiable:

    • Standard QFT treats all momentum modes equally in the measure,
      leading to UV divergences.

    • Here, each momentum mode is mapped to a bitstring encoding a
      discretized field configuration.

    • The Shannon entropy of this bitstring (computed via QBitwave)
      is used to suppress low-information (structureless) configurations.

If entropy vanishes in the UV, the weighted integral may converge
without renormalization.

This script computes and compares:

    (1) The standard divergent vacuum fluctuation integral
    (2) The entropy-weighted version

and plots both as a function of the momentum cutoff Λ.

This is a *numerical experiment* designed to pass or fail cleanly.

Author:
    Juha Meskanen (concept)
    ChatGPT (implementation assistance)

License:
    MIT
"""

import numpy as np
import matplotlib.pyplot as plt

from qbitwave import QBitwave


# ---------------------------------------------------------------------
# Configuration parameters
# ---------------------------------------------------------------------

L = 256                # Bitstring length (information capacity)
bits_per_site = 8      # Bits used to encode each spatial sample
m = 1.0                # Scalar field mass
beta = 10.0            # Entropy suppression strength
k_max = 50.0           # UV cutoff Λ
n_k = 400              # Number of momentum samples
n_realizations = 500   # Monte Carlo realizations per k


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def generate_field_bitstring(k: float, L: int) -> list:
    """
    Generate a bitstring encoding a scalar field configuration
    associated with momentum scale k.

    Interpretation:
        • Low k → smooth, correlated structure
        • High k → random, uncorrelated noise

    This reflects the intuition that UV modes correspond to
    structureless information.

    Args:
        k (float): momentum magnitude
        L (int): bitstring length

    Returns:
        list[int]: bitstring of length L
    """
    x = np.linspace(0, 2 * np.pi, L, endpoint=False)

    # Smooth component dominates at low k
    smooth = np.sin(k * x)

    # Noise dominates at high k
    noise = np.random.normal(scale=k / (k + 1), size=L)

    field = smooth + noise

    # Quantize field values to bits
    field_norm = (field - field.min()) / (field.ptp() + 1e-12)
    integers = np.floor(field_norm * (2**bits_per_site - 1)).astype(int)

    bitstring = []
    for val in integers:
        bits = [(val >> i) & 1 for i in reversed(range(bits_per_site))]
        bitstring.extend(bits)

    return bitstring[:L]


def entropy_weight(k: float) -> float:
    """
    Compute entropy-based suppression weight for momentum k.

    This performs a Monte Carlo average over bitstring realizations
    to reduce sampling noise.

    Args:
        k (float): momentum magnitude

    Returns:
        float: exp(-beta * <H>)
    """
    entropies = []

    for _ in range(n_realizations):
        bits = generate_field_bitstring(k, L)
        qb = QBitwave(bitstring=bits)
        entropies.append(qb.bit_entropy())

    H_avg = np.mean(entropies)
    return np.exp(-beta * H_avg)


# ---------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------

def run_experiment():
    """
    Compute standard and entropy-weighted vacuum integrals.

    Returns:
        tuple of np.ndarray:
            k_values,
            standard_integral,
            entropy_weighted_integral
    """
    k_values = np.linspace(0.1, k_max, n_k)

    I_std = np.zeros_like(k_values)
    I_info = np.zeros_like(k_values)

    for i, k in enumerate(k_values):
        integrand = k**2 / np.sqrt(k**2 + m**2)
        weight = entropy_weight(k)

        I_std[i] = integrand
        I_info[i] = weight * integrand

        print(f"k={k:.2f}  H-weight={weight:.5f}")

    # Cumulative integrals
    dk = k_values[1] - k_values[0]
    return (
        k_values,
        np.cumsum(I_std) * dk,
        np.cumsum(I_info) * dk
    )


# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------

def plot_results(k, I_std, I_info):
    """
    Plot standard vs entropy-weighted integrals.

    Args:
        k (np.ndarray): momentum values
        I_std (np.ndarray): standard integral
        I_info (np.ndarray): entropy-weighted integral
    """
    plt.figure(figsize=(9, 6))

    plt.plot(k, I_std, label="Standard QFT integral", linewidth=2)
    plt.plot(k, I_info, label="Entropy-weighted integral", linewidth=2)

    plt.xlabel("Momentum cutoff Λ")
    plt.ylabel("Vacuum fluctuation integral")
    plt.title("Entropy-Regularized QFT Vacuum Integral")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    k, I_std, I_info = run_experiment()
    plot_results(k, I_std, I_info)
