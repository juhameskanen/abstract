"""
qft_divergence_test.py

QFT Divergence Test using QBitwave
----------------------------------

This module simulates a 1D scalar field on a lattice to test the
entropy-weighted suppression of UV divergences using the QBitwave framework.

High-frequency modes in the field correspond to high-information, poorly
compressible configurations. QBitwave automatically suppresses these
contributions via emergent wavefunction amplitudes.

The simulation generates multiple random realizations of the field,
maps them to bitstrings, constructs QBitwave objects, and computes
three key measures:
    1. Raw bitstring Shannon entropy
    2. Wavefunction compressibility
    3. Bit-wavefunction coherence (KL divergence)

Plots are produced to visualize how high-frequency modes are naturally
suppressed, illustrating the entropy-weighted path integral concept.

Author:
-------
(c) 2019-2025 Juha Meskanen
"""

from typing import List
import numpy as np
import matplotlib.pyplot as plt
from qbitwave import QBitwave


def field_to_bitstring(phi_x: np.ndarray, bits_per_site: int = 8) -> List[int]:
    """
    Convert a discretized scalar field to a flattened bitstring.

    Args:
        phi_x (np.ndarray): 1D array of field values in [-1, 1]
        bits_per_site (int): Number of bits used to encode each lattice site

    Returns:
        List[int]: Flattened bitstring representing the field
    """
    int_vals = ((phi_x + 1) / 2 * (2**bits_per_site - 1)).astype(int)
    bitstring = []
    for val in int_vals:
        bits = [(val >> i) & 1 for i in reversed(range(bits_per_site))]
        bitstring.extend(bits)
    return bitstring


def simulate_qft_divergence(L: int = 64,
                            n_realizations: int = 20,
                            k_step: int = 4,
                            bits_per_site: int = 8):
    """
    Run a simulation of a 1D scalar field to test QFT divergence suppression.

    Args:
        L (int): Number of lattice sites
        n_realizations (int): Number of random field realizations per k_max
        k_step (int): Step size for UV cutoff k_max values
        bits_per_site (int): Bits used to encode each lattice site

    Returns:
        dict: Dictionary with keys 'k_max', 'entropy_raw', 'compressibility', 'coherence'
    """
    k_max_values = np.arange(1, L//2 + 1, k_step)
    entropy_raw = []
    compressibility = []
    coherence_vals = []

    for k_max in k_max_values:
        e_entropy, e_compress, e_coherence = [], [], []

        for _ in range(n_realizations):
            # Generate random 1D scalar field
            x = np.arange(L)
            phi_x = np.zeros(L)
            for k in range(1, k_max + 1):
                A_k = np.random.randn()
                phi_x += A_k * np.sin(2 * np.pi * k * x / L)
            phi_x /= np.max(np.abs(phi_x)) + 1e-10  # normalize to [-1,1]

            # Convert to bitstring
            bitstring = field_to_bitstring(phi_x, bits_per_site=bits_per_site)

            # Construct QBitwave
            qbw = QBitwave(bitstring=bitstring)

            # Compute measures
            e_entropy.append(qbw.bit_entropy())
            e_compress.append(qbw.compressibility())
            e_coherence.append(qbw.coherence())

        # Store averages
        entropy_raw.append(np.mean(e_entropy))
        compressibility.append(np.mean(e_compress))
        coherence_vals.append(np.mean(e_coherence))

    return {
        "k_max": k_max_values,
        "entropy_raw": entropy_raw,
        "compressibility": compressibility,
        "coherence": coherence_vals
    }


def plot_simulation_results(results: dict):
    """
    Plot entropy, compressibility, and coherence as functions of UV cutoff.

    Args:
        results (dict): Output dictionary from `simulate_qft_divergence`
    """
    plt.figure(figsize=(10, 6))
    plt.plot(results['k_max'], results['entropy_raw'], label="Raw Bitstring Entropy")
    plt.plot(results['k_max'], results['compressibility'], label="Wavefunction Compressibility")
    plt.plot(results['k_max'], results['coherence'], label="Bit-Wavefunction Coherence")
    plt.xlabel("UV cutoff (k_max)")
    plt.ylabel("Measure")
    plt.title("Entropy-Weighted Suppression of Divergent Modes")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Run the simulation and plot
    sim_results = simulate_qft_divergence()
    plot_simulation_results(sim_results)
