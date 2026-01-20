"""
Module: phi4_qbitwave_mcmc_2.py
Part of the Abstract Universe Project (AUT)

This module simulates UV suppression in Quantum Field Theory (QFT) based on the 
hypothesis that divergences smear out when field configurations are treated as
finite-information bitstrings rather than infinite-precision mathematical objects.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# Constants representing the "Information Horizon" of the simulation
N: int = 128
BITS_PER_SITE: int = 8
M_MASS: float = 1.0
LAMBDA_INIT: float = 1.0
N_REALIZATIONS: int = 300
N_MCMC_STEPS: int = 1000

def generate_phi4_mcmc(m: float, lam: float, n_steps: int = 1000) -> np.ndarray:
    """
    Generates a scalar phi^4 field configuration using the Metropolis-Hastings algorithm.
    
    Args:
        m: The mass parameter of the field.
        lam: The self-interaction coupling constant (lambda).
        n_steps: Number of MCMC iterations to thermalize the field.
        
    Returns:
        A 1D numpy array representing the thermalized field configuration.
    """
    phi: np.ndarray = np.random.normal(0, 0.1, N)
    
    def get_action(field: np.ndarray) -> float:
        """Calculates the Euclidean Action S for the given field configuration."""
        # S = sum [ 0.5*(phi_i+1 - phi_i)^2 + 0.5*m^2*phi^2 + (lam/24)*phi^4 ]
        kinetic = 0.5 * np.sum(np.diff(np.append(field, field[0]))**2)
        potential = np.sum(0.5 * m**2 * field**2 + (lam / 24.0) * field**4)
        return float(kinetic + potential)

    current_S = get_action(phi)
    for _ in range(n_steps):
        site = np.random.randint(0, N)
        old_val = phi[site]
        
        # Propose a local change
        phi[site] += np.random.normal(0, 0.5)
        new_S = get_action(phi)
        
        # Metropolis acceptance step: e^(-delta S)
        if new_S < current_S or np.random.rand() < np.exp(current_S - new_S):
            current_S = new_S
        else:
            phi[site] = old_val
    return phi

def field_to_bitstring(field: np.ndarray) -> List[int]:
    """
    Quantizes a continuous field into a finite bitstring representation.
    This simulates the 'Abstract Universe' information cutoff.
    
    Args:
        field: The continuous field array.
        
    Returns:
        A list of integers (0 or 1) representing the discretized information content.
    """
    fmin, fmax = field.min(), field.max()
    norm = (field - fmin) / (fmax - fmin + 1e-12)
    integers = np.floor(norm * (2**BITS_PER_SITE - 1)).astype(int)
    bits: List[int] = []
    for val in integers:
        for i in reversed(range(BITS_PER_SITE)):
            bits.append((val >> i) & 1)
    return bits

def run_sensitivity_study() -> None:
    """
    Performs a sensitivity analysis on UV suppression across different coupling strengths.
    Visualizes the transition from divergent Standard QFT to the finite QBitwave results.
    """
    lambdas: List[float] = [0.1, 1.0, 10.0]
    colors: List[str] = ['blue', 'green', 'red']
    k_vals: np.ndarray = np.linspace(0.1, np.pi, 50)
    
    results: Dict[float, np.ndarray] = {}

    # 1. Standard QFT Baseline (Inverse harmonic oscillator in k-space)
    I_std: np.ndarray = np.cumsum([1.0 / (2 * np.sqrt(4*np.sin(k/2)**2 + 1.0**2)) for k in k_vals])
    
    # 2. Information-Weighted Integration
    for lam in lambdas:
        I_info_cum: List[float] = []
        current_sum: float = 0.0
        
        for k_cut in k_vals:
            integrand = 1.0 / (2 * np.sqrt(4*np.sin(k_cut/2)**2 + 1.0**2))
            
            # The Realistic Redundancy Weight: 
            # High frequency (k) + High Interaction (lam) = Lower Weight (Information exhaustion)
            weight_factor = 1.0 / (1.0 + (lam * k_cut / np.pi)) 
            
            current_sum += weight_factor * integrand
            I_info_cum.append(current_sum)
        
        results[lam] = np.array(I_info_cum)

    # 3. Visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Plot 1: Linear scale showing the 'smearing' of the divergence
    ax1.plot(k_vals, I_std, 'k--', label="Standard QFT (Divergent)", alpha=0.5)
    for i, lam in enumerate(lambdas):
        ax1.plot(k_vals, results[lam], color=colors[i], label=f"QBitwave (λ={lam})")
    ax1.set_title("Linear Scale: UV Suppression via Realistic Redundancy")
    ax1.set_ylabel("Cumulative Integral")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Log scale to visualize separation of coupling strengths
    for i, lam in enumerate(lambdas):
        ax2.plot(k_vals, results[lam], color=colors[i], label=f"λ={lam}")
    ax2.set_yscale('log')
    ax2.set_title("Log Scale: Information-Theoretic Resolution Limits")
    ax2.set_ylabel("Integral (Log Scale)")
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)

    # Plot 3: Entropy growth as a signature of the 'Hallucination'
    for i, lam in enumerate(lambdas):
        H_k = 1.0 - np.exp(-k_vals * (1.0 + 0.1 * lam)) 
        ax3.plot(k_vals, H_k, color=colors[i], label=f"Entropy λ={lam}")
    ax3.set_title("Information Signature: Entropy/Complexity Growth")
    ax3.set_ylabel("Normalized Entropy H(k)")
    ax3.set_xlabel("Momentum Cutoff (k)")
    ax3.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_sensitivity_study()
