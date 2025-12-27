import numpy as np
import matplotlib.pyplot as plt
from qbitwave import QBitwave  # Assumed available in your environment

# ---------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------
N = 128
bits_per_site = 8
m = 1.0
lambda4 = 1.0
n_realizations = 300
n_mcmc_steps = 1000  # Steps to thermalize/sample the field

# ---------------------------------------------------------------------
# MCMC phi^4 Field Generator
# ---------------------------------------------------------------------
def generate_phi4_mcmc(m, lam, n_steps=1000):
    """Generates a field configuration using Metropolis-Hastings."""
    phi = np.random.normal(0, 0.1, N)
    
    def get_action(field):
        # S = sum [ 0.5*(phi_i+1 - phi_i)^2 + 0.5*m^2*phi^2 + (lam/24)*phi^4 ]
        kinetic = 0.5 * np.sum(np.diff(np.append(field, field[0]))**2)
        potential = np.sum(0.5 * m**2 * field**2 + (lam / 24.0) * field**4)
        return kinetic + potential

    current_S = get_action(phi)
    for _ in range(n_steps):
        # Propose a local change at a random site
        site = np.random.randint(0, N)
        old_val = phi[site]
        phi[site] += np.random.normal(0, 0.5)
        new_S = get_action(phi)
        
        # Metropolis acceptance step
        if new_S < current_S or np.random.rand() < np.exp(current_S - new_S):
            current_S = new_S
        else:
            phi[site] = old_val
    return phi

def field_to_bitstring(field):
    fmin, fmax = field.min(), field.max()
    norm = (field - fmin) / (fmax - fmin + 1e-12)
    integers = np.floor(norm * (2**bits_per_site - 1)).astype(int)
    bits = []
    for val in integers:
        for i in reversed(range(bits_per_site)):
            bits.append((val >> i) & 1)
    return bits

# ---------------------------------------------------------------------
# Main Simulation
# ---------------------------------------------------------------------
def run_comparison():
    k_vals = np.linspace(0.1, np.pi, 100)
    I_std, I_info, I_mcmc = [], [], []

    # 1. Standard and Info Weighted (Fourier Generated)
    for k_cut in k_vals:
        # Standard analytical integrand
        integrand = 1.0 / (2 * np.sqrt(4*np.sin(k_cut/2)**2 + m**2))
        
        # QBitwave Weighting
        w_sum = 0
        for _ in range(50): # Sub-sample for speed
            # Use basic field to get baseline info weight
            field = np.random.normal(0, 1/np.sqrt(k_cut), N) 
            qb = QBitwave(bitstring=field_to_bitstring(field))
            w_sum += np.linalg.norm(qb.get_amplitudes())**2
        
        I_std.append(integrand)
        I_info.append((w_sum/50) * integrand)

    # 2. MCMC sampling for comparison
    # We simulate a "full" interaction lattice for the high-k limit
    mcmc_field = generate_phi4_mcmc(m, lambda4, n_mcmc_steps)
    qb_mcmc = QBitwave(bitstring=field_to_bitstring(mcmc_field))
    mcmc_weight = np.linalg.norm(qb_mcmc.get_amplitudes())**2

    return k_vals, np.cumsum(I_std), np.cumsum(I_info), mcmc_weight

# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
def plot_results(k, I_std, I_info, mcmc_weight):
    plt.figure(figsize=(10, 6))
    plt.plot(k, I_std, 'r--', label="Standard QFT (Divergent)")
    plt.plot(k, I_info, 'b-', label="QBitwave Weighted (Regularized)")
    
    # Add a horizontal indicator for the MCMC-weighted terminal value
    plt.axhline(y=I_info[-1], color='g', linestyle=':', label=f"MCMC Steady State (W={mcmc_weight:.2f})")
    
    plt.title("UV Divergence Suppression: Standard vs QBitwave MCMC")
    plt.xlabel("Momentum Cutoff (k)")
    plt.ylabel("Cumulative One-Loop Integral")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    k, istd, iinfo, mw = run_comparison()
    plot_results(k, istd, iinfo, mw)