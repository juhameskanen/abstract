import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import argparse

def main():
    """
    Cosmological Bit-Cost Calculator
    -------------------------------
    Maps the emergence of physical structures (Protons, Atoms) to the 
    total entropy budget of the universe (in bits).
    
    Logic:
    1. X-axis is Log10(Total Bits in Universe).
    2. mu (Complexity) is derived from historical epoch peaks.
    3. Gravity is modeled as the derivative of structural discovery (dN/dH).
    """
    
    parser = argparse.ArgumentParser(description="Cosmological Bit-Cost Calculator")
    parser.add_argument("--sigma", type=float, default=0.35, help="Sharpness of Phase Transition")
    args = parser.parse_args()

    # X-axis: Log10 of Universal Entropy (Total Bits)
    # Range: 50 (Early) to 130 (Future Heat Death)
    x_log10 = np.linspace(50, 130, 1000) 
    
    plt.figure(figsize=(14, 8))

    # --- Level 1: Protons (Hadron Epoch) ---
    # Target Peak: 10^80 bits. 
    # mu = ln(Peak) + sigma^2. We use ln(10^80) = 80 * ln(10)
    mu_l1 = (80 * np.log(10)) + args.sigma**2
    
    # Calculate lognormal PDF. scale = exp(mu)
    # We evaluate at 10^x_log10
    y_l1 = lognorm.pdf(10**x_log10, s=args.sigma, scale=np.exp(mu_l1))
    
    # --- Level 3: Atoms (Recombination) ---
    # Target Peak: 10^88 bits.
    mu_l3 = (88 * np.log(10)) + args.sigma**2
    y_l3 = lognorm.pdf(10**x_log10, s=args.sigma, scale=np.exp(mu_l3))

    # --- Normalization ---
    # We normalize to 1.0 to see the relative "Epochs" clearly
    y1_norm = y_l1 / np.max(y_l1)
    y3_norm = y_l3 / np.max(y_l3)
    
    # Plotting using raw strings (r"...") to prevent LaTeX escape errors
    plt.plot(x_log10, y1_norm, color='purple', lw=3, 
             label=r"Proton Emergence ($\mu \approx$ " + f"{mu_l1:.1f})")
    plt.plot(x_log10, y3_norm, color='blue', lw=3, 
             label=r"Atom Emergence ($\mu \approx$ " + f"{mu_l3:.1f})")

    # --- The Gravity Prediction (Derivative) ---
    # Gravity is the RATE of new structure discovery (dN/dH)
    gravity = np.gradient(y1_norm, x_log10)
    gravity_scaled = gravity / np.max(np.abs(gravity))
    plt.plot(x_log10, gravity_scaled, color='green', linestyle='--', alpha=0.6, 
             label=r"Gravitational Pressure ($\propto dN/dH$)")

    # --- Milestones & Annotations ---
    plt.axvline(80, color='gray', linestyle=':', alpha=0.5)
    plt.text(80.5, 0.5, "Hadron Epoch\n($10^{80}$ bits)", rotation=90, va='center')
    
    plt.axvline(88, color='gray', linestyle=':', alpha=0.5)
    plt.text(88.5, 0.5, "Recombination\n($10^{88}$ bits)", rotation=90, va='center')
    
    # Present Day: ~10^122 bits (Bekenstein-Hawking bound for Hubble volume)
    plt.axvline(122, color='red', lw=2, alpha=0.6)
    plt.text(122.5, 0.8, "Present Day\n($10^{122}$ bits)", color='red', fontweight='bold')

    # --- Formatting ---
    plt.title("Universal Evolution: Structural Emergence vs. Bit Budget", fontsize=16)
    plt.xlabel(r"Total Universal Entropy ($\log_{10}$ of Bits)", fontsize=14)
    plt.ylabel("Normalized Density / Force", fontsize=14)
    plt.grid(True, alpha=0.2)
    plt.legend(loc="upper left", frameon=True)
    
    # Add explanatory note
    plt.annotate('The "Muttering" Era:\nGravity approaches zero\nas discovery slows.', 
                 xy=(122, 0.05), xytext=(105, 0.2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()