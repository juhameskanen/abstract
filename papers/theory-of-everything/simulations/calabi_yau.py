import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def compute_spectral_cost(data):
    """
    Computes the 'bit-cost' of the manifold by analyzing its 
    frequency components via FFT.
    """
    # Perform FFT on the coordinates
    fft_vals = np.fft.fftn(data)
    magnitude = np.abs(fft_vals)
    
    # Normalize to treat as a probability distribution (for entropy)
    prob_dist = magnitude / np.sum(magnitude)
    
    # Spectral Entropy as a proxy for bit-cost (Complexity)
    # Higher entropy = more high-frequency modes = higher bit cost
    spectral_entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-12))
    return spectral_entropy

def generate_calabi_yau(n, k_max, steps=50):
    """
    Generates a projection of a quintic-style manifold.
    n: The degree of the polynomial (complexity of folds).
    k_max: The number of k-th roots to sum (internal structure).
    """
    phi = np.linspace(0, 2*np.pi, steps)
    theta = np.linspace(0, 2*np.pi, steps)
    phi, theta = np.meshgrid(phi, theta)

    # We treat the manifold as a complex signal
    # z1^n + z2^n = 1
    x, y, z = np.zeros_like(phi), np.zeros_like(phi), np.zeros_like(phi)
    
    for k in range(k_max):
        # Projecting different roots of the complex equation
        # This adds the "folds" that increase spectral cost
        x += np.cos(phi) * np.cos((theta + 2*np.pi*k)/n)
        y += np.sin(phi) * np.cos((theta + 2*np.pi*k)/n)
        z += np.sin((theta + 2*np.pi*k)/n)

    coords = np.stack([x, y, z])
    return x, y, z, coords

def main():
    parser = argparse.ArgumentParser(description="Calabi-Yau Spectral Complexity Tool")
    parser.add_argument("-n", type=int, default=5, help="Degree of the manifold (folds)")
    parser.add_argument("-k", type=int, default=2, help="Number of internal root structures")
    parser.add_argument("--res", type=int, default=50, help="Grid resolution")
    args = parser.parse_args()

    # Generate the manifold
    x, y, z, coords = generate_calabi_yau(args.n, args.k, args.res)
    
    # Compute the "Spectral Price"
    cost = compute_spectral_cost(coords)
    
    print(f"--- Spectral Analysis ---")
    print(f"Manifold Parameters: n={args.n}, k={args.k}")
    print(f"Computed Spectral Cost: {cost:.4f} bits")
    print(f"Measure (2^-L): {2**(-cost):.4e}")

    # Visual Output
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, cmap='magma', edgecolors='k', lw=0.1)
    ax.set_title(f"Complexity L = {cost:.2f} bits")
    ax.set_axis_off()
    plt.show()

if __name__ == "__main__":
    main()