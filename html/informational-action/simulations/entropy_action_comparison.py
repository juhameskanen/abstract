import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def calculate_spectral_entropy(signal: np.ndarray) -> float:
    """
    Measures the 'Wavefunction Complexity' using Shannon Entropy of the PSD.
    """
    fft_vals = np.abs(np.fft.rfft(signal))
    psd = fft_vals**2
    psd_norm = psd / (np.sum(psd) + 1e-12)
    # Entropy H = -sum(p * log2(p))
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return float(entropy)

def calculate_compressibility(signal: np.ndarray, threshold_ratio: float = 0.01) -> float:
    """
    Measures the 'Measure' or 'Statistical Weight' of the path.
    1.0 = Pure Sine (Max Measure), 0.0 = White Noise (Zero Measure).
    """
    fft_vals = np.abs(np.fft.rfft(signal))
    max_val = np.max(fft_vals)
    # How many components are needed to describe this "universe"?
    significant_components = np.sum(fft_vals > (max_val * threshold_ratio))
    return 1.0 - (significant_components / len(fft_vals))

def run_experiment(args):
    noise_levels = np.linspace(args.min_noise, args.max_noise, args.points)
    x = np.linspace(0, 1, args.bits)
    h_max = np.log2(len(np.fft.rfft(x))) # The Shannon Ceiling
    
    results = {"noise": [], "action": [], "complexity": [], "measure": [], "h_max": h_max}

    print(f"Generating Data (Bits: {args.bits}, Samples: {args.samples})...")
    for n in noise_levels:
        batch_action, batch_complexity, batch_measure = [], [], []
        
        for _ in range(args.samples):
            # The path: Smooth Metric + Quantum Fluctuations
            path = np.sin(2 * np.pi * x) + np.random.normal(0, n, args.bits)
            
            # 1. Hawking Action (Sum of squared gradients)
            action = np.sum(np.gradient(path)**2)
            
            # 2. Informational Complexity (Entropy)
            complexity = calculate_spectral_entropy(path)
            
            # 3. Probabilistic Measure (Compressibility)
            measure = calculate_compressibility(path, args.threshold)
            
            batch_action.append(action)
            batch_complexity.append(complexity)
            batch_measure.append(measure)
            
        results["noise"].append(n)
        results["action"].append(np.mean(batch_action))
        results["complexity"].append(np.mean(batch_complexity))
        results["measure"].append(np.mean(batch_measure))
        print(f"Progress: {n/args.max_noise:.1%}", end='\r')

    return results

def visualize(results):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 13), sharex=True)

    # --- TOP PLOT: COST DUALITY ---
    ax1.plot(results["noise"], results["action"], 'r-', linewidth=2, label='Hawking Action (Euclidean I)')
    ax1.set_ylabel("Geometric Action (I)", color='r', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='r')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(results["noise"], results["complexity"], 'b--', linewidth=2, label='Spectral Entropy (H)')
    ax1_twin.axhline(y=results["h_max"], color='black', linestyle=':', alpha=0.5, label='Shannon Limit')
    ax1_twin.set_ylabel("Informational Entropy (Bits)", color='b', fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor='b')
    
    ax1.set_title("UPPER: Geometric Action vs. Informational Saturation", fontsize=14, pad=15)
    ax1.legend(loc='upper left')
    ax1_twin.legend(loc='lower right')

    # --- BOTTOM PLOT: MEASURE SUPPRESSION ---
    # We create a pseudo-Hawking measure (e^-I) to show the overlap
    action_array = np.array(results["action"])
    # Normalized e^-I for visual comparison
    hawking_weight = np.exp(-0.05 * (action_array - np.min(action_array))) 

    ax2.plot(results["noise"], results["measure"], 'g-', linewidth=3, label='QBitwave Compressibility (P)')
    ax2.plot(results["noise"], hawking_weight, 'm:', linewidth=2, label='Hawking Measure (e^-I approximation)')
    
    ax2.set_ylabel("Statistical Measure (Probability)", fontsize=12)
    ax2.set_xlabel("Geometric Irregularity (Noise Level \u03C3)", fontsize=12)
    ax2.set_title("LOWER: Emergent Born Rule & Path Suppression", fontsize=14, pad=15)
    ax2.legend()
    ax2.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bits', type=int, default=1024, help='Resolution of the bit-substrate')
    parser.add_argument('--samples', type=int, default=100, help='Samples per noise level')
    parser.add_argument('--points', type=int, default=60, help='Number of data points')
    parser.add_argument('--min_noise', type=float, default=0.01)
    parser.add_argument('--max_noise', type=float, default=2.5)
    parser.add_argument('--threshold', type=float, default=0.02, help='Sensitivity of compressibility')
    args = parser.parse_args()

    data = run_experiment(args)
    visualize(data)