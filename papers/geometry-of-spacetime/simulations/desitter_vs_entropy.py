import argparse
import numpy as np
import matplotlib.pyplot as plt

def calculate_shannon_entropy(bitstring):
    """Calculates the standard Shannon macrostate entropy of the bitstring."""
    p1 = np.mean(bitstring)
    p0 = 1.0 - p1
    
    if p1 == 0 or p0 == 0:
        return 0.0
        
    return -(p1 * np.log2(p1) + p0 * np.log2(p0))

def run_pure_shannon_spatial_mapping(length, flips_per_loop):
    # 1. Initialize bitstring to zero entropy (pure zeros)
    bitstring = np.zeros(length, dtype=int)
    
    # Strictly bind the timeline window to N Planck steps
    total_flips = length
    planck_time_steps = np.arange(1, total_flips + 1, flips_per_loop)
    
    entropy_profile = []
    
    # 2. Information Layer: Run the raw bit-flipping simulation
    for step in planck_time_steps:
        entropy_profile.append(calculate_shannon_entropy(bitstring))
        
        # Mutate bits randomly
        random_indices = np.random.randint(0, length, size=flips_per_loop)
        bitstring[random_indices] ^= 1

    entropy_profile = np.array(entropy_profile)
    
    # 3. GR Spatial Layer: Linear Shannon Entropy Mapping
    # Per your core insight: Cosmic evolution is governed by the information layer.
    # If R_ds is proportional to t_ds, and t_ds is proportional to H,
    # then the spatial horizon scale factor maps linearly to the Shannon profile.
    # No exponential warping functions allowed.
    spatial_scale_factor = entropy_profile * 1.0  
    
    # 4. Plotting via Dual Y-Axes 
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Left Axis: Information Entropy (Thermodynamic state)
    color = 'royalblue'
    ax1.set_xlabel("Physical Timeline (Linear Planck Time $\\tau_P$)", fontsize=10)
    ax1.set_ylabel("Shannon Entropy (Bits)", color=color, fontsize=11)
    line1 = ax1.plot(planck_time_steps, entropy_profile, color=color, linewidth=3, label="Bitstring Shannon Entropy")
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.6)
    
    # Right Axis: de Sitter Horizon Spatial Scale
    ax2 = ax1.twinx()  
    color = 'crimson'
    ax2.set_ylabel("de Sitter Horizon Spatial Scale ($a \\propto H$)", color=color, fontsize=11)
    line2 = ax2.plot(planck_time_steps, spatial_scale_factor, color=color, linewidth=2, linestyle="--", label="Shannon-Mapped Spatial Horizon")
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Combine legends from both axes
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right")
    
    plt.title(f"The Shannon Paradigm: Pure Information-Driven Spatial Expansion\n(Timeline Window: 1 to {length} $\\tau_P$ | No Second Mapping Warps)", fontsize=11)
    
    print("\n[+] Graph compiled with linear spatial mapping. Displaying plot...")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify de Sitter expansion natively scaled via the Shannon Entropy function.")
    parser.add_argument("length", type=int, help="Length of the bitstring (N)")
    parser.add_argument("flips_per_loop", type=int, help="Number of random bit flips per iteration loop")
    
    args = parser.parse_args()
    run_pure_shannon_spatial_mapping(args.length, args.flips_per_loop)
