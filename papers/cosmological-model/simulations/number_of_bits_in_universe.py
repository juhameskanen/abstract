import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. DEFINE CONSTANTS & SOLVE FOR LOG2(n)
# -------------------------------------------------------------------------
T_inflation = 1.86e8      # Planck times to end of inflation
D_inflation = 6.19e60     # Planck lengths (entropy) at end of inflation

# Age of the universe: ~13.8 billion years converted to Planck times
T_current = 8.08e60       

# From our derivation: log2(n) = log2(T) + D/T
log2_n = np.log2(T_inflation) + (D_inflation / T_inflation)

# -------------------------------------------------------------------------
# 2. DEFINE THE ENTROPY FUNCTION
# -------------------------------------------------------------------------
def calculate_entropy(t, log2_n):
    """
    Computes H(t) = t * (log2_n - log2(t))
    Using float64, this is safe from overflow up to t ~ 10^250 because 
    log2_n is ~10^52.
    """
    return t * (log2_n - np.log2(t))

# -------------------------------------------------------------------------
# 3. GENERATE PLOT DATA (Logarithmic scale)
# -------------------------------------------------------------------------
# We will plot from t = 10^0 to 10^65 Planck times
t_vector = np.logspace(0, 65, 1000)
H_vector = calculate_entropy(t_vector, log2_n)

# Compute values for our markers
H_inflation = calculate_entropy(T_inflation, log2_n)
H_current = calculate_entropy(T_current, log2_n)

# -------------------------------------------------------------------------
# 4. PLOTTING
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6), dpi=100)

# Plot the universal entropy curve
plt.plot(t_vector, H_vector, label="System Entropy $H(t)$", color="#1f77b4", linewidth=2.5)

# Plot Inflation Marker (Double backslashes escape LaTeX properly)
plt.scatter(T_inflation, H_inflation, color="#d62728", s=100, zorder=5, 
            label=f"End of Inflation\n$t = {T_inflation:.2e}$\n$H \\approx {H_inflation:.2e}$")

# Plot Current Day Marker
plt.scatter(T_current, H_current, color="#2ca02c", s=100, zorder=5, 
            label=f"Current Day (~13.8 Gyr)\n$t \\approx {T_current:.2e}$\n$H \\approx {H_current:.2e}$")

# Annotate the markers with lines pointing to the axes for clarity
plt.axvline(T_inflation, color="#d62728", linestyle="--", alpha=0.5)
plt.axhline(H_inflation, color="#d62728", linestyle="--", alpha=0.5)
plt.axvline(T_current, color="#2ca02c", linestyle="--", alpha=0.5)
plt.axhline(H_current, color="#2ca02c", linestyle="--", alpha=0.5)

# Formatting axes
plt.xscale('log')
plt.yscale('log')

plt.title("Cosmic Entropy Evolution (Ehrenfest Multi-Clock Model)", fontsize=14, pad=15)
plt.xlabel("Time (Planck Ticks $t$)", fontsize=12)
plt.ylabel("Total Entropy (Bits / Planck Volumes $H$)", fontsize=12)

# Fixed the bitwise XOR (^) bug here:
# Use float representations (1e65 and 1e115) instead of large integers
plt.xlim(1, 1e65)
plt.ylim(1, 1e115)

plt.grid(True, which="both", ls=":", alpha=0.5)
plt.legend(loc="upper left", fontsize=10, framealpha=0.9)
plt.tight_layout()

# Show the plot
plt.show()

# Print exact values
print(f"At End of Inflation:")
print(f"  t = {T_inflation:.4e} Planck times")
print(f"  H = {H_inflation:.4e} bits")
print("\nAt Present Day:")
print(f"  t = {T_current:.4e} Planck times")
print(f"  H = {H_current:.4e} bits")
