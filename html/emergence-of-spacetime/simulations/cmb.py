import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# --- Simulation parameters ---
mu = 1.5       # location parameter of the lognormal
sigma = 0.5    # scale parameter
x = np.linspace(0.01, 1.0, 500)  # normalized spatial scale
y_pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))  # lognormal PDF

# Convert density to log scale for heatmap
log_density = np.log(y_pdf + 1e-8)  # avoid log(0)

# Make a 2D heatmap by repeating the vector
heatmap = np.tile(log_density, (100, 1))

# --- Particle counts vs entropy (toy model) ---
entropy = np.linspace(0, 1, 500)
particle_counts = np.exp(-(np.log(entropy + 0.01) - mu)**2 / (2 * sigma**2))
particle_counts *= 10  # scale counts for visualization

# --- Plotting ---
plt.figure(figsize=(10, 6))

# Heatmap of lognormal PDF
plt.imshow(heatmap, extent=[0, 1, 0, 3], origin='lower', aspect='auto', cmap='viridis', alpha=0.8)
plt.colorbar(label='Log Density of Emergent Structures')

# Overlay particle counts curve (scaled to fit Y-axis)
plt.plot(entropy, np.log(particle_counts + 1e-8), color='red', lw=2, label='Log Particle Count vs Entropy')

# Highlight the mode
plt.scatter([0.5], [1.8], color='yellow', s=120, edgecolor='black', label='Mode (0.5, 1.8)')

# Labels and title
plt.xlabel('Normalized Spatial Scale / Entropy')
plt.ylabel('Log of Structure Count / Log Density')
plt.title('Emergence of Structures from Increasing Entropy (Lognormal Distribution)')
plt.legend()
plt.show()
