"""
Minimal GR Spacetime Curvature Simulation
========================================

This program visualizes a phenomenological model of spatial curvature in General
Relativity. It simulates a heavy mass moving through a 3D coordinate grid, 
displacing the grid vertices toward the center of mass using an inverse-distance 
geometric stretching function. This mimics the spatial distortion seen in the 
spatial slice of a Schwarzschild metric, rendering the output to an MP4 video.

Copyright 2001 ... 2026 - Juha Meskanen
The Abstract Universe Project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Grid Configuration
grid_size = 21         # Resolution of the spatial grid
box_range = 10         # Coordinate boundaries (-10 to +10)
num_frames = 120       # Animation duration

# Generate a uniform, unwarped 3D coordinate lattice
x = np.linspace(-box_range, box_range, grid_size)
y = np.linspace(-box_range, box_range, grid_size)
z = np.linspace(-box_range, box_range, grid_size)
X_orig, Y_orig, Z_orig = np.meshgrid(x, y, z, indexing='ij')

# 2. Setup 3D Matplotlib Environment (Fixed the tuple unpacking bug here)
fig, ax = plt.subplots(figsize=(8, 7), subplot_kw={"projection": "3d"})
ax.view_init(elev=25, azim=-45)

# Physics parameters
mass_strength = 4.5    # Determines the intensity of the grid bending
softening = 0.8        # Prevents infinite division singularity at the core
orbit_radius = 5.0

def update(frame):
    ax.cla() # Clear axis for the fresh state update
    ax.set_axis_off()
    ax.set_xlim(-box_range, box_range)
    ax.set_ylim(-box_range, box_range)
    ax.set_zlim(-box_range, box_range)
    
    # Calculate the moving mass position (Circular trajectory over time)
    theta = 2 * np.pi * (frame / num_frames)
    mx = orbit_radius * np.cos(theta)
    my = orbit_radius * np.sin(theta)
    mz = 0.0 # Keeping the mass in the XY plane
    
    # Compute Euclidean distance from every grid point to the mass center
    dx = X_orig - mx
    dy = Y_orig - my
    dz = Z_orig - mz
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Schwarzschild-inspired spatial displacement multiplier
    # Vertices are pulled inward toward the mass based on proximity
    displacement = mass_strength / (distance + softening)**2
    
    X_warped = X_orig - dx * displacement
    Y_warped = Y_orig - dy * displacement
    Z_warped = Z_orig - dz * displacement
    
    # 3. Render the Warped Spatial Slice
    # Slice through the center of the Z-axis to show clear cross-sectional bending
    mid_idx = grid_size // 2
    
    # Draw longitudinal and latitudinal grid lines of the distorted plane
    for i in range(grid_size):
        ax.plot(X_warped[i, :, mid_idx], Y_warped[i, :, mid_idx], Z_warped[i, :, mid_idx], 
                color='cyan', alpha=0.6, lw=1)
        ax.plot(X_warped[:, i, mid_idx], Y_warped[:, i, mid_idx], Z_warped[:, i, mid_idx], 
                color='cyan', alpha=0.6, lw=1)
        
    # Draw a few vertical tether lines to show the 3D depth pull
    for i in range(0, grid_size, 4):
        for j in range(0, grid_size, 4):
            ax.plot(X_warped[i, j, :], Y_warped[i, j, :], Z_warped[i, j, :], 
                    color='gray', alpha=0.2, lw=0.7)

    # Plot the mass object itself
    ax.scatter([mx], [my], [mz], color='white', s=180, edgecolors='blue', zorder=10)
    
    ax.set_title("Warped Spacetime Metric (Spatial Grid)", color='white', y=0.95)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    return []

# 4. Compile Video
writer = animation.FFMpegWriter(fps=30, codec='libx264')
print("Compiling GR metric distortion video...")
ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True)
ani.save("spacetime_curvature.mp4", writer=writer)
print("Done! Video saved as spacetime_curvature.mp4")
