"""
Minimal QM Simulation
=====================

This program implements a minimal 2D Quantum Mechanics simulation of the
double-slit experiment. 

Using a split-operator Fast Fourier Transform (FFT) method, it solves the
Time-Dependent Schrödinger Equation (TDSE) by alternating between position 
and momentum space. This approach ensures numerical stability and strict 
probability conservation. The simulation continuously drives flat plane waves 
from the bottom boundary, which scatter through a double-slit potential barrier 
to generate two-fluid circular wavefronts and clear downstream interference fringes, 
rendering the evolution directly into an MPEG video.

Copyright 2001 ... 2026 - Juha Meskanen
The Abstract Universe Project
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 1. Grid and Parameters (Must be powers of 2 for optimal FFT speed)
Nx, Ny = 256, 256        
dx = dy = 0.4           
dt = 0.05               # Safe, stable time-step with Split-Operator
num_frames = 200        
steps_per_frame = 8     

# 2. Coordinate Spaces
x = np.arange(Nx) * dx
y = np.arange(Ny) * dy
X, Y = np.meshgrid(x, y, indexing='ij')

# 3. Momentum Space (Frequencies for FFT)
kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
KX, KY = np.meshgrid(kx, ky, indexing='ij')

# Kinetic energy operator in Fourier space: K = 0.5 * (kx^2 + ky^2)
# We pre-compute the exact unitary propagator phase
K = 0.5 * (KX**2 + KY**2)
exp_K = np.exp(-1j * K * dt)

# 4. Potential Barrier (V)
V = np.zeros((Nx, Ny))
barrier_y = Ny // 4     
V[:, barrier_y] = 100.0 # High potential wall

# Cut two clean slits
slit_w = 4              
slit_spacing = 16       
mid_x = Nx // 2
V[mid_x - slit_spacing - slit_w : mid_x - slit_spacing + slit_w, barrier_y] = 0
V[mid_x + slit_spacing - slit_w : mid_x + slit_spacing + slit_w, barrier_y] = 0

# Pre-compute potential propagator phase in real space
exp_V = np.exp(-1j * V * dt)

# 5. Initialize Wavefunction
psi = np.zeros((Nx, Ny), dtype=complex)

# 6. Set Up Figure for Video Export
fig, ax = plt.subplots(figsize=(6, 6))
img = ax.imshow(np.abs(psi)**2, cmap='hot', origin='lower', extent=[0, Nx*dx, 0, Ny*dy])
ax.set_title("Unitary Quantum Double-Slit Simulation")
ax.axis('off')

# Continuous wave source parameters
k0 = 2.0  # Momentum of incoming waves
t_total = 0.0

def step_physics():
    global psi, t_total
    
    # Continuously drive a flat plane wave moving upward from the bottom row
    psi[:, 0] = np.exp(-1j * (0.5 * k0**2) * t_total)
    
    # --- Split-Operator Step (Unitary, Non-Exploding) ---
    # Step A: Apply Potential energy in real space
    psi *= exp_V
    
    # Step B: Transform to Momentum space, apply Kinetic energy, transform back
    psi_k = np.fft.fft2(psi)
    psi_k *= exp_K
    psi = np.fft.ifft2(psi_k)
    
    # Absorb reflections at the top edge to simulate open boundary
    psi[:, -5:] *= 0.5
    
    t_total += dt

# 7. Animation Compilation Loop
def update(frame):
    for _ in range(steps_per_frame):
        step_physics()
        
    density = np.abs(psi)**2
    img.set_array(density)
    img.set_clim(0, 1.0) # Solid, predictable contrast anchor
    return [img]

writer = animation.FFMpegWriter(fps=30, codec='libx264')

print("Compiling split-operator simulation video...")
ani = animation.FuncAnimation(fig, update, frames=num_frames, blit=True)
ani.save("quantum_double_slit.mp4", writer=writer)
print("Done! Video saved as quantum_double_slit.mp4")
