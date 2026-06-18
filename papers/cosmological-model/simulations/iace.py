"""
IAME Cosmological Engine: Empirical Fine-Tuning Edition
===================================================================
An informational dashboard mapping Dark Matter, Hadron, and Neutrino 
structural sinks against the unwarped Causal Planck Time timeline.

Features expanded slider ranges for Position and Spread (Sigma) to allow 
precise replication of historical cosmic expansion and the Hubble Tension.

Copyright 2026 - Juha Meskanen, The Abstract Universe Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def lognormal_profile(x, amplitude, peak_x, sigma):
    """
    Generates a lognormal matter peak positioned explicitly in Planck-time space.
    Normalizes the peak height to precisely match the amplitude slider.
    """
    if amplitude <= 0 or sigma <= 0: 
        return np.zeros_like(x)
    
    x_safe = np.maximum(x, 1e-5)
    mu = np.log(peak_x) + sigma**2
    
    raw = (1.0 / (x_safe * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x_safe) - mu)**2) / (2 * sigma**2))
    max_raw = (1.0 / (peak_x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(peak_x) - mu)**2) / (2 * sigma**2))
    
    return (raw / max_raw) * amplitude

# Domain extended to 200 units to view the long-tail behavior where acceleration occurs
bit_flips = np.linspace(0.001, 200, 1500)
natural_slope = 0.03 

# --- INITIAL EMPIRICAL TUNING SET POINTS ---
init_n_max    = 60.0     
init_dm_amp   = 12.0     # Dark matter dominates the early structure
init_dm_pos   = 2.0      
init_dm_sig   = 0.80     # Wide spread to simulate long-term stability

init_had_amp  = 8.0      # Sharp hadronization spike
init_had_pos  = 8.0      
init_had_sig  = 0.20     # Narrower spread (abrupt formation phase)

init_neu_amp  = 10.0     # Decoupling signature
init_neu_pos  = 25.0     
init_neu_sig  = 0.40     

# Create workspace
fig, ax = plt.subplots(figsize=(14, 9))
plt.subplots_adjust(bottom=0.45)
fig.suptitle("IAME: Empirical Tuning Dashboard & Metric Evolution", fontsize=14, fontweight='bold')

# --- INITIAL PROFILE GENERATION ---
de_sitter = init_n_max * (1.0 - np.exp(-natural_slope * bit_flips))

dark_matter = lognormal_profile(bit_flips, init_dm_amp, init_dm_pos, init_dm_sig)
hadrons     = lognormal_profile(bit_flips, init_had_amp, init_had_pos, init_had_sig)
neutrinos   = lognormal_profile(bit_flips, init_neu_amp, init_neu_pos, init_neu_sig)

real_universe = de_sitter - (dark_matter + hadrons + neutrinos)

# --- PLOTTING LAYERS ---
line_ds,   = ax.plot(bit_flips, de_sitter, label="Pure De Sitter Background S(t)", color='cyan', linestyle='--', lw=2)
line_dm,   = ax.plot(bit_flips, dark_matter, label="Dark Matter Profile Sink", color='olive', linestyle=':', lw=1.5)
line_had,  = ax.plot(bit_flips, hadrons, label="Hadron Density Sink", color='magenta', linestyle=':', lw=1.5)
line_neu,  = ax.plot(bit_flips, neutrinos, label="Neutrino Density Sink", color='purple', linestyle=':', lw=1.5)
line_real, = ax.plot(bit_flips, real_universe, label="Real Universe Geometry (Emergent Space)", color='gold', lw=3)

# Configure Viewport Geometry
ax.set_xlabel("True Cosmic Time (Planck Ticks)")
ax.set_ylabel("Spatial Horizon Size (Holographic Bit Capacity)")
ax.set_facecolor('#04040c')
ax.set_ylim(-5, init_n_max * 1.1)
ax.grid(True, color='#111125', linestyle='-')
ax.legend(loc="upper left", facecolor='#090915', edgecolor='#333344', labelcolor='white')

# --- INTERACTIVE CONTROL SLIDERS (HIGH-PRECISION RANGES) ---
ax_color = 'lightgoldenrodyellow'

# Total Bit Budget Slider
s_n_max = Slider(plt.axes([0.15, 0.36, 0.25, 0.02], facecolor=ax_color), 'Total Bits Max', 20.0, 200.0, valinit=init_n_max)

# Dark Matter Controls (Column 1)
s_dm_amp  = Slider(plt.axes([0.55, 0.36, 0.12, 0.02], facecolor=ax_color), 'Dark Matter Amp', 0.0, 30.0, valinit=init_dm_amp)
s_dm_pos  = Slider(plt.axes([0.55, 0.30, 0.12, 0.02], facecolor=ax_color), 'DM Peak Pos', 0.1, 30.0, valinit=init_dm_pos)
s_dm_sig  = Slider(plt.axes([0.55, 0.24, 0.12, 0.02], facecolor=ax_color), 'DM Spread (Sig)', 0.05, 2.5, valinit=init_dm_sig)

# Hadron Controls (Column 2)
s_had_amp = Slider(plt.axes([0.80, 0.36, 0.12, 0.02], facecolor=ax_color), 'Hadron Amp', 0.0, 30.0, valinit=init_had_amp)
s_had_pos = Slider(plt.axes([0.80, 0.30, 0.12, 0.02], facecolor=ax_color), 'Hadron Peak Pos', 1.0, 50.0, valinit=init_had_pos)
s_had_sig = Slider(plt.axes([0.80, 0.24, 0.12, 0.02], facecolor=ax_color), 'Hadron Spread (Sig)', 0.05, 2.5, valinit=init_had_sig)

# Neutrino Controls (Bottom Rows)
s_neu_amp = Slider(plt.axes([0.25, 0.12, 0.15, 0.02], facecolor=ax_color), 'Neutrino Amp', 0.0, 30.0, valinit=init_neu_amp)
s_neu_pos = Slider(plt.axes([0.50, 0.12, 0.15, 0.02], facecolor=ax_color), 'Neutrino Peak Pos', 5.0, 100.0, valinit=init_neu_pos)
s_neu_sig = Slider(plt.axes([0.75, 0.12, 0.15, 0.02], facecolor=ax_color), 'Neutrino Spread (Sig)', 0.05, 2.5, valinit=init_neu_sig)

def update(val):
    n_max = s_n_max.val
    
    new_ds = n_max * (1.0 - np.exp(-natural_slope * bit_flips))
    
    new_dm  = lognormal_profile(bit_flips, s_dm_amp.val, s_dm_pos.val, s_dm_sig.val)
    new_had = lognormal_profile(bit_flips, s_had_amp.val, s_had_pos.val, s_had_sig.val)
    new_neu = lognormal_profile(bit_flips, s_neu_amp.val, s_neu_pos.val, s_neu_sig.val)
    
    new_real = new_ds - (new_dm + new_had + new_neu)
    
    line_ds.set_ydata(new_ds)
    line_dm.set_ydata(new_dm)
    line_had.set_ydata(new_had)
    line_neu.set_ydata(new_neu)
    line_real.set_ydata(new_real)
    
    ax.set_ylim(-5, n_max * 1.1)
    fig.canvas.draw_idle()

s_n_max.on_changed(update)
s_dm_amp.on_changed(update)
s_dm_pos.on_changed(update)
s_dm_sig.on_changed(update)
s_had_amp.on_changed(update)
s_had_pos.on_changed(update)
s_had_sig.on_changed(update)
s_neu_amp.on_changed(update)
s_neu_pos.on_changed(update)
s_neu_sig.on_changed(update)

plt.show()
