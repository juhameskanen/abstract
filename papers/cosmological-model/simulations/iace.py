"""
IAME Cosmological Engine: Absolute Manual Scale Edition
===================================================================
An informational dashboard mapping structural sinks against a unified
Causal Planck Time timeline.

FIXES:
- Manual Right Y-Axis calculation (No more broken secondary_yaxis tools).
- Robust X-Axis label parser tracking both small seconds and massive years.
- Dynamic horizontal expansion (RUSH_RATE scales inversely with bit expansion).

Copyright 2026 - Juha Meskanen, The Abstract Universe Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def lognormal_profile(x, amplitude, peak_x, sigma):
    """Generates an informational matter peak mapped explicitly in Planck-time space."""
    if amplitude <= 0 or sigma <= 0: 
        return np.zeros_like(x)
    x_safe = np.maximum(x, 1e-5)
    mu = np.log(peak_x) + sigma**2
    raw = (1.0 / (x_safe * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x_safe) - mu)**2) / (2 * sigma**2))
    max_raw = (1.0 / (peak_x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(peak_x) - mu)**2) / (2 * sigma**2))
    return (raw / max_raw) * amplitude

# --- PHYSICAL SCALE ANCHORS ---
SCALE_FACTOR        = 1e43         # 1 Unit = 10^43 Planck Ticks
PLANCK_TIME_TO_SEC  = 5.3912e-44
PLANCK_LENGTH_TO_LY = 1.7084e-51
SEC_TO_YEAR         = 1.0 / (365.25 * 24 * 3600)

# Exact locations on our linear X-axis grid
MARKER_1S_X   = 1.0 / (PLANCK_TIME_TO_SEC * SCALE_FACTOR)        # ~1.855 units
MARKER_6M_X   = 360.0 / (PLANCK_TIME_TO_SEC * SCALE_FACTOR)      # ~667.7 units
MARKER_AGE_X  = 13.78e9 / (SCALE_FACTOR * PLANCK_TIME_TO_SEC * (365.25 * 24 * 3600))

# --- INITIAL VISUAL CONFIGURATION ---
init_n_max    = 3.0      # Sets the initial visual scale of the Y-axis
init_x_max    = 10.0     # Initial X-axis zoom

init_dm_amp   = 0.4     
init_dm_pos   = 0.5      
init_dm_sig   = 0.80     

init_had_amp  = 0.3      
init_had_pos  = 1.2      
init_had_sig  = 0.30     

init_neu_amp  = 0.5      
init_neu_pos  = MARKER_1S_X    
init_neu_sig  = 0.40     

# Create canvas workspace with an explicit dual twin-axis setup
fig, ax = plt.subplots(figsize=(14, 9.5))
ax_ly = ax.twinx()  # Dedicated right Y-axis object
plt.subplots_adjust(bottom=0.45, right=0.82)
fig.suptitle("IAME: Fixed Precision Space-Time Horizon Engine", fontsize=14, fontweight='bold')

# Base Domain Setup
bit_flips = np.linspace(0.001, init_x_max, 1500)

# Calculate the pristine statistical expansion profile scaling with init_n_max
# Velocity dynamically dampens as the system expands, allowing the curve to stretch
dynamic_rush = 0.6 / (init_n_max / 3.0)
de_sitter = init_n_max * (1.0 - np.exp(-dynamic_rush * bit_flips))
dark_matter = lognormal_profile(bit_flips, init_dm_amp, init_dm_pos, init_dm_sig)
hadrons     = lognormal_profile(bit_flips, init_had_amp, init_had_pos, init_had_sig)
neutrinos   = lognormal_profile(bit_flips, init_neu_amp, init_neu_pos, init_neu_sig)
real_universe = de_sitter - (dark_matter + hadrons + neutrinos)

# Plot Active Geometry Layers
line_ds,   = ax.plot(bit_flips, de_sitter, label="Pure De Sitter Background S(t)", color='cyan', linestyle='--', lw=2)
line_dm,   = ax.plot(bit_flips, dark_matter, label="Dark Matter Density Sink", color='olive', linestyle=':', lw=1.5)
line_had,  = ax.plot(bit_flips, hadrons, label="Hadron Density Sink", color='magenta', linestyle=':', lw=1.5)
line_neu,  = ax.plot(bit_flips, neutrinos, label="Neutrino Density Sink", color='purple', linestyle=':', lw=1.5)
line_real, = ax.plot(bit_flips, real_universe, label="Real Universe Geometry", color='gold', lw=3)

# Configure Left Metric Ruler
ax.set_ylabel("Spatial Horizon Size / Entropy (Units of $10^{90}\\ bits / l_P$)", color='white')
ax.set_facecolor('#04040c')
ax.set_xlim(0, init_x_max)
ax.set_ylim(-0.1, init_n_max * 1.1)
ax.tick_params(colors='white')
ax.grid(True, color='#111125', linestyle='-')

# Configure Right Metric Ruler (Manually converted light-years display)
ax_ly.set_ylabel("Spatial Horizon Size (Light-Years)", color='lightgray')
ax_ly.set_ylim(-0.1 * (10**90) * PLANCK_LENGTH_TO_LY, (init_n_max * 1.1) * (10**90) * PLANCK_LENGTH_TO_LY)
ax_ly.tick_params(colors='lightgray')

# --- STATIC REAL-WORLD MARKERS ---
v_1s  = ax.axvline(x=MARKER_1S_X, color='red', linestyle='-.', alpha=0.7, lw=1.5)
v_6m  = ax.axvline(x=MARKER_6M_X, color='orange', linestyle='-.', alpha=0.7, lw=1.5)
v_age = ax.axvline(x=MARKER_AGE_X, color='green', linestyle='-.', alpha=0.7, lw=1.5)

# Text labels that sit statically next to their respective markers
t_1s  = ax.text(MARKER_1S_X + 0.1, init_n_max * 0.2, "1s", color='red', fontsize=10, fontweight='bold')
t_6m  = ax.text(MARKER_6M_X + 0.1, init_n_max * 0.4, "6 min", color='orange', fontsize=10, fontweight='bold')
t_age = ax.text(MARKER_AGE_X + 0.1, init_n_max * 0.6, "13.78B Years", color='green', fontsize=10, fontweight='bold')

def get_time_axis_string(max_units):
    """Clean precision parser preventing the floating-point truncation down to zero."""
    total_seconds = float(max_units) * SCALE_FACTOR * PLANCK_TIME_TO_SEC
    if total_seconds < 31557600.0:
        return f"True Cosmic Time ($10^{{43}}\\ \\tau_P$ Ticks)  |  [Active Window Max = {total_seconds:.4f} Seconds]"
    total_years = total_seconds * SEC_TO_YEAR
    if total_years >= 1e9:
        return f"True Cosmic Time ($10^{{43}}\\ \\tau_P$ Ticks)  |  [Active Window Max = {total_years/1e9:.4f} Billion Years]"
    return f"True Cosmic Time ($10^{{43}}\\ \\tau_P$ Ticks)  |  [Active Window Max = {total_years:.2f} Years]"

ax.set_xlabel(get_time_axis_string(init_x_max), color='white')
ax.legend(loc="upper left", facecolor='#090915', edgecolor='#333344', labelcolor='white')

# --- INTERACTIVE CONTROL SLIDERS ---
ax_color = 'lightgoldenrodyellow'

s_n_max = Slider(plt.axes([0.15, 0.36, 0.25, 0.02], facecolor=ax_color), 'Total Bits Max', 1.0, 100.0, valinit=init_n_max)

s_dm_amp  = Slider(plt.axes([0.55, 0.36, 0.12, 0.02], facecolor=ax_color), 'Dark Matter Amp', 0.0, 10.0, valinit=init_dm_amp)
s_dm_pos  = Slider(plt.axes([0.55, 0.30, 0.12, 0.02], facecolor=ax_color), 'DM Peak Pos', 0.01, 10.0, valinit=init_dm_pos)
s_dm_sig  = Slider(plt.axes([0.55, 0.24, 0.12, 0.02], facecolor=ax_color), 'DM Spread (Sig)', 0.05, 5.0, valinit=init_dm_sig)

s_had_amp = Slider(plt.axes([0.80, 0.36, 0.12, 0.02], facecolor=ax_color), 'Hadron Amp', 0.0, 10.0, valinit=init_had_amp)
s_had_pos = Slider(plt.axes([0.80, 0.30, 0.12, 0.02], facecolor=ax_color), 'Hadron Peak Pos', 0.1, 10.0, valinit=init_had_pos)
s_had_sig = Slider(plt.axes([0.80, 0.24, 0.12, 0.02], facecolor=ax_color), 'Hadron Spread (Sig)', 0.05, 5.0, valinit=init_had_sig)

s_neu_amp = Slider(plt.axes([0.25, 0.12, 0.15, 0.02], facecolor=ax_color), 'Neutrino Amp', 0.0, 10.0, valinit=init_neu_amp)
s_neu_pos = Slider(plt.axes([0.50, 0.12, 0.15, 0.02], facecolor=ax_color), 'Neutrino Peak Pos', 0.1, 10.0, valinit=init_neu_pos)
s_neu_sig = Slider(plt.axes([0.75, 0.12, 0.15, 0.02], facecolor=ax_color), 'Neutrino Spread (Sig)', 0.05, 5.0, valinit=init_neu_sig)

def update(val):
    n_max = s_n_max.val
    
    # Scale x axis limit linearly with n_max
    dynamic_max_time = n_max * 2.0
    dynamic_timeline = np.linspace(0.001, dynamic_max_time, 1500)
    
    # Stretch out the entropic saturation curve horizontally so it spreads with the space axis extension
    dynamic_rush = 0.6 / (n_max / 3.0)
    
    new_ds = n_max * (1.0 - np.exp(-dynamic_rush * dynamic_timeline))
    new_dm  = lognormal_profile(dynamic_timeline, s_dm_amp.val, s_dm_pos.val, s_dm_sig.val)
    new_had = lognormal_profile(dynamic_timeline, s_had_amp.val, s_had_pos.val, s_had_sig.val)
    new_neu = lognormal_profile(dynamic_timeline, s_neu_amp.val, s_neu_pos.val, s_neu_sig.val)
    new_real = new_ds - (new_dm + new_had + new_neu)
    
    line_ds.set_xdata(dynamic_timeline)
    line_ds.set_ydata(new_ds)
    line_dm.set_xdata(dynamic_timeline)
    line_dm.set_ydata(new_dm)
    line_had.set_xdata(dynamic_timeline)
    line_had.set_ydata(new_had)
    line_neu.set_xdata(dynamic_timeline)
    line_neu.set_ydata(new_neu)
    line_real.set_xdata(dynamic_timeline)
    line_real.set_ydata(new_real)
    
    # Clean left limits
    ax.set_xlim(0, dynamic_max_time)
    ax.set_ylim(-0.1, n_max * 1.1)
    
    # Manually re-anchor right Y-axis values to match the true physical order of magnitude
    ax_ly.set_ylim(-0.1 * (10**90) * PLANCK_LENGTH_TO_LY, (n_max * 1.1) * (10**90) * PLANCK_LENGTH_TO_LY)
    
    t_1s.set_y(n_max * 0.2)
    t_6m.set_y(n_max * 0.4)
    t_age.set_y(n_max * 0.6)
    
    ax.set_xlabel(get_time_axis_string(dynamic_max_time))
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
