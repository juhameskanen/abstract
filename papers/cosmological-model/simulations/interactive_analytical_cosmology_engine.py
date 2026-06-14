"""
IAME Interactive Analytical Cosmology Engine (Conserved & Balanced Edition)
===========================================================================
An interactive field dashboard mapping lognormal particle spectrums to a 
strictly conserved relational spacetime metric. Uses dual-axis scaling
to prevent the matter spectrum from being visually crushed by the massive
total information pool.

Copyright 2026 - Juha Meskanen, The Abstract Universe Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def lognormal_field(t, amplitude, mu, sigma):
    """
    Generates a continuous cosmic density field over time.
    The amplitude parameter is mathematically scaled so it represents the 
    TRUE peak height of the curve, making slider adjustments intuitive.
    """
    if sigma <= 0: return np.zeros_like(t)
    t_safe = np.maximum(t, 1e-5)
    # Calculate the raw lognormal profile
    raw_field = (1.0 / (t_safe * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(t_safe) - mu)**2) / (2 * sigma**2))
    # Find the theoretical maximum value to normalize against
    peak_t = np.exp(mu - sigma**2)
    max_val = (1.0 / (peak_t * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(peak_t) - mu)**2) / (2 * sigma**2))
    
    return (raw_field / max_val) * amplitude

# Vectorize time domain across the universal lifespan
time_ticks = np.linspace(0.01, 1000, 1000)

# --- INITIAL COSMIC CONDITIONS ---
init_total_info = 5000.0   # Total finite bit budget of the universe
init_hadron     = [1200.0, 5.0, 0.5]  # Peak height, Mu, Sigma
init_atom       = [800.0,  5.8, 0.4]
init_compound   = [400.0,  6.5, 0.3]

# Create figure and setup layout
fig, (ax_fields, ax_expansion) = plt.subplots(1, 2, figsize=(15, 6.5))
plt.subplots_adjust(bottom=0.35)

fig.suptitle("IAME Field Cosmology: Strictly Conserved Relational Dashboard", fontsize=14, fontweight='bold')

# Create a twin axis for the left plot so matter fields don't get squashed
ax_matter = ax_fields.twinx()

# --- INITIAL FIELD CALCULATIONS WITH CONSERVATION ---
h_dens = lognormal_field(time_ticks, *init_hadron)
a_dens = lognormal_field(time_ticks, *init_atom)
c_dens = lognormal_field(time_ticks, *init_compound)

total_matter = h_dens + a_dens + c_dens
# Spacetime fabric is the EXACT leftover of the information pool
f_dens = np.maximum(0.1, init_total_info - total_matter)

cosmic_radius = total_matter / f_dens
if np.max(cosmic_radius) > 0:
    cosmic_radius /= np.max(cosmic_radius)

# --- INITIAL PLOTTING ---
# Primary Left Axis: Spacetime Fabric
line_f, = ax_fields.plot(time_ticks, f_dens, label="Spacetime Fabric (Vacuum)", color='gray', linestyle=':', lw=2)
ax_fields.set_ylabel("Fabric Information Density (Bits)", color='gray')
ax_fields.tick_params(axis='y', labelcolor='gray')
ax_fields.set_ylim(0, init_total_info * 1.1)

# Secondary Left Axis: Matter Spectrums
line_h, = ax_matter.plot(time_ticks, h_dens, label="L1 Hadrons", color='cyan', lw=1.5)
line_a, = ax_matter.plot(time_ticks, a_dens, label="L2 Atoms", color='magenta', lw=1.5)
line_c, = ax_matter.plot(time_ticks, c_dens, label="L3 Compounds", color='lime', lw=1.5)
ax_matter.set_ylabel("Matter Structure Density (Bits)", color='magenta')
ax_matter.tick_params(axis='y', labelcolor='magenta')

ax_fields.set_title("Conserved Cosmic Density Fields")
ax_fields.set_xlabel("Cosmic Time")
ax_fields.set_facecolor('#020205')

# Consolidate legends from both axes
lines_left = [line_f, line_h, line_a, line_c]
labels_left = [l.get_label() for l in lines_left]
ax_fields.legend(lines_left, labels_left, loc="upper right")

# Right Plot: Scale Factor R(t)
line_r, = ax_expansion.plot(time_ticks, cosmic_radius, color='orange', lw=3, label="Derived Scale Factor R(t)")
ax_expansion.set_title("Resulting Cosmological Expansion Profile")
ax_expansion.set_xlabel("Cosmic Time")
ax_expansion.set_ylabel("Relative Space Radius")
ax_expansion.set_facecolor('#020205')
ax_expansion.set_ylim(-0.05, 1.05)
ax_expansion.legend(loc="upper left")

# --- INTERACTIVE SLIDERS SETUP ---
ax_color = 'lightgoldenrodyellow'

# Global Pool Control
s_univ_info = Slider(plt.axes([0.1, 0.22, 0.3, 0.02], facecolor=ax_color), 'Total Info Pool', 2000.0, 10000.0, valinit=init_total_info)

# Matter Amplitudes (True Peak Heights)
s_h_amp = Slider(plt.axes([0.55, 0.22, 0.3, 0.02], facecolor=ax_color), 'Hadron Peak', 0.0, 2500.0, valinit=init_hadron[0])
s_a_amp = Slider(plt.axes([0.55, 0.18, 0.3, 0.02], facecolor=ax_color), 'Atom Peak', 0.0, 2000.0, valinit=init_atom[0])
s_c_amp = Slider(plt.axes([0.55, 0.14, 0.3, 0.02], facecolor=ax_color), 'Compound Peak', 0.0, 1500.0, valinit=init_compound[0])

# Global Matter Timing
s_m_mu  = Slider(plt.axes([0.55, 0.08, 0.3, 0.02], facecolor=ax_color), 'Matter Era Peak (Mu)', 3.0, 8.0, valinit=init_hadron[1])
s_m_sig = Slider(plt.axes([0.1, 0.08, 0.3, 0.02], facecolor=ax_color), 'Matter Spread (Sig)', 0.1, 2.0, valinit=init_hadron[2])

def update(val):
    total_pool = s_univ_info.val
    h_amplitude = s_h_amp.val
    a_amplitude = s_a_amp.val
    c_amplitude = s_c_amp.val
    m_mu = s_m_mu.val
    m_sig = s_m_sig.val
    
    # Generate true-peak matter fields
    new_h = lognormal_field(time_ticks, h_amplitude, m_mu, m_sig)
    new_a = lognormal_field(time_ticks, a_amplitude, m_mu + 0.8, m_sig * 0.8)
    new_c = lognormal_field(time_ticks, c_amplitude, m_mu + 1.5, m_sig * 0.6)
    
    new_total_matter = new_h + new_a + new_c
    
    # Enforce strict conservation law: fabric takes whatever is left over
    new_f = np.maximum(0.1, total_pool - new_total_matter)
    
    # Relational Scale Factor calculation
    new_radius = new_total_matter / new_f
    if np.max(new_radius) > 0:
        new_radius /= np.max(new_radius)
        
    # Update all line vectors
    line_f.set_ydata(new_f)
    line_h.set_ydata(new_h)
    line_a.set_ydata(new_a)
    line_c.set_ydata(new_c)
    line_r.set_ydata(new_radius)
    
    # Cleanly rescale the primary axis (Fabric)
    ax_fields.set_ylim(0, total_pool * 1.1)
    
    # Dynamic autoscaling for the secondary axis (Matter Spectrum)
    ax_matter.relim()
    ax_matter.autoscale_view(True, True, True)
    
    fig.canvas.draw_idle()

s_univ_info.on_changed(update)
s_h_amp.on_changed(update)
s_a_amp.on_changed(update)
s_c_amp.on_changed(update)
s_m_mu.on_changed(update)
s_m_sig.on_changed(update)

plt.show()
