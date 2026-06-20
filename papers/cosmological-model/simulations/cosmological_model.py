"""
IAME Cosmological Engine: Correct Counting Formula Edition
===================================================================
An informational dashboard mapping structural sinks against a unified
Causal Planck Time timeline.

The relational scale factor follows the counting argument from Paper IV:

    Resolution(t) = (n - m(t)) + k(t)
    R(t) = Resolution(t) / n

where n is the total bit count, m(t) = sum_j w_j * k_j(t) is the number
of bits consumed by matter structures, and k(t) = sum_j k_j(t) is the
total count of composite matter entities.

Each matter level j has bit-width w_j (set by slider). The net resolution
loss per entity is (w_j - 1): w_j bits consumed, 1 entity created.

Copyright 2026 - Juha Meskanen, The Abstract Universe Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---------------------------------------------------------------------------
# Planck unit conversion constants
# ---------------------------------------------------------------------------
PLANCK_TIME_SEC  = 5.3912e-44   # seconds per Planck time
PLANCK_LEN_M     = 1.6162e-35   # metres per Planck length
M_PER_LY         = 9.461e15     # metres per light-year

def lognormal_profile(x, amplitude, peak_x, sigma):
    """
    Lognormal entity count k_j(t) normalised so peak = amplitude.
    Represents the abundance of one matter hierarchy level.
    """
    if amplitude <= 0 or sigma <= 0:
        return np.zeros_like(x)
    x_safe = np.maximum(x, 1e-9)
    mu = np.log(peak_x) + sigma ** 2
    raw     = np.exp(-((np.log(x_safe) - mu) ** 2) / (2 * sigma ** 2)) / (x_safe * sigma * np.sqrt(2 * np.pi))
    raw_max = np.exp(-((np.log(peak_x)  - mu) ** 2) / (2 * sigma ** 2)) / (peak_x  * sigma * np.sqrt(2 * np.pi))
    return (raw / raw_max) * amplitude

def compute_R(t, n, k1, k2, k3, w1, w2, w3):
    """
    Relational scale factor from Paper IV counting argument.

        m(t)          = w1*k1 + w2*k2 + w3*k3   (bits consumed)
        free_fabric   = max(n - m(t), 0)          (free bits)
        k_total       = k1 + k2 + k3              (composite entities)
        Resolution(t) = free_fabric + k_total
        R(t)          = Resolution(t) / n

    Limits:
        k=0, m=0  =>  R = 1          (De Sitter)
        k=1, m=n  =>  R = 1/n ~ 0   (Schwarzschild)
    """
    m = w1 * k1 + w2 * k2 + w3 * k3
    free_fabric = np.maximum(n - m, 0.0)
    k_total     = k1 + k2 + k3
    resolution  = free_fabric + k_total
    return resolution / n

def de_sitter_curve(t, n):
    """
    Pure De Sitter background in units of n (normalised to 1).
    Entropy saturation curve S(t) = 1 - exp(-k*t) scaled by n,
    then divided by n => approaches 1 asymptotically.
    k chosen so curve reaches ~0.95 at t = n*ln(n) (Paper III time span).
    """
    t_span = max(n * np.log(max(n, 2)), 1.0)
    k = 3.0 / t_span
    return 1.0 - np.exp(-k * t)

# ---------------------------------------------------------------------------
# Initial parameters
# ---------------------------------------------------------------------------
init_n    = 184.0   # bits (Paper III value for observable universe)

# Matter level bit-widths (resolution loss per entity = w - 1)
init_w1, init_w2, init_w3 = 2.0, 4.0, 8.0

# Lognormal parameters: (amplitude in bits, peak position in Planck times, sigma)
# Amplitudes are fractions of n
init_dm_amp,  init_dm_pos,  init_dm_sig  = 0.10, 0.20, 0.80
init_had_amp, init_had_pos, init_had_sig = 0.15, 0.50, 0.30
init_neu_amp, init_neu_pos, init_neu_sig = 0.05, 1.80, 0.40

# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 9.5))
ax_r    = ax.twinx()   # right axis: light-years
plt.subplots_adjust(bottom=0.48, right=0.82)
fig.patch.set_facecolor('#04040c')
fig.suptitle("IAME: Relational Space-Time Horizon Engine (Paper IV)",
             fontsize=14, fontweight='bold', color='white')

def make_timeline(n):
    """x-axis runs from 0 to n*ln(n) Planck times (Paper III time span)."""
    t_max = n * np.log(max(n, 2))
    return np.linspace(1e-6, t_max, 2000), t_max

t, t_max = make_timeline(init_n)

# Compute initial curves
k1 = lognormal_profile(t, init_dm_amp  * init_n, init_dm_pos  * t_max, init_dm_sig)
k2 = lognormal_profile(t, init_had_amp * init_n, init_had_pos * t_max, init_had_sig)
k3 = lognormal_profile(t, init_neu_amp * init_n, init_neu_pos * t_max, init_neu_sig)
ds = de_sitter_curve(t, init_n)
R  = compute_R(t, init_n, k1, k2, k3, init_w1, init_w2, init_w3)

# Matter-dominated curve: De Sitter minus consumed bits fraction
# (intermediate view: bits consumed but relational entity count not yet added back)
m_frac          = (init_w1*k1 + init_w2*k2 + init_w3*k3) / init_n
matter_dominated = np.maximum(ds - m_frac, 0.0)

# Plot
line_ds,   = ax.plot(t, ds,              label="Pure De Sitter  $S(t) = 1 - e^{-kt}$",
                     color='cyan',    linestyle='--', lw=2)
line_md,   = ax.plot(t, matter_dominated, label="Matter-dominated  $S(t) - m(t)/n$",
                     color='deepskyblue', linestyle='-.', lw=1.8)
line_k1,   = ax.plot(t, k1/init_n,      label="Matter level $L_1$ (density sink / n)",
                     color='olive',   linestyle=':',  lw=1.5)
line_k2,   = ax.plot(t, k2/init_n,      label="Matter level $L_2$ (density sink / n)",
                     color='magenta', linestyle=':',  lw=1.5)
line_k3,   = ax.plot(t, k3/init_n,      label="Matter level $L_3$ (density sink / n)",
                     color='purple',  linestyle=':',  lw=1.5)
line_real, = ax.plot(t, R,               label="$R(t)$ — Relational scale factor",
                     color='gold',    lw=3)

# Reference markers at t = 1s, 6min, age of universe in Planck times
T_1S  = 1.0              / PLANCK_TIME_SEC
T_6M  = 360.0            / PLANCK_TIME_SEC
T_AGE = 13.78e9 * 365.25 * 86400 / PLANCK_TIME_SEC

for T, col, lbl in [(T_1S, 'red', '1 s'), (T_6M, 'orange', '6 min'),
                    (T_AGE, 'limegreen', '13.8 Gyr')]:
    ax.axvline(x=T, color=col, linestyle='-.', alpha=0.6, lw=1.2)
    ax.text(T * 1.01, 0.05, lbl, color=col, fontsize=9, fontweight='bold',
            transform=ax.get_xaxis_transform())

# Axes formatting
ax.set_facecolor('#04040c')
ax.set_xlim(0, t_max)
ax.set_ylim(-0.05, 1.15)
ax.set_xlabel(f"Bit-flip time  $t$  [Planck times $\\tau_P$]   "
              f"(range $0 \\to n\\ln n = {t_max:.2e}\\,\\tau_P$)",
              color='white')
ax.set_ylabel("Normalised resolution  $R(t) = \\mathrm{Resolution}(t)\\,/\\,n$",
              color='white')
ax.tick_params(colors='white')
ax.grid(True, color='#111125')
ax.legend(loc="upper left", facecolor='#090915', edgecolor='#333344',
          labelcolor='white', fontsize=9)

# Right axis: spatial horizon in light-years  (R * n * l_P converted to ly)
def r_to_ly(r_val, n):
    return r_val * n * PLANCK_LEN_M / M_PER_LY

ly_ticks = np.linspace(0, 1.15, 8)
ax_r.set_ylim(-0.05, 1.15)
ax_r.set_yticks(ly_ticks)
ax_r.set_yticklabels([f"{r_to_ly(v, init_n):.2e}" for v in ly_ticks], color='lightgray')
ax_r.set_ylabel("Spatial horizon  $R(t)\\cdot n\\cdot\\ell_P$  [light-years]",
                color='lightgray')

# ---------------------------------------------------------------------------
# Sliders
# ---------------------------------------------------------------------------
C = 'lightgoldenrodyellow'


s_n     = Slider(plt.axes([0.10, 0.40, 0.25, 0.018], facecolor=C), '$n$ (bits)',
                 10.0, 500.0, valinit=init_n, valstep=1.0)

s_dm_a  = Slider(plt.axes([0.50, 0.40, 0.12, 0.018], facecolor=C), '$L_1$ Amp',
                 0.0, 0.5, valinit=init_dm_amp)
s_dm_p  = Slider(plt.axes([0.50, 0.35, 0.12, 0.018], facecolor=C), '$L_1$ Peak',
                 0.01, 0.99, valinit=init_dm_pos)
s_dm_s  = Slider(plt.axes([0.50, 0.30, 0.12, 0.018], facecolor=C), '$L_1$ $\\sigma$',
                 0.05, 2.0, valinit=init_dm_sig)
s_w1    = Slider(plt.axes([0.50, 0.25, 0.12, 0.018], facecolor=C), '$w_1$ (bits)',
                 2.0, 20.0, valinit=init_w1, valstep=1.0)

s_had_a = Slider(plt.axes([0.70, 0.40, 0.12, 0.018], facecolor=C), '$L_2$ Amp',
                 0.0, 0.5, valinit=init_had_amp)
s_had_p = Slider(plt.axes([0.70, 0.35, 0.12, 0.018], facecolor=C), '$L_2$ Peak',
                 0.01, 0.99, valinit=init_had_pos)
s_had_s = Slider(plt.axes([0.70, 0.30, 0.12, 0.018], facecolor=C), '$L_2$ $\\sigma$',
                 0.05, 2.0, valinit=init_had_sig)
s_w2    = Slider(plt.axes([0.70, 0.25, 0.12, 0.018], facecolor=C), '$w_2$ (bits)',
                 2.0, 20.0, valinit=init_w2, valstep=1.0)

s_neu_a = Slider(plt.axes([0.10, 0.18, 0.12, 0.018], facecolor=C), '$L_3$ Amp',
                 0.0, 0.5, valinit=init_neu_amp)
s_neu_p = Slider(plt.axes([0.10, 0.13, 0.12, 0.018], facecolor=C), '$L_3$ Peak',
                 0.01, 0.99, valinit=init_neu_pos)
s_neu_s = Slider(plt.axes([0.10, 0.08, 0.12, 0.018], facecolor=C), '$L_3$ $\\sigma$',
                 0.05, 2.0, valinit=init_neu_sig)
s_w3    = Slider(plt.axes([0.10, 0.03, 0.12, 0.018], facecolor=C), '$w_3$ (bits)',
                 2.0, 20.0, valinit=init_w3, valstep=1.0)

def update(_):
    n    = s_n.val
    w1, w2, w3 = s_w1.val, s_w2.val, s_w3.val

    t_new, t_max_new = make_timeline(n)

    # Lognormal peaks are fractions of t_max so they scale with n
    k1_new = lognormal_profile(t_new, s_dm_a.val  * n, s_dm_p.val  * t_max_new, s_dm_s.val)
    k2_new = lognormal_profile(t_new, s_had_a.val * n, s_had_p.val * t_max_new, s_had_s.val)
    k3_new = lognormal_profile(t_new, s_neu_a.val * n, s_neu_p.val * t_max_new, s_neu_s.val)
    ds_new = de_sitter_curve(t_new, n)
    R_new  = compute_R(t_new, n, k1_new, k2_new, k3_new, w1, w2, w3)
    m_frac_new = (w1*k1_new + w2*k2_new + w3*k3_new) / n
    md_new = np.maximum(ds_new - m_frac_new, 0.0)

    line_ds.set_data(t_new, ds_new)
    line_md.set_data(t_new, md_new)
    line_k1.set_data(t_new, k1_new / n)
    line_k2.set_data(t_new, k2_new / n)
    line_k3.set_data(t_new, k3_new / n)
    line_real.set_data(t_new, R_new)

    ax.set_xlim(0, t_max_new)

    # Update right axis tick labels for new n
    ax_r.set_yticklabels([f"{r_to_ly(v, n):.2e}" for v in ly_ticks], color='lightgray')

    ax.set_xlabel(f"Bit-flip time  $t$  [Planck times $\\tau_P$]   "
                  f"(range $0 \\to n\\ln n = {t_max_new:.2e}\\,\\tau_P$)",
                  color='white')
    fig.canvas.draw_idle()

for s in [s_n, s_dm_a, s_dm_p, s_dm_s, s_w1,
          s_had_a, s_had_p, s_had_s, s_w2,
          s_neu_a, s_neu_p, s_neu_s, s_w3]:
    s.label.set_color(C)     # Fixes the label text (e.g., '$n$ (bits)')
    s.valtext.set_color(C)   # Fixes the numerical value text next to the slider
    s.on_changed(update)


    
plt.show()
