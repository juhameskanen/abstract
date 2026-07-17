from __future__ import annotations

import argparse
import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import comb

T_PLANCK_YR: float = 1.71e-51
T_AGE_YR: float = 13.8e9

FloatArray = NDArray[np.float64]

def years_to_tbf(t_yr: FloatArray | float, t_today: float) -> FloatArray | float:
    ln_min = np.log(T_PLANCK_YR)
    ln_max = np.log(T_AGE_YR)
    return t_today * (np.log(np.maximum(t_yr, T_PLANCK_YR)) - ln_min) / (ln_max - ln_min)

def p_of_tau(tau: FloatArray, k_rate: float) -> FloatArray:
    return 0.5 * (1.0 - np.exp(-k_rate * tau))

def analytical_entropy(rho: FloatArray) -> FloatArray:
    r = np.clip(rho, 1e-15, 1.0 - 1e-15)
    return -(r * np.log2(r) + (1.0 - r) * np.log2(1.0 - r))

def universal_structure_salience(rho: FloatArray, w: int = 20) -> FloatArray:
    """Computes the expected combinatorial surprise of localized configurations."""
    salience = np.zeros_like(rho)
    K_vals = np.arange(1, w + 1)
    combinations = comb(w, K_vals)
    
    for idx, r_global in enumerate(rho):
        if r_global <= 1e-12 or r_global >= 0.5 - 1e-4:
            salience[idx] = 0.0
            continue
            
        r_g = np.clip(r_global, 1e-12, 0.5)
        p_K = combinations * (r_g ** K_vals) * ((1.0 - r_g) ** (w - K_vals))
        p_K = np.clip(p_K, 1e-15, 1.0)
        
        salience[idx] = np.sum(-p_K * np.log(p_K))
        
    if np.max(salience) > 0:
        salience = salience / np.max(salience)
        
    return salience

def run_simulation(
    steps: int = 3000,
    t_max_bits_ratio: float = 0.5,
    g_factor: float = 1.0
):
    k_rate = 2.0  
    t_bf = np.linspace(1e-9, t_max_bits_ratio, steps)
    rho = p_of_tau(t_bf, k_rate)
    entropy = analytical_entropy(rho)

    # Calculate raw first-principles surprise curve
    p_typical_structure = universal_structure_salience(rho, w=20)
    
    # G acts as a structural resistance/damping exponent
    damped_salience = p_typical_structure ** g_factor
    
    # NEW PARADIGM: Spacetime IS the typicality curve
    spacetime_profile = damped_salience

    # Emergent characteristic coherence scale
    coherence_scale = 1.0 / np.maximum(rho, 1e-3)

    # Anthropic peak calculation anchoring "Now"
    peak_idx = int(np.argmax(spacetime_profile))
    t_bf_max_sim = t_bf[-1]
    t_today = (t_bf[peak_idx] / t_bf_max_sim) * 74.0

    return t_bf, entropy, spacetime_profile, coherence_scale, t_today, peak_idx

def plot_results(t_bf, entropy, spacetime_profile, coherence_scale, t_today, peak_idx, output_path):
    fig, (ax_st, ax_met) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Relational Spacetime Profile (Spacetime = Typicality)", fontsize=12, fontweight="bold")

    tick_years = np.array([1e-40, 1e-30, 1e-20, 1e-10, 1e-4, 1e3, 1e9, 13.8e9])
    tick_labels = ["10\u207b\u2074\u2070", "10\u207b\u00b3\u2070", "10\u207b\u00b2\u2070",
                   "10\u207b\u00b9\u2070", "10\u207b\u2074", "10\u00b3", "10\u2079", "13.8B yr"]
    
    tick_coords = years_to_tbf(tick_years, t_today)

    t_bf_max_sim = t_bf[-1]
    t_mapped_for_plot = (t_bf / t_bf_max_sim) * (t_today * (t_bf_max_sim / t_bf[peak_idx]))
    max_visual_limit = t_mapped_for_plot[-1]

    # --- Left Panel: Emergent Spatial Bubble (Minkowski View) ---
    ax_st.set_facecolor("#020205")
    ax_st.set_xlabel("Physical Time (years, log scale via t\u2192ln t)")
    ax_st.set_ylabel("Relational Spatial Volume Profile")
    
    ax_st.set_xlim(0, max_visual_limit)
    ax_st.set_xticks(tick_coords)
    ax_st.set_xticklabels(tick_labels, fontsize=7)
    
    # Fills the actual bubble expanding and contracting according to the profile
    ax_st.fill_between(t_mapped_for_plot, -spacetime_profile / 2, spacetime_profile / 2, color="gainsboro", alpha=0.15)
    ax_st.plot(t_mapped_for_plot, spacetime_profile / 2, color="white", lw=2)
    ax_st.plot(t_mapped_for_plot, -spacetime_profile / 2, color="white", lw=2)
    
    t_peak_visual = t_mapped_for_plot[peak_idx]
    ax_st.axvline(t_peak_visual, color="lime", lw=1.5, ls="--", label="Anthropic Peak (Now)")
    ax_st.legend(loc="upper right", facecolor="#111115", edgecolor="gray", labelcolor="white", fontsize=8)

    # --- Right Panel: Mechanics View ---
    ax_met.set_facecolor("#0a0a0a")
    ax_met.set_xlabel("Physical Time (years, log scale via t\u2192ln t)")
    ax_met.set_ylabel("Normalized Value")
    
    ax_met.set_xlim(0, max_visual_limit)
    ax_met.set_xticks(tick_coords)
    ax_met.set_xticklabels(tick_labels, fontsize=7)

    ax_met.plot(t_mapped_for_plot, entropy, color="red", lw=1.5, ls="--", label="Universal Entropy S(\u03c4)")
    ax_met.plot(t_mapped_for_plot, spacetime_profile, color="magenta", lw=2.5, label="Spacetime / Structural Salience Profile")
    ax_met.axvline(t_peak_visual, color="lime", lw=1.5, ls="--", label="Anthropic Peak")
    
    ax_scale = ax_met.twinx()
    ax_scale.plot(t_mapped_for_plot, coherence_scale, color="cyan", lw=1.0, ls=":", label="Coherence Scale (\u03bb)")
    ax_scale.set_ylabel("Coherence Length (\u03bb)", color="cyan")
    ax_scale.tick_params(axis='y', labelcolor='cyan')
    ax_scale.set_yscale('log')

    ax_met.set_ylim(-0.05, 1.05)
    ax_met.legend(loc="upper left", fontsize=8, facecolor="#111115", edgecolor="gray", labelcolor="white")
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved figure -> {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-G", "--g_factor", type=float, default=1.0)
    parser.add_argument("-t", "--t_max_bits", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="anthropic_spacetime.png")
    args = parser.parse_args()

    t_bf, entropy, spacetime_profile, coherence_scale, t_today, peak_idx = run_simulation(
        t_max_bits_ratio=args.t_max_bits, g_factor=args.g_factor
    )

    plot_results(t_bf, entropy, spacetime_profile, coherence_scale, t_today, peak_idx, args.output)

if __name__ == "__main__":
    main()
