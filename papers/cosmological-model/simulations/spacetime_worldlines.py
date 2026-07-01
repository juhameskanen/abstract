"""
Emergent Spacetime Worldlines — Faithful Implementation
===================================================================
Honors Paper III (Emergent Spacetime Structures from Zero-Entropy
Initial States) and Paper IV (Space-time Curvature as an
Information-Theoretic Structure).

Level hierarchy (as established in the morning session):
  L0  = total unfolded entropy = spacetime fabric (the rope itself)
  L_DM = dark matter: first-order structures to emerge from the
         entropy gradient. Hypothesis: lognormal-shaped, peaks early.
         No observational data exists for this curve; it is a
         PREDICTION of the framework, not an input.
  L_nu  = neutrinos: real cosmological data from JSON spectrum file.
  L_bar = baryons: real cosmological data from JSON spectrum file.
         (L3/atoms excluded per earlier discussion: same baryon
          number, just a different binding state — not a separate toll.)

Time-axis mapping (the ONLY sanctioned bridge, Paper III):
  t → ln t substitution means physical time maps log-uniformly onto
  bit-flip ticks:
      t_bf = t_bf_max * (ln(t_yr) - ln(t_min_yr))
                      / (ln(t_age_yr) - ln(t_min_yr))
  Anchors: t_bf=0 <-> Planck time (~5.4e-44 s = 1.7e-51 yr),
           t_bf=t_bf_max <-> age of universe (13.8e9 yr).
  This avoids the float64 overflow of t_phys=t0*exp(t_bf) over
  n*ln(n) ticks while preserving the log-time physics.

Amplitude scaling: number densities from the JSON cannot be
  converted to bit-counts without E=mc^2 (not yet derived in the
  framework). Each level is therefore normalized to its peak value
  and scaled by a CLI amplitude argument. This is an acknowledged
  placeholder, explicitly flagged.

Copyright 2026 - Juha Meskanen, The IAME Collaboration
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# ===========================================================================
# TIME-AXIS MAPPING  (Paper III: t -> ln t substitution)
# ===========================================================================

# Planck time in years (5.391e-44 s / 3.156e7 s/yr)
T_PLANCK_YR = 1.71e-51
T_AGE_YR    = 13.8e9

def years_to_tbf(t_yr, t_bf_max):
    """
    Map physical years -> bit-flip ticks via the log-uniform bridge
    sanctioned by Paper III's t -> ln(t) substitution.
    t_yr=T_PLANCK_YR  ->  t_bf=0
    t_yr=T_AGE_YR     ->  t_bf=t_bf_max
    """
    ln_min = np.log(T_PLANCK_YR)
    ln_max = np.log(T_AGE_YR)
    return t_bf_max * (np.log(np.maximum(t_yr, T_PLANCK_YR)) - ln_min) / (ln_max - ln_min)

def tbf_to_years(t_bf, t_bf_max):
    """Inverse of years_to_tbf."""
    ln_min = np.log(T_PLANCK_YR)
    ln_max = np.log(T_AGE_YR)
    return np.exp(ln_min + t_bf / t_bf_max * (ln_max - ln_min))


# ===========================================================================
# JSON SPECTRUM LOADER
# ===========================================================================

class JsonSpectrumLoader:
    """
    Loads the observed baryonic spectrum JSON and interpolates each
    level's number-density curve onto a t_bf grid.

    Amplitude scaling note: because E=mc^2 has not yet been derived
    within the framework, there is no principled way to convert
    number densities (per m^3) into bit-counts. Each curve is
    therefore normalized to peak=1 and scaled by the caller's
    amplitude argument. This is an explicit placeholder.
    """

    def __init__(self, json_path, t_bf_max):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.t_bf_max = t_bf_max

    def get_curve(self, level_key, t_bf_grid):
        """
        Return abundance curve for level_key (e.g. 'L1', 'L2'),
        normalized to peak=1, interpolated onto t_bf_grid.
        """
        pts = self.data["levels"][level_key]["abundances"]
        t_yr  = np.array([p["time_years"]    for p in pts])
        n_den = np.array([p["density_per_m3"] for p in pts])

        # Convert years -> t_bf
        t_bf_pts = years_to_tbf(t_yr, self.t_bf_max)

        # Interpolate in log-density space (spans many orders of magnitude)
        log_n = np.log10(n_den + 1e-300)
        order = np.argsort(t_bf_pts)
        interp = PchipInterpolator(t_bf_pts[order], log_n[order],
                                   extrapolate=False)

        log_n_grid = interp(t_bf_grid)
        # Outside data range -> 0 (no extrapolation: don't invent data)
        n_grid = np.where(np.isfinite(log_n_grid),
                          10.0 ** log_n_grid, 0.0)
        n_grid = np.clip(n_grid, 0.0, None)

        # Normalize to peak=1
        peak = n_grid.max()
        if peak > 0:
            n_grid /= peak
        return n_grid


# ===========================================================================
# CORE COUNTING-EQUATION MACHINERY
# ===========================================================================

def entropy_saturation(t_bf, k_rate, I_max=1.0):
    """
    S(t) = I_max * (1 - exp(-k*t))
    Paper III, sec. 'Comparison to General Relativity'.
    """
    return I_max * (1.0 - np.exp(-k_rate * t_bf))


def lognormal_abundance(t_bf, peak_t_bf, sigma_ln, amplitude, k_rate):
    """
    Lognormal-shaped abundance curve gated by the entropy gradient.
    Used ONLY for dark matter (L_DM), for which no observational data
    exists — this is a PREDICTION/HYPOTHESIS of the framework.
    """
    t = np.maximum(t_bf, 1e-12)
    mu_ln = np.log(peak_t_bf) + sigma_ln ** 2
    shape = (np.exp(-(np.log(t) - mu_ln) ** 2 / (2 * sigma_ln ** 2))
             / (t * sigma_ln * np.sqrt(2 * np.pi)))
    gate  = np.exp(-k_rate * t_bf)
    gated = shape * gate
    peak  = gated.max()
    return gated / peak * amplitude if peak > 0 else gated


# ===========================================================================
# SIMULATION
# ===========================================================================

def run_simulation(args):
    n = args.n_bits
    t_bf_max = args.t_bf_max if args.t_bf_max is not None else n * np.log(n)
    k_rate   = args.k_rate   if args.k_rate   is not None else (
        -np.log(1.0 - args.sat_fraction) / t_bf_max)

    t_bf    = np.linspace(0.0, t_bf_max, args.steps)
    S       = entropy_saturation(t_bf, k_rate)
    unfolded = S * n

    # --- L_DM: dark matter hypothesis (lognormal) ---
    N_dm = lognormal_abundance(
        t_bf, args.peak_dm_frac * t_bf_max,
        args.sigma_dm, args.amp_dm, k_rate)

    # --- L_nu, L_bar: real data from JSON (or fallback lognormals) ---
    if args.spectrum:
        loader = JsonSpectrumLoader(args.spectrum, t_bf_max)
        N_nu  = loader.get_curve("L1", t_bf) * args.amp_nu
        N_bar = loader.get_curve("L2", t_bf) * args.amp_bar
        data_source = f"JSON: {args.spectrum}"
    else:
        # Fallback: lognormals with explicit warning
        print("WARNING: no --spectrum JSON supplied; using illustrative "
              "lognormals for L_nu and L_bar.")
        N_nu  = lognormal_abundance(
            t_bf, args.peak_nu_frac  * t_bf_max, args.sigma_nu,
            args.amp_nu,  k_rate)
        N_bar = lognormal_abundance(
            t_bf, args.peak_bar_frac * t_bf_max, args.sigma_bar,
            args.amp_bar, k_rate)
        data_source = "illustrative lognormals (no JSON supplied)"

    # --- Bits consumed: DM + neutrinos + baryons ---
    # Amplitude scaling is a placeholder until E=mc^2 is derived.
    m_bits  = args.w_dm * N_dm + args.w_nu * N_nu + args.w_bar * N_bar
    k_count = N_dm + N_nu + N_bar

    overshoot = m_bits - unfolded
    if np.any(overshoot > 0):
        print(f"WARNING: structure demand exceeds unfolded budget on "
              f"{np.mean(overshoot>0):.1%} of timeline "
              f"(max overshoot {overshoot.max():.2f} bits). "
              f"Clamping — reduce amplitudes or bit-widths.")
    m_bits = np.minimum(m_bits, unfolded)

    fabric = np.clip(unfolded - m_bits, 0.0, n)
    R      = np.clip((fabric + k_count) / n, 0.0, 1.0)

    return dict(t_bf=t_bf, t_bf_max=t_bf_max, k_rate=k_rate, S=S,
                unfolded=unfolded, N_dm=N_dm, N_nu=N_nu, N_bar=N_bar,
                m_bits=m_bits, k_count=k_count, fabric=fabric, R=R,
                data_source=data_source)


# ===========================================================================
# WORLDLINE CONSTRUCTION
# ===========================================================================

def build_worldlines(N_k, n_slots, seed):
    rng = np.random.default_rng(seed)
    y_comoving = np.linspace(-0.5, 0.5, n_slots)
    slot_rank  = rng.permutation(n_slots)
    N_frac     = N_k / max(N_k.max(), 1e-12)
    active_count = np.clip((N_frac * n_slots).astype(int), 0, n_slots)
    active_mask  = slot_rank[None, :] < active_count[:, None]
    return y_comoving, active_mask


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_results(sim, args):
    n    = args.n_bits
    t_bf = sim["t_bf"]
    envelope = sim["fabric"] / n

    y_dm,  active_dm  = build_worldlines(sim["N_dm"],  args.slots_dm,  seed=1)
    y_nu,  active_nu  = build_worldlines(sim["N_nu"],  args.slots_nu,  seed=2)
    y_bar, active_bar = build_worldlines(sim["N_bar"], args.slots_bar, seed=3)

    # x-axis tick labels in physical years for readability
    t_bf_max = sim["t_bf_max"]
    tick_years = [1e-40, 1e-30, 1e-20, 1e-10, 1e-4, 1e3, 1e9, 13.8e9]
    tick_tbf   = [years_to_tbf(t, t_bf_max) for t in tick_years]
    tick_labels = ["10⁻⁴⁰", "10⁻³⁰", "10⁻²⁰", "10⁻¹⁰",
                   "10⁻⁴", "10³", "10⁹", "13.8G"]
    # only keep ticks that fall inside the plot range
    valid = [(tb, lb) for tb, lb in zip(tick_tbf, tick_labels)
             if 0 <= tb <= t_bf_max]
    tick_tbf_v, tick_labels_v = zip(*valid) if valid else ([], [])

    fig, (ax_st, ax_met) = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle("Emergent Spacetime Worldlines — IAME Framework\n"
                 f"L0=fabric  L_DM=dark matter (hypothesis)  "
                 f"L_nu/L_bar={sim['data_source']}",
                 fontsize=11, fontweight="bold")

    # ---- Left: Minkowski spacetime diagram ----
    ax_st.set_facecolor("#020205")
    ax_st.set_xlabel("Physical time (years, log scale via t→ln t mapping)")
    ax_st.set_ylabel("Comoving y × fabric(t)/n   (spatial extent)")
    ax_st.set_xticks(tick_tbf_v)
    ax_st.set_xticklabels(tick_labels_v, fontsize=7)

    ax_st.fill_between(t_bf, -envelope/2, envelope/2,
                        color="gainsboro", alpha=0.15)
    ax_st.plot(t_bf,  envelope/2, color="white", lw=2.2, alpha=0.9)
    ax_st.plot(t_bf, -envelope/2, color="white", lw=2.2, alpha=0.9)

    # DM worldlines: yellow — early, dominant
    for i, y0 in enumerate(y_dm):
        mask = active_dm[:, i]
        if mask.any():
            ax_st.plot(t_bf[mask], y0*envelope[mask],
                        color="yellow", lw=0.5, alpha=0.25)
    # Neutrino worldlines: cyan
    for i, y0 in enumerate(y_nu):
        mask = active_nu[:, i]
        if mask.any():
            ax_st.plot(t_bf[mask], y0*envelope[mask],
                        color="cyan", lw=0.6, alpha=0.35)
    # Baryon worldlines: magenta
    for i, y0 in enumerate(y_bar):
        mask = active_bar[:, i]
        if mask.any():
            ax_st.plot(t_bf[mask], y0*envelope[mask],
                        color="magenta", lw=0.9, alpha=0.45)

    ax_st.plot([], [], color="white",   lw=2.2, label="L0 spacetime fabric")
    ax_st.plot([], [], color="yellow",  lw=1.2, label="L_DM dark matter [hypothesis]")
    ax_st.plot([], [], color="cyan",    lw=1.2, label="L_nu neutrinos [JSON data]")
    ax_st.plot([], [], color="magenta", lw=1.5, label="L_bar baryons [JSON data]")
    ax_st.legend(loc="upper left", facecolor="#111115", edgecolor="gray",
                  labelcolor="white", fontsize=8)
    ax_st.text(0.02, 0.02,
               f"2^{n:.0f} ≈ {2.0**n:.2e} Planck lengths at full saturation",
               transform=ax_st.transAxes, color="gray", fontsize=7)

    # ---- Right: metric curves ----
    ax_met.set_facecolor("#0a0a0a")
    ax_met.set_xlabel("Physical time (years, log scale)")
    ax_met.set_ylabel("Fraction of n  (normalized)")
    ax_met.set_xticks(tick_tbf_v)
    ax_met.set_xticklabels(tick_labels_v, fontsize=7)

    ax_met.plot(t_bf, sim["S"],            color="red",     lw=1.5, ls="--",
                label="S(t)  total entropy [exact]")
    ax_met.plot(t_bf, sim["fabric"]/n,     color="white",   lw=3.0,
                label="fabric(t)/n  L0 [bold]")
    ax_met.plot(t_bf, sim["R"],            color="orange",  lw=1.5,
                label="R(t) = (fabric+k)/n  [Paper IV]")
    ax_met.plot(t_bf, sim["N_dm"] / max(sim["N_dm"].max(), 1e-12),
                color="yellow",  lw=1.2, alpha=0.8,
                label="L_DM [normalized, hypothesis]")
    ax_met.plot(t_bf, sim["N_nu"] / max(sim["N_nu"].max(), 1e-12),
                color="cyan",    lw=1.2, alpha=0.8,
                label="L_nu [normalized, JSON data]")
    ax_met.plot(t_bf, sim["N_bar"] / max(sim["N_bar"].max(), 1e-12),
                color="magenta", lw=1.2, alpha=0.8,
                label="L_bar [normalized, JSON data]")

    ax_met.set_ylim(-0.05, 1.15)
    ax_met.legend(loc="upper left", fontsize=8, facecolor="#111115",
                   edgecolor="gray", labelcolor="white")
    ax_met.set_title(
        f"k={sim['k_rate']:.5f}  t_bf_max={sim['t_bf_max']:.0f}\n"
        f"NOTE: amplitudes are placeholders (E=mc² not yet derived)",
        fontsize=8)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, facecolor="white")
    print(f"Saved → {args.output}")
    print(f"t_bf_max = {sim['t_bf_max']:.2f}  (n·ln n for n={n})")
    print(f"fabric(t)/n range: [{(sim['fabric']/n).min():.4f}, "
          f"{(sim['fabric']/n).max():.4f}]")
    print(f"max m(t)/n consumed: {(sim['m_bits']/n).max():.4f}")
    print(f"Data source: {sim['data_source']}")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description="IAME emergent-spacetime worldline demonstrator.")

    # Universe parameters
    p.add_argument("--n_bits", type=float, default=184.0)
    p.add_argument("--t_bf_max", type=float, default=None,
                   help="Default: n·ln(n)")
    p.add_argument("--sat_fraction", type=float, default=0.99)
    p.add_argument("--k_rate", type=float, default=None)
    p.add_argument("--steps", type=int, default=3000)

    # JSON data source
    p.add_argument("--spectrum", type=str, default=None,
                   help="Path to observed_baryonic_spectrum.json. "
                        "If omitted, falls back to illustrative lognormals.")

    # L_DM: dark matter (lognormal hypothesis — no data exists)
    p.add_argument("--peak_dm_frac", type=float, default=0.10,
                   help="DM lognormal peak as fraction of t_bf_max "
                        "(peaks early — first structures to form).")
    p.add_argument("--sigma_dm",   type=float, default=0.5)
    p.add_argument("--amp_dm",     type=float, default=10.0,
                   help="DM amplitude (placeholder; no E=mc^2 yet).")
    p.add_argument("--w_dm",       type=float, default=2.0,
                   help="Bit-width per DM entity (placeholder).")

    # L_nu: neutrinos (from JSON L1)
    p.add_argument("--amp_nu",      type=float, default=6.0,
                   help="Neutrino amplitude scaler (placeholder).")
    p.add_argument("--w_nu",        type=float, default=1.5,
                   help="Bit-width per neutrino entity (placeholder).")
    p.add_argument("--peak_nu_frac",  type=float, default=0.15,
                   help="Fallback lognormal peak (only if no JSON).")
    p.add_argument("--sigma_nu",    type=float, default=0.5)

    # L_bar: baryons (from JSON L2)
    p.add_argument("--amp_bar",     type=float, default=5.0,
                   help="Baryon amplitude scaler (placeholder).")
    p.add_argument("--w_bar",       type=float, default=3.0,
                   help="Bit-width per baryon entity (placeholder).")
    p.add_argument("--peak_bar_frac", type=float, default=0.35,
                   help="Fallback lognormal peak (only if no JSON).")
    p.add_argument("--sigma_bar",   type=float, default=0.6)

    # Worldline rendering
    p.add_argument("--slots_dm",  type=int, default=50)
    p.add_argument("--slots_nu",  type=int, default=50)
    p.add_argument("--slots_bar", type=int, default=40)

    p.add_argument("--output", type=str, default="../figures/spacetime_worldlines.png")
    args = p.parse_args()

    sim = run_simulation(args)
    plot_results(sim, args)


if __name__ == "__main__":
    main()
