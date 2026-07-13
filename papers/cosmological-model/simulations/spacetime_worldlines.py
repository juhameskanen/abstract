"""
Emergent Spacetime Worldlines — Faithful Implementation
===================================================================

Recursive emergence cascade: L_DM -> L_nu -> L_bar, each level
condensing from its parent when that parent reaches sufficient density.
All levels eventually return to zero at entropy saturation.
The JSON loader is retained as an optional --spectrum flag for
Lambda-CDM reference comparison only.

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
  bit-flip ticks. The observable window [T_Planck, T_age] maps to
  [0, t_today], NOT to [0, t_bf_max]. The remainder of the timeline
  [t_today, t_bf_max] is the universe's future — entropy not yet spent.
  t_today is estimated from theory + observation: the universe is
  currently accelerating, implying we have just passed the peak of the
  DM lognormal and are still in the young phase (S(t_today) < 0.5).
  Default: t_today = 74 bit-flip ticks (S ≈ 0.30), consistent with
  "we have more entropy left to spend than what we have consumed."

Amplitude scaling: number densities from the JSON cannot be
  converted to bit-counts without E=mc^2 (not yet derived in the
  framework). Each level is therefore normalized to its peak value
  and scaled by a CLI amplitude argument. This is an acknowledged
  placeholder, explicitly flagged.

Copyright 2026 - Juha Meskanen
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

T_PLANCK_YR = 1.71e-51
T_AGE_YR    = 13.8e9


# ===========================================================================
# TIME-AXIS MAPPING
# ===========================================================================

def years_to_tbf(t_yr, t_today):
    """[T_Planck, T_age] -> [0, t_today] log-uniformly (t -> ln t, Paper III)."""
    ln_min = np.log(T_PLANCK_YR)
    ln_max = np.log(T_AGE_YR)
    return t_today * (np.log(np.maximum(t_yr, T_PLANCK_YR)) - ln_min) / (ln_max - ln_min)


# ===========================================================================
# ENTROPY AND LOGNORMAL MACHINERY
# ===========================================================================

def entropy_saturation(t_bf, k_rate, I_max=1.0):
    """S(t) = I_max*(1 - exp(-k*t))  [Paper III, exact]."""
    return I_max * (1.0 - np.exp(-k_rate * t_bf))


def lognormal_gated(t_bf, peak_t_bf, sigma_ln, amplitude, k_rate,
                    parent=None):
    """
    Lognormal abundance curve gated by:
      (a) entropy gradient exp(-k*t)  — no structure from white noise
      (b) parent abundance (optional) — child can't precede its parent

    peak_t_bf : location of the lognormal MODE in bit-flip time
    parent    : normalized parent curve in [0,1]; child is multiplied
                by this so it cannot rise before the parent does
    """
    t = np.maximum(t_bf, 1e-12)
    mu_ln = np.log(peak_t_bf) + sigma_ln ** 2
    shape = (np.exp(-(np.log(t) - mu_ln) ** 2 / (2 * sigma_ln ** 2))
             / (t * sigma_ln * np.sqrt(2 * np.pi)))
    gate = np.exp(-k_rate * t_bf)
    if parent is not None:
        gate = gate * parent          # child gated by parent density
    gated = shape * gate
    peak = gated.max()
    return gated / peak * amplitude if peak > 0 else gated


# ===========================================================================
# OPTIONAL JSON LOADER (Lambda-CDM reference only)
# ===========================================================================

class JsonSpectrumLoader:
    def __init__(self, json_path, t_today):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.t_today = t_today

    def get_curve(self, level_key, t_bf_grid):
        pts   = self.data["levels"][level_key]["abundances"]
        t_yr  = np.array([p["time_years"]    for p in pts])
        n_den = np.array([p["density_per_m3"] for p in pts])
        t_bf_pts = years_to_tbf(t_yr, self.t_today)
        log_n = np.log10(n_den + 1e-300)
        order = np.argsort(t_bf_pts)
        interp = PchipInterpolator(t_bf_pts[order], log_n[order],
                                   extrapolate=False)
        log_n_grid = interp(t_bf_grid)
        t_bf_min_data = t_bf_pts[order][0]
        n_grid = np.where(
            np.isfinite(log_n_grid) & (t_bf_grid >= t_bf_min_data),
            10.0 ** log_n_grid, 0.0)
        n_grid = np.clip(n_grid, 0.0, None)
        peak = n_grid.max()
        return n_grid / peak if peak > 0 else n_grid


# ===========================================================================
# SIMULATION
# ===========================================================================

def run_simulation(args):
    n        = args.n_bits
    t_bf_max = args.t_bf_max if args.t_bf_max is not None else n * np.log(n)
    k_rate   = args.k_rate   if args.k_rate   is not None else (
               -np.log(1.0 - args.sat_fraction) / t_bf_max)
    t_today  = args.t_today

    t_bf     = np.linspace(0.0, t_bf_max, args.steps)
    S        = entropy_saturation(t_bf, k_rate)
    unfolded = S * n

    # --- L_DM: first-order condensate, driven by entropy gradient alone ---
    N_dm = lognormal_gated(
        t_bf, args.peak_dm * t_bf_max, args.sigma_dm,
        args.amp_dm, k_rate, parent=None)

    # --- L_nu: second-order, driven by entropy gradient AND L_DM density ---
    # Peak delayed by delta_nu relative to L_DM peak; width narrowed.
    peak_nu = args.peak_dm * t_bf_max + args.delta_nu * t_bf_max
    N_dm_norm = N_dm / max(N_dm.max(), 1e-12)
    N_nu = lognormal_gated(
        t_bf, peak_nu, args.sigma_dm * args.compression,
        args.amp_dm * args.ratio, k_rate, parent=N_dm_norm)

    # --- L_bar: third-order, driven by entropy gradient AND L_nu density ---
    peak_bar = peak_nu + args.delta_bar * t_bf_max
    N_nu_norm = N_nu / max(N_nu.max(), 1e-12)
    N_bar = lognormal_gated(
        t_bf, peak_bar, args.sigma_dm * args.compression ** 2,
        args.amp_dm * args.ratio ** 2, k_rate, parent=N_nu_norm)

    # --- Optional Lambda-CDM reference curves (not used in fabric calc) ---
    ref_nu = ref_bar = None
    if args.spectrum:
        loader  = JsonSpectrumLoader(args.spectrum, t_today)
        ref_nu  = loader.get_curve("L1", t_bf) * args.amp_dm * args.ratio
        ref_bar = loader.get_curve("L2", t_bf) * args.amp_dm * args.ratio ** 2

    # --- Bits consumed and fabric ---
    m_bits  = args.w_dm * N_dm + args.w_dm * args.ratio * N_nu + \
              args.w_dm * args.ratio ** 2 * N_bar
    k_count = N_dm + N_nu + N_bar

    overshoot = m_bits - unfolded
    if np.any(overshoot > 0):
        print(f"WARNING: overshoot on {np.mean(overshoot>0):.1%} of "
              f"timeline (max {overshoot.max():.2f} bits). Clamping.")
    m_bits  = np.minimum(m_bits, unfolded)
    fabric  = np.clip(unfolded - m_bits, 0.0, n)
    R       = np.clip((fabric + k_count) / n, 0.0, 1.0)

    # --- Ratio check at t_today ---
    idx_now   = np.argmin(np.abs(t_bf - t_today))
    dm_now    = N_dm[idx_now]
    bar_now   = N_bar[idx_now]
    fabric_now = fabric[idx_now]
    ratio_dm_bar    = dm_now  / max(bar_now,    1e-12)
    ratio_dm_fabric = dm_now  / max(fabric_now, 1e-12)

    return dict(t_bf=t_bf, t_bf_max=t_bf_max, t_today=t_today,
                k_rate=k_rate, S=S, unfolded=unfolded,
                N_dm=N_dm, N_nu=N_nu, N_bar=N_bar,
                ref_nu=ref_nu, ref_bar=ref_bar,
                m_bits=m_bits, k_count=k_count, fabric=fabric, R=R,
                ratio_dm_bar=ratio_dm_bar,
                ratio_dm_fabric=ratio_dm_fabric)


# ===========================================================================
# WORLDLINES
# ===========================================================================

def build_worldlines(N_k, n_slots, seed):
    rng          = np.random.default_rng(seed)
    y_comoving   = np.linspace(-0.5, 0.5, n_slots)
    slot_rank    = rng.permutation(n_slots)
    N_frac       = N_k / max(N_k.max(), 1e-12)
    active_count = np.clip((N_frac * n_slots).astype(int), 0, n_slots)
    active_mask  = slot_rank[None, :] < active_count[:, None]
    return y_comoving, active_mask


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_results(sim, args):
    n        = args.n_bits
    t_bf     = sim["t_bf"]
    t_bf_max = sim["t_bf_max"]
    envelope = sim["fabric"] / n

    y_dm,  active_dm  = build_worldlines(sim["N_dm"],  args.slots_dm,  seed=1)
    y_nu,  active_nu  = build_worldlines(sim["N_nu"],  args.slots_nu,  seed=2)
    y_bar, active_bar = build_worldlines(sim["N_bar"], args.slots_bar, seed=3)

    # x-axis ticks in physical years
    tick_years  = [1e-40, 1e-30, 1e-20, 1e-10, 1e-4, 1e3, 1e9, 13.8e9]
    tick_tbf    = [years_to_tbf(t, sim["t_today"]) for t in tick_years]
    tick_labels = ["10⁻⁴⁰","10⁻³⁰","10⁻²⁰","10⁻¹⁰","10⁻⁴","10³","10⁹","now"]
    valid = [(tb, lb) for tb, lb in zip(tick_tbf, tick_labels)
             if 0 <= tb <= t_bf_max]
    tick_tbf_v, tick_labels_v = zip(*valid) if valid else ([], [])

    ratio_str = (f"DM/bar={sim['ratio_dm_bar']:.1f}  "
                 f"DM/fabric={sim['ratio_dm_fabric']:.1f}  "
                 f"(target both ≈ 5)")

    fig, (ax_st, ax_met) = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle(
        "Emergent Spacetime Worldlines — Recursive Cascade\n"
        f"L_DM → L_nu → L_bar  |  compression={args.compression}  "
        f"ratio={args.ratio}  |  {ratio_str}",
        fontsize=10, fontweight="bold")

    # ---- Left: Minkowski diagram ----
    ax_st.set_facecolor("#020205")
    ax_st.set_xlabel("Physical time (years, log scale via t→ln t)")
    ax_st.set_ylabel("Comoving y × fabric(t)/n")
    ax_st.set_xticks(tick_tbf_v)
    ax_st.set_xticklabels(tick_labels_v, fontsize=7)

    ax_st.fill_between(t_bf, -envelope/2, envelope/2,
                        color="gainsboro", alpha=0.15)
    ax_st.plot(t_bf,  envelope/2, color="white", lw=2.2, alpha=0.9)
    ax_st.plot(t_bf, -envelope/2, color="white", lw=2.2, alpha=0.9)

    for i, y0 in enumerate(y_dm):
        mask = active_dm[:, i]
        if mask.any():
            ax_st.plot(t_bf[mask], y0*envelope[mask],
                        color="yellow", lw=0.5, alpha=0.25)
    for i, y0 in enumerate(y_nu):
        mask = active_nu[:, i]
        if mask.any():
            ax_st.plot(t_bf[mask], y0*envelope[mask],
                        color="cyan", lw=0.6, alpha=0.35)
    for i, y0 in enumerate(y_bar):
        mask = active_bar[:, i]
        if mask.any():
            ax_st.plot(t_bf[mask], y0*envelope[mask],
                        color="magenta", lw=0.9, alpha=0.45)

    ax_st.plot([], [], color="white",   lw=2,   label="L0 spacetime fabric")
    ax_st.plot([], [], color="yellow",  lw=1.2, label="L_DM dark matter")
    ax_st.plot([], [], color="cyan",    lw=1.2, label="L_nu neutrinos")
    ax_st.plot([], [], color="magenta", lw=1.5, label="L_bar baryons")
    ax_st.plot([], [], color="lime",    lw=1.5, ls="--",
                label=f"now (t={sim['t_today']:.0f})")

    t_now    = sim["t_today"]
    env_now  = np.interp(t_now, t_bf, envelope)
    ax_st.axvline(t_now, color="lime", lw=1.5, ls="--", alpha=0.85)
    ax_st.annotate("now", xy=(t_now, env_now*0.48),
                   xytext=(t_now + t_bf_max*0.02, env_now*0.38),
                   color="lime", fontsize=8,
                   arrowprops=dict(arrowstyle="->", color="lime", lw=1.0))
    ax_st.legend(loc="upper left", facecolor="#111115", edgecolor="gray",
                  labelcolor="white", fontsize=8)
    ax_st.text(0.02, 0.02,
               f"2^{n:.0f} ≈ {2.0**n:.2e} Planck lengths at saturation",
               transform=ax_st.transAxes, color="gray", fontsize=7)

    # ---- Right: metric curves ----
    ax_met.set_facecolor("#0a0a0a")
    ax_met.set_xlabel("Physical time (years, log scale)")
    ax_met.set_ylabel("Fraction of n  (normalized)")
    ax_met.set_xticks(tick_tbf_v)
    ax_met.set_xticklabels(tick_labels_v, fontsize=7)

    ax_met.plot(t_bf, sim["S"],         color="red",    lw=1.5, ls="--",
                label="S(t) total entropy [exact]")
    ax_met.plot(t_bf, sim["fabric"]/n,  color="white",  lw=3.0,
                label="fabric(t)/n  L0 [bold]")
    ax_met.plot(t_bf, sim["R"],         color="orange", lw=1.5,
                label="R(t) = (fabric+k)/n  [Paper IV]")

    N_dm_n  = sim["N_dm"]  / max(sim["N_dm"].max(),  1e-12)
    N_nu_n  = sim["N_nu"]  / max(sim["N_nu"].max(),  1e-12)
    N_bar_n = sim["N_bar"] / max(sim["N_bar"].max(), 1e-12)
    ax_met.plot(t_bf, N_dm_n,  color="yellow",  lw=1.5, label="L_DM [norm]")
    ax_met.plot(t_bf, N_nu_n,  color="cyan",    lw=1.2, label="L_nu [norm]")
    ax_met.plot(t_bf, N_bar_n, color="magenta", lw=1.2, label="L_bar [norm]")

    # Optional Lambda-CDM reference overlays (dashed, same colour)
    if sim["ref_nu"] is not None:
        ref_nu_n = sim["ref_nu"] / max(sim["ref_nu"].max(), 1e-12)
        ax_met.plot(t_bf, ref_nu_n,  color="cyan",    lw=0.8,
                    ls=":", alpha=0.5, label="L_nu ΛCDM ref")
    if sim["ref_bar"] is not None:
        ref_bar_n = sim["ref_bar"] / max(sim["ref_bar"].max(), 1e-12)
        ax_met.plot(t_bf, ref_bar_n, color="magenta", lw=0.8,
                    ls=":", alpha=0.5, label="L_bar ΛCDM ref")

    ax_met.set_ylim(-0.05, 1.15)
    ax_met.legend(loc="upper left", fontsize=7, facecolor="#111115",
                   edgecolor="gray", labelcolor="white")

    S_today = np.interp(sim["t_today"], t_bf, sim["S"])
    ax_met.axvline(sim["t_today"], color="lime", lw=1.5, ls="--", alpha=0.85)
    ax_met.text(sim["t_today"] + t_bf_max*0.01, 0.52,
                f"now\nS={S_today:.2f}\nt={sim['t_today']:.0f}",
                color="lime", fontsize=7, va="center")
    ax_met.set_title(
        f"k={sim['k_rate']:.5f}  t_bf_max={t_bf_max:.0f}\n"
        f"{ratio_str}\n"
        f"NOTE: amplitudes placeholder (E=mc² not yet derived)",
        fontsize=7)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, facecolor="white")
    print(f"Saved → {args.output}")
    print(f"Ratio DM/bar    at t_today: {sim['ratio_dm_bar']:.2f}  (observed ~5)")
    print(f"Ratio DM/fabric at t_today: {sim['ratio_dm_fabric']:.2f}  (target ~5?)")
    print(f"S(t_today) = {S_today:.3f}  (30% entropy consumed)")
    print(f"fabric(t)/n range: [{(sim['fabric']/n).min():.4f}, "
          f"{(sim['fabric']/n).max():.4f}]")


# ===========================================================================
# CLI
# ===========================================================================

def main():
    p = argparse.ArgumentParser(
        description="Recursive emergent-spacetime worldline demonstrator.")

    p.add_argument("--n_bits",       type=float, default=184.0)
    p.add_argument("--t_bf_max",     type=float, default=None,
                   help="Default: n·ln(n)")
    p.add_argument("--sat_fraction", type=float, default=0.99)
    p.add_argument("--k_rate",       type=float, default=None)
    p.add_argument("--steps",        type=int,   default=3000)
    p.add_argument("--t_today",      type=float, default=74.0,
                   help="Bit-flip ticks = present day (S≈0.30 at default 74).")

    # Recursive cascade parameters
    p.add_argument("--peak_dm",    type=float, default=0.10,
                   help="L_DM peak as fraction of t_bf_max.")
    p.add_argument("--sigma_dm",   type=float, default=0.6,
                   help="L_DM lognormal width.")
    p.add_argument("--amp_dm",     type=float, default=10.0,
                   help="L_DM peak amplitude (bits).")
    p.add_argument("--w_dm",       type=float, default=2.0,
                   help="Bit-width per L_DM entity.")
    p.add_argument("--delta_nu",   type=float, default=0.08,
                   help="L_nu peak delay relative to L_DM, as fraction of t_bf_max.")
    p.add_argument("--delta_bar",  type=float, default=0.06,
                   help="L_bar peak delay relative to L_nu, as fraction of t_bf_max.")
    p.add_argument("--compression",type=float, default=0.7,
                   help="Width compression per recursive level (<1 = narrower).")
    p.add_argument("--ratio",      type=float, default=0.2,
                   help="Amplitude ratio per recursive level. "
                        "ratio=0.2 gives DM:nu:bar = 1:0.2:0.04. "
                        "Tune toward DM/bar ≈ 5 at t_today.")

    # Optional Lambda-CDM reference overlay
    p.add_argument("--spectrum",   type=str,   default=None,
                   help="Optional JSON spectrum for Lambda-CDM reference overlay.")

    # Worldline rendering
    p.add_argument("--slots_dm",  type=int, default=60)
    p.add_argument("--slots_nu",  type=int, default=50)
    p.add_argument("--slots_bar", type=int, default=40)
    p.add_argument("--output",    type=str,
                   default="spacetime_worldlines.png")

    args = p.parse_args()
    sim  = run_simulation(args)
    plot_results(sim, args)


if __name__ == "__main__":
    main()
