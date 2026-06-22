"""
poc_spectral_vs_euclidean.py
============================
Proof-of-concept for the conjecture (Meskanen 2026, §4):

    C_s(Ψ) ∝ S_Euclidean(Ψ)

Physical setting
----------------
Wheeler-DeWitt minisuperspace with scale factor a(τ) and scalar field φ(τ)
in imaginary (Euclidean) time τ ∈ [0, τ_max].  The Euclidean action is:

    S_E = ∫ [ -a + κ (ȧ²/a) + ½a φ̇² + ½a³φ² + (Λ/3)a³ ] dτ  +  2a(τ_max)

with κ=0.15, Λ=1.  The classical (Hartle-Hawking no-boundary) solution is:

    a*(τ) = a_max · sin(ω τ),   ω = √(Λ/3),   a_max = 1/ω

Fourier basis and boundary conditions
--------------------------------------
Paths are parameterised as finite Fourier series in the natural basis of this
problem — the eigenfunctions that satisfy a(0)=0 and ȧ(τ_max)=0:

    f_k(τ) = sin((2k-1) π τ / (2 τ_max)),   k = 1, 2, …, K_max

k=1 reproduces the Hawking solution exactly.  Higher k add higher-frequency
corrections.  Both a(τ) and φ(τ) use the same basis with independent
coefficients A_k and B_k.

Why this basis matters for C_s
--------------------------------
C_s is computed directly from the Fourier coefficients in this natural basis,
NOT from a DFT of the discretised path.  A DFT of a non-periodic signal
(which every path on [0, τ_max] is, relative to the DFT's assumed
periodicity) suffers spectral leakage: a single sine wave fills dozens of DFT
bins and produces artificially high C_s.  Computing C_s in the problem's own
basis is both physically correct and numerically honest.

Spectral complexity in the natural basis
-----------------------------------------
Each active coefficient (A_k, B_k) corresponds to a mode with frequency:

    ω_k = (2k-1) π / (2 τ_max)

C_s is then the total frequency cost of all active modes:

    C_s = Σ_{k active in a} ω_k/Δω  +  Σ_{k active in φ} ω_k/Δω

where a mode is 'active' if it carries more than a threshold fraction of the
total power in its field.  The minimum C_s path is the one that achieves the
required boundary conditions using the lowest-frequency modes — which is
exactly k=1, the Hawking solution.

Confound elimination
---------------------
The previous POC confounded three variables: noise amplitude, C_s, and S_E
all increased together.  This ensemble eliminates that confound by design:

    - Coefficients A_k, B_k are drawn independently per mode
    - Amplitudes are drawn from a log-uniform distribution so that
      high-k modes can have LARGE amplitudes (high S_E, moderate C_s)
      and low-k modes can have small amplitudes (low S_E, low C_s)
    - We additionally report partial Spearman ρ(C_s, S_E | roughness)
      where roughness = Σ_k k·|A_k| is an explicit proxy for path
      roughness, to verify the correlation survives after controlling
      for it

Primary output
--------------
    Spearman ρ(C_s, S_E) across 20,000 paths
    Partial ρ(C_s, S_E | roughness) — confound-controlled
    Does argmin C_s recover the Hawking solution?
    Does argmin S_E recover the Hawking solution?
    Scatter plot (log scale), profiles, path comparison
"""

from __future__ import annotations

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr, pearsonr

# ─────────────────────────────────────────────────────────────────────────────
# Physical parameters
# ─────────────────────────────────────────────────────────────────────────────

LAMBDA   : float = 1.0
KAPPA    : float = 0.15          # coefficient of ȧ²/a term
OMEGA    : float = float(np.sqrt(LAMBDA / 3.0))
TAU_MAX  : float = float(np.pi / (2.0 * OMEGA))
A_MAX    : float = 1.0 / OMEGA   # Hawking amplitude

N_TAU    : int   = 800           # grid points along τ
K_MAX    : int   = 12            # number of basis modes available per field
N_PATHS  : int   = 20_000        # ensemble size
SEED     : int   = 42

# Fidelity threshold for mode activity: a mode is 'active' if its power
# fraction exceeds this value.
POWER_THRESHOLD : float = 1e-4

# ─────────────────────────────────────────────────────────────────────────────
# Basis and grid
# ─────────────────────────────────────────────────────────────────────────────

tau  : np.ndarray = np.linspace(1e-6, TAU_MAX, N_TAU)
dtau : float      = float(tau[1] - tau[0])

# Natural frequencies of basis modes
# ω_k = (2k-1)π / (2 τ_max),  k = 1, …, K_max
omega_k : np.ndarray = (2*np.arange(1, K_MAX+1) - 1) * np.pi / (2.0 * TAU_MAX)

# Δω: minimum frequency resolution = gap between adjacent basis frequencies
delta_omega : float = float(omega_k[1] - omega_k[0])   # = π / τ_max, uniform

# Basis function matrix B[k, n] = f_k(τ_n)
B_mat : np.ndarray = np.array([
    np.sin((2*k - 1) * np.pi * tau / (2.0 * TAU_MAX))
    for k in range(1, K_MAX + 1)
])   # shape (K_max, N_tau)


# ─────────────────────────────────────────────────────────────────────────────
# Euclidean action
# ─────────────────────────────────────────────────────────────────────────────

def euclidean_action(a: np.ndarray, phi: np.ndarray) -> float:
    """Compute S_E for paths a(τ), φ(τ) on the shared grid.

    Args:
        a:   Scale factor path, shape (N_tau,).  Clipped to ≥ 1e-4.
        phi: Scalar field path, shape (N_tau,).

    Returns:
        S_E as a float.
    """
    a    = np.clip(a, 1e-4, None)
    da   = np.gradient(a,   dtau)
    dphi = np.gradient(phi, dtau)
    integrand = (
        -a
        + KAPPA * (da**2) / a
        + 0.5 * a * dphi**2
        + 0.5 * a**3 * phi**2
        + (LAMBDA / 3.0) * a**3
    )
    return float(np.sum(integrand) * dtau + 2.0 * a[-1])


# ─────────────────────────────────────────────────────────────────────────────
# Spectral complexity in the natural basis
# ─────────────────────────────────────────────────────────────────────────────

def spectral_complexity_natural(
    A_coeffs: np.ndarray,
    B_coeffs: np.ndarray,
) -> tuple[float, int, float]:
    """Compute C_s directly from Fourier coefficients in the natural basis.

    A mode k is 'active' if its power fraction exceeds POWER_THRESHOLD.
    C_s is the sum of ω_k/Δω over all active modes across both fields.

    This avoids DFT spectral leakage entirely: a single basis mode k=1
    has exactly one active frequency ω_1, giving C_s = ω_1/Δω = 1.

    Args:
        A_coeffs: Coefficients for a(τ), shape (K_max,).
        B_coeffs: Coefficients for φ(τ), shape (K_max,).

    Returns:
        Tuple of (C_s, n_active_modes, roughness).
        roughness = Σ_k (2k-1) · (|A_k| + |B_k|) — explicit confound proxy.
    """
    # Power fractions per mode per field
    pow_a   = A_coeffs**2
    pow_phi = B_coeffs**2
    tot_a   = pow_a.sum()   + 1e-30
    tot_phi = pow_phi.sum() + 1e-30

    active_a   = pow_a   / tot_a   > POWER_THRESHOLD
    active_phi = pow_phi / tot_phi > POWER_THRESHOLD

    # Frequency cost of active modes (ω_k / Δω)
    freq_costs = omega_k / delta_omega   # = [1, 3, 5, 7, …] by construction

    cs_a   = float(np.sum(freq_costs[active_a]))
    cs_phi = float(np.sum(freq_costs[active_phi])) if tot_phi > 1e-28 else 0.0
    cs     = cs_a + cs_phi

    n_active = int(active_a.sum() + active_phi.sum())

    # Roughness: frequency-weighted amplitude sum (explicit confound proxy)
    roughness = float(np.sum((2*np.arange(1,K_MAX+1)-1) * (np.abs(A_coeffs) + np.abs(B_coeffs))))

    return cs, n_active, roughness


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_ensemble(
    n_paths : int,
    rng     : np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw n_paths random paths with coefficients decorrelated from roughness.

    Design principles to eliminate the confound:
    - Amplitudes drawn log-uniformly so high-k modes can be large (breaking
      the implicit 'large k → small amplitude' assumption of noise-addition).
    - Number of active modes drawn independently of amplitude magnitude.
    - Classical path included as path index 0.

    Returns:
        Dict with keys: A_coeffs (n_paths, K_max), B_coeffs (n_paths, K_max),
        S_E (n_paths,), C_s (n_paths,), n_modes (n_paths,), roughness (n_paths,).
    """
    A_all = np.zeros((n_paths, K_MAX))
    B_all = np.zeros((n_paths, K_MAX))

    # Path 0: classical Hawking (k=1 only, φ=0)
    A_all[0, 0] = A_MAX
    B_all[0, :] = 0.0

    # Paths 1…n_paths-1: random
    for i in range(1, n_paths):
        # Independently choose how many modes are active for a and φ
        n_active_a   = rng.integers(1, K_MAX + 1)
        n_active_phi = rng.integers(0, K_MAX + 1)   # 0 = φ=0

        # Choose which modes are active (random subset)
        idx_a   = rng.choice(K_MAX, size=n_active_a,   replace=False)
        idx_phi = rng.choice(K_MAX, size=n_active_phi, replace=False) if n_active_phi > 0 else []

        # Draw amplitudes log-uniformly in [0.05, 3.0] — decouples amplitude from k
        amp_a   = np.exp(rng.uniform(np.log(0.05), np.log(3.0), n_active_a))
        amp_phi = np.exp(rng.uniform(np.log(0.05), np.log(2.0), n_active_phi)) if n_active_phi > 0 else []

        # Random signs
        amp_a   *= rng.choice([-1, 1], size=n_active_a)
        if n_active_phi > 0:
            amp_phi *= rng.choice([-1, 1], size=n_active_phi)

        A_all[i, idx_a] = amp_a
        if n_active_phi > 0:
            B_all[i, idx_phi] = amp_phi

    # Build paths and evaluate costs
    S_E_arr    = np.zeros(n_paths)
    Cs_arr     = np.zeros(n_paths)
    nm_arr     = np.zeros(n_paths, dtype=int)
    rough_arr  = np.zeros(n_paths)

    print(f"  Evaluating {n_paths:,} paths …")
    t0 = time.time()
    for i in range(n_paths):
        a_path   = np.clip(B_mat.T @ A_all[i], 1e-3, None)
        phi_path = B_mat.T @ B_all[i]

        S_E_arr[i] = euclidean_action(a_path, phi_path)
        Cs_arr[i], nm_arr[i], rough_arr[i] = spectral_complexity_natural(
            A_all[i], B_all[i])

        if (i+1) % 5000 == 0:
            print(f"    {i+1:>6,} / {n_paths:,}  ({time.time()-t0:.1f}s)")

    print(f"  Done in {time.time()-t0:.1f}s")
    return dict(
        A_coeffs=A_all, B_coeffs=B_all,
        S_E=S_E_arr, C_s=Cs_arr,
        n_modes=nm_arr, roughness=rough_arr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Partial Spearman correlation (confound control)
# ─────────────────────────────────────────────────────────────────────────────

def partial_spearman(x: np.ndarray, y: np.ndarray,
                     z: np.ndarray) -> tuple[float, float]:
    """Spearman ρ(x, y | z): correlation after removing linear effect of z.

    Ranks x, y, z then computes residuals of rank(x) ~ rank(z) and
    rank(y) ~ rank(z), then correlates the residuals.

    Args:
        x, y: Variables of interest.
        z:    Confound variable to control for.

    Returns:
        (partial_rho, p_value) via Pearson on rank residuals.
    """
    from scipy.stats import rankdata
    rx = rankdata(x).astype(float)
    ry = rankdata(y).astype(float)
    rz = rankdata(z).astype(float)

    def residuals(a, b):
        # OLS residuals of a regressed on b
        b_c = b - b.mean()
        beta = np.dot(b_c, a - a.mean()) / (np.dot(b_c, b_c) + 1e-30)
        return a - (a.mean() + beta * b_c)

    res_x = residuals(rx, rz)
    res_y = residuals(ry, rz)
    return pearsonr(res_x, res_y)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def make_figures(
    data          : dict[str, np.ndarray],
    rho_full      : float,
    rho_partial   : float,
    p_partial     : float,
    idx_min_cs    : int,
    idx_min_se    : int,
    output_dir    : str = ".",
) -> None:
    """Produce and save all figures."""

    S_E  = data["S_E"]
    C_s  = data["C_s"]
    A_all = data["A_coeffs"]
    B_all = data["B_coeffs"]

    # ── Figure 1: scatter S_E vs C_s ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Clip extreme outliers for display only
    se_p5, se_p95 = np.percentile(S_E, [1, 99])
    cs_p5, cs_p95 = np.percentile(C_s, [1, 99])
    mask = (S_E >= se_p5) & (S_E <= se_p95) & (C_s >= cs_p5) & (C_s <= cs_p95)

    for ax, xscale, yscale, title_extra in [
        (axes[0], 'linear', 'linear', 'Linear scale'),
        (axes[1], 'log',    'log',    'Log scale'),
    ]:
        ax.scatter(S_E[mask & (np.arange(len(S_E)) > 0)],
                   C_s[mask & (np.arange(len(S_E)) > 0)],
                   s=3, alpha=0.25, color='steelblue', label='Random paths')
        ax.scatter([S_E[0]], [C_s[0]], s=200, marker='*',
                   color='red', zorder=6, label='Hawking (k=1)')
        ax.scatter([S_E[idx_min_cs]], [C_s[idx_min_cs]], s=150, marker='D',
                   color='blue', zorder=6, label=f'Min C_s path')
        ax.scatter([S_E[idx_min_se]], [C_s[idx_min_se]], s=150, marker='^',
                   color='green', zorder=6, label=f'Min S_E path')
        if xscale == 'log':
            ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Euclidean action $S_E$', fontsize=11)
        ax.set_ylabel('Spectral complexity $C_s$', fontsize=11)
        ax.set_title(f'{title_extra} — Spearman ρ={rho_full:.4f}\n'
                     f'Partial ρ(C_s,S_E|roughness)={rho_partial:.4f} '
                     f'(p={p_partial:.2e})', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Conjecture test: $C_s \\propto S_E$ in WdW minisuperspace\n'
                 f'N={N_PATHS:,} paths, K_max={K_MAX}, log-uniform amplitudes',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'scatter_cs_vs_se.png'), dpi=150,
                bbox_inches='tight')
    print(f"  Saved scatter_cs_vs_se.png")

    # ── Figure 2: minimum paths vs Hawking ───────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))

    hawk_a   = np.clip(B_mat.T @ A_all[0], 1e-3, None)
    hawk_phi = B_mat.T @ B_all[0]

    min_cs_a   = np.clip(B_mat.T @ A_all[idx_min_cs], 1e-3, None)
    min_cs_phi = B_mat.T @ B_all[idx_min_cs]

    min_se_a   = np.clip(B_mat.T @ A_all[idx_min_se], 1e-3, None)
    min_se_phi = B_mat.T @ B_all[idx_min_se]

    for ax, field_hawk, field_cs, field_se, ylabel, title in [
        (axes2[0], hawk_a,   min_cs_a,   min_se_a,
         'Scale factor $a(\\tau)$', 'Scale factor paths'),
        (axes2[1], hawk_phi, min_cs_phi, min_se_phi,
         'Scalar field $\\phi(\\tau)$', 'Scalar field paths'),
    ]:
        ax.plot(tau, field_hawk, 'r-',  lw=2.5, label='Hawking (k=1)')
        ax.plot(tau, field_cs,   'b--', lw=1.8, label=f'Min $C_s$ (idx={idx_min_cs})')
        ax.plot(tau, field_se,   'g:',  lw=1.8, label=f'Min $S_E$ (idx={idx_min_se})')
        ax.set_xlabel('$\\tau$', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig2.suptitle('Minimum-cost paths vs Hawking solution', fontsize=11)
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'minimum_paths.png'), dpi=150,
                 bbox_inches='tight')
    print(f"  Saved minimum_paths.png")

    # ── Figure 3: C_s and S_E distributions + roughness partial ─────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4))

    axes3[0].hist(C_s[C_s < np.percentile(C_s, 99)], bins=60,
                  color='steelblue', alpha=0.7, edgecolor='none')
    axes3[0].axvline(C_s[0],         color='red',  lw=2, label='Hawking')
    axes3[0].axvline(C_s[idx_min_cs],color='blue', lw=2, ls='--', label='Min C_s')
    axes3[0].set_xlabel('$C_s$'); axes3[0].set_title('C_s distribution')
    axes3[0].legend(fontsize=8)

    axes3[1].hist(S_E[S_E < np.percentile(S_E, 99)], bins=60,
                  color='salmon', alpha=0.7, edgecolor='none')
    axes3[1].axvline(S_E[0],         color='red',  lw=2, label='Hawking')
    axes3[1].axvline(S_E[idx_min_se],color='green',lw=2, ls='--', label='Min S_E')
    axes3[1].set_xlabel('$S_E$'); axes3[1].set_title('$S_E$ distribution')
    axes3[1].legend(fontsize=8)

    # Partial correlation residuals scatter
    from scipy.stats import rankdata
    rough = data["roughness"]
    rx = rankdata(S_E).astype(float);  rx -= rx.mean()
    ry = rankdata(C_s).astype(float);  ry -= ry.mean()
    rz = rankdata(rough).astype(float); rz -= rz.mean()
    beta_x = np.dot(rz, rx)/(np.dot(rz,rz)+1e-30)
    beta_y = np.dot(rz, ry)/(np.dot(rz,rz)+1e-30)
    res_x = rx - beta_x*rz
    res_y = ry - beta_y*rz
    subsample = np.random.default_rng(0).integers(0, len(res_x), 3000)
    axes3[2].scatter(res_x[subsample], res_y[subsample],
                     s=4, alpha=0.3, color='steelblue')
    axes3[2].set_xlabel('rank($S_E$) residual | roughness')
    axes3[2].set_ylabel('rank($C_s$) residual | roughness')
    axes3[2].set_title(f'Partial correlation\nρ={rho_partial:.4f}  p={p_partial:.2e}')
    axes3[2].grid(True, alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'distributions.png'), dpi=150,
                 bbox_inches='tight')
    print(f"  Saved distributions.png")
    plt.close('all')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(output_dir: str = ".") -> None:
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 65)
    print("Conjecture POC: C_s ∝ S_Euclidean  (Meskanen 2026)")
    print("=" * 65)
    print(f"  Λ={LAMBDA}, κ={KAPPA}, ω={OMEGA:.4f}, τ_max={TAU_MAX:.4f}")
    print(f"  K_max={K_MAX}, N_τ={N_TAU}, N_paths={N_PATHS:,}")
    print(f"  Natural basis frequencies ω_k/Δω = [1, 3, 5, …, {2*K_MAX-1}]")
    print()

    # Hawking reference values
    A_hawk = np.zeros(K_MAX); A_hawk[0] = A_MAX
    B_hawk = np.zeros(K_MAX)
    a_hawk = np.clip(B_mat.T @ A_hawk, 1e-3, None)
    phi_hawk = B_mat.T @ B_hawk
    S_hawk = euclidean_action(a_hawk, phi_hawk)
    Cs_hawk, nm_hawk, _ = spectral_complexity_natural(A_hawk, B_hawk)
    print(f"  Hawking solution:  S_E={S_hawk:.4f},  C_s={Cs_hawk:.1f},  "
          f"modes={nm_hawk}  (k=1 only)")
    print()

    # Generate ensemble
    print("Generating ensemble …")
    rng  = np.random.default_rng(SEED)
    data = generate_ensemble(N_PATHS, rng)

    S_E  = data["S_E"]
    C_s  = data["C_s"]
    rough = data["roughness"]

    # ── Statistics ────────────────────────────────────────────────────────────
    rho_full, p_full   = spearmanr(C_s, S_E)
    rho_partial, p_partial = partial_spearman(S_E, C_s, rough)

    # Filter finite values only (some paths may have extreme S_E)
    finite = np.isfinite(S_E) & np.isfinite(C_s)
    rho_fin, _ = spearmanr(C_s[finite], S_E[finite])

    idx_min_cs = int(np.argmin(C_s))
    idx_min_se = int(np.argmin(S_E[finite]))

    # How do minimum paths rank on the OTHER metric?
    rank_cs_by_se = int((S_E < S_E[idx_min_cs]).sum()) + 1
    rank_se_by_cs = int((C_s < C_s[idx_min_se]).sum()) + 1

    # Coefficient comparison: do min paths look like Hawking?
    def dominant_mode(idx):
        A = data["A_coeffs"][idx]
        B = data["B_coeffs"][idx]
        k_a   = int(np.argmax(np.abs(A))) + 1
        k_phi = int(np.argmax(np.abs(B))) + 1 if np.any(B != 0) else None
        return k_a, k_phi

    k_a_cs, k_phi_cs = dominant_mode(idx_min_cs)
    k_a_se, k_phi_se = dominant_mode(idx_min_se)

    print()
    print("=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"  Total paths evaluated        : {N_PATHS:,}")
    print(f"  Finite-valued paths          : {finite.sum():,}")
    print()
    print(f"  Spearman ρ(C_s, S_E)         : {rho_full:.5f}  (p={p_full:.2e})")
    print(f"  Partial ρ(C_s, S_E|roughness): {rho_partial:.5f}  (p={p_partial:.2e})")
    print(f"  (Partial ρ controls for path roughness as explicit confound)")
    print()
    print(f"  Hawking path (idx=0):")
    print(f"    S_E={S_E[0]:.4f}  C_s={C_s[0]:.1f}")
    print(f"    Rank by S_E: #{int((S_E<S_E[0]).sum())+1}  "
          f"Rank by C_s: #{int((C_s<C_s[0]).sum())+1}  (out of {N_PATHS:,})")
    print()
    print(f"  Min-C_s path (idx={idx_min_cs}):")
    print(f"    C_s={C_s[idx_min_cs]:.1f}  S_E={S_E[idx_min_cs]:.4f}")
    print(f"    Dominant modes: a→k={k_a_cs}, φ→k={k_phi_cs}")
    print(f"    Rank by S_E: #{rank_cs_by_se}  (out of {N_PATHS:,})")
    print()
    print(f"  Min-S_E path (idx={idx_min_se}):")
    print(f"    S_E={S_E[idx_min_se]:.4f}  C_s={C_s[idx_min_se]:.1f}")
    print(f"    Dominant modes: a→k={k_a_se}, φ→k={k_phi_se}")
    print(f"    Rank by C_s: #{rank_se_by_cs}  (out of {N_PATHS:,})")
    print()

    # Interpretation
    print("INTERPRETATION")
    print("-" * 65)
    if abs(rho_partial) > 0.3 and p_partial < 0.01:
        print("  Partial correlation survives roughness control.")
        print("  C_s and S_E share structure beyond common sensitivity to path roughness.")
        print("  This SUPPORTS the conjecture.")
    elif abs(rho_partial) < 0.15:
        print("  Partial correlation collapses after roughness control.")
        print("  The full correlation was driven by the common confound.")
        print("  This does NOT support the conjecture as stated.")
    else:
        print("  Partial correlation is moderate. Conjecture partially supported.")
        print("  Further investigation needed.")

    if k_a_cs == 1:
        print(f"  Min-C_s path has dominant mode k=1 (Hawking). ✓")
    else:
        print(f"  Min-C_s path has dominant mode k={k_a_cs} (not Hawking). ✗")

    if k_a_se == 1:
        print(f"  Min-S_E path has dominant mode k=1 (Hawking). ✓")
    else:
        print(f"  Min-S_E path has dominant mode k={k_a_se} (not Hawking). ✗")

    print()
    print("Saving figures …")
    make_figures(data, rho_full, rho_partial, p_partial,
                 idx_min_cs, idx_min_se, output_dir=output_dir)
    print("Done.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--n_paths",    type=int, default=N_PATHS)
    args = p.parse_args()
    N_PATHS = args.n_paths
    main(output_dir=args.output_dir)
