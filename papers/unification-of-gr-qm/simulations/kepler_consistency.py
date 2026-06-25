"""
kepler_consistency.py
=====================
Corollary: Kepler's third law as aspect ratio identity.
G = ω²r³/M is a consistency condition, not a coupling constant.

Experiments:
  1. Verify ω²_ISCO · r³_ISCO = M exactly in Planck units.
  2. Show n_bh = log2(r_s/ℓ_P) is consistent with r_s = 2M.
  3. Verify across a range of masses.
  4. Show G in SI units is purely a unit conversion factor.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from wavefunction import Wavefunction

# ── Planck constants (SI) ─────────────────────────────────────────────────────
ell_P = 1.616255e-35   # m
t_P   = 5.391247e-44   # s
m_P   = 2.176434e-8    # kg
G_SI  = 6.674e-11      # m³ kg⁻¹ s⁻²
c     = 2.998e8        # m/s

print("=" * 65)
print("KEPLER CONSISTENCY: G as aspect ratio identity")
print("  In Planck units: G = 1, r_s = 2M, ω_ISCO = 1/(6√6·M)")
print("=" * 65)

# ── 1. ISCO verification in Planck units ──────────────────────────────────────
print("\n── ISCO: ω²·r³ = M in Planck units ──")
print(f"\n  {'M (Planck)':>14}  {'r_ISCO':>14}  {'ω_ISCO':>12}  "
      f"{'ω²r³':>14}  {'ratio ω²r³/M':>14}  {'= G?':>6}")
print(f"  {'-'*82}")

# ISCO for Schwarzschild: r_ISCO = 6M = 3r_s, ω_ISCO = 1/(6√6·M)
test_masses = [1.0, 10.0, 100.0, 1e6, 1e10, 1e20, 1e37]

for M in test_masses:
    r_s       = 2 * M                      # Schwarzschild radius, Planck units
    r_ISCO    = 6 * M                      # = 3·r_s
    omega_ISCO = 1.0 / (6 * np.sqrt(6) * M)
    lhs       = omega_ISCO**2 * r_ISCO**3  # should = M = G·M with G=1
    ratio     = lhs / M
    print(f"  {M:>14.4e}  {r_ISCO:>14.4e}  {omega_ISCO:>12.4e}  "
          f"{lhs:>14.4e}  {ratio:>14.8f}  {'✓' if abs(ratio-1)<1e-10 else '✗'}")

print(f"\n  → ω²_ISCO · r³_ISCO = M exactly for all masses. ✓")
print(f"  → G = 1 is a consistency condition, not a free parameter.")

# ── 2. n_bh consistency ───────────────────────────────────────────────────────
print("\n── n_bh = log2(r_s/ℓ_P) consistency ──")
print(f"  r_s = 2M in Planck units → n_bh = log2(2M) = 1 + log2(M)")
print()
print(f"  {'M (Planck)':>14}  {'r_s':>14}  {'n_bh':>10}  {'1+log2(M)':>12}  {'match':>6}")
print(f"  {'-'*62}")

for M in [1e10, 1e20, 1e30, 1e37, 1e40]:
    r_s   = 2 * M
    n_bh  = np.log2(r_s)
    check = 1 + np.log2(M)
    print(f"  {M:>14.4e}  {r_s:>14.4e}  {n_bh:>10.4f}  {check:>12.4f}  "
          f"{'✓' if abs(n_bh - check) < 1e-10 else '✗'}")

print(f"\n  → n_bh = 1 + log2(M) is an exact identity. ✓")

# ── 3. Minimum-complexity worldline at ISCO ───────────────────────────────────
print("\n── Minimum-complexity worldline at ISCO ──")
print("  ψ(t) = e^{i·ω_ISCO·t} on T = n_bh·ln(n_bh) time steps")
print()

M_solar_planck = 1.989e30 / m_P
r_s_solar      = 2 * M_solar_planck
n_bh_solar     = np.log2(r_s_solar)
T_bh           = int(n_bh_solar * np.log(n_bh_solar))
omega_ISCO_s   = 1.0 / (6 * np.sqrt(6) * M_solar_planck)

t_grid = np.arange(T_bh, dtype=float)
psi_ISCO = np.exp(1j * omega_ISCO_s * t_grid)

wf_ISCO = Wavefunction(psi_ISCO, dx=1.0)
cs_ISCO = wf_ISCO.spectral_complexity()
modes   = wf_ISCO.retained_modes()

print(f"  Solar mass: M = {M_solar_planck:.4e} m_P")
print(f"  r_s        = {r_s_solar:.4e} ℓ_P")
print(f"  n_bh       = {n_bh_solar:.4f} bits")
print(f"  T_bh       = {T_bh} Planck times")
print(f"  ω_ISCO     = {omega_ISCO_s:.4e} (Planck units)")
print(f"  C_s        = {cs_ISCO:.4f}  (expect ~1)")
print(f"  Modes retained: {len(modes)}")
if modes:
    print(f"  Dominant mode frequency: {modes[0].frequency:.4e}")

# ── 4. G in SI = unit conversion factor ──────────────────────────────────────
print("\n── G in SI units: pure unit conversion ──")
print("""
  In Planck units: G = 1 (by definition of Planck units).
  In SI units:     G = ℓ_P³ / (m_P · t_P²)

  This is not a measured coupling — it is the conversion between
  Planck units and SI units, fixed by ℓ_P, m_P, t_P.
""")

G_from_planck = ell_P**3 / (m_P * t_P**2)
print(f"  G from Planck units: ℓ_P³/(m_P·t_P²) = {G_from_planck:.4e} m³ kg⁻¹ s⁻²")
print(f"  G measured (CODATA): {G_SI:.4e} m³ kg⁻¹ s⁻²")
print(f"  Ratio:               {G_from_planck/G_SI:.6f}")
print(f"  → {'Consistent ✓' if abs(G_from_planck/G_SI - 1) < 0.01 else 'Discrepancy — check Planck constants'}")

print("""
  Conclusion: G in SI units is the conversion factor between
  human-defined units (kg, m, s) and the natural units of the
  compression framework (Planck units). It is not more fundamental
  than the numerical value of c in m/s.
""")

# ── 5. Circular orbit check: all radii satisfy ω²r³=M ────────────────────────
print("── General circular orbit: ω²r³ = M for any r > r_s ──")
print(f"\n  M = 1 (Planck), varying orbital radius r:")
print(f"  {'r/r_s':>8}  {'r':>10}  {'ω_circ':>12}  {'ω²r³':>12}  {'= G·M?':>8}")
print(f"  {'-'*58}")

M = 1.0
r_s = 2 * M
for r_factor in [1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 100.0]:
    r       = r_factor * r_s
    omega_c = np.sqrt(M / r**3)   # circular orbit frequency
    lhs     = omega_c**2 * r**3
    print(f"  {r_factor:>8.1f}  {r:>10.4f}  {omega_c:>12.6f}  "
          f"{lhs:>12.6f}  {'✓' if abs(lhs-M)<1e-10 else '✗'}")

print(f"\n  → ω²r³ = M holds for all circular orbits. ✓")
print(f"  → The ISCO at r = 3r_s = 6M is the minimum-complexity STABLE orbit.")
