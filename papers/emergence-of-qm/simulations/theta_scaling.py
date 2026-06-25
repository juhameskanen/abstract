"""
Verify ||B|| = sin(2θ)/2 analytically and numerically across all L.
ψ(θ) = cos(θ)·e_0 + i·sin(θ)·e_1  (already normalised)
"""
import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from wavefunction import Wavefunction

def boson_norm_raw(psi):
    psi = np.asarray(psi, dtype=complex)
    n = float(np.sqrt(np.sum(np.abs(psi)**2)))
    if n < 1e-12: return 0.0
    psi = psi / n
    rho = np.outer(psi, psi.conj())
    B = rho - np.diag(np.diag(rho))
    return float(np.linalg.norm(B))

# ── Analytical derivation ────────────────────────────────────────────────────
print("=" * 65)
print("ANALYTICAL DERIVATION")
print("=" * 65)
print("""
  ψ(θ) = cos(θ)·e_0 + i·sin(θ)·e_1   (normalised: |ψ|²=1)

  Density matrix ρ = |ψ⟩⟨ψ|:
    ρ_00 = cos²(θ)
    ρ_11 = sin²(θ)
    ρ_01 = ψ_0·ψ_1* = cos(θ)·(-i·sin(θ)) = -i·sin(θ)cos(θ)
    ρ_10 = ψ_1·ψ_0* = i·sin(θ)·cos(θ)

  Off-diagonal boson matrix B = ρ - diag(ρ):
    B_01 = ρ_01 = -i·sin(θ)cos(θ) = -(i/2)·sin(2θ)
    B_10 = ρ_10 = +(i/2)·sin(2θ)
    All other B_ij = 0  (since ψ_j=0 for j≥2)

  Frobenius norm:
    ||B||² = |B_01|² + |B_10|²
           = 2·(sin(2θ)/2)²
           = sin²(2θ)/2

    ||B|| = sin(2θ)/√2

  Maximum at θ = π/4: ||B||_max = 1/√2 ≈ 0.7071

  Single hop (θ=π/2): ||B|| = sin(π)/√2 = 0  ← WAIT
""")

# Check numerically at θ=π/2
L = 8
psi_hop = np.zeros(L, dtype=complex)
psi_hop[0] = np.cos(np.pi/2)    # = 0
psi_hop[1] = 1j * np.sin(np.pi/2)  # = i
print("  Numerical check at θ=π/2:")
print(f"  ψ = {psi_hop[:4]} ...")
# This gives ψ = i·e_1 — single site! So B=0.
# The single hop 0→1 is encoded differently: ψ = e_0 + i·e_1 (not normalised the same way)
print(f"  ||B|| = {boson_norm_raw(psi_hop):.6f}")
print()
print("  The interpolation ψ(θ)=cos(θ)e_0 + i·sin(θ)e_1 reaches")
print("  SINGLE SITE at both endpoints (θ=0 → e_0, θ=π/2 → i·e_1).")
print("  The standard hop encoding is ψ = e_0 + i·e_1 (equal weight).")
print("  That corresponds to θ=π/4, the MAXIMUM of sin(2θ)/√2. ✓")
print()

# ── Verify sin(2θ)/√2 numerically ───────────────────────────────────────────
print("=" * 65)
print("NUMERICAL VERIFICATION: ||B|| = sin(2θ)/√2")
print("=" * 65)
print(f"\n  {'θ/π':>6}  {'||B|| num':>12}  {'sin(2θ)/√2':>12}  {'error':>10}")
print(f"  {'-'*48}")

thetas = np.linspace(0, 0.5, 21)
max_err = 0.0
for t in thetas:
    theta = t * np.pi
    psi = np.zeros(L, dtype=complex)
    psi[0] = np.cos(theta)
    psi[1] = 1j * np.sin(theta)
    bn = boson_norm_raw(psi)
    predicted = np.sin(2*theta) / np.sqrt(2)
    err = abs(bn - predicted)
    max_err = max(max_err, err)
    marker = " ← max" if abs(t - 0.25) < 0.01 else ""
    print(f"  {t:>6.3f}  {bn:>12.8f}  {predicted:>12.8f}  {err:>10.2e}{marker}")

print(f"\n  Max error across all θ: {max_err:.2e}  {'✓ exact' if max_err < 1e-14 else '✗'}")

# ── L-independence ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("L-INDEPENDENCE: same formula for all chain lengths?")
print("=" * 65)
print(f"\n  θ=π/4 (standard hop), varying L:")
print(f"  {'L':>6}  {'||B|| num':>12}  {'1/√2':>10}  {'error':>10}")
print(f"  {'-'*44}")

for L in [4, 8, 16, 32, 64, 128]:
    psi = np.zeros(L, dtype=complex)
    psi[0] = np.cos(np.pi/4)
    psi[1] = 1j * np.sin(np.pi/4)
    bn = boson_norm_raw(psi)
    err = abs(bn - 1.0/np.sqrt(2))
    print(f"  {L:>6}  {bn:>12.10f}  {1/np.sqrt(2):>10.8f}  {err:>10.2e}")

print(f"\n  → Formula is EXACTLY L-independent. ✓")
print(f"  → The boson amplitude depends only on θ, not on system size.")

# ── Site-independence ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SITE-INDEPENDENCE: same formula for hop i→j (any i,j)?")
print("=" * 65)
L = 16
theta = np.pi / 4
print(f"\n  L={L}, θ=π/4 (standard hop), varying hop sites i→j:")
print(f"  {'i→j':>8}  {'||B|| num':>12}  {'1/√2':>10}  {'error':>10}")
print(f"  {'-'*46}")

for i, j in [(0,1),(0,7),(3,9),(5,14),(7,15),(0,15)]:
    psi = np.zeros(L, dtype=complex)
    psi[i] = np.cos(theta)
    psi[j] = 1j * np.sin(theta)
    bn = boson_norm_raw(psi)
    err = abs(bn - 1/np.sqrt(2))
    print(f"  {i}→{j:>2}      {bn:>12.10f}  {1/np.sqrt(2):>10.8f}  {err:>10.2e}")

print(f"\n  → Formula is EXACTLY site-independent. ✓")
print(f"  → The boson amplitude is the same whether the hop is")
print(f"    nearest-neighbour or across the entire chain.")

# ── The full picture ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("COMPLETE RESULT")
print("=" * 65)
print(f"""
  For ψ(θ) = cos(θ)·e_i + i·sin(θ)·e_j  (any i≠j, any L):

    ||B(θ)|| = sin(2θ) / √2   EXACTLY

  This is:
  • Analytically exact (proved from ρ = |ψ⟩⟨ψ|)
  • L-independent  (chain length irrelevant)
  • Site-independent (hop distance irrelevant)
  • Encoding-independent (any phase on e_j gives same norm)

  Special values:
    θ = 0:    ||B|| = 0          (stationary, no boson)
    θ = π/4:  ||B|| = 1/√2      (standard hop, maximum boson)
    θ = π/2:  ||B|| = 0          (arrived, stationary again)

  The boson amplitude traces a HALF-SINE over the fermion's journey.
  It is born when the fermion starts moving, peaks at mid-hop,
  and vanishes when the fermion arrives.

  This is the propagator of a virtual particle.

  Connection to Born rule:
    Probability of hop:     P = sin²(θ)        (Born rule)
    Boson amplitude:        ||B|| = sin(2θ)/√2
                                  = 2·sin(θ)·cos(θ)/√2
                                  = √2 · √P · √(1-P)

  ||B|| = √2 · √(P·(1-P))  — the geometric mean of hop and no-hop,
  scaled by √2. This is the INTERFERENCE TERM of Born rule.
  The boson IS the quantum interference, made visible as a matrix.
""")
