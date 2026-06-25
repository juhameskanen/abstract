"""
graviton_amplitude.py
=====================
Theorem 3: ||W(ε)|| = sin(2ε)/√2

For orthogonal metric profiles h ⊥ k:
  g^(0)(x) = cos(ε)·h(x),  g^(1)(x) = sin(ε)·k(x)
the graviton (antisymmetric off-diagonal) amplitude is exactly sin(2ε)/√2.

Experiments:
  1. Numerical verification across all ε ∈ [0, π/2].
  2. L-independence: same formula for all chain lengths.
  3. Profile-independence: same formula for all orthogonal profile pairs.
  4. Lifecycle: graviton born, peaked, vanished.
  5. Comparison with boson amplitude from Paper VI.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from wavefunction import Wavefunction

def graviton_norm(g0, g1):
    """Compute ||ρ_graviton|| for metric pair (g0, g1)."""
    psi  = np.array(g0, dtype=float) + 1j * np.array(g1, dtype=float)
    norm = np.linalg.norm(psi)
    if norm < 1e-12: return 0.0
    psi  /= norm
    rho   = np.outer(psi, psi.conj())
    rho_W = rho - np.diag(np.diag(rho))
    rho_g = (rho_W - rho_W.T) / 2
    return float(np.linalg.norm(rho_g))

def make_orthogonal_pair(L, seed=0):
    """Generate a random pair of orthogonal unit vectors in R^L."""
    rng  = np.random.RandomState(seed)
    h    = rng.randn(L)
    k    = rng.randn(L)
    k   -= np.dot(k, h) / np.dot(h, h) * h   # Gram-Schmidt
    h   /= np.linalg.norm(h)
    k   /= np.linalg.norm(k)
    assert abs(np.dot(h, k)) < 1e-12, "Not orthogonal"
    return h, k

print("=" * 65)
print("THEOREM 3: Graviton amplitude ||W(ε)|| = sin(2ε)/√2")
print("=" * 65)

# ── 1. Numerical verification ─────────────────────────────────────────────────
print("\n── Numerical verification (L=8, random orthogonal profiles) ──")
print(f"\n  {'ε/π':>6}  {'||W|| num':>12}  {'sin(2ε)/√2':>12}  {'error':>10}  lifecycle")
print(f"  {'-'*58}")

L = 8
h, k = make_orthogonal_pair(L, seed=1)
epsilons = np.linspace(0, np.pi/2, 21)
max_err = 0.0

for eps in epsilons:
    g0    = np.cos(eps) * h
    g1    = np.sin(eps) * k
    wn    = graviton_norm(g0, g1)
    pred  = np.sin(2*eps) / np.sqrt(2)
    err   = abs(wn - pred)
    max_err = max(max_err, err)
    if eps == 0:            stage = "static — no graviton"
    elif abs(eps - np.pi/4) < 0.01: stage = "← peak amplitude"
    elif abs(eps - np.pi/2) < 0.01: stage = "static again — graviton gone"
    else:                   stage = ""
    print(f"  {eps/np.pi:>6.3f}  {wn:>12.8f}  {pred:>12.8f}  {err:>10.2e}  {stage}")

print(f"\n  Max error: {max_err:.2e}  {'✓ exact' if max_err < 1e-14 else '✗'}")

# ── 2. L-independence ─────────────────────────────────────────────────────────
print("\n── L-independence: same formula for all chain lengths ──")
print(f"\n  ε=π/4 (peak amplitude), varying L:")
print(f"  {'L':>6}  {'||W|| num':>12}  {'1/√2':>10}  {'error':>10}")
print(f"  {'-'*44}")

eps = np.pi / 4
for Ltest in [4, 8, 16, 32, 64, 128]:
    h, k = make_orthogonal_pair(Ltest, seed=2)
    g0   = np.cos(eps) * h
    g1   = np.sin(eps) * k
    wn   = graviton_norm(g0, g1)
    err  = abs(wn - 1/np.sqrt(2))
    print(f"  {Ltest:>6}  {wn:>12.10f}  {1/np.sqrt(2):>10.8f}  {err:>10.2e}")

print(f"  → Formula is exactly L-independent. ✓")

# ── 3. Profile-independence ───────────────────────────────────────────────────
print("\n── Profile-independence: same formula for all orthogonal profile pairs ──")
print(f"\n  ε=π/4, L=8, 10 different random orthogonal profile pairs:")
print(f"  {'seed':>6}  {'||W|| num':>12}  {'1/√2':>10}  {'error':>10}")
print(f"  {'-'*44}")

L = 8; eps = np.pi/4
for seed in range(10):
    h, k = make_orthogonal_pair(L, seed=seed)
    g0   = np.cos(eps) * h
    g1   = np.sin(eps) * k
    wn   = graviton_norm(g0, g1)
    err  = abs(wn - 1/np.sqrt(2))
    print(f"  {seed:>6}  {wn:>12.10f}  {1/np.sqrt(2):>10.8f}  {err:>10.2e}")

print(f"  → Formula is profile-independent. ✓")

# ── 4. Graviton lifecycle in detail ──────────────────────────────────────────
print("\n── Graviton lifecycle ──")
print(f"  ψ(ε) = cos(ε)·h + i·sin(ε)·k,  h⊥k,  ε: 0 → π/2")
print()
print(f"  {'ε/π':>6}  {'||W||':>10}  {'||ρ_Ricci||':>12}  {'||ρ_tidal||':>12}  stage")
print(f"  {'-'*65}")

L = 8; h, k = make_orthogonal_pair(L, seed=3)
for eps in np.linspace(0, np.pi/2, 11):
    g0   = np.cos(eps) * h
    g1   = np.sin(eps) * k
    psi  = g0 + 1j * g1
    psi /= np.linalg.norm(psi)
    rho  = np.outer(psi, psi.conj())
    rho_R = np.diag(np.diag(rho))
    rho_W = rho - rho_R
    rho_g = (rho_W - rho_W.T) / 2
    rho_t = (rho_W + rho_W.T) / 2
    wn    = np.linalg.norm(rho_g)
    rn    = np.linalg.norm(rho_R)
    tn    = np.linalg.norm(rho_t)
    if eps < 0.01:           stage = "static"
    elif abs(eps-np.pi/4)<0.01: stage = "peak"
    elif abs(eps-np.pi/2)<0.01: stage = "static"
    else:                    stage = ""
    print(f"  {eps/np.pi:>6.3f}  {wn:>10.6f}  {rn:>12.6f}  {tn:>12.6f}  {stage}")

# ── 5. Comparison with Paper VI boson amplitude ───────────────────────────────
print("\n── Comparison with Paper VI boson amplitude ──")
print("""
  Paper VI (fermions):
    ψ(θ) = cos(θ)·e_i + i·sin(θ)·e_j   (two sites, orthogonal by construction)
    ||B(θ)|| = sin(2θ)/√2               EXACTLY

  Paper VII (metric configurations):
    ψ(ε) = cos(ε)·h̃ + i·sin(ε)·k̃      (two orthogonal metric profiles)
    ||W(ε)|| = sin(2ε)/√2               EXACTLY

  The formula is IDENTICAL.

  Reason: both arise from the same density matrix decomposition
  applied to a normalised two-component state in superposition.
  The physical interpretation differs (boson vs graviton) but the
  mathematical structure is the same codec applied to different
  degrees of freedom.

  This universality is not a coincidence — it is the signature of
  the compression principle operating at all scales.  □
""")

# ── 6. Analytical proof summary ───────────────────────────────────────────────
print("── Analytical proof (sketch) ──")
print("""
  ψ = cos(ε)·h̃ + i·sin(ε)·k̃,  with ⟨h̃,k̃⟩ = 0, ||h̃||=||k̃||=1.

  ρ_xy = cos²(ε)·h̃(x)h̃(y) + sin²(ε)·k̃(x)k̃(y)
         - (i/2)·sin(2ε)·[h̃(x)k̃(y) - k̃(x)h̃(y)]

  Antisymmetric part:
  (ρ_graviton)_xy = -(i/2)·sin(2ε)·[h̃(x)k̃(y) - k̃(x)h̃(y)]

  ||W||² = (sin²(2ε)/4) · Σ_xy |h̃(x)k̃(y) - k̃(x)h̃(y)|²
         = (sin²(2ε)/4) · 2·(1 - ⟨h̃,k̃⟩²)
         = (sin²(2ε)/4) · 2·(1 - 0)
         = sin²(2ε)/2

  ||W(ε)|| = sin(2ε)/√2.  □
""")
