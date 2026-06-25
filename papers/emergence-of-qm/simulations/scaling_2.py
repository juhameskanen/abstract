"""
Deep dive into Exp 4 momentum spectrum.
k is NOT linear in d — it looks like it peaks at d=L/2 and is symmetric.
This is the dispersion relation of a tight-binding model: k = 2·arcsin(d/L)?
Or is it the chord of a circle? Let's find out.
"""
import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from wavefunction import Wavefunction

def make_wf_hop(L, d):
    f0 = np.zeros(L); f0[0] = 1.0
    f1 = np.zeros(L); f1[d] = 1.0
    psi = f0 + 1j * f1
    return Wavefunction(psi, dx=1.0)

def get_k(wf, L):
    psi = wf.psi
    rho = np.outer(psi, psi.conj())
    B = rho - np.diag(np.diag(rho))
    B_H = (B + B.conj().T) / 2
    evals, evecs = np.linalg.eigh(B_H)
    idx = np.argmax(np.abs(evals))
    v = evecs[:, idx]
    phases = np.unwrap(np.angle(v))
    x = np.arange(L, dtype=float)
    return float(np.polyfit(x, phases, 1)[0])

print("L=8 momentum spectrum vs candidate dispersion relations")
print()
L = 8
print(f"{'d':>4} {'k_meas':>9} {'2π/L·d':>9} {'π/L·d':>9} {'2arcsin(d/L)':>14} {'arctan(d/√(L²-d²))':>20}")
print("-" * 72)

for d in range(1, L):
    wf = make_wf_hop(L, d)
    k = get_k(wf, L)
    k1 = 2*np.pi/L * d                          # linear
    k2 = np.pi/L * d                             # half
    k3 = 2*np.arcsin(d/L) if d<=L else 0        # arcsin
    k4 = np.arctan2(d, np.sqrt(L**2 - d**2))    # arctan / chord
    print(f"{d:>4} {k:>9.4f} {k1:>9.4f} {k2:>9.4f} {k3:>14.4f} {k4:>20.4f}")

# Now look at the ACTUAL phase pattern of the dominant eigenvector
print("\n\nPhase patterns of dominant boson eigenvector for each hop d (L=8):")
print()
for d in [1, 2, 3, 4]:
    wf = make_wf_hop(L, d)
    psi = wf.psi
    rho = np.outer(psi, psi.conj())
    B = rho - np.diag(np.diag(rho))
    B_H = (B + B.conj().T) / 2
    evals, evecs = np.linalg.eigh(B_H)
    idx = np.argmax(np.abs(evals))
    v = evecs[:, idx]
    print(f"d={d}: amplitudes={np.round(np.abs(v),3)}  phases/π={np.round(np.angle(v)/np.pi,3)}")
    print(f"     eigenvalues={np.round(evals,4)}")
    print()

# The key insight: for a single hop 0→d, psi = (e0 + i*ed)/√2
# where e0, ed are unit vectors. The boson matrix is:
# B_ij = ψi* ψj for i≠j
# ψ = [1,0,...,0,i,0,...]/√2  (1 at 0, i at d)
# So B_{0,d} = ψ0*.ψd = (1/√2)*(i/√2) = i/2
#    B_{d,0} = ψd*.ψ0 = (-i/√2)*(1/√2) = -i/2
# ALL other B entries = 0 (since ψj=0 for j≠0,d)
# So the boson matrix is EXACTLY a 2x2 antisymmetric block embedded in LxL
print("Analytical check: for hop 0→d, boson matrix is a 2×2 block")
print("B_{0,d} = +i/2,  B_{d,0} = -i/2,  all other entries = 0")
print("This is ALWAYS purely antisymmetric, regardless of L or d.")
print()
print("Eigenvalues of this 2×2 block: ±1/2")
print("Eigenvectors: (e_0 ± i·e_d)/√2  — superposition of the two sites")
print()
print("The 'momentum' we measure is just the phase gradient of (e_0 + i·e_d)/√2")
print("across ALL L sites — but only 2 are non-zero.")
print("So the linear fit is fitting phase through mostly-zero entries.")
print("The true momentum is encoded in the PAIR (0, d), not a gradient.")
print()

# Better momentum measure: phase difference between the two active sites
print("Better momentum measure: k = angle(B_{0,d}) / d = phase of correlation / hop distance")
print()
for d in range(1, L):
    wf = make_wf_hop(L, d)
    psi = wf.psi
    rho = np.outer(psi, psi.conj())
    B = rho - np.diag(np.diag(rho))
    phase_corr = np.angle(B[0, d])   # phase of the off-diagonal element
    k_true = phase_corr / d if d > 0 else 0
    print(f"  d={d}: B[0,d] = {B[0,d]:.4f}  phase = {phase_corr:.4f}  k=phase/d = {k_true:.4f}  k·d/π = {k_true*d/np.pi:.4f}")

