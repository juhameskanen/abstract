# Focused analysis of the top candidate and the perfect antisymmetric case
import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from wavefunction import Wavefunction

def frame_to_array(frame_int, n=4):
    return np.array([(frame_int >> i) & 1 for i in range(n)], dtype=float)

def video_to_wf(seq):
    f0 = frame_to_array(seq[0])
    f1 = frame_to_array(seq[1])
    psi = f0 + 1j * f1
    if np.all(np.abs(psi) == 0):
        psi = np.ones(4, dtype=complex)
    return Wavefunction(psi, dx=1.0)

# ── The perfectly antisymmetric case: (1, 2) ─────────────────────────────────
print("=" * 60)
print("PERFECTLY ANTISYMMETRIC BOSON: video (1, 2)")
print("frames: [1 0 0 0] → [0 1 0 0]")
print("=" * 60)

wf12 = video_to_wf((1, 2))
psi = wf12.psi
print(f"\nψ = {np.round(psi, 4)}")
print(f"C_s = {wf12.spectral_complexity():.4f}")

rho = np.outer(psi, psi.conj())
B = rho - np.diag(np.diag(rho))
print(f"\nBoson matrix B:")
print(np.round(B, 4))

sym  = (B + B.T) / 2
asym = (B - B.T) / 2
print(f"\nSymmetric part ||sym||  = {np.linalg.norm(sym):.6f}")
print(f"Antisymmetric part ||asym|| = {np.linalg.norm(asym):.6f}")
print(f"→ Purely antisymmetric: {np.linalg.norm(sym) < 1e-10}")

B_H = (B + B.conj().T) / 2
evals, evecs = np.linalg.eigh(B_H)
print(f"\nEigenvalues: {np.round(evals, 4)}")
idx = np.argmax(np.abs(evals))
v = evecs[:, idx]
print(f"Dominant eigenvector amplitudes: {np.round(np.abs(v), 4)}")
print(f"Dominant eigenvector phases/π:   {np.round(np.angle(v)/np.pi, 4)}")

# momentum
phases = np.unwrap(np.angle(v))
x = np.arange(4, dtype=float)
k = np.polyfit(x, phases, 1)[0]
print(f"Momentum k = {k:.4f}  (= {k/np.pi:.4f}π,  = 2π×{k/(2*np.pi):.4f})")

# plane wave overlap
plane = np.exp(1j * k * x); plane /= np.linalg.norm(plane)
print(f"|⟨v | plane wave k={k:.3f}⟩| = {abs(np.dot(v.conj(), plane)):.6f}")

# What pixel transition does this represent?
print(f"\nPixel interpretation:")
print(f"  Frame 0: pixel 0 is ON,  pixels 1,2,3 OFF  → single fermion at site 0")
print(f"  Frame 1: pixel 1 is ON,  pixels 0,2,3 OFF  → single fermion at site 1")
print(f"  → fermion hops from site 0 to site 1")
print(f"  → boson = the MEDIATOR of that hop")
print(f"  → antisymmetry = fermionic exchange signature")

# ── Compare: what is (2,1) — the reverse hop? ────────────────────────────────
print("\n" + "=" * 60)
print("REVERSE HOP: video (2, 1)")
print("frames: [0 1 0 0] → [1 0 0 0]")
print("=" * 60)

wf21 = video_to_wf((2, 1))
psi21 = wf21.psi
rho21 = np.outer(psi21, psi21.conj())
B21 = rho21 - np.diag(np.diag(rho21))
B_H21 = (B21 + B21.conj().T) / 2
evals21, evecs21 = np.linalg.eigh(B_H21)
idx21 = np.argmax(np.abs(evals21))
v21 = evecs21[:, idx21]
phases21 = np.unwrap(np.angle(v21))
k21 = np.polyfit(np.arange(4, dtype=float), phases21, 1)[0]
print(f"\nMomentum k = {k21:.4f}  (opposite direction: {k21:.4f} vs {k:.4f})")
print(f"→ SAME boson, opposite momentum — particle vs antiparticle?")

# ── The (3,12) candidate: block hop ──────────────────────────────────────────
print("\n" + "=" * 60)
print("BLOCK HOP: video (3, 12)")
print("frames: [1 1 0 0] → [0 0 1 1]")
print("=" * 60)
wf312 = video_to_wf((3, 12))
print(f"C_s = {wf312.spectral_complexity():.4f}")
psi312 = wf312.psi
print(f"ψ = {np.round(psi312, 4)}")
rho312 = np.outer(psi312, psi312.conj())
B312 = rho312 - np.diag(np.diag(rho312))
B_H312 = (B312 + B312.conj().T)/2
evals312, evecs312 = np.linalg.eigh(B_H312)
idx312 = np.argmax(np.abs(evals312))
v312 = evecs312[:, idx312]
phases312 = np.unwrap(np.angle(v312))
k312 = np.polyfit(np.arange(4, dtype=float), phases312, 1)[0]
print(f"Momentum k = {k312:.4f}  = {k312/(2*np.pi):.4f} × 2π")
print(f"Eigenvalues: {np.round(evals312, 4)}")
print(f"Dominant eigenvector phases/π: {np.round(np.angle(v312)/np.pi, 4)}")
