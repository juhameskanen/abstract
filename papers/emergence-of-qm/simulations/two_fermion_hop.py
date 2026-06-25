"""
Two-fermion hop analysis.
Does the universal π/2 phase break for two fermions?
What replaces it? Is the breaking physically meaningful?
"""
import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from wavefunction import Wavefunction

def make_wf(f0, f1):
    psi = np.array(f0, dtype=float) + 1j * np.array(f1, dtype=float)
    if np.all(np.abs(psi) == 0):
        psi = np.ones(len(f0), dtype=complex)
    return Wavefunction(psi, dx=1.0)

def boson_matrix(wf):
    psi = wf.psi
    rho = np.outer(psi, psi.conj())
    return rho - np.diag(np.diag(rho))

def analyse_B(B, label=""):
    """Extract all off-diagonal phases and amplitudes."""
    L = B.shape[0]
    entries = []
    for i in range(L):
        for j in range(i+1, L):
            val = B[i, j]
            if abs(val) > 1e-10:
                entries.append({
                    'i': i, 'j': j,
                    'amp': abs(val),
                    'phase': np.angle(val),
                    'phase_pi': np.angle(val) / np.pi,
                    'val': val
                })
    return entries

def print_entries(entries, label):
    print(f"\n{label}")
    print(f"  {'(i,j)':>8}  {'|B_ij|':>8}  {'phase':>8}  {'phase/π':>8}  {'B_ij':>20}")
    print(f"  {'-'*60}")
    for e in entries:
        print(f"  ({e['i']},{e['j']}):  {e['amp']:>8.4f}  {e['phase']:>8.4f}  "
              f"{e['phase_pi']:>8.4f}  {e['val'].real:>+8.4f}{e['val'].imag:>+8.4f}j")

L = 8

# ── 1-fermion reference ───────────────────────────────────────────────────────
print("=" * 65)
print("REFERENCE: 1-fermion hop 0→1  (universal π/2 phase)")
print("=" * 65)
f0 = np.zeros(L); f0[0] = 1.0
f1 = np.zeros(L); f1[1] = 1.0
wf1 = make_wf(f0, f1)
B1 = boson_matrix(wf1)
e1 = analyse_B(B1)
print_entries(e1, "1-fermion 0→1")
print(f"\n  Eigenvalues: {np.round(np.linalg.eigvalsh((B1+B1.conj().T)/2), 4)}")

# ── 2-fermion adjacent block hop ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("2-FERMION HOPS — varying configurations")
print("=" * 65)

cases = [
    ("adjacent block  01→12", [0,1], [1,2]),
    ("adjacent block  12→23", [1,2], [2,3]),
    ("spread hop      02→13", [0,2], [1,3]),
    ("spread hop      03→14", [0,3], [1,4]),
    ("same-dir long   01→23", [0,1], [2,3]),
    ("opposite        01→67", [0,1], [6,7]),
    ("crossing        03→41", [0,3], [4,1]),  # one goes right, one left
]

phase_summary = []

for label, sites_from, sites_to in cases:
    f0 = np.zeros(L); f1 = np.zeros(L)
    for s in sites_from: f0[s] = 1.0
    for s in sites_to:   f1[s] = 1.0
    wf = make_wf(f0, f1)
    B = boson_matrix(wf)
    entries = analyse_B(B)
    evals = np.linalg.eigvalsh((B + B.conj().T)/2)
    print_entries(entries, f"2f {label}")
    print(f"  Eigenvalues: {np.round(evals, 4)}")
    phases = [e['phase_pi'] for e in entries]
    amps   = [e['amp']      for e in entries]
    phase_summary.append((label, phases, amps, evals))

# ── Key question: what phases appear? ────────────────────────────────────────
print("\n" + "=" * 65)
print("PHASE SUMMARY — what phases appear in 2-fermion bosons?")
print("(1-fermion always gives -π/2 = -0.5π)")
print("=" * 65)
print(f"\n  {'Case':>30}  {'Phases/π':>30}  {'Amplitudes'}")
print(f"  {'-'*80}")
for label, phases, amps, evals in phase_summary:
    ph_str = " ".join(f"{p:+.3f}" for p in sorted(set(round(p,3) for p in phases)))
    am_str = " ".join(f"{a:.3f}" for a in sorted(set(round(a,4) for a in amps)))
    print(f"  {label:>30}  {ph_str:>30}  {am_str}")

# ── The crossing case: one fermion goes right, one goes left ──────────────────
print("\n" + "=" * 65)
print("SPECIAL: CROSSING HOP — fermion A: 0→4, fermion B: 3→1")
print("(fermions pass through each other — Pauli violation?)")
print("=" * 65)
f0c = np.zeros(L); f0c[0]=1; f0c[3]=1
f1c = np.zeros(L); f1c[4]=1; f1c[1]=1
wfc = make_wf(f0c, f1c)
Bc = boson_matrix(wfc)
ec = analyse_B(Bc)
print_entries(ec, "crossing 03→41")
evalsc = np.linalg.eigvalsh((Bc+Bc.conj().T)/2)
print(f"\n  Eigenvalues: {np.round(evalsc, 4)}")

# Compare: non-crossing version 03→14
f0n = np.zeros(L); f0n[0]=1; f0n[3]=1
f1n = np.zeros(L); f1n[1]=1; f1n[4]=1
wfn = make_wf(f0n, f1n)
Bn = boson_matrix(wfn)
en = analyse_B(Bn)
print_entries(en, "non-crossing 03→14")
evalsn = np.linalg.eigvalsh((Bn+Bn.conj().T)/2)
print(f"\n  Eigenvalues: {np.round(evalsn, 4)}")

print(f"\n  Crossing eigenvalue sum:     {sum(abs(evalsc)):.4f}")
print(f"  Non-crossing eigenvalue sum: {sum(abs(evalsn)):.4f}")
print(f"  → {'DIFFERENT — crossing is distinguishable from non-crossing' if abs(sum(abs(evalsc))-sum(abs(evalsn)))>1e-6 else 'SAME — crossing indistinguishable'}")

# ── What does Pauli exclusion look like in this framework? ────────────────────
print("\n" + "=" * 65)
print("PAULI TEST: two fermions trying to occupy the SAME site")
print("=" * 65)
for config in [
    ("same site 0→0 (both)", [0,0], [0,0]),
    ("both hop to same",     [0,1], [2,2]),
    ("one stays one hops to occupied", [0,1], [0,0]),
]:
    label, sf, st = config
    f0p = np.zeros(L); f1p = np.zeros(L)
    for s in sf: f0p[s] = 1.0   # double occupancy just adds amplitude
    for s in st: f1p[s] = 1.0
    try:
        wfp = make_wf(f0p, f1p)
        Bp = boson_matrix(wfp)
        ep = analyse_B(Bp)
        evalsp = np.linalg.eigvalsh((Bp+Bp.conj().T)/2)
        norm_val = float(np.sqrt(np.sum(np.abs(f0p + 1j*f1p)**2)))
        print(f"\n  {label}")
        print(f"  ||ψ|| before normalisation = {norm_val:.4f}")
        print(f"  Eigenvalues: {np.round(evalsp,4)}")
        print_entries(ep, f"  boson matrix")
    except Exception as ex:
        print(f"\n  {label} → ERROR: {ex}")

