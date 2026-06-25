"""
Scaling Experiment
==================
Test whether the boson momentum quantisation k = 2π·m/L holds
as we scale up:
  - Chain length L = 4, 6, 8, 12, 16 sites
  - Single fermion hop (nearest neighbour and longer range)
  - Two-fermion hop
  - Three frames (two hops in sequence)
 
For each case we extract:
  - Boson momentum k from dominant eigenvector
  - Expected k = 2π·m/L for integer m
  - Deviation from quantisation
  - Antisymmetry score
  - Propagation score
"""
 
import numpy as np
import sys
sys.path.insert(0, '/home/claude')
from wavefunction import Wavefunction
 
# ── helpers ──────────────────────────────────────────────────────────────────
 
def make_wf(frames, dx=1.0):
    """
    Encode a sequence of frames as a wavefunction.
    For 2 frames: ψ = f0 + i·f1  (as before)
    For 3 frames: ψ = f0 + i·f1, then second wf = f1 + i·f2
                  return both; we analyse each transition.
    """
    if len(frames) == 2:
        f0 = np.array(frames[0], dtype=float)
        f1 = np.array(frames[1], dtype=float)
        psi = f0 + 1j * f1
        if np.all(np.abs(psi) == 0):
            psi = np.ones(len(f0), dtype=complex)
        return [Wavefunction(psi, dx=dx)]
    else:
        wfs = []
        for i in range(len(frames) - 1):
            f0 = np.array(frames[i],   dtype=float)
            f1 = np.array(frames[i+1], dtype=float)
            psi = f0 + 1j * f1
            if np.all(np.abs(psi) == 0):
                psi = np.ones(len(f0), dtype=complex)
            wfs.append(Wavefunction(psi, dx=dx))
        return wfs
 
 
def boson_matrix(wf):
    psi = wf.psi
    rho = np.outer(psi, psi.conj())
    return rho - np.diag(np.diag(rho))
 
 
def analyse_boson(B, L):
    """Extract momentum, antisymmetry, propagation from boson matrix B."""
    # Antisymmetry
    sym  = (B + B.T) / 2
    asym = (B - B.T) / 2
    s = np.linalg.norm(sym)
    a = np.linalg.norm(asym)
    total = s + a
    antisym_score = float(a / total) if total > 1e-12 else 0.0
 
    # Dominant eigenvector of Hermitian part
    B_H = (B + B.conj().T) / 2
    evals, evecs = np.linalg.eigh(B_H)
    idx = np.argmax(np.abs(evals))
    v = evecs[:, idx]
    dominant_eval = float(evals[idx])
 
    # Momentum from phase gradient
    phases = np.unwrap(np.angle(v))
    x = np.arange(L, dtype=float)
    coeffs = np.polyfit(x, phases, 1)
    k_measured = float(coeffs[0])
 
    # Nearest quantised k = 2π·m/L
    m_float = k_measured * L / (2 * np.pi)
    m_nearest = round(m_float)
    k_quantised = 2 * np.pi * m_nearest / L
    quant_dev = abs(k_measured - k_quantised)
 
    # Plane wave overlap
    plane = np.exp(1j * k_measured * x)
    plane /= np.linalg.norm(plane)
    overlap = float(abs(np.dot(v.conj(), plane)))
 
    # Amplitude uniformity
    amps = np.abs(v)
    amp_uni = 1.0 - float(np.std(amps) / (np.mean(amps) + 1e-12))
    amp_uni = max(0.0, amp_uni)
 
    return {
        'k':            k_measured,
        'k_q':          k_quantised,
        'm':            m_nearest,
        'quant_dev':    quant_dev,
        'antisym':      antisym_score,
        'overlap':      overlap,
        'amp_uni':      amp_uni,
        'eval_dom':     dominant_eval,
        'evals':        evals,
    }
 
 
def single_hop_frame(L, site_from, site_to):
    """Single fermion at site_from → site_to."""
    f0 = np.zeros(L); f0[site_from] = 1.0
    f1 = np.zeros(L); f1[site_to]   = 1.0
    return [f0, f1]
 
 
def block_hop_frame(L, sites_from, sites_to):
    """Block of fermions hop together."""
    f0 = np.zeros(L)
    f1 = np.zeros(L)
    for s in sites_from: f0[s] = 1.0
    for s in sites_to:   f1[s] = 1.0
    return [f0, f1]
 
 
# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Single fermion hop, vary chain length L and hop distance d
# ══════════════════════════════════════════════════════════════════════════════
 
print("=" * 72)
print("EXP 1: SINGLE FERMION HOP — momentum quantisation vs chain length")
print("       hop: site 0 → site d,  expected k = 2π·m/L")
print("=" * 72)
print(f"\n{'L':>4} {'d':>4} {'k_meas':>9} {'m':>4} {'k_quant':>9} {'dev':>8} "
      f"{'antisym':>8} {'overlap':>8} {'C_s':>6}")
print("-" * 75)
 
for L in [4, 6, 8, 12, 16]:
    for d in [1, 2, L//2]:
        if d >= L: continue
        frames = single_hop_frame(L, 0, d)
        wfs = make_wf(frames)
        wf = wfs[0]
        B = boson_matrix(wf)
        r = analyse_boson(B, L)
        cs = wf.spectral_complexity()
        print(f"{L:>4} {d:>4} {r['k']:>9.4f} {r['m']:>4} {r['k_q']:>9.4f} "
              f"{r['quant_dev']:>8.4f} {r['antisym']:>8.4f} {r['overlap']:>8.4f} {cs:>6.2f}")
    print()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Two-fermion block hop — does boson change?
# ══════════════════════════════════════════════════════════════════════════════
 
print("=" * 72)
print("EXP 2: TWO-FERMION BLOCK HOP — boson changes with fermion number?")
print("=" * 72)
print(f"\n{'L':>4} {'hop':>14} {'k_meas':>9} {'m':>4} {'k_quant':>9} "
      f"{'antisym':>8} {'overlap':>8} {'C_s':>6}")
print("-" * 72)
 
for L in [4, 8, 12]:
    # 1-fermion hop
    frames1 = single_hop_frame(L, 0, 1)
    wf1 = make_wf(frames1)[0]
    B1 = boson_matrix(wf1)
    r1 = analyse_boson(B1, L)
    cs1 = wf1.spectral_complexity()
    print(f"{L:>4} {'0→1 (1f)':>14} {r1['k']:>9.4f} {r1['m']:>4} {r1['k_q']:>9.4f} "
          f"{r1['antisym']:>8.4f} {r1['overlap']:>8.4f} {cs1:>6.2f}")
 
    # 2-fermion block hop  0,1 → 1,2
    frames2 = block_hop_frame(L, [0,1], [1,2])
    wf2 = make_wf(frames2)[0]
    B2 = boson_matrix(wf2)
    r2 = analyse_boson(B2, L)
    cs2 = wf2.spectral_complexity()
    print(f"{L:>4} {'01→12 (2f)':>14} {r2['k']:>9.4f} {r2['m']:>4} {r2['k_q']:>9.4f} "
          f"{r2['antisym']:>8.4f} {r2['overlap']:>8.4f} {cs2:>6.2f}")
 
    # 2-fermion non-adjacent hop  0,2 → 1,3
    frames3 = block_hop_frame(L, [0,2], [1,3])
    wf3 = make_wf(frames3)[0]
    B3 = boson_matrix(wf3)
    r3 = analyse_boson(B3, L)
    cs3 = wf3.spectral_complexity()
    print(f"{L:>4} {'02→13 (2f)':>14} {r3['k']:>9.4f} {r3['m']:>4} {r3['k_q']:>9.4f} "
          f"{r3['antisym']:>8.4f} {r3['overlap']:>8.4f} {cs3:>6.2f}")
    print()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Three frames — two sequential hops, same boson?
# ══════════════════════════════════════════════════════════════════════════════
 
print("=" * 72)
print("EXP 3: THREE FRAMES — fermion hops twice. Same boson each step?")
print("       L=8, fermion: site0 → site1 → site2")
print("=" * 72)
 
L = 8
f0 = np.zeros(L); f0[0] = 1.0
f1 = np.zeros(L); f1[1] = 1.0
f2 = np.zeros(L); f2[2] = 1.0
 
frames_3 = [f0, f1, f2]
wfs = make_wf(frames_3)
 
print(f"\nTransition 0→1 (frame 0 to frame 1):")
B01 = boson_matrix(wfs[0])
r01 = analyse_boson(B01, L)
cs01 = wfs[0].spectral_complexity()
print(f"  k = {r01['k']:.4f}  m = {r01['m']}  k_q = {r01['k_q']:.4f}  "
      f"antisym = {r01['antisym']:.4f}  overlap = {r01['overlap']:.4f}  C_s = {cs01:.2f}")
 
print(f"\nTransition 1→2 (frame 1 to frame 2):")
B12 = boson_matrix(wfs[1])
r12 = analyse_boson(B12, L)
cs12 = wfs[1].spectral_complexity()
print(f"  k = {r12['k']:.4f}  m = {r12['m']}  k_q = {r12['k_q']:.4f}  "
      f"antisym = {r12['antisym']:.4f}  overlap = {r12['overlap']:.4f}  C_s = {cs12:.2f}")
 
print(f"\nAre the two bosons identical?")
print(f"  Same k:       {abs(r01['k'] - r12['k']) < 0.001}")
print(f"  Same antisym: {abs(r01['antisym'] - r12['antisym']) < 0.001}")
print(f"  Same C_s:     {abs(cs01 - cs12) < 0.001}")
print(f"  → {'YES — same boson emitted at each step' if abs(r01['k']-r12['k'])<0.01 else 'NO — boson changes'}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Momentum spectrum — all single hops on L=8
# ══════════════════════════════════════════════════════════════════════════════
 
print("\n" + "=" * 72)
print("EXP 4: FULL MOMENTUM SPECTRUM — all hop distances on L=8")
print("       Does k scale linearly with hop distance?")
print("=" * 72)
print(f"\n{'hop d':>6} {'k_meas':>9} {'k/d':>9} {'m':>4} {'antisym':>8}")
print("-" * 45)
 
L = 8
k_values = []
for d in range(1, L):
    frames = single_hop_frame(L, 0, d)
    wf = make_wf(frames)[0]
    B = boson_matrix(wf)
    r = analyse_boson(B, L)
    k_values.append((d, r['k'], r['antisym']))
    print(f"{d:>6} {r['k']:>9.4f} {r['k']/d if d>0 else 0:>9.4f} {r['m']:>4} {r['antisym']:>8.4f}")
 
# Check linearity
ks = np.array([kv[1] for kv in k_values])
ds = np.array([kv[0] for kv in k_values])
coeffs = np.polyfit(ds, ks, 1)
residuals = ks - np.polyval(coeffs, ds)
print(f"\nLinear fit k = {coeffs[0]:.4f}·d + {coeffs[1]:.4f}")
print(f"R² = {1 - np.var(residuals)/np.var(ks):.6f}")
print(f"Expected slope for k=2π/L: {2*np.pi/L:.4f}")
print(f"Measured slope:            {coeffs[0]:.4f}")
print(f"Ratio measured/expected:   {coeffs[0]/(2*np.pi/L):.4f}")
