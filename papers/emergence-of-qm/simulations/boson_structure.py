"""
Boson Structure Experiment
==========================
Building on minimal_gif.py. For each 2-frame video we already know the
boson signal norm. Now we ask THREE questions about the off-diagonal
(boson candidate) matrix:
 
  Q1. SYMMETRY    — is the off-diagonal symmetric (spin-0) or
                    antisymmetric (spin-1 candidate)?
 
  Q2. PROPAGATION — does the boson signal have a spatial momentum?
                    We extract this by diagonalising the off-diagonal
                    matrix and looking at its eigenvector structure.
 
  Q3. QUANTISATION — does the boson signal come in discrete units
                     related to C_s?
 
We then look for videos where all three properties point to a
photon-like object: antisymmetric, propagating, quantised.
"""
 
import numpy as np
from itertools import product
import sys
sys.path.insert(0, '/home/claude')
from wavefunction import Wavefunction
 
N_PIXELS = 4
N_FRAMES = 2
N_CONFIGS = 2 ** N_PIXELS
 
 
# ── helpers ──────────────────────────────────────────────────────────────────
 
def frame_to_array(frame_int, n=N_PIXELS):
    return np.array([(frame_int >> i) & 1 for i in range(n)], dtype=float)
 
 
def video_to_wavefunction(frame_sequence):
    f0 = frame_to_array(frame_sequence[0])
    f1 = frame_to_array(frame_sequence[1])
    psi = f0 + 1j * f1
    if np.all(np.abs(psi) == 0):
        psi = np.ones(N_PIXELS, dtype=complex)
    return Wavefunction(psi, dx=1.0)
 
 
def boson_matrix(wf):
    """Full off-diagonal density matrix — the boson candidate."""
    psi = wf.psi
    rho = np.outer(psi, psi.conj())
    return rho - np.diag(np.diag(rho))
 
 
def symmetry_score(B):
    """
    +1 = perfectly symmetric  (spin-0 candidate)
    -1 = perfectly antisymmetric (spin-1 candidate)
     0 = mixed
    Computed as (sym_norm - asym_norm) / total_norm.
    """
    sym  = (B + B.T) / 2
    asym = (B - B.T) / 2
    s = np.linalg.norm(sym)
    a = np.linalg.norm(asym)
    total = s + a
    if total < 1e-12: return 0.0
    return float((s - a) / total)
 
 
def propagation_score(B):
    """
    Diagonalise B (Hermitian part) and measure how 'plane-wave-like'
    the dominant eigenvector is.
    A plane wave over N sites has equal amplitudes and linearly
    increasing phase: ψ_k(x) = e^{ikx}/√N.
    Score = 1 means perfect plane wave, 0 means localised.
    We use the Hermitian part B_H = (B + B†)/2.
    """
    B_H = (B + B.conj().T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(B_H)
    # dominant eigenvector (largest |eigenvalue|)
    idx = np.argmax(np.abs(eigenvalues))
    v = eigenvectors[:, idx]
    # plane-wave test: all amplitudes equal?
    amps = np.abs(v)
    amp_uniformity = 1.0 - float(np.std(amps) / (np.mean(amps) + 1e-12))
    amp_uniformity = max(0.0, amp_uniformity)
    # phase linearity test: does phase increase linearly with site?
    phases = np.unwrap(np.angle(v))
    x = np.arange(len(phases), dtype=float)
    if len(x) > 1:
        coeffs = np.polyfit(x, phases, 1)
        residuals = phases - np.polyval(coeffs, x)
        phase_linearity = 1.0 - float(np.std(residuals) / (np.pi + 1e-12))
        phase_linearity = max(0.0, phase_linearity)
    else:
        phase_linearity = 1.0
    return float(amp_uniformity * phase_linearity), float(coeffs[0]) if len(x) > 1 else 0.0
 
 
def quantisation_score(B, cs):
    """
    Is the boson signal norm related to C_s by a simple rational?
    Score = 1 if ||B|| ≈ m * 2^{-C_s} for small integer m.
    """
    norm = np.linalg.norm(B)
    if norm < 1e-12: return 0.0, 0
    unit = 2.0 ** (-cs) if cs > 0 else 1.0
    ratio = norm / unit
    nearest_int = round(ratio)
    if nearest_int == 0: return 0.0, 0
    score = 1.0 - abs(ratio - nearest_int) / nearest_int
    return float(max(0.0, score)), int(nearest_int)
 
 
# ── main experiment ───────────────────────────────────────────────────────────
 
all_videos = list(product(range(N_CONFIGS), repeat=N_FRAMES))
 
results = []
for video in all_videos:
    wf  = video_to_wavefunction(video)
    cs  = wf.spectral_complexity()
    B   = boson_matrix(wf)
    bn  = float(np.linalg.norm(B))
 
    sym  = symmetry_score(B)
    prop, momentum = propagation_score(B)
    qscore, qnum   = quantisation_score(B, cs)
 
    # composite "photon score": antisymmetric + propagating + quantised
    # antisymmetry → sym score should be NEGATIVE (asym dominates)
    antisym = max(0.0, -sym)
    photon_score = antisym * prop * (qscore if qscore > 0 else 0.01)
 
    results.append({
        'video':        video,
        'cs':           cs,
        'bn':           bn,
        'sym':          sym,
        'antisym':      antisym,
        'prop':         prop,
        'momentum':     momentum,
        'qscore':       qscore,
        'qnum':         qnum,
        'photon_score': photon_score,
        'f0':           frame_to_array(video[0]).astype(int),
        'f1':           frame_to_array(video[1]).astype(int),
    })
 
# ── Q1: SYMMETRY ─────────────────────────────────────────────────────────────
print("=" * 70)
print("Q1. SYMMETRY OF BOSON MATRIX")
print("    +1 = symmetric (spin-0)   -1 = antisymmetric (spin-1 candidate)")
print("=" * 70)
print(f"\n{'Video':>10}  {'C_s':>6}  {'||B||':>8}  {'Sym score':>10}  {'Character'}")
print("-" * 60)
 
# show spread across symmetry values
by_sym = sorted(results, key=lambda r: r['sym'])
shown = set()
for r in by_sym[:5] + by_sym[-5:]:
    key = round(r['sym'], 2)
    if key in shown: continue
    shown.add(key)
    char = 'antisymmetric' if r['sym'] < -0.5 else ('symmetric' if r['sym'] > 0.5 else 'mixed')
    print(f"{str(r['video']):>10}  {r['cs']:>6.2f}  {r['bn']:>8.4f}  {r['sym']:>10.4f}  {char}")
 
# ── Q2: PROPAGATION ───────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Q2. PROPAGATION — plane-wave score of dominant boson eigenvector")
print("    1.0 = perfect plane wave (propagating)  0.0 = localised")
print("=" * 70)
print(f"\n{'Video':>10}  {'C_s':>6}  {'Prop score':>11}  {'Momentum k':>12}  pixels")
print("-" * 65)
 
by_prop = sorted(results, key=lambda r: -r['prop'])
for r in by_prop[:10]:
    print(f"{str(r['video']):>10}  {r['cs']:>6.2f}  {r['prop']:>11.4f}  {r['momentum']:>12.4f}  {r['f0']}→{r['f1']}")
 
# ── Q3: QUANTISATION ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Q3. QUANTISATION — is ||B|| = m × 2^{-C_s} for integer m?")
print("=" * 70)
print(f"\n{'Video':>10}  {'C_s':>6}  {'||B||':>8}  {'Q score':>8}  {'m':>4}  {'2^{-Cs}':>10}")
print("-" * 60)
 
by_q = sorted(results, key=lambda r: (-r['qscore'], r['cs']))
shown_cs = set()
for r in by_q[:12]:
    key = round(r['cs'], 1)
    if key in shown_cs: continue
    shown_cs.add(key)
    unit = 2.0**(-r['cs']) if r['cs'] > 0 else 1.0
    print(f"{str(r['video']):>10}  {r['cs']:>6.2f}  {r['bn']:>8.4f}  {r['qscore']:>8.4f}  {r['qnum']:>4}  {unit:>10.6f}")
 
# ── PHOTON CANDIDATES ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("PHOTON CANDIDATES — antisymmetric + propagating + quantised")
print("=" * 70)
print(f"\n{'Video':>10}  {'C_s':>6}  {'Antisym':>8}  {'Prop':>8}  {'Q':>8}  {'Photon score':>13}  pixels")
print("-" * 80)
 
by_photon = sorted(results, key=lambda r: -r['photon_score'])
for r in by_photon[:10]:
    print(f"{str(r['video']):>10}  {r['cs']:>6.2f}  "
          f"{r['antisym']:>8.4f}  {r['prop']:>8.4f}  {r['qscore']:>8.4f}  "
          f"{r['photon_score']:>13.6f}  {r['f0']}→{r['f1']}")
 
# ── BOSON MATRIX OF TOP CANDIDATE ────────────────────────────────────────────
print("\n" + "=" * 70)
print("BOSON MATRIX OF TOP PHOTON CANDIDATE")
print("=" * 70)
 
top = by_photon[0]
wf_top = video_to_wavefunction(top['video'])
B_top  = boson_matrix(wf_top)
 
print(f"\nVideo: {top['video']}  frames: {top['f0']} → {top['f1']}")
print(f"C_s = {top['cs']:.4f}   ||B|| = {top['bn']:.6f}")
print(f"Symmetry score = {top['sym']:.4f}  (negative = antisymmetric)")
print(f"Propagation score = {top['prop']:.4f}  momentum k = {top['momentum']:.4f}")
print(f"Quantisation score = {top['qscore']:.4f}  m = {top['qnum']}")
print(f"\nBoson matrix B (off-diagonal density matrix):")
print(np.round(B_top, 4))
 
# Eigendecomposition
B_H = (B_top + B_top.conj().T) / 2
evals, evecs = np.linalg.eigh(B_H)
print(f"\nEigenvalues of B_H: {np.round(evals, 4)}")
print(f"Dominant eigenvector (boson mode):")
v = evecs[:, np.argmax(np.abs(evals))]
print(f"  amplitudes: {np.round(np.abs(v), 4)}")
print(f"  phases:     {np.round(np.angle(v), 4)} rad")
print(f"  phases/π:   {np.round(np.angle(v)/np.pi, 4)}")
 
# Is it a plane wave? Print explicitly
print(f"\nPlane wave check (e^{{ikx}}, k={top['momentum']:.4f}):")
x = np.arange(N_PIXELS, dtype=float)
plane = np.exp(1j * top['momentum'] * x)
plane /= np.linalg.norm(plane)
overlap = abs(np.dot(v.conj(), plane))
print(f"  |⟨boson mode | plane wave⟩| = {overlap:.6f}  (1.0 = exact plane wave)")
