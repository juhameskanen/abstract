
"""
Minimal GIF experiment: 2x2 pixels, 2 frames.
Find minimum-complexity wavefunction over all frame sequences.
Decompose into fermionic (single-site) vs correlation (boson candidate) terms.
"""

import numpy as np
from itertools import product
from wavefunction import Wavefunction

# ─── 1. State space ───────────────────────────────────────────────────────────
# 2x2 grid = 4 pixels, each 0 or 1.
# A "frame" is a 4-bit string. 2 frames = an 8-bit "video".
# We encode a 2-frame video as a wavefunction over the 4-pixel space
# by treating the sequence of frames as a time-ordered signal.

N_PIXELS = 4          # 2x2
N_FRAMES = 2
N_CONFIGS = 2**N_PIXELS  # 16 possible frames

def frame_to_array(frame_int, n=N_PIXELS):
    """Integer → binary pixel array, e.g. 5 → [0,1,0,1]"""
    return np.array([(frame_int >> i) & 1 for i in range(n)], dtype=float)

def video_to_wavefunction(frame_sequence):
    """
    Encode a sequence of frames as a complex wavefunction.
    Strategy: treat pixel index as spatial coordinate x.
    Frame 0 → real part contribution, Frame 1 → imaginary part contribution.
    This is the minimal complex encoding of a 2-frame video.
    Result: ψ(x) = f0(x) + i·f1(x), normalised.
    """
    f0 = frame_to_array(frame_sequence[0])
    f1 = frame_to_array(frame_sequence[1])
    psi = f0 + 1j * f1
    # Handle zero wavefunction (all-zero frames)
    if np.all(np.abs(psi) == 0):
        psi = np.ones(N_PIXELS, dtype=complex)
    return Wavefunction(psi, dx=1.0)

# ─── 2. Enumerate all 2-frame videos ─────────────────────────────────────────
print("=" * 60)
print("MINIMAL GIF EXPERIMENT: 2×2 pixels, 2 frames")
print("=" * 60)

all_videos = list(product(range(N_CONFIGS), repeat=N_FRAMES))
print(f"\nTotal 2-frame videos: {len(all_videos)}")

# Score each video by wavefunction complexity
results = []
for video in all_videos:
    wf = video_to_wavefunction(video)
    cs = wf.spectral_complexity()
    sw = wf.solomonoff_weight()
    results.append((video, wf, cs, sw))

results.sort(key=lambda r: r[2])  # sort by complexity ascending

# ─── 3. Show minimum complexity videos ───────────────────────────────────────
print("\n── Top 10 minimum-complexity videos ──")
print(f"{'Frame0':>8} {'Frame1':>8} {'C_s':>8} {'S-weight':>12}  pixels_f0 → pixels_f1")
print("-" * 70)
for video, wf, cs, sw in results[:10]:
    f0 = frame_to_array(video[0]).astype(int)
    f1 = frame_to_array(video[1]).astype(int)
    print(f"{video[0]:>8} {video[1]:>8} {cs:>8.3f} {sw:>12.6f}  {f0} → {f1}")

# ─── 4. Decompose minimum wavefunction ───────────────────────────────────────
print("\n── Decomposition of minimum-complexity wavefunction ──")
best_video, best_wf, best_cs, best_sw = results[0]
f0 = frame_to_array(best_video[0])
f1 = frame_to_array(best_video[1])
psi = best_wf.psi

print(f"\nBest video: frames {best_video[0]} → {best_video[1]}")
print(f"Frame 0 pixels: {f0.astype(int)}")
print(f"Frame 1 pixels: {f1.astype(int)}")
print(f"C_s = {best_cs:.4f}")
print(f"\nψ(x) = {psi}")

# Fermionic term: product state (no correlations)
# ψ_fermi(x) = independent per-site amplitudes
# = |amplitude at each pixel| — what you'd get with no inter-pixel correlations
fermi_amplitudes = np.abs(psi)
fermi_phases = np.angle(psi)
print(f"\nFermionic (per-site) amplitudes: {fermi_amplitudes}")
print(f"Fermionic (per-site) phases:     {fermi_phases}")

# Bosonic residual: what's LEFT after removing per-site structure
# = inter-site correlations = off-diagonal density matrix elements
rho = np.outer(psi, psi.conj())
print(f"\nDensity matrix ρ = |ψ⟩⟨ψ|:")
print(np.round(rho, 4))

# Diagonal = fermionic (single-site occupation probabilities)
diag = np.diag(np.diag(rho))
print(f"\nFermionic part (diagonal ρ):")
print(np.round(diag, 4))

# Off-diagonal = bosonic candidate (inter-site correlations)
offdiag = rho - diag
print(f"\nBosonic candidate (off-diagonal ρ — inter-site correlations):")
print(np.round(offdiag, 4))

offdiag_norm = np.linalg.norm(offdiag)
print(f"\n||off-diagonal|| = {offdiag_norm:.6f}")
if offdiag_norm < 1e-10:
    print("→ No correlations: product state, no boson emerged.")
else:
    print("→ Non-zero correlations: BOSON CANDIDATE PRESENT.")

# ─── 5. Now find the video where boson signal is STRONGEST ───────────────────
print("\n\n── Videos ranked by boson signal strength (off-diagonal norm) ──")
boson_results = []
for video, wf, cs, sw in results:
    psi = wf.psi
    rho = np.outer(psi, psi.conj())
    offdiag = rho - np.diag(np.diag(rho))
    boson_signal = np.linalg.norm(offdiag)
    boson_results.append((video, cs, sw, boson_signal))

boson_results.sort(key=lambda r: (-r[3], r[1]))  # max boson signal, min complexity

print(f"\n{'Frame0':>8} {'Frame1':>8} {'C_s':>8} {'Boson signal':>14}")
print("-" * 50)
for video, cs, sw, bs in boson_results[:10]:
    f0 = frame_to_array(video[0]).astype(int)
    f1 = frame_to_array(video[1]).astype(int)
    print(f"{video[0]:>8} {video[1]:>8} {cs:>8.3f} {bs:>14.6f}  {f0}→{f1}")

# ─── 6. The key question: does minimum complexity REQUIRE boson signal? ───────
print("\n\n── Key question: complexity vs boson signal ──")
print("(Is there a trade-off? Does compressibility require correlations?)\n")
print(f"{'C_s':>8} {'Boson signal':>14} {'Video':>16}")
print("-" * 45)
# Show unique complexity levels
seen_cs = set()
for video, cs, sw, bs in sorted(boson_results, key=lambda r: r[1]):
    cs_round = round(cs, 2)
    if cs_round not in seen_cs:
        seen_cs.add(cs_round)
        print(f"{cs:>8.3f} {bs:>14.6f}  {video[0]:>3}→{video[1]:<3}")
    if len(seen_cs) > 12:
        break

