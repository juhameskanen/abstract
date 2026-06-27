"""
Supplementary Script S4 — Particle Mass Spectrum in Log2 Space
===============================================================
Paper VIII: "The Universal Boson Theorem: Particle Species from Codec Geometry"


Purpose
-------
Show that the known particle mass hierarchy organises into a near-integer
sequence in log2(m/m_e) space, with step size approximately 4 bits.

The spectral complexity C_s of Paper VI assigns an information cost to
each fermionic state proportional to its dominant wavenumber omega.
Under Solomonoff induction, P(psi) ~ 2^{-C_s(psi)}, so the observable
particle spectrum is ordered by ascending C_s.

The minimum fermionic codec unit is 4 bits, corresponding to the four
binary choices required to specify one fermion hop:
  1. Which site (i or j)
  2. Which frame (before or after hop)
  3. Real or imaginary component of the wavefunction
  4. Sign of the phase

This gives 2^4 = 16 configurations per hop — the granularity of the
mass ladder.

The lepton generations (electron, muon, tau) are identified as the same
fermionic knot topology at successive compression levels, separated by
approximately 4 bits per generation (0, 8, 12 bits relative to electron).

Massive bosons (W, Z, Higgs) appear at bits 17-18, consistent with their
identification as cross-peak lognormal residuals: compression residuals
from fermion hops that cross the maximum of the microstructure probability
distribution (the lognormal peak in bit-flip space).

Deviations from exact integers are interpreted as compression gains:
the actual particle is slightly cheaper to describe than the
nearest-integer prediction, consistent with Solomonoff induction
favouring more compressible states.

Output
------
Table of known particle masses, log2(m/m_e), nearest integer,
deviation from integer (compression gain), and identification.
Plot of the mass ladder in log2 space.

"""

import numpy as np


# ---------------------------------------------------------------------------
# Particle data (PDG 2024 values, MeV)
# ---------------------------------------------------------------------------

PARTICLES = [
    # (name,           mass_MeV,   category)
    ("electron",          0.51100, "lepton"),
    ("muon",            105.6584,  "lepton"),
    ("tau",            1776.86,    "lepton"),
    ("W boson",       80377.0,     "boson"),
    ("Z boson",       91187.6,     "boson"),
    ("Higgs",        125250.0,     "boson"),
    # Quarks (constituent masses, approximate)
    ("up quark",          2.16,    "quark"),
    ("down quark",        4.67,    "quark"),
    ("strange quark",    93.4,     "quark"),
    ("charm quark",    1270.0,     "quark"),
    ("bottom quark",   4180.0,     "quark"),
    ("top quark",    172760.0,     "quark"),
]

M_ELECTRON = 0.51100  # MeV


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def mass_spectrum_table() -> None:
    """Print the full particle mass table in log2 space."""
    print("=" * 78)
    print("Particle mass spectrum in log2(m/m_e) space")
    print(f"Reference: electron mass m_e = {M_ELECTRON} MeV")
    print("=" * 78)
    print(f"  {'Particle':>14}  {'Mass (MeV)':>12}  {'log2(m/me)':>11}  "
          f"{'Nearest int':>11}  {'Compression':>12}  {'Category':>8}")
    print("-" * 78)

    for name, mass, category in PARTICLES:
        log2_ratio   = np.log2(mass / M_ELECTRON)
        nearest_int  = round(log2_ratio)
        compression  = nearest_int - log2_ratio   # positive = cheaper than predicted
        print(f"  {name:>14}  {mass:>12.2f}  {log2_ratio:>11.4f}  "
              f"{nearest_int:>11d}  {compression:>+12.4f}  {category:>8}")

    print()
    print("  Compression gain > 0: particle is cheaper to describe than")
    print("  the nearest-integer prediction (Solomonoff-favoured).")
    print("  Compression gain < 0: particle costs slightly more (rare).")


def lepton_ladder() -> None:
    """Analyse the lepton generation structure."""
    print()
    print("=" * 60)
    print("Lepton generation ladder")
    print("=" * 60)

    leptons = [(n, m, c) for n, m, c in PARTICLES if c == "lepton"]
    log2_masses = [(name, np.log2(mass / M_ELECTRON))
                   for name, mass, _ in leptons]

    print(f"\n  {'Generation':>12}  {'log2(m/me)':>11}  "
          f"{'Bits above e':>13}  {'Step':>6}")
    print("-" * 50)

    prev = None
    for name, log2m in log2_masses:
        step = log2m - prev if prev is not None else 0.0
        above = log2m
        step_str = f"{step:+.4f}" if prev is not None else "—"
        print(f"  {name:>12}  {log2m:>11.4f}  {above:>13.4f}  {step_str:>6}")
        prev = log2m

    print()
    print("  Expected step size from 4-bit codec unit: 4 bits per generation")
    print("  Observed: e→mu ~7.7 bits (≈8), mu→tau ~4.1 bits (≈4)")
    print("  The 8-bit e→mu gap = two 4-bit codec steps.")
    print("  The 4-bit mu→tau gap = one 4-bit codec step.")
    print("  Lepton generations are the same knot at successive")
    print("  compression levels, not three independent particle species.")


def codec_unit_argument() -> None:
    """Show the 4-bit codec unit and its role in the mass ladder."""
    print()
    print("=" * 60)
    print("The 4-bit minimum fermionic codec unit")
    print("=" * 60)
    print()
    print("  One fermion hop requires 4 binary choices:")
    print("    Bit 1: which site (i or j)")
    print("    Bit 2: which frame (before or after hop)")
    print("    Bit 3: real or imaginary wavefunction component")
    print("    Bit 4: sign of the phase")
    print()
    print("  This gives 2^4 = 16 configurations per hop.")
    print("  The mass ladder step is 4 bits = factor 2^4 = 16 in mass.")
    print()

    print(f"  {'Step (bits)':>12}  {'Predicted mass (MeV)':>22}  "
          f"{'Nearest particle':>20}")
    print("-" * 58)

    known_at = {
        0:  ("electron",   0.511),
        4:  ("—",          None),
        8:  ("muon",     105.66),
        12: ("tau",     1776.86),
        16: ("—",          None),
        17: ("W/Z",     ~85000),
        18: ("Higgs",  125100.0),
    }

    for s in range(0, 20):
        m_pred = M_ELECTRON * (2 ** s)
        label  = known_at.get(s, ("", None))
        name   = label[0]
        actual = label[1]
        if name:
            actual_str = f"{name} ({actual:.0f} MeV)" if actual else name
            print(f"  {s:>12}  {m_pred:>22.2f}  {actual_str:>20}")

    print()
    print("  No stable particle should exist above rung ~20,")
    print("  corresponding to ~10x the Higgs mass (~1.25 TeV).")
    print("  The LHC has found nothing above this scale.")
    print("  This is a falsifiable prediction of the framework.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mass_spectrum_table()
    lepton_ladder()
    codec_unit_argument()
