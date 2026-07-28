"""
Supplementary Script S4 — Particle Mass Spectrum in Log2 Space
===============================================================
Paper VIII: "Fermion Structure and Particle Classification from Codec Geometry"

Purpose
-------
Report the known particle masses in log2(m/m_e) space, and report a
separately-motivated combinatorial 4-bit argument, side by side --
WITHOUT asserting a derived connection between them. See Section 8
("Mass and the Missing Generation Index") of the paper and the
companion script audit_mass_ladder.py, which shows in detail why no
such connection is currently derivable.

What this script does NOT claim
--------------------------------
An earlier version of this script (and of the paper) presented these
numbers as a "mass ladder" prediction of the framework, including a
claimed falsifiable bound on new-particle masses. That claim has been
withdrawn (see Section 8 of the paper and audit_mass_ladder.py):
  - The (n,m) classification of Section 5 has no third parameter that
    could distinguish electron/muon/tau, all of which sit in the same
    class (2,1). No construction here computes C_s for "generation 2"
    or "generation 3" and gets 8 or 12 bits as output.
  - The two observed lepton gaps (7.69 and 4.07 bits) are not equal to
    each other and do not both round cleanly to a shared integer unit.
  - The "4-bit unit" is a separate, independently-motivated combinatorial
    count (four binary choices per fermion hop) that is not derived from,
    or connected by any construction to, the C_s formula of Paper VI or
    to these mass gaps.
This script therefore reports the raw log2 mass ratios and the 4-bit
combinatorial count as two independent, unconnected pieces of
information. Any apparent near-integer alignment is noted as an
observation, not evidence of a derivation.

Output
------
Table of known particle masses and log2(m/m_e).
The 4-bit combinatorial argument, reported separately.
No mass predictions and no falsifiability claims.

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
    """Print the raw particle mass table in log2 space. No claims of
    derivation -- see module docstring and audit_mass_ladder.py."""
    print("=" * 78)
    print("Particle masses in log2(m/m_e) space (raw observation, not a derived quantity)")
    print(f"Reference: electron mass m_e = {M_ELECTRON} MeV")
    print("=" * 78)
    print(f"  {'Particle':>14}  {'Mass (MeV)':>12}  {'log2(m/me)':>11}  "
          f"{'Nearest int':>11}  {'Deviation':>12}  {'Category':>8}")
    print("-" * 78)

    for name, mass, category in PARTICLES:
        log2_ratio   = np.log2(mass / M_ELECTRON)
        nearest_int  = round(log2_ratio)
        deviation    = nearest_int - log2_ratio
        print(f"  {name:>14}  {mass:>12.2f}  {log2_ratio:>11.4f}  "
              f"{nearest_int:>11d}  {deviation:>+12.4f}  {category:>8}")

    print()
    print("  'Deviation' is nearest-integer minus observed log2 ratio.")
    print("  No claim is made that particles should land on integers;")
    print("  this is a description of the raw numbers, not a fit or a")
    print("  compression-cost calculation. See audit_mass_ladder.py.")


def lepton_ladder() -> None:
    """Report the raw lepton log2 mass gaps. Does not assert they are
    equally-spaced or connected to a codec unit -- see audit_mass_ladder.py,
    which shows the two gaps (7.69, 4.07 bits) are unequal and do not both
    round cleanly to a shared integer unit."""
    print()
    print("=" * 60)
    print("Lepton log2 mass gaps (raw observation)")
    print("=" * 60)

    leptons = [(n, m, c) for n, m, c in PARTICLES if c == "lepton"]
    log2_masses = [(name, np.log2(mass / M_ELECTRON))
                   for name, mass, _ in leptons]

    print(f"\n  {'Particle':>12}  {'log2(m/me)':>11}  {'Gap from prev':>14}")
    print("-" * 50)

    prev = None
    for name, log2m in log2_masses:
        step = log2m - prev if prev is not None else 0.0
        step_str = f"{step:+.4f}" if prev is not None else "—"
        print(f"  {name:>12}  {log2m:>11.4f}  {step_str:>14}")
        prev = log2m

    print()
    print("  The two gaps (7.69, 4.07 bits) are NOT equal to each other,")
    print("  and 7.69/4 = 1.92 is not an integer, so they do not both")
    print("  round cleanly to a shared 4-bit unit. All three charged")
    print("  leptons occupy the single (n,m)=(2,1) class of Section 5;")
    print("  the classification has no third parameter to distinguish")
    print("  them. A generation index is an open problem (Section 10),")
    print("  not something this script demonstrates.")


def codec_unit_argument() -> None:
    """Show the 4-bit combinatorial count as an independently-motivated,
    UNCONNECTED argument -- not a prediction and not a fit to the masses
    above. See audit_mass_ladder.py for why no link is currently derivable."""
    print()
    print("=" * 60)
    print("A 4-bit combinatorial count (independent of the mass data above)")
    print("=" * 60)
    print()
    print("  One fermion hop involves 4 binary choices:")
    print("    Bit 1: which site (i or j)")
    print("    Bit 2: which frame (before or after hop)")
    print("    Bit 3: real or imaginary wavefunction component")
    print("    Bit 4: sign of the phase")
    print()
    print("  This gives 2^4 = 16 configurations per hop -- a combinatorial")
    print("  observation about the codec, not a mass formula.")
    print()
    print("  This count is NOT connected by any derivation in this")
    print("  framework to the C_s formula of Paper VI, nor to the log2")
    print("  mass gaps reported above. Presenting both numbers together")
    print("  is not evidence that they are related; see Section 8 of the")
    print("  paper and audit_mass_ladder.py for the full audit of why this")
    print("  connection does not currently exist and no mass-scale or")
    print("  new-particle predictions can be derived from it.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mass_spectrum_table()
    lepton_ladder()
    codec_unit_argument()
