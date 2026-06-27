"""
Supplementary Script S3 — Charge, Neutrinos, and Colour Confinement
====================================================================
Paper VIII: "Fermion-Boson Duality: Particle Classification from
Codec Geometry"
IAME Collaboration, Juha Meskanen, June 2026

Purpose
-------
Demonstrate three codec derivations:

1. CHARGE FROM WINDING NUMBER (n=2 sector)
   Within the n=2 sector, the winding number m acts as the charge
   selector:
     m=0: zero phase winding, zero electromagnetic coupling -> neutrino
     m=1: one phase winding, unit electromagnetic coupling -> charged lepton

   Electric charge is the U(1) winding number within the n=2 sector.
   This gives a clean algorithmic rule with no inconsistency:
   - Neutrino (n=2, m=0): charge 0
   - Charged lepton (n=2, m=1): charge 1
   Both emerge from the same two-site codec geometry.

2. FRACTIONAL OCCUPANCY (n=3 sector)
   Within the n=3 sector, each site carries probability |psi_k|^2 = 1/3.
   This is the per-site occupancy of a quark in a colour triplet.
   The charge mechanism here is distinct from the n=2 sector:
   n=3 charge arises from per-site occupancy, not winding number.
   A unified charge formula covering both sectors is an open problem.

3. COLOUR CONFINEMENT (codec argument)
   The photon boson matrix (n=2, m=1) is purely antisymmetric.
   Its best projection onto a self-contained symmetry class has zero
   residual -- the photon is a free asymptotic state.

   The gluon boson matrix (n=3, m=1) has mixed symmetry.
   Its best projection onto either self-contained class leaves a
   nonzero residual (~0.408), confirmed to floating-point precision.
   A mixed-symmetry object always requires reference to the fermion
   configuration from which it arose -- it is codec-confined.
   Colour confinement is a structural property of the codec (conjecture).

Output
------
Section 1: n=2 sector -- neutrino vs charged lepton from winding number.
Section 2: n=3 sector -- per-site occupancy and colour symmetry.
Section 3: Symmetry classification and confinement projection residuals.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def boson_matrix(psi: np.ndarray) -> np.ndarray:
    """B = rho - diag(rho) for normalised pure state psi."""
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    return rho - np.diag(np.diag(rho))


def winding_state(n: int, m: int) -> np.ndarray:
    """n-site Fourier mode with winding number m.

    For n=2, uses the physical hop encoding from Paper VII:
      m=0: (1, 1)/sqrt(2)  -- symmetric, neutrino
      m=1: (1, i)/sqrt(2)  -- antisymmetric, photon/charged lepton
    For n>=3, uses discrete Fourier modes exp(2*pi*i*m*k/n)/sqrt(n).
    """
    if n == 2:
        if m == 0:
            return np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        else:
            return np.array([1.0, 1j], dtype=complex) / np.sqrt(2)
    k = np.arange(n)
    return np.exp(1j * 2 * np.pi * m * k / n) / np.sqrt(n)


def symmetry_label(B: np.ndarray, tol: float = 1e-10) -> str:
    """Classify symmetry of B."""
    if np.max(np.abs(B + B.T)) < tol:
        return "antisymmetric"
    if np.max(np.abs(B - B.T)) < tol:
        return "symmetric"
    return "mixed"


# ---------------------------------------------------------------------------
# Section 1: n=2 sector -- charge from winding number
# ---------------------------------------------------------------------------

def n2_sector() -> None:
    """Show that m distinguishes neutrino (m=0) from charged lepton (m=1).

    Both states live in the n=2 codec geometry with identical norms.
    The winding number m is the charge selector within this sector:
      m=0: no phase winding -> no electromagnetic coupling -> neutrino
      m=1: one phase winding -> unit EM coupling -> charged lepton

    This gives a clean algorithmic rule without inconsistency.
    """
    print("=" * 65)
    print("Section 1: n=2 sector -- charge from winding number")
    print("=" * 65)
    print()
    print("Both neutrino and charged lepton live in the n=2 codec.")
    print("The winding number m selects the charge:")
    print()

    for m, particle, charge in [(0, "neutrino", 0), (1, "charged lepton", 1)]:
        psi   = winding_state(2, m)
        B     = boson_matrix(psi)
        sym   = symmetry_label(B)
        probs = np.abs(psi) ** 2
        F_norm = np.linalg.norm(probs)
        B_norm = np.linalg.norm(B)

        print(f"  (n=2, m={m}): {particle}   charge = {charge}")
        print(f"    psi        = {np.round(psi, 4)}")
        print(f"    |psi_k|^2  = {np.round(probs, 4)}")
        print(f"    B symmetry = {sym}")
        print(f"    ||F||      = {F_norm:.6f}  (= 1/sqrt(2) = {1/np.sqrt(2):.6f})")
        print(f"    ||B||      = {B_norm:.6f}  (= 1/sqrt(2))")
        print(f"    ||F||^2 + ||B||^2 = {F_norm**2 + B_norm**2:.6f}")
        print()

    print("  Key result: norms are IDENTICAL for both states.")
    print("  The charge distinction is carried entirely by the")
    print("  phase winding m, not by the norm structure.")
    print()
    print("  Charge rule within n=2 sector:")
    print("    charge = m  (winding number)")
    print()
    print("  No additional postulate required. The neutrino and")
    print("  charged lepton are the two minimum-C_s states of")
    print("  the same two-site codec.")


# ---------------------------------------------------------------------------
# Section 2: n=3 sector -- per-site occupancy and colour symmetry
# ---------------------------------------------------------------------------

def n3_sector() -> None:
    """Show per-site occupancy = 1/3 and exact colour symmetry for n=3.

    The charge mechanism in the n=3 sector is distinct from n=2:
    here it arises from per-site occupancy, not winding number.
    A unified charge formula is an open problem.
    """
    print("=" * 65)
    print("Section 2: n=3 sector -- occupancy and colour symmetry")
    print("=" * 65)
    print()
    print("Per-site occupancy = 1/n = 1/3 for all n=3 states:")
    print()

    for m, label in [(0, "Higgs candidate (scalar)"),
                     (1, "gluon (confined)")]:
        psi   = winding_state(3, m)
        B     = boson_matrix(psi)
        probs = np.abs(psi) ** 2
        sym   = symmetry_label(B)
        print(f"  (n=3, m={m}): {label}")
        print(f"    |psi_k|^2  = {np.round(probs, 6)}")
        print(f"    B symmetry = {sym}")
        print(f"    ||B||      = {np.linalg.norm(B):.6f}"
              f"  (exact: sqrt(2/3) = {np.sqrt(2/3):.6f})")
        print()

    print("  Colour symmetry of gluon (n=3, m=1):")
    psi = winding_state(3, 1)
    B   = boson_matrix(psi)
    for i in range(3):
        for j in range(3):
            if i != j:
                mag   = abs(B[i, j])
                phase = np.angle(B[i, j]) / np.pi
                print(f"    |B[{i},{j}]| = {mag:.6f}   phase = {phase:.4f}*pi")

    mags  = [abs(B[i,j]) for i in range(3) for j in range(3) if i != j]
    equal = np.allclose(mags, mags[0], atol=1e-12)
    print(f"\n  All |B_ij| equal (colour symmetry): {equal}")
    print(f"  Each = 1/n = 1/3 = {1/3:.6f}:  "
          f"{np.allclose(mags, 1/3, atol=1e-12)}")
    print()
    print("  NOTE: The charge mechanism differs between sectors.")
    print("  n=2: charge = winding number m")
    print("  n=3: charge = per-site occupancy 1/3")
    print("  A unified charge formula is an open problem.")


# ---------------------------------------------------------------------------
# Section 3: Symmetry classification and confinement
# ---------------------------------------------------------------------------

def confinement() -> None:
    """Quantify the mixed symmetry of the gluon and its confinement.

    Projection onto the nearest self-contained symmetry class:
      B_antisym = (B - B^T) / 2
      B_sym     = (B + B^T) / 2
    Residual = ||B - B_proj|| measures how far B is from being free.
    """
    print("=" * 65)
    print("Section 3: Symmetry classification and confinement")
    print("=" * 65)
    print()
    print(f"  {'(n,m)':>7}  {'Particle':>20}  {'Symmetry':>16}  "
          f"{'Min residual':>14}  {'Confined':>9}")
    print("-" * 75)

    cases = [
        (1, 0, "vacuum/Pauli"),
        (2, 0, "neutrino"),
        (2, 1, "photon"),
        (3, 0, "Higgs candidate"),
        (3, 1, "gluon"),
        (4, 2, "spin-2 (open)"),
    ]

    for n, m, label in cases:
        psi  = winding_state(n, m)
        B    = boson_matrix(psi)
        sym  = symmetry_label(B)

        B_as   = (B - B.T) / 2
        B_s    = (B + B.T) / 2
        res_as = np.linalg.norm(B - B_as)
        res_s  = np.linalg.norm(B - B_s)
        min_r  = min(res_as, res_s)
        conf   = min_r > 1e-10

        print(f"  ({n},{m}):  {label:>20}  {sym:>16}  "
              f"{min_r:>14.4e}  {'YES' if conf else 'no':>9}")

    print()
    print("  Photon (n=2,m=1): residual = 0 -- free asymptotic state.")
    print("  Gluon  (n=3,m=1): residual > 0 -- cannot be projected onto")
    print("  any self-contained symmetry class without information loss.")
    print("  The gluon propagator always references its source fermion.")
    print("  This is the codec argument for colour confinement (conjecture).")
    print()
    print("  n=4 spin-2 object: symmetric, free -- physical ID open.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    n2_sector()
    print()
    n3_sector()
    print()
    confinement()
