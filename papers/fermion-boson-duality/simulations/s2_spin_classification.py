"""
Supplementary Script S2 — Fermion Classification by Winding Number
====================================================================
Paper VIII: "Fermion Structure and Particle Classification from Codec Geometry"


Purpose
-------
Show that the SYMMETRY of the fermion's compression residual B is
determined by the winding number m of the fermion's phase pattern
around the n-site ring, while the NORM ||B|| = sqrt((n-1)/n) is
unaffected by the choice of m (Theorem 1).

The natural phase patterns on an n-site ring are the discrete Fourier
modes:

    phi_k^(m) = 2*pi*m*k/n,   k = 0, ..., n-1

where m is the winding number (number of complete phase revolutions)
of the FERMION's own wavefunction. These are the minimum-spectral-
complexity states on the ring, ordered by ascending m (ascending C_s
cost).

IMPORTANT (fermion-first framing): B is a compression residual, not a
particle. Every row below classifies a fermion complexity class (n, m).
The "SM correlation label" is the Standard Model name for the pattern
of fermion-event correlation that class produces -- it is not the name
of an independently-existing boson. See the paper's Introduction and
Discussion for why. An earlier version of this script labelled these
rows as if B itself were "the photon" / "the gluon" etc.; that phrasing
has been corrected here to match the paper's current terminology.

Residual symmetry -> correlation type:
  - Antisymmetric (B = -B^T):  vector-like correlation (e.g. photon-, gluon-like)
  - Symmetric     (B =  B^T):  scalar/tensor-like correlation
  - Mixed (neither):            no self-contained symmetry class; the
                                 fermion cannot be described without
                                 reference to its source configuration
                                 (confinement, Section 7 of the paper)

The fermion classification (Table 2 of the paper):

    (n=2, m=1): Charged lepton  — antisymmetric residual; SM correlation label: photon
    (n=3, m=0): Scalar sector   — symmetric residual;     SM correlation label: Higgs sector (conjecture)
    (n=3, m=1): Quark           — mixed-symmetry residual; SM correlation label: gluon (confined)
    (n=4, m=2): Spin-2 class    — symmetric residual;      SM correlation label: open (graviton?)

Output
------
For each (n, m) pair: norm, symmetry type, eigenvalues of iB,
and the corresponding SM correlation label (not a particle claim).

"""

import numpy as np
from typing import Tuple


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def boson_matrix(psi: np.ndarray) -> np.ndarray:
    """Compute B = rho - diag(rho) for normalised pure state psi.
    B is the fermion's compression residual, not a particle."""
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    return rho - np.diag(np.diag(rho))


def winding_state(n: int, m: int) -> np.ndarray:
    """Construct the fermion's winding-m Fourier mode on an n-site ring.

    For n >= 3, uses the discrete Fourier mode:
        psi_k = (1/sqrt(n)) * exp(i * 2*pi*m*k / n)

    For n=2, the Fourier basis collapses (exp(i*pi*k) = (1,-1)/sqrt(2),
    which gives a real B matrix and misses the physical charged-lepton
    hop encoding). The physical hop state from Paper VII is used instead:
        m=0: (1, 1)/sqrt(2)  -- neutral fermion (neutrino), symmetric B
        m=1: (1, i)/sqrt(2)  -- charged lepton (hop encoding), antisymmetric B

    This is not a special case but a clarification: the physical hop
    encoding (1, exp(i*pi/2)) corresponds to winding m=1 on the
    complex circle with step pi/2 rather than pi, i.e. a quarter-turn
    per hop rather than a half-turn.  The quarter-turn is the universal
    phase of Paper VII.

    Args:
        n: Number of sites.
        m: Winding number, 0 <= m <= n//2.

    Returns:
        Normalised complex wavefunction, shape (n,).
    """
    if n == 2:
        # Physical hop encoding from Paper VII
        if m == 0:
            return np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
        else:  # m == 1: charged lepton, antisymmetric B
            return np.array([1.0, 1j], dtype=complex) / np.sqrt(2)
    k = np.arange(n)
    phases = 2 * np.pi * m * k / n
    return np.exp(1j * phases) / np.sqrt(n)


def symmetry_type(B: np.ndarray, tol: float = 1e-10) -> Tuple[str, str]:
    """Classify the symmetry of B and infer the correlation's spin-like type.

    Args:
        B:   Boson matrix, shape (n, n).
        tol: Numerical tolerance for symmetry check.

    Returns:
        (symmetry_label, spin_label): descriptive strings.
    """
    antisym_err = np.max(np.abs(B + B.T))
    sym_err     = np.max(np.abs(B - B.T))

    if antisym_err < tol:
        return "antisymmetric (B = -B^T)", "spin-1"
    if sym_err < tol:
        return "symmetric     (B =  B^T)", "spin-0/2"
    return "mixed (neither symmetric nor antisymmetric)", "confined"


def sm_correlation_label(n: int, m: int) -> str:
    """Return the Standard Model correlation label for (n, m).

    This is NOT a claim that the fermion class (n,m) IS this particle;
    it is the label an observer using Standard Model language would
    attach to the correlation pattern of that fermion class. See the
    paper's Introduction/Discussion and Table 2.
    """
    table = {
        (2, 0): "no correlation (neutral fermion / neutrino)",
        (2, 1): "photon (correlation label; charged lepton is the fermion)",
        (3, 0): "Higgs sector (conjecture)",
        (3, 1): "gluon (correlation label; quark is the fermion, confined)",
        (4, 0): "no correlation (scalar)",
        (4, 1): "---",
        (4, 2): "graviton candidate (correlation label; open)",
    }
    return table.get((n, m), "---")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def classify_bosons(n_max: int = 5) -> None:
    """Classify all (n, m) fermion complexity classes up to n_max sites.

    For each pair, prints: norm, symmetry, eigenvalues of iB,
    and the SM correlation label (not a particle identification).

    Args:
        n_max: Maximum number of sites to analyse.
    """
    print("=" * 72)
    print("Fermion classification by (n, m): residual norm and symmetry")
    print("Theorem 1: ||B|| = sqrt((n-1)/n) for all m")
    print("=" * 72)

    for n in range(2, n_max + 1):
        print(f"\nn = {n} sites   [exact ||B|| = sqrt({n-1}/{n}) = {np.sqrt((n-1)/n):.6f}]")
        print("-" * 60)

        for m in range(n // 2 + 1):
            psi = winding_state(n, m)
            B   = boson_matrix(psi)

            norm      = np.linalg.norm(B)
            exact     = np.sqrt((n - 1) / n)
            sym, spin = symmetry_type(B)
            eigs      = np.sort(np.linalg.eigvalsh(1j * B))
            label     = sm_correlation_label(n, m)

            print(f"  m={m}:  ||B||={norm:.6f}  (exact={exact:.6f})  "
                  f"error={abs(norm-exact):.1e}")
            print(f"        Symmetry: {sym}")
            print(f"        Spin:     {spin}")
            print(f"        Eigenvalues of iB: {np.round(eigs, 4)}")
            print(f"        SM correlation label: {label}")

    print()
    print("=" * 72)
    print("Summary: fermion classification (n, m) -> SM correlation label")
    print("=" * 72)
    print(f"  {'(n,m)':>8}  {'||B||':>10}  {'Spin':>10}  SM correlation label")
    print("-" * 60)
    key_pairs = [(2,1), (3,0), (3,1), (4,2)]
    for n, m in key_pairs:
        psi  = winding_state(n, m)
        B    = boson_matrix(psi)
        norm = np.linalg.norm(B)
        _, spin = symmetry_type(B)
        label = sm_correlation_label(n, m)
        print(f"  ({n},{m}):      {norm:>10.6f}  {spin:>10}  {label}")


# ---------------------------------------------------------------------------
# Off-diagonal structure: colour symmetry of the (3,1) quark class
# ---------------------------------------------------------------------------

def gluon_colour_structure() -> None:
    """Show that the 3-site m=1 fermion (quark) has exact colour symmetry
    in its compression residual.

    All off-diagonal entries |B_ij| are equal to 1/3.
    This is the codec realisation of SU(3) colour symmetry:
    the quark's residual correlates equally with all three colour pairs
    (the SM correlation label for this pattern is "gluon").
    """
    print()
    print("=" * 60)
    print("Colour structure of the (n=3, m=1) quark class")
    print("=" * 60)

    n, m = 3, 1
    psi = winding_state(n, m)
    B   = boson_matrix(psi)

    print(f"\nWavefunction: psi = {np.round(psi, 4)}")
    print(f"\nBoson matrix B (real part):\n{np.round(np.real(B), 6)}")
    print(f"\nBoson matrix B (imag part):\n{np.round(np.imag(B), 6)}")
    print(f"\nOff-diagonal magnitudes |B_ij|:")

    magnitudes = []
    for i in range(n):
        for j in range(n):
            if i != j:
                mag   = abs(B[i, j])
                phase = np.angle(B[i, j]) / np.pi
                magnitudes.append(mag)
                print(f"  |B[{i},{j}]| = {mag:.6f}   phase = {phase:.4f}*pi")

    equal = np.allclose(magnitudes, magnitudes[0], atol=1e-12)
    print(f"\nAll magnitudes equal (colour symmetry): {equal}")
    print(f"Each magnitude = 1/n = 1/3 = {1/3:.6f}:  "
          f"{np.allclose(magnitudes, 1/3, atol=1e-12)}")
    print("\nInterpretation: the quark's residual correlates equally with")
    print("all three colour pairs (01), (02), (12) -- the SM correlation")
    print("label for this pattern is 'gluon'. SU(3) colour symmetry is not")
    print("imposed; it follows from the equal-superposition codec constraint.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    classify_bosons(n_max=5)
    gluon_colour_structure()
