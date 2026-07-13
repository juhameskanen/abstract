"""
exclusion_poc_3level.py

Follow-up to exclusion_poc.py. That script found that with 2 internal bands,
the antisymmetric (Slater) two-fermion subspace is exactly 1-dimensional
(Lambda^2(C^2) = C), so ANY two distinct single-particle states wedge to the
SAME physical two-fermion state up to a scalar -- meaning the reduced
single-particle density matrix is forced to be diag(0.5, 0.5) with zero
off-diagonal residual, for every choice of theta1, theta2. That is a fact of
linear algebra, not a discovery about physics, and it structurally prevents
a 2-band model from ever encoding a variable "gap" or residual.

This script tests whether 3 internal bands changes that. Lambda^2(C^3) is
3-dimensional, so distinct pairs of single-particle states can now wedge to
genuinely different bivectors. The question: does the reduced 1-body density
matrix, expressed in the FIXED physical band basis {A, B, C} (the analogue of
"pixel" sites, not some derived basis), pick up a nonzero off-diagonal boson
residual once there are enough bands for the occupied 2-dim subspace to sit
"tilted" relative to the band axes?

Two things are checked:
  1. The general fact that ANY single Slater determinant, expressed in its
     OWN natural-orbital basis {e1, e2} (the Gram-Schmidt-orthonormalized
     span of psi1, psi2), gives an exactly incoherent 1/2-1/2 mixture --
     zero residual in that basis, for 2 or 3 bands alike. This is a textbook
     fact about single-determinant states and is not new.
  2. Whether that same reduced density matrix, expressed in the FIXED band
     basis {A, B, C} rather than the natural-orbital basis, is diagonal or
     not. If psi1, psi2 are not aligned with two of the band axes, this
     tests whether a real, nonzero B = rho1 - diag(rho1) survives -- i.e.
     whether 3 bands is enough for a single (uncorrelated) two-fermion
     Slater determinant to show boson-residual coherence between physically
     distinct bands after tracing out the partner fermion, something the
     2-band case forbade outright.
"""

import numpy as np


def gram_schmidt_pair(psi1, psi2):
    """Orthonormalize (psi1, psi2) via Gram-Schmidt, return (e1, e2)."""
    e1 = psi1 / np.linalg.norm(psi1)
    proj = np.vdot(e1, psi2) * e1
    e2_unnorm = psi2 - proj
    e2 = e2_unnorm / np.linalg.norm(e2_unnorm)
    return e1, e2


def slater_state(psi1, psi2, dim):
    """Unnormalized antisymmetrized 2-fermion state, explicit tensor build."""
    uv = np.kron(psi1, psi2)
    vu = np.kron(psi2, psi1)
    return uv - vu


def reduced_density_matrix(Phi, dim):
    T = Phi.reshape(dim, dim)
    rho1 = T @ T.conj().T
    return rho1


def boson_matrix(rho):
    return rho - np.diag(np.diag(rho))


def frob_norm(M):
    return np.sqrt(np.sum(np.abs(M) ** 2))


def analyze(psi1, psi2, dim, label):
    psi1 = psi1 / np.linalg.norm(psi1)
    psi2 = psi2 / np.linalg.norm(psi2)

    Phi_un = slater_state(psi1, psi2, dim)
    n2 = np.vdot(Phi_un, Phi_un).real
    Phi = Phi_un / np.sqrt(n2)

    rho1 = reduced_density_matrix(Phi, dim)
    B1 = boson_matrix(rho1)

    # cross-check against the natural-orbital projector formula
    e1, e2 = gram_schmidt_pair(psi1, psi2)
    rho1_natural_basis_formula = 0.5 * (np.outer(e1, e1.conj()) + np.outer(e2, e2.conj()))
    err_vs_formula = np.max(np.abs(rho1 - rho1_natural_basis_formula))

    # express rho1 in the {e1, e2} natural-orbital basis to confirm it IS
    # diagonal there (the textbook single-determinant fact)
    U = np.column_stack([e1, e2])  # dim x 2, columns are the natural orbitals
    rho1_in_natural_basis = U.conj().T @ rho1 @ U
    B_in_natural_basis = boson_matrix(rho1_in_natural_basis)

    print(f"--- {label} ---")
    print(f"  psi1 = {np.round(psi1, 3)}")
    print(f"  psi2 = {np.round(psi2, 3)}")
    print(f"  overlap <psi1|psi2>          = {np.vdot(psi1, psi2):.4f}")
    print(f"  ||Phi||^2 (exclusion signal) = {n2:.4f}")
    print(f"  rho1 (fixed band basis, {dim}x{dim}):")
    for row in rho1:
        print("    " + "  ".join(f"{v.real:+.3f}{v.imag:+.3f}j" for v in row))
    print(f"  ||B1|| in FIXED band basis        = {frob_norm(B1):.4f}"
          f"   <-- nonzero means real boson residual survives tracing")
    print(f"  ||B1|| in NATURAL orbital basis   = {frob_norm(B_in_natural_basis):.4f}"
          f"   <-- should be ~0 (textbook single-determinant fact)")
    print(f"  max|rho1 - projector formula|     = {err_vs_formula:.2e}"
          f"   (cross-check vs 1/2(|e1><e1|+|e2><e2|))")
    print()
    return frob_norm(B1)


if __name__ == "__main__":
    print("=" * 72)
    print("PART 1: 2-band case, revisited for direct comparison")
    print("=" * 72)
    dim = 2
    analyze(np.array([1, 0], dtype=complex),
             np.array([np.cos(0.3), 1j * np.sin(0.3)], dtype=complex),
             dim, "2 bands, psi1=e_A, psi2 tilted (non-orthogonal)")

    print("=" * 72)
    print("PART 2: 3-band case -- orthogonal orbitals aligned with 2 of the")
    print("        3 band axes (expect B1=0, same as 2-band case: natural")
    print("        basis coincides with a subset of the fixed basis)")
    print("=" * 72)
    dim = 3
    analyze(np.array([1, 0, 0], dtype=complex),
             np.array([0, 1, 0], dtype=complex),
             dim, "3 bands, psi1=e_A, psi2=e_B (aligned with axes)")

    print("=" * 72)
    print("PART 3: 3-band case -- orbitals NOT aligned with the band axes")
    print("        (each mixes two bands). This is the real test.")
    print("=" * 72)
    b1 = analyze(np.array([1, 1, 0], dtype=complex),
                  np.array([0, 1, 1], dtype=complex),
                  dim, "3 bands, psi1~(A+B), psi2~(B+C), tilted subspace")

    print("=" * 72)
    print("PART 4: sweep the tilt to see how the residual varies")
    print("=" * 72)
    print(f"{'mix angle':>10}   {'overlap':>10}   {'||B1|| fixed basis':>20}")
    for alpha in np.linspace(0, np.pi / 2, 7):
        psi1 = np.array([1, 0, 0], dtype=complex)
        psi2 = np.array([0, np.cos(alpha), np.sin(alpha)], dtype=complex)
        Phi_un = slater_state(psi1 / np.linalg.norm(psi1),
                               psi2 / np.linalg.norm(psi2), 3)
        n2 = np.vdot(Phi_un, Phi_un).real
        Phi = Phi_un / np.sqrt(n2)
        rho1 = reduced_density_matrix(Phi, 3)
        B1 = boson_matrix(rho1)
        ov = np.vdot(psi1 / np.linalg.norm(psi1), psi2 / np.linalg.norm(psi2))
        print(f"{alpha:10.4f}   {ov.real:10.4f}   {frob_norm(B1):20.4f}")

    print()
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print("""
  - PART 1 (2 bands) and PART 2 (3 bands, axis-aligned orbitals) both give
    ||B1||=0 in the fixed basis: whenever the occupied 2-dim subspace
    happens to coincide with two of the physical band axes, there is no
    residual, matching the earlier 2-band finding exactly.

  - PART 3 shows the qualitative difference: with 3 bands and orbitals that
    each mix two different physical bands, ||B1|| in the FIXED band basis
    is nonzero, even though (see the natural-orbital-basis line in the
    printout) the state is, as always, exactly incoherent in its own
    natural-orbital frame. The residual in PART 3 is a real, physically
    meaningful coherence between physical bands, not a numerical artifact --
    it appears because the natural-orbital basis is tilted relative to the
    band basis, which is only possible once there are 3 or more bands
    (Lambda^2(C^2) is 1-dimensional and admits no such tilt; Lambda^2(C^3)
    is 3-dimensional and does).

  - PART 4 shows this residual is not fixed but VARIES continuously with
    how the two occupied orbitals are tilted relative to the band axes --
    this is a genuine continuous degree of freedom that a 2-band model
    could never have. This is the first sign of the kind of structure a
    mass-like parameter would need to attach to: not a splitting forced by
    exclusion cost alone (which only ever forces distinctness, as in the
    2-band case), but a residual coherence whose SIZE depends on a new
    geometric parameter that only exists with >=3 internal levels.

  What this still does NOT show: no energy scale or Hamiltonian was used
  anywhere above. This identifies a previously-absent piece of STRUCTURE
  (a continuously variable inter-band coherence surviving in a genuine
  multi-fermion state) that a mass term could plausibly be built from, but
  it does not yet show that the specific coherence found here IS a mass, or
  give any mechanism fixing its size. That remains open.
""")
