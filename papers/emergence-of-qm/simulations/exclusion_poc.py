"""
exclusion_poc.py

Tests whether the same mechanism that produced Pauli exclusion for spatial
sites in Paper VII Experiment 4 (double occupancy => psi localizes onto a
single basis vector => B = 0, zero compression residual) also operates on
the internal band degree of freedom introduced in Experiment 7 -- and
whether it can do any work toward explaining a mass splitting, without
importing m by hand this time.

Setup
-----
Single-fermion states at one site, over 2 internal bands {A, B}, using the
existing Paper VII convention:

    psi(theta) = cos(theta) e_A + i sin(theta) e_B

Two fermions are placed at the same site (same 2-band system) with
individual angles theta1, theta2, and properly antisymmetrized (Slater
determinant), exactly as required by fermion statistics -- this is the
only physically legitimate way to put two fermions "in the same place":

    Phi(theta1, theta2) = psi(theta1) (x) psi(theta2) - psi(theta2) (x) psi(theta1)

This is unnormalized. Its squared norm is computed both by explicit tensor
construction (no shortcuts) and by the analytic overlap formula, and the two
are cross-checked.

We then:
  1. Confirm ||Phi||^2 -> 0 as theta1 -> theta2 (Pauli exclusion appearing
     in the band sector exactly as it did in the site sector of Exp 4:
     attempting to put two fermions in the identical internal state is
     informationally forbidden -- zero norm, zero compression residual).
  2. Define an exclusion cost C_exclusion = -log2(||Phi||^2), the natural
     MDL/Solomonoff cost of describing this configuration, and show it
     diverges as theta1 -> theta2.
  3. Compute the Solomonoff weight w(theta1,theta2) = 2^{-C_exclusion} =
     ||Phi||^2 (up to normalization) over the SINGLE free parameter
     Delta = theta1 - theta2, and find which Delta is most probable.
  4. Compute the reduced single-particle density matrix (tracing out
     particle 2) and its own boson-residual B = rho_reduced -
     diag(rho_reduced), to see how the ORIGINAL Exp-6/7 formalism behaves
     under antisymmetrization.

Note that this does not produce a numerical value for a mass
splitting energy, and no Hamiltonian/energy scale is introduced anywhere in
this script. It tests only the counting/exclusion argument, isolated from
any dynamical assumption.
"""

import numpy as np

def psi(theta):
    """Single-fermion state on 2 bands, Paper VII / Exp 7 convention."""
    return np.array([np.cos(theta), 1j * np.sin(theta)], dtype=complex)


def slater_state(theta1, theta2):
    """
    Unnormalized antisymmetrized 2-fermion state on 2 bands, built by
    explicit tensor product (no shortcuts), living in C^2 (x) C^2 = C^4.
    Basis order: |AA>, |AB>, |BA>, |BB>.
    """
    u = psi(theta1)
    v = psi(theta2)
    uv = np.kron(u, v)
    vu = np.kron(v, u)
    return uv - vu


def overlap(theta1, theta2):
    """<psi(theta1)|psi(theta2)>, computed explicitly."""
    u = psi(theta1)
    v = psi(theta2)
    return np.vdot(u, v)  # np.vdot conjugates the first argument


def reduced_density_matrix(Phi):
    """
    Trace out particle 2 from the (normalized) 2-fermion state Phi living
    in C^2 (x) C^2, basis order |AA>,|AB>,|BA>,|BB>, returning the 2x2
    reduced density matrix for particle 1 over bands {A,B}.
    """
    T = Phi.reshape(2, 2)  # T[i,j] = amplitude for (particle1=i, particle2=j)
    rho1 = T @ T.conj().T  # sum over particle-2 index
    return rho1


def boson_matrix(rho):
    return rho - np.diag(np.diag(rho))


def frob_norm(M):
    return np.sqrt(np.sum(np.abs(M) ** 2))


def run():
    print("=" * 72)
    print("Step 1-2: overlap, norm, and exclusion cost vs Delta = theta1-theta2")
    print("=" * 72)

    theta1 = 0.7  # arbitrary fixed reference angle
    deltas = np.linspace(1e-4, np.pi / 2, 400)

    max_overlap_err = 0.0
    max_norm_err = 0.0

    norms_sq = np.zeros_like(deltas)
    costs = np.zeros_like(deltas)

    for i, d in enumerate(deltas):
        theta2 = theta1 - d  # Delta = theta1 - theta2

        # explicit overlap, cross-checked against analytic cos(Delta)
        ov = overlap(theta1, theta2)
        ov_analytic = np.cos(d)
        max_overlap_err = max(max_overlap_err, abs(ov - ov_analytic))

        # explicit Slater determinant norm^2, cross-checked against 2 sin^2(Delta)
        Phi = slater_state(theta1, theta2)
        n2 = np.vdot(Phi, Phi).real
        n2_analytic = 2 * np.sin(d) ** 2
        max_norm_err = max(max_norm_err, abs(n2 - n2_analytic))

        norms_sq[i] = n2
        # exclusion cost in bits; guard the d->0 endpoint numerically
        costs[i] = -np.log2(n2) if n2 > 0 else np.inf

    print(f"max |overlap - cos(Delta)|        : {max_overlap_err:.3e}")
    print(f"max |norm^2 - 2 sin^2(Delta)|      : {max_norm_err:.3e}")
    print()
    print("Behavior as Delta -> 0 (two fermions forced toward IDENTICAL")
    print("internal state):")
    for d_test in [0.5, 0.1, 0.01, 0.001, 0.0001]:
        n2 = 2 * np.sin(d_test) ** 2
        cost = -np.log2(n2) if n2 > 0 else np.inf
        print(f"  Delta={d_test:<8.4f}  ||Phi||^2={n2:.6e}   cost={cost:8.3f} bits")
    print()
    print("  => ||Phi||^2 -> 0 and cost -> infinity as Delta -> 0.")
    print("     Yippee, this is Pauli exclusion in the band sector: forcing two")
    print("     fermions into the same internal superposition is exactly")
    print("     as forbidden here as forcing them onto the same spatial")
    print("     site was in Experiment 4 -- same mechanism, same formalism,")
    print("     no new assumption introduced.")
    print()

    print("=" * 72)
    print("Step 3: Solomonoff weight over Delta -- which separation is")
    print("        most probable, using only the exclusion cost (no energy")
    print("        term of any kind)?")
    print("=" * 72)
    weights = norms_sq / np.trapezoid(norms_sq, deltas)  # normalize to a density over Delta
    peak_idx = np.argmax(weights)
    print(f"  argmax weight at Delta = {deltas[peak_idx]:.4f} rad "
          f"(pi/2 = {np.pi/2:.4f})")
    print(f"  weight is monotonically increasing in Delta on (0, pi/2]: "
          f"{np.all(np.diff(weights) >= -1e-9)}")
    print()
    print("  => With no Hamiltonian and no imported mass parameter, the")
    print("     purely combinatorial/exclusion weight w(Delta) = ||Phi||^2")
    print("     is maximized at Delta = pi/2 -- full orthogonality between")
    print("     the two fermions' band-occupation. This reproduces, from")
    print("     MDL/Solomonoff weighting alone, the standard fact that two")
    print("     fermions on two bands settle into the one occupation-number")
    print("     state allowed by antisymmetry: one fermion purely in band A,")
    print("     one purely in band B.")
    print()

    print("=" * 72)
    print("Step 4: reduced single-particle density matrix and its boson")
    print("        residual, at a few representative separations")
    print("=" * 72)
    for d_test in [np.pi / 2, np.pi / 4, 0.2]:
        theta2 = theta1 - d_test
        Phi_un = slater_state(theta1, theta2)
        n2 = np.vdot(Phi_un, Phi_un).real
        Phi = Phi_un / np.sqrt(n2)
        rho1 = reduced_density_matrix(Phi)
        B1 = boson_matrix(rho1)
        print(f"  Delta={d_test:.4f}:")
        print(f"    rho1 diag (band populations) = "
              f"{np.real(np.diag(rho1))}")
        print(f"    ||B1|| (residual on reduced single-particle state) = "
              f"{frob_norm(B1):.4f}")
    print()
    print("  Interpretation: at Delta=pi/2 (the MDL-favored, fully")
    print("  antisymmetrized state), the reduced single-particle state is")
    print("  an equal mixture of bands A and B with no coherence left "
          "between")
    print("  them (mixed state, not a superposition) -- tracing out the")
    print("  partner fermion has consumed the coherence that would")
    print("  otherwise show up as a boson residual. This is a genuinely")
    print("  new observation, not previously in the paper: antisymmetrization")
    print("  converts what would be a coherent (superposition) single-particle")
    print("  boson signal into a classical mixture once the partner fermion")
    print("  is traced out.")
    print()

    print("=" * 72)
    print("Notes:")
    print("=" * 72)
    print("""
  - No energy scale, Hamiltonian, or numerical value for a mass splitting
    was introduced or derived here. Steps 1-4 use only counting/overlap
    (the exclusion cost), never a dynamical term.
  - The result "Delta=pi/2 is MDL-favored" explains why two fermions on two
    bands should end up in orthogonal internal states -- but says nothing
    about the energy associated with maintaining that separation, which is
    what a rest mass actually is, isn't it? (m in m*sigma_z is an energy, not an angle).
  - So this experiment closes the gap identified in Experiment 7's "Status
    of this result" only partially: it shows the exclusion/overlap
    functional does transplant cleanly from the spatial sector (Exp 4) to
    the band sector, and does force band separation from pure MDL
    reasoning with no imported assumption. However, it does not yet show that this
    separation carries a specific energy m, or that m*sigma_z is the
    correct operator form. Converting "Delta is forced to pi/2" into "the
    two bands differ by a specific rest-mass energy" is still open, and
    plausibly requires a genuinely new ingredient (e.g. relating cost-in-bits
    to cost-in-energy via some equivalent of kT or hbar*omega), not just a
    relabeling.
""")


if __name__ == "__main__":
    run()
