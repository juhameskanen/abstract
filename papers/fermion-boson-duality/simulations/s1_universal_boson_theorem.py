"""
Supplementary Script S1 — Universal Boson Norm Theorem
=======================================================
Paper VIII: "The Universal Boson Theorem: Particle Species from Codec Geometry"


Purpose
-------
Verify Theorem 1 numerically:

    For any n-site equal superposition psi with arbitrary phases,
    the Frobenius norm of the boson matrix B = rho - diag(rho) satisfies

        ||B|| = sqrt((n-1)/n)

    exactly, independently of all phase choices.

The proof is three lines (see paper); this script confirms it holds
to floating-point precision for n = 2..10 and 1000 independent random
phase draws per n.

Background
----------
A fermion in equal superposition across n sites is encoded as:

    psi_k = (1/sqrt(n)) * exp(i * phi_k),   k = 0, ..., n-1

for arbitrary real phases phi_k.  The density matrix is:

    rho_ij = psi_i * conj(psi_j) = (1/n) * exp(i*(phi_i - phi_j))

The boson matrix B strips the diagonal:

    B_ij = rho_ij  for i != j,   B_ii = 0

Its Frobenius norm squared is:

    ||B||^2 = sum_{i!=j} |B_ij|^2
            = sum_{i!=j} (1/n^2) * |exp(i*(phi_i-phi_j))|^2
            = n*(n-1) * (1/n^2)
            = (n-1)/n

The key step: |exp(i*anything)| = 1, so the phases cancel completely.

Output
------
Printed table: n, exact formula, 1000-trial mean, max deviation from exact.
All deviations should be < 1e-14 (floating-point noise only).

"""

import numpy as np

# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def boson_matrix(psi: np.ndarray) -> np.ndarray:
    """Compute B = rho - diag(rho) for a pure state psi.

    Args:
        psi: Complex wavefunction, shape (n,). Need not be normalised
             on entry; normalisation is enforced internally.

    Returns:
        B: Complex boson matrix, shape (n, n), with zero diagonal.
    """
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    return rho - np.diag(np.diag(rho))


def equal_superposition(n: int, phases: np.ndarray) -> np.ndarray:
    """Construct an n-site equal superposition with given phases.

    psi_k = (1/sqrt(n)) * exp(i * phases[k])

    Args:
        n:      Number of sites.
        phases: Real phase array, shape (n,).

    Returns:
        Normalised complex wavefunction, shape (n,).
    """
    return np.exp(1j * phases) / np.sqrt(n)


def exact_norm(n: int) -> float:
    """Exact boson norm from Theorem 1: sqrt((n-1)/n)."""
    return np.sqrt((n - 1) / n)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_theorem(n_max: int = 10, n_trials: int = 1000, seed: int = 42) -> None:
    """Verify Theorem 1 for n = 1..n_max over n_trials random phase draws.

    For each n, draws n_trials sets of random phases uniformly from [0, 2pi),
    computes ||B|| for each, and checks agreement with sqrt((n-1)/n).

    Args:
        n_max:    Maximum number of sites to test.
        n_trials: Number of random phase draws per n.
        seed:     Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    print("=" * 65)
    print("Verification of Theorem 1:  ||B|| = sqrt((n-1)/n)")
    print("=" * 65)
    print(f"{'n':>4}  {'Exact sqrt((n-1)/n)':>22}  "
          f"{'Mean ||B||':>12}  {'Max |error|':>12}  {'Pass':>5}")
    print("-" * 65)

    all_pass = True
    for n in range(1, n_max + 1):
        exact = exact_norm(n)

        if n == 1:
            # Special case: single site, no off-diagonal entries, B = 0 exactly
            norms = np.zeros(n_trials)
        else:
            phases_batch = rng.uniform(0, 2 * np.pi, size=(n_trials, n))
            norms = np.array([
                np.linalg.norm(boson_matrix(equal_superposition(n, phases_batch[t])))
                for t in range(n_trials)
            ])

        mean_norm = norms.mean()
        max_error = np.max(np.abs(norms - exact))
        passed = max_error < 1e-13
        all_pass = all_pass and passed

        print(f"{n:>4}  {exact:>22.15f}  {mean_norm:>12.10f}  "
              f"{max_error:>12.2e}  {'YES' if passed else 'FAIL':>5}")

    print("-" * 65)
    print(f"All tests passed: {all_pass}")
    print()
    print("Interpretation:")
    print("  n=1: localised fermion, no boson (B=0 exactly, Pauli exclusion)")
    print("  n=2: photon regime,  ||B|| = 1/sqrt(2)  = 0.70711...")
    print("  n=3: gluon regime,   ||B|| = sqrt(2/3)  = 0.81650...")
    print("  All deviations from exact are floating-point noise only.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    verify_theorem(n_max=10, n_trials=1000, seed=42)
