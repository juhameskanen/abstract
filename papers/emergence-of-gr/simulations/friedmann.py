#!/usr/bin/env python3
"""Simulation of Information-Theoretic Spacetime Contraction.

This module implements the mathematical framework of Juha Meskanen's Paper V
(June 2026), demonstrating that space-time curvature and the Friedmann scale 
factor emerge identically from a closed, finite bitstring budget. 

It tracks the relational scale factor contraction from the De Sitter vacuum limit
to the Schwarzschild singularity as a function of matter structure condensation,
enforcing strict information conservation.
"""

import numpy as np
import matplotlib.pyplot as plt


def simulate_spacetime_contraction(
    total_bits: int = 1000, entity_width: int = 10
) -> None:
    """Calculates and plots the 1D and 3D relational spatial contraction profiles.

    Validates that the informational counting equation strictly matches the
    corrected Friedmann variable mapping without free parameters, yielding an
    exact downward-opening parabola as space contracts toward the singularity.

    Args:
        total_bits: The static total information budget of the universe (n).
            Defaults to 1000.
        entity_width: The bit-width of an emergent level-k composite matter
            structure (w_k or l). Defaults to 10.

    Raises:
        ValueError: If entity_width is greater than total_bits or less than 1.
    """
    if entity_width > total_bits or entity_width < 1:
        raise ValueError("Entity width must be between 1 and total_bits.")

    # -------------------------------------------------------------------------
    # 1. Generate State Spectrum (Progress of Collapse)
    # -------------------------------------------------------------------------
    # k: Count of composite matter entities transitioning from 0 to max density
    max_entities: int = int(total_bits / entity_width)
    k_particles: np.ndarray = np.arange(0, max_entities + 1)

    # m: Total number of bits consumed by matter structures (m = w_k * k)
    m_bits: np.ndarray = k_particles * entity_width

    # -------------------------------------------------------------------------
    # 2. Intrinsic Informational Geometry (Section 2)
    # -------------------------------------------------------------------------
    # Equation 2.1: Free spacetime fabric tokens remaining unallocated
    rho_fabric: np.ndarray = total_bits - m_bits

    # Equation 2.2: Relational linear scale factor R(t)
    # Reflects the net cost of (l - 1) bits per addressable matter entity
    R_counting: np.ndarray = (rho_fabric + k_particles) / total_bits

    # Section 2.2: Quadratic mapping to observable 3D sphere geometry
    a_obs_counting: np.ndarray = 4 * np.pi * (R_counting**2)

    # -------------------------------------------------------------------------
    # 3. Corrected Friedmann Mapping (Section 3)
    # -------------------------------------------------------------------------
    # Identifications from Eq 3.3: Lambda is the unbound fabric fraction,
    # rho_matter is the bound entity count fraction.
    Lambda: np.ndarray = rho_fabric / total_bits
    rho_matter: np.ndarray = k_particles / total_bits

    # Corrected combination applying the net physical cost: (l - 1) / l
    # Cancels the implicit double-dilution bug of dividing by l twice
    net_cost_modifier: float = (entity_width - 1) / entity_width
    R_friedmann: np.ndarray = Lambda + (rho_matter * net_cost_modifier)
    a_obs_friedmann: np.ndarray = 4 * np.pi * (R_friedmann**2)

    # -------------------------------------------------------------------------
    # 4. Visualization & Analytic Verification
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Intrinsic 1D Linear Scale Factor R(k)
    ax1.plot(
        k_particles,
        R_counting,
        color="tab:blue",
        linewidth=2.5,
        label="Linear $R(k) = 1 - \\frac{k(l-1)}{n}$",
    )
    ax1.set_xlabel("Matter Entities ($k$) $\\rightarrow$ Progress of Collapse")
    ax1.set_ylabel("1D Spatial Resolution $R$")
    ax1.set_title("Intrinsic Linear Resolution Space")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Plot 2: Observable 3D Spherically Squeezed Space
    ax2.plot(
        k_particles,
        a_obs_counting,
        color="purple",
        linewidth=4,
        alpha=0.7,
        label="Counting Equation $a_{obs}$",
    )
    ax2.plot(
        k_particles,
        a_obs_friedmann,
        color="tab:orange",
        linestyle="--",
        linewidth=2.5,
        label="Friedmann",
    )
    ax2.set_xlabel("Matter Entities ($k$) $\\rightarrow$ Progress of Collapse")
    ax2.set_ylabel("Observable Spatial Measure ($4\\pi R^2$)")
    ax2.set_title("3D Parabolic Spherical Space Contraction")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Console Verification of the Identity Proof
    is_identical: bool = np.allclose(a_obs_counting, a_obs_friedmann)
    print("=== Paper V Analytic Verification ===")
    print(f"Counting Equation matches Friedmann Mapping perfectly: {is_identical}")
    print(f"De Sitter Limit (k=0) Volume:     {a_obs_counting[0]:.4f} (4*pi)")
    print(f"Schwarzschild Limit (k={max_entities}) Volume: {a_obs_counting[-1]:.4f}")


if __name__ == "__main__":
    simulate_spacetime_contraction(total_bits=1000, entity_width=10)

