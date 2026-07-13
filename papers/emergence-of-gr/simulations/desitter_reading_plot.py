"""
desitter_reading_plot.py

Companion figure script for "Space-time Curvature as an
Information-Theoretic Structure" (revised).

Background
----------
Paper III established the Ehrenfest bit-flip entropy-saturation curve:

    S(tau) = (n / 2) * (1 - exp(-2 * tau))

The "reciprocal reading" of this curve treats the relational scale
factor as the inverse of the remaining gap to equilibrium:

    g(tau)       = n/2 - S(tau) = (n/2) * exp(-2 * tau)
    R_rec(tau)   = (n/2) / g(tau) = exp(2 * tau)

Because g(tau) is EXACTLY exponential at every tau (not merely in some
asymptotic limit), the fractional growth rate of R_rec is exactly
constant for all tau:

    (1 / R_rec) * (dR_rec / dtau) = 2   for all tau

This is the defining property of de Sitter expansion: a genuinely
constant Hubble parameter H, obtained here as an identity of the same
curve used for the direct reading in milne_reading_plot.py, not through
a separately introduced exponential ansatz. See the paper's Discussion
section for the caveat that this uses the closed-form, infinite-
precision Ehrenfest curve; a finite bit budget n fluctuates near
equilibrium rather than approaching it smoothly, so exact constancy is
an idealisation that should be re-checked against stochastic
trajectories.

Output
------
Running this module as a script writes
../figures/desitter_reading.png: a two-panel figure showing (left)
R_rec(tau) on a log-linear axis, where exponential growth appears as a
straight line, and (right) the fractional growth rate
(dR_rec/dtau) / R_rec against tau, which should be flat at the
theoretical value of 2 across the full range tested.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

THEORETICAL_H: float = 2.0


def entropy_curve(n: float, tau: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ehrenfest bit-flip entropy-saturation curve S(tau), established
    in Paper III: S(tau) = (n/2) * (1 - exp(-2 * tau))."""
    return (n / 2.0) * (1.0 - np.exp(-2.0 * tau))


def gap_to_equilibrium(n: float, tau: NDArray[np.float64]) -> NDArray[np.float64]:
    """g(tau) = n/2 - S(tau) = (n/2) * exp(-2 * tau): the remaining
    entropy capacity before saturation."""
    return (n / 2.0) - entropy_curve(n, tau)


def reciprocal_reading(n: float, tau: NDArray[np.float64]) -> NDArray[np.float64]:
    """R_rec(tau) = (n/2) / g(tau) = exp(2 * tau): the relational scale
    factor read as the inverse of the remaining gap to equilibrium."""
    return (n / 2.0) / gap_to_equilibrium(n, tau)


def fractional_growth_rate(
    tau: NDArray[np.float64], R: NDArray[np.float64]
) -> NDArray[np.float64]:
    """(dR/dtau) / R via numerical differentiation, for comparison
    against the theoretical constant value of 2."""
    dR = np.gradient(R, tau)
    return dR / R


def plot_desitter_reading(
    n: float = 20_000.0,
    tau_max: float = 3.0,
    n_points: int = 2000,
    output_path: str = "../figures/desitter_reading.png",
) -> NDArray[np.float64]:
    """
    Build and save the reciprocal-reading figure: R_rec(tau) on a
    log-linear axis (left panel) and the fractional growth rate
    (dR_rec/dtau) / R_rec against tau (right panel), which should be
    flat at H = 2 across the full range. Returns the computed H(tau)
    array so callers can report the numerical spread around the
    theoretical value.
    """
    tau = np.linspace(1e-4, tau_max, n_points)
    R = reciprocal_reading(n, tau)
    H = fractional_growth_rate(tau, R)

    fig, (ax_growth, ax_rate) = plt.subplots(1, 2, figsize=(12, 5))

    ax_growth.semilogy(tau, R, color="tab:red", lw=2)
    ax_growth.set_xlabel(r"$\tau$ (entropy coordinate)")
    ax_growth.set_ylabel(r"$R_{\mathrm{rec}}(\tau)$ (log scale)")
    ax_growth.set_title("Reciprocal reading: exponential growth")
    ax_growth.grid(True, which="both", alpha=0.3)

    ax_rate.plot(tau, H, color="tab:red", lw=2, label=r"$(dR_{\mathrm{rec}}/d\tau)/R_{\mathrm{rec}}$")
    ax_rate.axhline(
        THEORETICAL_H, color="black", lw=1.0, ls="--",
        label=f"theoretical constant $H={THEORETICAL_H:.1f}$",
    )
    ax_rate.set_xlabel(r"$\tau$ (entropy coordinate)")
    ax_rate.set_ylabel(r"fractional growth rate")
    ax_rate.set_title("Exactly constant $H$ across all $\\tau$")
    ax_rate.set_ylim(THEORETICAL_H - 0.5, THEORETICAL_H + 0.5)
    ax_rate.legend(fontsize=9)
    ax_rate.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return H


if __name__ == "__main__":
    H = plot_desitter_reading()
    print(f"Fractional growth rate H(tau): min={H.min():.6f}, max={H.max():.6f}")
    print(f"Theoretical constant value: H = {THEORETICAL_H:.6f}")
