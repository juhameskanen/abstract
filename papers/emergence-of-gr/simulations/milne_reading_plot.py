"""
milne_reading_plot.py

Companion figure script for "Space-time Curvature as an
Information-Theoretic Structure" (revised).

Background
----------
Paper III established that an internal observer's entropy coordinate
tau (bit-flip ticks per bit, over a universe of n total bits) follows
the Ehrenfest bit-flip saturation curve:

    S(tau) = (n / 2) * (1 - exp(-2 * tau))

This is a bounded, relaxing process: S(0) = 0 (zero entropy, the
founding premise's singularity point) and S(tau) -> n/2 as tau -> inf
(maximum entropy, the horizon-like equilibrium). It never reaches
S = n; saturation occurs at exactly half the bit budget.

The "direct reading" of this curve treats the relational scale factor
as directly proportional to the entropy consumed so far:

    R_dir(tau) = S(tau) / (n / 2)

Near tau = 0, S(tau) ~ n * tau to leading order (each tick flips a
fresh bit with near-certainty before collisions become likely), so
R_dir(tau) grows approximately linearly in tau. This module fits a
power law R_dir(tau) ~ tau^p over an early window and compares the
fitted exponent p against three reference cosmological behaviours:

    p = 1     Milne universe   (empty, k = -1, Lambda = 0)
    p = 1/2   radiation-dominated Friedmann expansion
    p = 2/3   matter-dominated Friedmann expansion

The fitted exponent should sit close to p = 1 (Milne), and clearly
apart from 1/2 and 2/3 -- see the Discussion section of the paper for
the caveat that this checks only the scale-factor POWER LAW, not
curvature itself.

Output
------
Running this module as a script writes ../figures/milne_reading.png: a
log-log plot of R_dir(tau) over the early window, the fitted power
law, and the three reference exponents for comparison.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


REFERENCE_EXPONENTS: dict[str, float] = {
    "Milne (empty, k=-1)": 1.0,
    "radiation-dominated": 0.5,
    "matter-dominated": 2.0 / 3.0,
}


@dataclass(frozen=True)
class PowerLawFit:
    """Result of fitting R_dir(tau) ~ A * tau^p over an early window."""
    exponent: float
    log_amplitude: float

    @property
    def amplitude(self) -> float:
        return float(np.exp(self.log_amplitude))


def entropy_curve(n: float, tau: NDArray[np.float64]) -> NDArray[np.float64]:
    """Ehrenfest bit-flip entropy-saturation curve S(tau), established
    in Paper III: S(tau) = (n/2) * (1 - exp(-2 * tau))."""
    return (n / 2.0) * (1.0 - np.exp(-2.0 * tau))


def direct_reading(n: float, tau: NDArray[np.float64]) -> NDArray[np.float64]:
    """R_dir(tau) = S(tau) / (n/2): the relational scale factor read
    directly off the entropy consumed so far."""
    return entropy_curve(n, tau) / (n / 2.0)


def fit_early_power_law(
    n: float, tau_max: float = 0.1, n_points: int = 2000
) -> PowerLawFit:
    """Fit R_dir(tau) ~ tau^p over tau in (0, tau_max] via a log-log
    linear regression, returning the fitted exponent p."""
    tau = np.linspace(tau_max / n_points, tau_max, n_points)
    R = direct_reading(n, tau)
    p, log_amplitude = np.polyfit(np.log(tau), np.log(R), 1)
    return PowerLawFit(exponent=float(p), log_amplitude=float(log_amplitude))


def plot_milne_reading(
    n: float = 20_000.0,
    tau_max: float = 0.1,
    output_path: str = "../figures/milne_reading.png",
) -> PowerLawFit:
    """
    Build and save the direct-reading figure: R_dir(tau) on log-log
    axes over the early window, overlaid with the fitted power law and
    the three reference cosmological exponents (Milne, radiation,
    matter). Returns the fit result so callers (or the paper build) can
    report the fitted exponent alongside the figure.
    """
    fit = fit_early_power_law(n, tau_max)

    tau = np.linspace(tau_max / 2000, tau_max, 2000)
    R = direct_reading(n, tau)
    fitted_curve = fit.amplitude * tau ** fit.exponent

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(tau, R, color="tab:blue", lw=2, label=r"$R_{\mathrm{dir}}(\tau)$ (simulated)")
    ax.loglog(
        tau, fitted_curve, color="black", lw=1.2, ls="--",
        label=fr"fit: $\tau^{{{fit.exponent:.3f}}}$",
    )

    # reference exponents, each anchored to pass through the same point
    # at tau = tau_max for visual comparison of slope only
    anchor_tau = tau_max
    anchor_R = R[-1]
    for label, p_ref in REFERENCE_EXPONENTS.items():
        ref_curve = anchor_R * (tau / anchor_tau) ** p_ref
        ax.loglog(tau, ref_curve, lw=1.0, alpha=0.6, label=f"{label} ($p={p_ref:.3f}$)")

    ax.set_xlabel(r"$\tau$ (entropy coordinate)")
    ax.set_ylabel(r"$R_{\mathrm{dir}}(\tau)$")
    ax.set_title("Direct reading: early-$\\tau$ power law vs. reference exponents")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return fit


if __name__ == "__main__":
    result = plot_milne_reading()
    print(f"Fitted early-tau exponent: p = {result.exponent:.4f}")
    print("Reference exponents: Milne=1.000, radiation=0.500, matter=0.667")
