"""
milne_desitter_readings.py

Companion script for "Space-time Curvature as an Information-Theoretic
Structure". Demonstrates that two different ways of reading the
same Ehrenfest entropy-saturation curve S(tau) = (n/2)(1 - e^{-2 tau})
reproduce, as limits, the two vacuum (matter-free) solutions of General
Relativity that bound the Friedmann equation on the expansion side:

  - DIRECT reading:      R_dir(tau)  = S(tau) / (n/2)
      -> early-tau (tau << 1) power-law growth with exponent ~1,
         matching the Milne universe (a(t) ~ t, empty, k=-1, Lambda=0).

  - RECIPROCAL reading:  R_rec(tau)  = (n/2) / (n/2 - S(tau))
      -> EXACTLY constant fractional growth rate (dR/dtau)/R = 2 for
         ALL tau, matching the de Sitter universe (a(t) ~ e^{H t},
         Lambda > 0, rho = 0) not merely asymptotically but identically,
         since the "gap to equilibrium" n/2 - S(tau) is exactly
         proportional to e^{-2 tau} at every tau, not just for large tau.

Both readings come from the same underlying process (no separate
mechanism is introduced for each); they differ only in whether the
observer's notion of "resolution" tracks consumed entropy directly, or
the inverse of the remaining distance to equilibrium.

What this script does not do: derive the matter-filled middle regime
(the true Friedmann era), which requires the lognormal-like hump curve
m(S) relating matter-structure formation to entropy -- that mapping is
not yet derived and is explicitly left as future work. It also does not
show that the early-tau limit has the zero curvature of an exact Milne
solution -- only that the SCALE-FACTOR POWER LAW exponent matches; the
curvature itself (Riemann tensor analogue) has not been checked.
"""

import numpy as np


def entropy_curve(n, tau):
    """Ehrenfest bit-flip entropy-saturation curve, established and
    verified in prior work (Paper III / observer-emergence testbeds)."""
    return (n / 2.0) * (1.0 - np.exp(-2.0 * tau))


def direct_reading(n, tau):
    S = entropy_curve(n, tau)
    return S / (n / 2.0)


def reciprocal_reading(n, tau):
    S = entropy_curve(n, tau)
    gap = (n / 2.0) - S
    return (n / 2.0) / gap


def check_milne_exponent(n, tau_max_fraction=0.1, n_points=2000):
    """Fit R_dir(tau) ~ tau^p over an early window (tau << 1); compare to
    Milne's p=1, versus radiation (p=0.5) and matter (p=2/3) as
    alternatives it is NOT claiming to match."""
    tau_max = tau_max_fraction
    tau = np.linspace(1e-4, tau_max, n_points)
    R = direct_reading(n, tau)
    p, logA = np.polyfit(np.log(tau), np.log(R), 1)
    return p


def check_desitter_constant_H(n, tau_values):
    """Verify (dR/dtau)/R is exactly constant (=2) for the reciprocal
    reading, at a spread of tau values including large tau -- not just
    asymptotically."""
    tau = np.linspace(1e-4, max(tau_values) * 1.2, 20000)
    R = reciprocal_reading(n, tau)
    dR = np.gradient(R, tau)
    H = dR / R
    results = []
    for t in tau_values:
        idx = np.argmin(np.abs(tau - t))
        results.append((tau[idx], H[idx]))
    return results


if __name__ == "__main__":
    n = 20000

    print("=" * 72)
    print("DIRECT reading: early-tau power-law exponent")
    print("=" * 72)
    p = check_milne_exponent(n)
    print(f"  fitted exponent p = {p:.4f}")
    print(f"  Milne (empty, k=-1):      p = 1.000")
    print(f"  radiation-dominated:      p = 0.500")
    print(f"  matter-dominated:         p = 0.667")
    print(f"  -> matches Milne's power law, not radiation or matter.")
    print()

    print("=" * 72)
    print("RECIPROCAL reading: (dR/dtau)/R at a spread of tau (should be exactly 2)")
    print("=" * 72)
    for tau_val, H_val in check_desitter_constant_H(n, [0.1, 0.5, 1.0, 2.0, 2.9]):
        print(f"  tau={tau_val:.3f}   H={H_val:.6f}")
    print("  -> constant to numerical precision at EVERY tau tested, not just")
    print("     asymptotically: the reciprocal reading is exactly de-Sitter-like")
    print("     for this idealized (infinite-precision) Ehrenfest curve.")
    print()

    print("=" * 72)
    print("Caveats (see paper Sec. 4 for full discussion)")
    print("=" * 72)
    print("""
  - This uses the CLOSED-FORM Ehrenfest curve, not a finite-n stochastic
    simulation; a real finite chain fluctuates near equilibrium rather
    than approaching it smoothly, so the reciprocal reading's exact
    constancy is an idealization that should be re-checked against
    actual stochastic trajectories, not just the closed-form average.
  - Matching the SCALE FACTOR power law is not the same as matching
    curvature; whether the early-tau limit is truly flat (as Milne is)
    has not been checked here.
  - The matter-filled regime connecting these two limits requires the
    hump curve m(S), not yet derived from first principles.
""")
