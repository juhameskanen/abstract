#!/usr/bin/env python3
"""
Black Hole Information-Theoretic Calculator
Based on: "The Geometry of Spacetime as an Information-Theoretic Structure"
          by Juha Meskanen (2010..2026)

Usage:
    python blackhole_info.py <mass_in_solar_masses>
    python blackhole_info.py 1
    python blackhole_info.py 10 1e6 1e9
"""

import sys
import math

# ── Planck / physical constants ─────────────────────────────────────────────
G       = 6.674e-11        # m³ kg⁻¹ s⁻²
c       = 2.998e8          # m s⁻¹
hbar    = 1.055e-34        # J s
k_B     = 1.381e-23        # J K⁻¹
M_sun   = 1.989e30         # kg
l_P     = 1.616e-35        # m   (Planck length)
t_P     = 5.391e-44        # s   (Planck time)


# ── Core formulae ────────────────────────────────────────────────────────────

def schwarzschild_radius(mass_kg: float) -> float:
    """r_s = 2GM/c²  [metres]"""
    return 2 * G * mass_kg / c**2


def n_bh(r_s: float) -> float:
    """n_bh = log₂(r_s / l_P)  [bits] — information content of BH horizon"""
    return math.log2(r_s / l_P)


def spatial_resolution(n: float) -> float:
    """Δx_max = 2^n · l_P  [metres] — maximum spatial resolution"""
    # Use log-space to avoid overflow for large n
    log2_val = n  # log₂(2^n) = n
    log_metres = log2_val * math.log(2) + math.log(l_P)
    return math.exp(log_metres)


def temporal_resolution(n: float) -> float:
    """Δt_max = n·ln(n)·t_P  [seconds] — coupon-collector temporal resolution"""
    return n * math.log(n) * t_P


def aspect_ratio(n: float) -> float:
    """A(n) = 2^n / (n·ln n)  [dimensionless]"""
    # Compute in log space to handle large n
    log_A = n * math.log(2) - math.log(n) - math.log(math.log(n))
    return math.exp(log_A)


def bekenstein_hawking_entropy(r_s: float) -> float:
    """S_BH = A / (4 l_P²)  [bits]  where A = 4π r_s²"""
    A_BH = 4 * math.pi * r_s**2
    S_nats = A_BH / (4 * l_P**2)          # in natural units (nats)
    return S_nats / math.log(2)            # convert to bits


def model_surface_area(n: float) -> float:
    """A_model = (2^n_bh)² · l_P²  [m²]  (flat raster, pre-4π correction)"""
    # = (r_s / l_P)² · l_P² = r_s²  but computed cleanly from n
    log_A = 2 * n * math.log(2) + 2 * math.log(l_P)
    return math.exp(log_A)


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_sci(x: float, sig: int = 4) -> str:
    """Format as ±X.XXe±YY with given significant figures."""
    return f"{x:.{sig-1}e}"


def fmt_large_int(x: float) -> str:
    """Express very large numbers as 10^exponent."""
    if x <= 0:
        return "0"
    exp = math.log10(x)
    return f"10^{exp:.2f}"


# ── Main report ───────────────────────────────────────────────────────────────

def report(mass_solar: float) -> None:
    mass_kg  = mass_solar * M_sun
    r_s      = schwarzschild_radius(mass_kg)
    n        = n_bh(r_s)
    dx       = spatial_resolution(n)
    dt       = temporal_resolution(n)
    A_ratio  = aspect_ratio(n)
    S_bh_bits = bekenstein_hawking_entropy(r_s)
    A_model  = model_surface_area(n)
    A_bh     = 4 * math.pi * r_s**2
    geo_factor = A_bh / A_model          # should be ≈ 4π

    # ── coupon-collector exact vs approximation ──────────────────────────────
    # H_n = sum_{k=1}^{n} 1/k  ≈ ln(n) + γ   (γ = Euler–Mascheroni)
    gamma = 0.5772156649
    n_int = max(1, round(n))
    H_n   = sum(1/k for k in range(1, n_int + 1))
    dt_exact = n_int * H_n * t_P         # exact coupon-collector mean

    sep = "─" * 60

    print(f"\n{'═'*60}")
    print(f"  Black Hole: {mass_solar:,.4g} M☉")
    print(f"{'═'*60}")

    print(f"\n{'INPUT PARAMETERS':}")
    print(sep)
    print(f"  Mass                      {fmt_sci(mass_kg)} kg")
    print(f"  Schwarzschild radius r_s  {fmt_sci(r_s)} m")

    print(f"\n{'INFORMATION CONTENT':}")
    print(sep)
    print(f"  n_bh = log₂(r_s / l_P)   {n:.4f} bits")
    print(f"  (r_s / l_P)               {fmt_sci(r_s / l_P)}")

    print(f"\n{'RESOLUTIONS':}")
    print(sep)
    print(f"  Spatial   Δx = 2^n · l_P  {fmt_sci(dx)} m")
    print(f"            in Planck units  {fmt_large_int(dx / l_P)} l_P")
    print(f"  Temporal  Δt = n·ln(n)·t_P  {fmt_sci(dt)} s")
    print(f"            (approx, leading term)")
    print(f"            Δt exact (n·H_n)   {fmt_sci(dt_exact)} s")
    print(f"            in Planck units  {fmt_large_int(dt / t_P)} t_P")

    e_folds = n * math.log(2)   # N = ln(2^n) = n·ln2

    print(f"\n{'ASPECT RATIO':}")
    print(sep)
    print(f"  A(n) = 2^n / (n·ln n)    {fmt_large_int(A_ratio)}")
    print(f"  (= Δx [Planck] / Δt [Planck])")

    print(f"\n{'E-FOLDS':}")
    print(sep)
    print(f"  N = n·ln2 = ln(2^n)      {e_folds:.2f}")
    print(f"  (logarithmic expansion from 1 Planck cell to Δx)")

    print(f"\n{'ENTROPY (Bekenstein–Hawking)':}")
    print(sep)
    print(f"  Horizon area A_BH         {fmt_sci(A_bh)} m²")
    print(f"  S_BH = A/(4 l_P²)         {fmt_large_int(S_bh_bits)} bits")
    print(f"  S_BH (scientific)         {fmt_sci(S_bh_bits)} bits")

    print(f"\n{'MODEL vs BEKENSTEIN–HAWKING':}")
    print(sep)
    print(f"  A_model (flat raster)     {fmt_sci(A_model)} m²")
    print(f"  A_BH    (spherical)       {fmt_sci(A_bh)} m²")
    print(f"  A_BH / A_model            {geo_factor:.6f}")
    print(f"  4π                        {4*math.pi:.6f}")
    print(f"  Match (within 0.1%):      {'✓' if abs(geo_factor - 4*math.pi) < 0.001*4*math.pi else '✗'}")

    print(f"\n{'UNIVERSE CONTEXT (n_universe ≈ 184)':}")
    print(sep)
    n_universe = 184.0
    print(f"  n_bh / n_universe         {n / n_universe:.4f}")
    print(f"  A(n_bh) < A(n_universe):  {'✓' if A_ratio < aspect_ratio(n_universe) else '✗'}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Example: python blackhole_info.py 1")
        sys.exit(1)

    masses = []
    for arg in sys.argv[1:]:
        try:
            masses.append(float(arg))
        except ValueError:
            print(f"  ✗ Could not parse '{arg}' as a number. Skipping.")

    if not masses:
        print("No valid masses provided.")
        sys.exit(1)

    for m in masses:
        if m <= 0:
            print(f"  ✗ Mass must be positive (got {m}). Skipping.")
            continue
        report(m)


if __name__ == "__main__":
    main()
