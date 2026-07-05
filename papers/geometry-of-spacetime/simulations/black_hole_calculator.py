#!/usr/bin/env python3
"""
Black Hole Information-Theoretic Calculator (v2 — corrected)

Fixes vs v1:
  1. n is the Bekenstein-Hawking entropy in bits (S_BH), matching the
     definition used in doc1/doc2. 
  3. Temporal resolution uses doc1's local-flip-rate hypothesis
     (duration-per-flip scaled by T_Planck/T_H(M)) instead of a bare
     saturation-step-count formula (n/2, n ln n, n*3, ...). This is what
     actually reproduces the M^3 Hawking evaporation scaling; a step-count
     formula alone (any of them) cannot, since it's only ever O(n) or
     O(n log n) and the true evaporation time is O(n^1.5) relative to n's
     entropy scaling (M^2 -> M^3 requires an extra factor of M ~ n^0.5).
  4. H_n ~= ln(n) + gamma, which is accurate
     to ~1/(2n) and therefore exact for all practical purposes here.

Usage:
    python blackhole_info_v2.py <mass_in_solar_masses> [more masses...]
"""

import sys
import math

# ---- Planck / physical constants -------------------------------------------
G       = 6.674e-11        # m^3 kg^-1 s^-2
c       = 2.998e8          # m s^-1
hbar    = 1.055e-34        # J s
k_B     = 1.381e-23        # J K^-1
M_sun   = 1.989e30         # kg
l_P     = 1.616e-35        # m
t_P     = 5.391e-44        # s
GAMMA   = 0.5772156649015329  # Euler-Mascheroni constant

T_PLANCK = math.sqrt(hbar * c**5 / (G * k_B**2))  # Planck temperature, K


# ---- Core formulae -----------------------------------------------------------

def schwarzschild_radius(mass_kg: float) -> float:
    """r_s = 2GM/c^2  [metres]"""
    return 2 * G * mass_kg / c**2


def hawking_temperature(mass_kg: float) -> float:
    """T_H = hbar c^3 / (8 pi G M k_B)  [K]"""
    return hbar * c**3 / (8 * math.pi * G * mass_kg * k_B)


def entropy_bits(r_s: float) -> float:
    """
    n = S_BH in BITS.  This is THE n used throughout doc1/doc2 -- the
    length of the bitstring, not log2 of the radius.
    S_BH = A / (4 l_P^2) nats = A / (4 l_P^2 ln2) bits, A = 4 pi r_s^2.
    """
    A_BH = 4 * math.pi * r_s**2
    S_nats = A_BH / (4 * l_P**2)
    return S_nats / math.log(2)


def radius_from_n(n: float) -> float:
    """
    Inverse of entropy_bits(): r_s implied by n bits, in metres.
    r_s = l_P * sqrt(n ln2 / pi)
    NOTE: this is a tautological round-trip (n was derived FROM r_s), not an
    independent prediction. Useful only as a self-consistency sanity check.
    """
    return l_P * math.sqrt(n * math.log(2) / math.pi)


def flip_count_saturation(n: float, rule: str = "ehrenfest_relax") -> float:
    """
    Number of RANDOM BIT FLIPS to go from zero entropy to saturation.
    Multiple candidate rules -- all are O(n) or O(n log n); none of them,
    by themselves, can produce M^3 scaling (see module docstring). Included
    here only as reference / comparison, not as the final temporal model.

    - "ehrenfest_relax": n/2  (mean relaxation time of Ehrenfest urn)
    - "coupon_collector": n * (ln n + gamma)  (expected flips to hit every
       bit at least once; H_n asymptotic, exact summation infeasible for
       n ~ 1e77+)
    """
    if rule == "ehrenfest_relax":
        return n / 2
    elif rule == "coupon_collector":
        return n * (math.log(n) + GAMMA)
    else:
        raise ValueError(f"unknown rule {rule}")


def local_flip_rate_model(n: float, mass_kg: float) -> float:
    """
    doc1's resolution: duration-per-flip is NOT fixed at t_P. It scales
    with the *local* thermal timescale, i.e. inversely with Hawking
    temperature: duration_per_flip = t_P * (T_Planck / T_H(M)).

    T_model(M) = n(M) * t_P * (T_Planck / T_H(M))

    Since n ~ M^2 and T_Planck/T_H ~ M, this gives T_model ~ M^3,
    matching Hawking evaporation's mass scaling exactly (up to a constant).
    """
    T_H = hawking_temperature(mass_kg)
    duration_per_flip = t_P * (T_PLANCK / T_H)
    return n * duration_per_flip


def hawking_evap_time(mass_kg: float) -> float:
    """t_evap = 5120 pi G^2 M^3 / (hbar c^4)  [seconds]"""
    return 5120 * math.pi * G**2 * mass_kg**3 / (hbar * c**4)


# ---- Formatting ---------------------------------------------------------------

def fmt_sci(x: float, sig: int = 4) -> str:
    return f"{x:.{sig-1}e}"


SECONDS_PER_YEAR = 365.25 * 24 * 3600


def report(mass_solar: float) -> None:
    mass_kg = mass_solar * M_sun
    r_s     = schwarzschild_radius(mass_kg)
    n       = entropy_bits(r_s)                 # CORRECT n (entropy bits)
    T_H     = hawking_temperature(mass_kg)

    r_s_roundtrip = radius_from_n(n)

    flips_ehrenfest = flip_count_saturation(n, "ehrenfest_relax")
    flips_coupon    = flip_count_saturation(n, "coupon_collector")

    T_model = local_flip_rate_model(n, mass_kg)      # doc1's model, seconds
    t_evap  = hawking_evap_time(mass_kg)             # real GR/QFT result, seconds

    ratio = T_model / t_evap
    predicted_ratio = math.pi / (160 * math.log(2))

    sep = "-" * 62
    print(f"\n{'='*62}")
    print(f"  Black Hole: {mass_solar:,.4g} Msun")
    print(f"{'='*62}")

    print("\nINPUT / GEOMETRY")
    print(sep)
    print(f"  Mass                       {fmt_sci(mass_kg)} kg")
    print(f"  Schwarzschild radius r_s   {fmt_sci(r_s)} m")
    print(f"  Hawking temperature T_H    {fmt_sci(T_H)} K")

    print("\nINFORMATION CONTENT  (n = entropy in bits, S_BH)")
    print(sep)
    print(f"  n = S_BH                   {fmt_sci(n)} bits")
    print(f"  r_s reconstructed from n   {fmt_sci(r_s_roundtrip)} m  "
          f"(tautological check, should equal r_s above)")

    print("\nSATURATION FLIP-COUNT (reference only -- see note)")
    print(sep)
    print(f"  Ehrenfest relaxation (n/2)         {fmt_sci(flips_ehrenfest)} flips")
    print(f"  Coupon collector (n(ln n + gamma)) {fmt_sci(flips_coupon)} flips")
    print("  NOTE: neither of these, times a flat t_P, can reproduce M^3")
    print("        evaporation scaling -- see local flip-rate model below.")

    print("\nLOCAL FLIP-RATE MODEL vs REAL HAWKING EVAPORATION")
    print(sep)
    print(f"  T_model = n * t_P * (T_Planck/T_H)   {fmt_sci(T_model)} s"
          f"  ({fmt_sci(T_model/SECONDS_PER_YEAR)} yr)")
    print(f"  t_evap (real, 5120 pi G^2 M^3/hbar c^4) {fmt_sci(t_evap)} s"
          f"  ({fmt_sci(t_evap/SECONDS_PER_YEAR)} yr)")
    print(f"  T_model / t_evap                     {ratio:.6f}")
    print(f"  predicted constant pi/(160 ln2)       {predicted_ratio:.6f}")
    # Tolerance loosened to 1e-3: G, c, hbar, k_B are only given to 4 sig
    # figs above, which caps achievable precision well short of 1e-6.
    match = abs(ratio - predicted_ratio) < 1e-3 * predicted_ratio
    print(f"  Match:                                {'OK' if match else 'MISMATCH'}")
    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    masses = []
    for arg in sys.argv[1:]:
        try:
            masses.append(float(arg))
        except ValueError:
            print(f"  Could not parse '{arg}' as a number. Skipping.")
    if not masses:
        print("No valid masses provided.")
        sys.exit(1)
    for m in masses:
        if m <= 0:
            print(f"  Mass must be positive (got {m}). Skipping.")
            continue
        report(m)


if __name__ == "__main__":
    main()
