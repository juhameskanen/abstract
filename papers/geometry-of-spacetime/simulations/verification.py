# Verify all numerical claims in the paper against the equations
import numpy as np

# Paper's definitions
n = 184
ell_P = 1.616e-35  # m
t_P   = 5.39e-44   # s
G     = 6.674e-11
c     = 3e8
M_sun = 1.989e30

print("=== Verifying numerical claims ===\n")

# Section 3.1: Spatial resolution
delta_x_max = 2**n * ell_P
print(f"Spatial resolution: 2^{n} * ell_P")
print(f"  2^{n} = {2**n:.3e} Planck lengths")
print(f"  delta_x_max = {delta_x_max:.3e} m")
print()

# Section 3.2: Temporal resolution
T = n * np.log(n)
delta_t_max = T * t_P
print(f"Temporal resolution: n*ln(n) = {n}*ln({n}) = {T:.4f} Planck times")
print(f"  delta_t_max = {delta_t_max:.3e} s")
print()

# Section 3.3: Aspect ratio
A = 2**n / (n * np.log(n))
print(f"Aspect ratio A(n) = 2^n / (n*ln n) = {A:.3e}")
print()

# Section 4: Calibration
delta_t = 1e-35  # s
delta_x = 1e26   # m
delta_t_Pl = delta_t / t_P
delta_x_Pl = delta_x / ell_P
A_inf = delta_x_Pl / delta_t_Pl
print(f"Inflationary aspect ratio:")
print(f"  delta_t = {delta_t:.0e} s -> {delta_t_Pl:.3e} Planck times")
print(f"  delta_x = {delta_x:.0e} m -> {delta_x_Pl:.3e} Planck lengths")
print(f"  A_inf = {A_inf:.3e}")
print(f"  A(184) = {A:.3e}")
print(f"  Match: {abs(np.log10(A) - np.log10(A_inf)) < 0.1}")
print()

# Section 5: E-folds
e_folds_from_n = n * np.log(2)  # natural log units
print(f"E-folds from n: n*ln(2) = {e_folds_from_n:.1f} natural-log units")
print(f"  Stated in paper: ~127 natural-log units")
print(f"  Consistent: {abs(e_folds_from_n - 127) < 1}")
print()

# Section 6: Black hole
r_s = 2 * G * M_sun / c**2
n_bh = np.log2(r_s / ell_P)
print(f"Solar-mass black hole:")
print(f"  r_s = {r_s:.4f} m  (paper says 2950 m: {abs(r_s-2950)<1})")
print(f"  r_s/ell_P = {r_s/ell_P:.3e}  (paper says 1.83e38: {abs(r_s/ell_P-1.83e38)/1.83e38 < 0.02})")
print(f"  n_bh = log2(r_s/ell_P) = {n_bh:.2f}  (paper says ~127: {abs(n_bh-127)<1})")
print()

# Surface area prediction
A_model = (r_s/ell_P)**2 * ell_P**2
print(f"Surface area prediction:")
print(f"  A_model = (r_s/ell_P)^2 * ell_P^2 = r_s^2 = {A_model:.3e} m^2")
print(f"  Paper states: ~8.7e6 m^2  Check: {abs(A_model - 8.7e6)/8.7e6 < 0.01}")
A_BH = 4 * np.pi * r_s**2
print(f"  A_BH = 4*pi*r_s^2 = {A_BH:.3e} m^2  (paper: 1.09e8)")
print(f"  Ratio A_model/A_BH = {A_model/A_BH:.6f}  (paper: 1/4pi = {1/(4*np.pi):.6f})")
print()

# KEY CONSISTENCY CHECK: is n used consistently?
print("=== Consistency of n usage ===\n")
print("DEFINITION in paper:")
print("  n = number of BITS (integer, ~184)")
print("  2^n = number of CONFIGURATIONS = spatial resolution in Planck lengths")
print()
print("CHECKING each equation:")
print()
print("1. delta_x_max = 2^n * ell_P")
print(f"   n=184 (bits), 2^n = {2**n:.2e} (configurations)")
print(f"   CONSISTENT: n is the exponent, 2^n is what's multiplied by ell_P")
print()
print("2. A(n) = 2^n / (n*ln n)")
print(f"   numerator 2^n = {2**n:.2e} (spatial resolution)")
print(f"   denominator n*ln(n) = {n*np.log(n):.1f} (temporal resolution)")
print(f"   CONSISTENT")
print()
print("3. n_bh = log2(r_s/ell_P)")
print(f"   This defines n_bh = {n_bh:.2f} bits for the black hole")
print(f"   Then 2^n_bh = r_s/ell_P = {2**n_bh:.3e} -- which equals r_s in Planck units")
print(f"   CONSISTENT: n_bh is the bit count, 2^n_bh is the resolution")
print()
print("4. Surface area = 2^(2*n_bh) * ell_P^2")
print(f"   = (2^n_bh)^2 * ell_P^2 = (r_s/ell_P)^2 * ell_P^2 = r_s^2")
print(f"   = {r_s**2:.3e} m^2")
print(f"   CONSISTENT: squaring a 1D resolution gives 2D area")
print()

# Now check the abstract's wording
print("=== Abstract wording check ===")
print()
print("Abstract says: '2^n where n~184 bits'")
print("  -> n is the exponent, 2^n is the number of configurations")
print("  CORRECT")
print()
print("Abstract says: 'total information content n bits'")  
print("  -> n bits means the system has n binary degrees of freedom")
print("  -> This IMPLIES 2^n configurations")
print("  CORRECT and CONSISTENT")
print()
print("Potential confusion point:")
print("  The paper sometimes writes '2^n configurations' and sometimes")
print("  just 'n bits' - these are DIFFERENT quantities but both refer")
print("  to the SAME n=184. The paper never confuses them numerically.")
print()
print("One genuine ambiguity to check:")
print("  Section 3.1 says 'there are 2^n distinct configurations'")
print("  then 'we identify this with a count of Planck-length cells: 2^n * ell_P'")
print("  This is fine: 2^n configurations = 2^n spatial cells")
print("  But is it clear WHY configurations = spatial cells?")
print("  The paper asserts this via 'if space is the geometric interpretation'")
print("  which is the framework axiom - so it's stated as given, not derived")
