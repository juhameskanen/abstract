import math

G = 6.674e-11
c = 2.998e8
hbar = 1.0546e-34
k_B = 1.381e-23
Msun = 1.989e30
l_P = 1.616e-35
t_P = 5.391e-44
year = 3.156e7

T_Planck = math.sqrt(hbar*c**5/(G*k_B**2))

def schwarzschild_radius(M):
    return 2*G*M/c**2

def bits_from_radius(R_m):
    R_P = R_m/l_P
    return math.pi*R_P**2/math.log(2)

def hawking_temp(M):
    return hbar*c**3/(8*math.pi*G*M*k_B)

def hawking_evap_time(M):
    return 5120*math.pi*G**2*M**3/(hbar*c**4)

cases = {
    "Solar mass BH": 1*Msun,
    "Sgr A*": 4.3e6*Msun,
    "M87*": 6.5e9*Msun,
    "Largest known SMBH": 2e10*Msun,
}

print(f"{'Case':22s} {'T_H (K)':>12s} {'duration/flip (s)':>18s} {'model_time (yr)':>16s} {'t_evap (yr)':>14s} {'model/t_evap':>14s}")
for name, M in cases.items():
    Rs = schwarzschild_radius(M)
    n = bits_from_radius(Rs)
    T_H = hawking_temp(M)
    duration_per_flip = t_P * (T_Planck/T_H)   # rate ~ T_H  =>  duration ~ 1/T_H
    model_time_s = n * duration_per_flip
    model_time_yr = model_time_s/year
    t_evap_yr = hawking_evap_time(M)/year
    ratio = model_time_yr/t_evap_yr
    print(f"{name:22s} {T_H:12.3e} {duration_per_flip:18.3e} {model_time_yr:16.3e} {t_evap_yr:14.3e} {ratio:14.3e}")
