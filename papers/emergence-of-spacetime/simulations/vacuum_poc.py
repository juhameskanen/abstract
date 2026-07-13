import numpy as np

def simulate_hamming_weight(n, T_max, n_trials, rng):
    """Ehrenfest: n bits, one uniformly random bit flipped per tick, start
    all-zero. Track Hamming weight (S = number of 1-bits) over time,
    averaged over many independent trials."""
    S = np.zeros((n_trials, T_max+1), dtype=np.int32)
    bits = np.zeros((n_trials, n), dtype=np.int8)
    for t in range(1, T_max+1):
        idx = rng.integers(0, n, size=n_trials)
        bits[np.arange(n_trials), idx] ^= 1  # flip
        S[:, t] = bits.sum(axis=1)
    return S.mean(axis=0)

rng = np.random.default_rng(0)
n = 20000
T_max = 4000
n_trials = 200

S_avg = simulate_hamming_weight(n, T_max, n_trials, rng)
r_avg = S_avg / 2.0  # the r = S/2 mapping

# fit power law r(T) ~ A*T^p over the EARLY (small T, T << n) regime
T = np.arange(1, T_max+1)
early = T < n // 10  # well before saturation
logT = np.log(T[early])
logr = np.log(r_avg[1:][early])
p, logA = np.polyfit(logT, logr, 1)

print(f"n={n}, early-time power-law fit over T in [1,{int(n//10)}]:")
print(f"  r(T) ~ T^{p:.4f}   (Milne/empty predicts exponent 1, "
      f"radiation predicts 0.5, matter predicts 0.667)")
print()

# also check saturation matches the known closed-form Ehrenfest curve
tau = T_max / n
p_theory = 0.5 * (1 - np.exp(-2*tau))
S_theory_final = n * p_theory
print(f"at T={T_max} (tau={tau:.3f}): "
      f"simulated S={S_avg[-1]:.1f}   theory n*p(tau)={S_theory_final:.1f}")

# print a few sample points across the range to see the whole shape
print()
print(f"{'T':>8}{'S(T) sim':>12}{'r=S/2':>10}")
for Tval in [10, 50, 200, 1000, 2000, 4000]:
    print(f"{Tval:8d}{S_avg[Tval]:12.2f}{r_avg[Tval]:10.2f}")
