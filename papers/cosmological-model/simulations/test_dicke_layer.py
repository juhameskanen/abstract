"""
Tests for dicke_layer.py. Run directly: python3 test_dicke_layer.py

Verifies:
  1. Rank-1 blocks + exact hypergeom trace match, across several (n,k,w).
  2. entanglement_entropy() closed-form shortcut matches exact diagonalization.
  3. Flat-state sector weights equal Binomial(n,k,1/2) exactly (algebraic identity).
  4. tau_of_k / k_of_tau are exact inverses, and consistent with multiclock's p(tau).
  5. window_marginal cross-checked against multiclock.family_fractions_exact
     (f_fall / f_rise are just the tails of the same pmf).
"""

import numpy as np
from scipy.stats import hypergeom

import dicke_layer as dl
import multiclock as mc


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


def test_rank1_and_entropy_shortcut():
    cases = [(10, 5, 4), (12, 6, 3), (14, 7, 5), (16, 4, 6), (16, 12, 6)]
    for n_bits, k, w in cases:
        result = dl.verify_rank1_purity(n_bits, k, w)
        check(f"rank-1 blocks (n={n_bits},k={k},w={w})", result["max_rank_violation"] == 0)
        check(f"block trace matches hypergeom.pmf (n={n_bits},k={k},w={w})",
              result["max_trace_error"] < 1e-9)
        check(f"S_vN shortcut matches exact diagonalization (n={n_bits},k={k},w={w})",
              result["entropy_match"])


def test_flat_state_binomial_weights():
    n_bits = 14
    dim = 2 ** n_bits
    psi_flat = np.ones(dim) / np.sqrt(dim)
    sector_weight = np.zeros(n_bits + 1)
    for x in range(dim):
        k = bin(x).count("1")
        sector_weight[k] += psi_flat[x] ** 2
    expected = dl.sector_probability(n_bits, np.arange(n_bits + 1))
    check("flat-state sector weights == Binomial(n,k,1/2) exactly",
          np.allclose(sector_weight, expected, atol=1e-12))


def test_clock_inverse_and_consistency_with_multiclock():
    n_bits = 184
    for tau in [1.0, 50.0, 200.0, 500.0]:
        k = dl.k_of_tau(n_bits, tau)
        tau_back = dl.tau_of_k(n_bits, k)
        check(f"tau_of_k(k_of_tau(tau)) == tau  (tau={tau})", abs(tau_back - tau) < 1e-8)

        # cross-check against multiclock's own p(tau) -- same formula, must agree exactly
        p_mc = mc.p_of_tau(np.array([tau]), n_bits, mc.TRUE_K_RATE)[0]
        k_mc = n_bits * p_mc
        check(f"k_of_tau matches multiclock.p_of_tau*n  (tau={tau})", abs(k - k_mc) < 1e-8)


def test_window_marginal_matches_multiclock_family_fractions():
    n_bits, k, w = 184, 90, 12
    p = dl.window_marginal(n_bits, k, w)
    f_fall_dl = p[0]                      # P(j=0)
    j_thresh = int(np.ceil(w / 2.0))
    f_rise_dl = float(hypergeom.sf(j_thresh - 1, n_bits, k, w))

    f_fall_mc, f_bump_mc, f_rise_mc = mc.family_fractions_exact(n_bits, np.array([k]), w)
    check("window_marginal f_fall matches multiclock.family_fractions_exact",
          abs(f_fall_dl - f_fall_mc[0]) < 1e-12)
    check("window_marginal f_rise matches multiclock.family_fractions_exact",
          abs(f_rise_dl - f_rise_mc[0]) < 1e-12)
    check("window_marginal sums to 1", abs(np.sum(p) - 1.0) < 1e-9)


def test_pattern_probability_matches_exact_linear_algebra():
    """The load-bearing check: does a SPECIFIC ordered pattern's probability
    really equal hypergeom.pmf(a;n,k,w)/C(w,a), i.e. is the state uniform
    over orderings within a composition sector? Direct statevector proof."""
    n, k, w, a = 14, 7, 5, 2
    dim = 2 ** n
    psi = np.zeros(dim)
    configs = list(__import__("itertools").combinations(range(n), k))
    amp = 1 / np.sqrt(len(configs))
    for cfg in configs:
        idx = sum(1 << p for p in cfg)
        psi[idx] = amp
    psi = psi.reshape([2] * n)
    psi_mat = psi.reshape(2 ** w, 2 ** (n - w))
    rho_w = psi_mat @ psi_mat.T

    specific_patterns = [s for s in range(2 ** w) if bin(s).count("1") == a]
    probs = [rho_w[s, s] for s in specific_patterns]
    check("all specific orderings within a composition sector are equally likely",
          np.allclose(probs, probs[0], atol=1e-12))
    check("pattern_probability() matches the direct diagonal entry exactly",
          abs(probs[0] - dl.pattern_probability(n, k, a, w - a)) < 1e-12)


def test_pattern_probability_reproduces_wiki_three_fold_shape():
    """Cross-check against the independently-derived wiki Result 3: does the
    EXACT quantum formula (not the mean-field approximation) land in the
    same shape class for a>b / a=b / a<b, at real working parameters?"""
    n_bits = 184
    t_bf = np.linspace(1e-9, n_bits * np.log(n_bits), 2000)
    k_arr = np.clip(np.round(dl.k_of_tau(n_bits, t_bf)).astype(int), 0, n_bits)

    for a, b, expected_shape in [(4, 2, "monotonic_rise"), (3, 3, "monotonic_to_boundary"), (2, 4, "hump")]:
        check(f"pattern_shape({a},{b}) == {expected_shape}", dl.pattern_shape(a, b) == expected_shape)
        P = dl.pattern_probability(n_bits, k_arr, a, b)
        peak_idx = np.argmax(P)
        has_hump = peak_idx < len(t_bf) - 5 and P[peak_idx] > P[-1] * 1.05
        check(f"exact hump presence matches classification for a={a},b={b}",
              has_hump == (expected_shape == "hump"))


if __name__ == "__main__":
    test_rank1_and_entropy_shortcut()
    test_flat_state_binomial_weights()
    test_clock_inverse_and_consistency_with_multiclock()
    test_window_marginal_matches_multiclock_family_fractions()
    test_pattern_probability_matches_exact_linear_algebra()
    test_pattern_probability_reproduces_wiki_three_fold_shape()
    print("\nAll tests passed.")
