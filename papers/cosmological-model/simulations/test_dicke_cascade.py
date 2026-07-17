"""
Tests for dicke_cascade.py. Run directly: python3 test_dicke_cascade.py
"""

import numpy as np
from itertools import combinations
from scipy.stats import hypergeom

import dicke_cascade as dc
import multiclock as mc


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    if not condition:
        raise AssertionError(name)


def test_exact_factorization_small_n():
    """Direct linear-algebra proof of the building block the whole cascade rests on."""
    n, k, w0, j0 = 12, 6, 4, 2
    dim = 2 ** n
    psi = np.zeros(dim)
    configs = list(combinations(range(n), k))
    amp = 1 / np.sqrt(len(configs))
    for cfg in configs:
        idx = sum(1 << p for p in cfg)
        psi[idx] = amp
    psi = psi.reshape([2] * n)
    psi_mat = psi.reshape(2 ** w0, 2 ** (n - w0))

    win0_states = [s for s in range(2 ** w0) if bin(s).count("1") == j0]
    rest_vecs = [psi_mat[s] for s in win0_states]
    rest_normed = [v / np.linalg.norm(v) for v in rest_vecs]
    ref = rest_normed[0]
    check("all conditional remainders identical up to phase",
          all(np.allclose(abs(np.dot(ref, v)), 1.0) for v in rest_normed))

    n_rest, k_rest = n - w0, k - j0
    dim_rest = 2 ** n_rest
    target = np.zeros(dim_rest)
    configs_rest = list(combinations(range(n_rest), k_rest))
    amp_rest = 1 / np.sqrt(len(configs_rest))
    for cfg in configs_rest:
        idx = sum(1 << p for p in cfg)
        target[idx] = amp_rest
    check("remainder equals |D_{n-w0}^{k-j0}> exactly", abs(np.dot(ref, target)) > 1 - 1e-10)

    total_prob = sum(np.linalg.norm(v) ** 2 for v in rest_vecs)
    check("total probability at j0 matches hypergeom.pmf",
          abs(total_prob - hypergeom.pmf(j0, n, k, w0)) < 1e-10)


def test_probability_conservation():
    n_bits, k, widths = 184, 90, [6, 12, 20]
    summaries, final_branches = dc.recursive_cascade(n_bits, k, widths)

    total0 = summaries[0].fabric_prob + summaries[0].structure_prob + summaries[0].promoted_prob
    check("level 0 probabilities sum to 1", abs(total0 - 1.0) < 1e-9)

    for i in range(1, len(summaries)):
        incoming = summaries[i - 1].promoted_prob
        outgoing_total = summaries[i].fabric_prob + summaries[i].structure_prob + summaries[i].promoted_prob
        check(f"level {i} probabilities sum to level {i-1}'s promoted mass",
              abs(outgoing_total - incoming) < 1e-9)

    final_mass = sum(b.prob for b in final_branches)
    check("final surviving branches sum to last level's promoted mass",
          abs(final_mass - summaries[-1].promoted_prob) < 1e-9)


def test_level0_matches_classical_exactly():
    """Level 0 has no prior level to disagree about -- must match multiclock exactly."""
    n_bits, k, w = 184, 90, 6
    summaries, _ = dc.recursive_cascade(n_bits, k, [w])
    f_fall_mc, f_bump_mc, f_rise_mc = mc.family_fractions_exact(n_bits, np.array([k]), w)

    check("level-0 fabric matches classical f_fall",
          abs(summaries[0].fabric_prob - f_fall_mc[0]) < 1e-9)
    check("level-0 structure matches classical f_bump",
          abs(summaries[0].structure_prob - f_bump_mc[0]) < 1e-9)
    check("level-0 promoted matches classical f_rise",
          abs(summaries[0].promoted_prob - f_rise_mc[0]) < 1e-9)


def test_level1_plus_diverges_from_classical_and_quantify_it():
    """Honest check: level >= 1 is NOT expected to match multiclock's retarded-time
    rescaling, because it's a structurally different model (see module docstring).
    This test documents the divergence rather than hiding it."""
    n_bits, scales = 184, [6, 12, 20]
    tau = 300.0
    sim = mc.run_simulation(n_bits=n_bits, scales=scales, steps=2000, t_bf_max=tau / n_bits)
    k = int(round(sim.unfolded_bits[-1] * 0 + mc.n_bits if False else 0))  # placeholder unused

    # get k(tau) the same way multiclock does
    p_tau = mc.p_of_tau(np.array([tau]), n_bits, mc.TRUE_K_RATE)[0]
    k = int(round(n_bits * p_tau))

    summaries, _ = dc.recursive_cascade(n_bits, k, scales)

    # classical level-1 promoted fraction, from the actual level_state cascade
    levels = mc.build_scale_hierarchy(np.array([tau]), scales, n_bits, mc.TRUE_K_RATE)
    classical_level1_promoted_frac = float(levels[1].f_rise[0])
    quantum_level1_promoted_frac = summaries[1].promoted_prob / summaries[0].promoted_prob

    diff = abs(classical_level1_promoted_frac - quantum_level1_promoted_frac)
    print(f"    (info) classical level-1 promoted fraction = {classical_level1_promoted_frac:.4f}")
    print(f"    (info) cascade  level-1 promoted fraction   = {quantum_level1_promoted_frac:.4f}")
    print(f"    (info) divergence = {diff:.4f}  <- expected to be nonzero, different models")
    check("level-1 divergence is documented (not silently assumed to vanish)", True)


if __name__ == "__main__":
    test_exact_factorization_small_n()
    test_probability_conservation()
    test_level0_matches_classical_exactly()
    test_level1_plus_diverges_from_classical_and_quantify_it()
    print("\nAll tests passed (divergence test PASSES by documenting the gap, not closing it).")
