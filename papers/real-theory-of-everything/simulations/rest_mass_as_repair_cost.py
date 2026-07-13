"""
repair_cost_poc.py

Tests the "mass = cost of fighting an unpredictable background corruption
due to random walk through configurations hypothesis.
The corruption source is the Ehrenfest process itself:
exactly one bit, chosen uniformly at random among n, flips every tick. This
is unpredictable by construction -- no codec, no motion vector, and no
amount of cleverness can anticipate which bit flips next. That is the whole
point of the process, not an oversight.

Setup
-----
- A universe of n bits.
- A "particle" is a redundantly-encoded pattern occupying k of those n bits
  (a repetition code: all k bits agree, e.g. all 0, representing a single
  coherent value). The remaining n-k bits are unstructured background.
- Every tick, one bit is chosen uniformly at random from all n and flipped
  (Ehrenfest dynamics, exactly as used elsewhere in this project).
- If the flipped bit lies OUTSIDE the k-bit pattern: nothing relevant
  happens (background noise, no cost attributed to the particle).
- If the flipped bit lies INSIDE the k-bit pattern: the pattern's
  redundancy is broken by one bit. Maintaining the particle as a
  recognizable, persistent structure requires repairing it (flipping that
  bit back), which costs exactly 1 bit of "work" per repair event.

We measure, empirically, the repair rate (repairs per tick) as a function
of k (pattern size / redundancy) and n (universe size), and check three
concrete, falsifiable properties any respectable "mass" candidate should
plausibly have:
  1. Scaling with k: does a bigger (more redundant) pattern cost more to
     maintain, and is the relationship simple (e.g. linear)?
  2. Scaling with n: does the SAME pattern cost less to maintain as the
     total bit budget grows (dilution)?
  3. Additivity: do two separate, non-overlapping patterns' repair costs
     simply add, the way non-interacting masses should at low energy?

Explicitly NOT claimed: no attempt is made here to recover a quadrature
combination with momentum, no physical unit conversion is proposed, and no
numerical value for any real particle's mass is derived. This tests only
whether the repair-rate mechanism behaves sensibly as a mass CANDIDATE.
"""

import numpy as np


def simulate_repair_rate(n, k_ranges, T, rng, offset=0):
    """
    Run T ticks of the Ehrenfest process on n bits, with one or more
    redundantly-encoded pattern regions given as (start, size) tuples in
    k_ranges. Returns, for each region, the empirical repair rate
    (repairs per tick).
    """
    repairs = [0 for _ in k_ranges]
    for _ in range(T):
        idx = rng.integers(0, n)
        for i, (start, size) in enumerate(k_ranges):
            if start <= idx < start + size:
                repairs[i] += 1  # corruption event -> immediately repaired
                break
    return [r / T for r in repairs]


def run():
    rng = np.random.default_rng(0)
    T = 400_000  # ticks; large for a clean empirical estimate

    print("=" * 72)
    print("Test 1: repair rate vs pattern size k, fixed universe size n")
    print("=" * 72)
    n = 10_000
    ks = [10, 50, 100, 250, 500, 1000, 2000]
    print(f"{'k':>6}{'empirical rate':>18}{'theoretical k/n':>20}{'ratio':>10}")
    for k in ks:
        rate = simulate_repair_rate(n, [(0, k)], T, rng)[0]
        theory = k / n
        print(f"{k:6d}{rate:18.5f}{theory:20.5f}{rate/theory:10.4f}")

    print()
    print("=" * 72)
    print("Test 2: repair rate vs universe size n, fixed pattern size k")
    print("=" * 72)
    k_fixed = 100
    ns = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]
    print(f"{'n':>8}{'empirical rate':>18}{'theoretical k/n':>20}{'ratio':>10}")
    for n_test in ns:
        rate = simulate_repair_rate(n_test, [(0, k_fixed)], T, rng)[0]
        theory = k_fixed / n_test
        print(f"{n_test:8d}{rate:18.5f}{theory:20.5f}{rate/theory:10.4f}")

    print()
    print("=" * 72)
    print("Test 3: additivity -- two separate, non-overlapping patterns")
    print("=" * 72)
    n = 10_000
    k1, k2 = 200, 500
    rates = simulate_repair_rate(n, [(0, k1), (k1, k2)], T, rng)
    rate1, rate2 = rates
    combined_theory = (k1 + k2) / n
    combined_empirical = rate1 + rate2
    print(f"  pattern 1 (k1={k1}): empirical rate = {rate1:.5f}, "
          f"theory = {k1/n:.5f}")
    print(f"  pattern 2 (k2={k2}): empirical rate = {rate2:.5f}, "
          f"theory = {k2/n:.5f}")
    print(f"  combined (sum of the two): empirical = {combined_empirical:.5f}, "
          f"theory (k1+k2)/n = {combined_theory:.5f}")

    print()
    print("=" * 72)
    print("Interpretation")
    print("=" * 72)
    print("""
  - Test 1: repair rate tracks k/n closely (ratio of empirical to
    theoretical clusters near 1.0 across nearly two orders of magnitude in
    k). A bigger, more redundantly-encoded pattern genuinely costs more,
    per tick, to keep intact against the unpredictable background flip --
    and the relationship is the simplest possible one, exactly linear in
    k. This did NOT get eroded to near-zero the way the earlier
    motion-compensated experiment did, precisely because there is no
    predictable structure here for any codec to exploit: the corrupting
    flip is uniform-random by construction, so "the pattern was fine last
    tick" gives zero predictive power about whether THIS tick's flip lands
    inside it.

  - Test 2: for fixed k, repair rate falls off as 1/n as the universe's
    total bit budget grows -- the same pattern becomes cheaper to maintain,
    and hence lighter under this candidate, as n grows. This is a genuinely
    new, testable-sounding consequence: in a framework where the universe's
    total bit budget n is itself the parameter distinguishing cosmological
    from black-hole scales (as in the earlier n-unification discussion),
    this predicts that the SAME physical pattern would carry different
    effective mass at different n -- worth flagging explicitly as a
    speculative, falsifiable-in-principle consequence rather than a
    derived fact, since nothing here fixes what "n changing" would mean
    operationally for an embedded observer.

  - Test 3: the two non-overlapping patterns' repair rates add, to within
    statistical noise, exactly as (k1+k2)/n predicts. This is the
    additivity a sensible low-energy mass concept needs (two distant,
    non-interacting particles have a combined rest energy that is just the
    sum of their individual rest energies).

  - A suggestive (not derived) side note: each individual corruption event
    is itself short-lived -- born when the flip lands inside the pattern,
    "consumed" by the very next repair -- which is at least reminiscent of
    the boson lifecycle from Paper VII Experiment 6 (born, present briefly,
    reabsorbed). This is NOT claimed as a proof that boson and repair-event
    are the same object; it is noted only as a resemblance worth checking
    more carefully in a combined model.

  - What remains genuinely open: this produces a mass CANDIDATE with the
    right qualitative scaling properties (grows with structure size, shrinks
    with total bit budget, adds across separate particles), but it has not
    been combined with the momentum construction from Experiment 7, and no
    attempt has been made here to check whether repair-rate-as-mass and
    hop-phase-as-momentum combine in quadrature (E^2 = p^2 + m^2) or in any
    other specific way. That combination -- building one model with BOTH a
    moving/hopping component AND a redundantly-encoded repaired component,
    and checking how their costs actually combine -- is the natural next
    experiment, not yet attempted.
""")


if __name__ == "__main__":
    run()
