"""
tick_budget_poc.py

Tests a specific, concrete version of the "matter contracts space AND time"
idea, restricted here to time only, as a candidate for the coupling that
combined_poc.py showed was missing (motion cost and repair cost combining
in quadrature requires SOMETHING to trade off against something else; here
we test whether a bit-locking mass structure trades off against the
UNIVERSE'S OWN raw tick rate, producing a slowdown of "real" (observable)
change elsewhere -- a mechanism structurally similar to gravitational time
dilation, arrived at from pure bit-budget competition, nothing imported
from GR).

Setup
-----
n bits total. One uniformly random bit-flip proposal per tick, exactly the
Ehrenfest process used throughout. Three disjoint regions:

  - a "mass block" of m bits: any proposal landing here is ABSORBED --
    it does not produce any observable change (the block is treated as a
    fixed, locked structure that resists being altered). This tick is
    "wasted" from the point of view of everything outside the block.
  - a "pattern" of k bits (the repair_cost_poc.py mass CANDIDATE): any
    proposal landing here is a genuine corruption event requiring repair,
    exactly as before.
  - the remaining n-m-k bits: unstructured background; a proposal landing
    here is a "real" (non-absorbed, non-corrupting) background tick.

Three things are measured over T raw universe-ticks:
  1. The LOCAL CLOCK RATE: the fraction of raw ticks that produce ANY real
     (non-absorbed) change anywhere outside the mass block. Prediction:
     (n-m)/n, i.e. this should fall as the mass block grows, exactly the
     qualitative signature of gravitational time dilation (more mass
     nearby => slower effective local clock), derived here purely from
     tick-budget competition, not assumed from GR.
  2. The pattern's ABSOLUTE repair rate (repairs per RAW universe-tick).
     Prediction: k/n, UNCHANGED by the presence of the mass block, since
     corruption events depend only on whether the proposal lands in the
     k-bit region, which the mass block does not touch.
  3. The pattern's repair rate PER REAL (non-absorbed) TICK -- i.e., in
     units of the observer's OWN experienced time, where absorbed ticks
     don't count as elapsed time at all. Prediction: k/(n-m), which RISES
     as the mass block grows even though the absolute rate (2) does not --
     the same physical process looks faster when measured against a
     slowed local clock, the toy analogue of blueshift for an observer
     deep in a gravity well.
"""

import numpy as np


def simulate(n, m, k, T, rng):
    idx = rng.integers(0, n, size=T)
    absorbed = idx < m
    corrupted = (idx >= m) & (idx < m + k)
    real_ticks = np.sum(~absorbed)
    corruption_count = np.sum(corrupted)
    return real_ticks, corruption_count


def run():
    rng = np.random.default_rng(0)
    n = 20_000
    k = 100
    T = 500_000

    print("=" * 96)
    print("Local clock rate and pattern repair rate vs mass-block size m")
    print("=" * 96)
    header = (f"{'m/n':>8}{'local clock rate':>20}{'theory (n-m)/n':>18}"
              f"{'abs repair rate':>18}{'theory k/n':>14}"
              f"{'repair/real-tick':>18}{'theory k/(n-m)':>18}")
    print(header)

    m_fractions = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    for frac in m_fractions:
        m = int(frac * (n - k))  # keep room for the pattern
        real_ticks, corruption_count = simulate(n, m, k, T, rng)

        local_clock_rate = real_ticks / T
        local_clock_theory = (n - m) / n

        abs_repair_rate = corruption_count / T
        abs_repair_theory = k / n

        repair_per_real = corruption_count / real_ticks if real_ticks > 0 else float("nan")
        repair_per_real_theory = k / (n - m) if (n - m) > 0 else float("inf")

        print(f"{frac:8.2f}{local_clock_rate:20.5f}{local_clock_theory:18.5f}"
              f"{abs_repair_rate:18.6f}{abs_repair_theory:14.6f}"
              f"{repair_per_real:18.6f}{repair_per_real_theory:18.6f}")

    print()
    print("=" * 96)
    print("Interpretation")
    print("=" * 96)
    print("""
  - Local clock rate tracks (n-m)/n closely across the full range, falling
    from 1.0 (no mass) toward 0 as the mass block consumes nearly the
    entire bit budget (m/n -> 1). This IS a genuine, computable, non-
    imported time-dilation-like effect: a structure that locks up a larger
    share of the universe's finite per-tick "attention" leaves
    correspondingly less of that attention for everything else, so the
    rate of real, observable change elsewhere slows. As m/n -> 1 the local
    clock rate -> 0, the toy analogue of an observer's proper time nearly
    stopping relative to the outside as a horizon is approached.

  - The pattern's ABSOLUTE repair rate stays flat at k/n regardless of m,
    confirming it is untouched by the mass block's presence in raw
    universe-tick terms -- corruption events only care about whether the
    proposal lands in the pattern's own k bits.

  - But the pattern's repair rate PER REAL TICK rises with m, tracking
    k/(n-m): the SAME absolute process looks like it is happening faster
    once time is measured in the observer's own (slowed) experienced
    ticks rather than raw universe ticks. This is the toy version of
    blueshift -- processes elsewhere look sped up to an observer whose own
    clock has been slowed by nearby mass.

  IMPORTANT CAVEAT -- what this does and does NOT establish:

  This is a GRAVITATIONAL-style time dilation mechanism (mass slows a
  local clock via bit-budget competition), and it is a clean, honest,
  computable result on its own terms. It is NOT the same thing as the
  KINEMATIC (velocity-based) time dilation that combined_poc.py showed was
  missing for combining motion-cost and repair-cost into anything
  quadrature-like. Those are two distinct effects in real GR too (though
  related by the same metric). This experiment shows mass-as-bit-lock
  CAN suppress a local clock rate through pure resource competition, using
  nothing beyond what the framework already has (the Ehrenfest process).
  It does NOT yet show that MOTION (a pattern hopping through the lattice)
  produces an analogous suppression of its OWN repair rate -- that is a
  separate hypothesis, structurally similar but not established here, and
  would need its own dedicated test: does a hopping pattern's bit budget
  competition with the SAME background process reduce ITS OWN repair rate
  as its hop distance grows? That is the natural next experiment if this
  thread is worth continuing.
""")


if __name__ == "__main__":
    run()
