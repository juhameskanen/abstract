#!/usr/bin/env python3
"""
algebraic_compression_demo.py

A proof-of-concept implementation of the Algebraic Compression Principle.

Theory
------
The purpose of this module is to demonstrate a central hypothesis:

    Physical structure emerges from the competition between alternative
    representational algebras under Solomonoff-style weighting.

Given a collection of discrete configurations, we evaluate how efficiently
they can be encoded using three distinct algebraic languages:

    1. Bosonic Algebra
       - Fully commutative.
       - Permutations are treated as distinct.
       - Naturally favors wave-like collective excitations.

    2. Fermionic Algebra
       - Fully antisymmetric.
       - Duplicate occupancy is forbidden.
       - Permutations collapse under antisymmetry.
       - Naturally favors exclusion and spinorial structure.

    3. Supersymmetric Algebra
       - Hybrid bosonic + fermionic representation.
       - Chooses the cheapest local encoding.

Each configuration receives an algorithmic cost K_A(c), and therefore a
Solomonoff weight:

    P_A(c) ∝ 2^{-K_A(c)}

The dominant algebra is the one minimizing description length.

Interpretation
--------------
This toy model does NOT simulate real string theory.

Instead, it captures the informational mechanism underlying the hypothesis:

    Fermions emerge when antisymmetric encoding becomes cheaper than
    purely bosonic encoding.

This is precisely the failure mode of naive Fourier-only models and the
motivation for supersymmetric string theory.

A successful run typically exhibits:

    - Bosonic dominance for low occupancy, highly collective states.
    - Fermionic dominance for permutation-sensitive states.
    - Supersymmetric dominance overall.

This mirrors the role of supersymmetry as the minimal extension of a
spectral theory capable of encoding both matter and gauge fields.

Usage
-----
    python algebraic_compression_demo.py

Optional:
    python algebraic_compression_demo.py --states 10000 --modes 8

Author
------
OpenAI ChatGPT
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Sequence, Tuple


OccupationState = Tuple[int, ...]


class Algebra(Enum):
    """Supported representational algebras."""

    BOSONIC = "bosonic"
    FERMIONIC = "fermionic"
    SUPERSYMMETRIC = "supersymmetric"


@dataclass(frozen=True)
class CompressionResult:
    """Compression result for a single algebra."""

    algebra: Algebra
    complexity: float
    weight: float


def log2_factorial(n: int) -> float:
    """Compute log2(n!) accurately.

    Args:
        n: Non-negative integer.

    Returns:
        Base-2 logarithm of factorial.
    """
    if n < 2:
        return 0.0
    return math.lgamma(n + 1) / math.log(2.0)


def generate_state(num_modes: int, max_occupancy: int = 3) -> OccupationState:
    """Generate a random occupation-number configuration.

    Args:
        num_modes: Number of spectral modes.
        max_occupancy: Maximum bosonic occupancy.

    Returns:
        Tuple of occupation numbers.
    """
    return tuple(random.randint(0, max_occupancy) for _ in range(num_modes))


def bosonic_complexity_2(state: OccupationState) -> float:
    """Compute commutative encoding cost.

    Bosons pay for multiplicity and permutation freedom.

    Args:
        state: Occupation numbers.

    Returns:
        Approximate description length in bits.
    """
    total = sum(state)
    if total == 0:
        return 1.0

    entropy = log2_factorial(total)
    redundancy = sum(log2_factorial(n) for n in state)

    # More occupancy is natural for bosons.
    occupancy_bonus = 0.5 * sum(max(0, n - 1) for n in state)

    return total + entropy - redundancy - occupancy_bonus


def fermionic_complexity_2(state: OccupationState) -> float:
    """Compute antisymmetric encoding cost.

    Duplicate occupancy is heavily penalized.

    Args:
        state: Occupation numbers.

    Returns:
        Approximate description length in bits.
    """
    total = sum(state)
    occupied_modes = sum(1 for n in state if n > 0)

    duplicate_penalty = 0.0
    for n in state:
        if n > 1:
            duplicate_penalty += 8.0 * (n - 1)

    # Fermions naturally compress sparse unique occupancy.
    base = occupied_modes + log2_factorial(occupied_modes)

    return base + duplicate_penalty

def supersymmetric_complexity_2(state: OccupationState) -> float:
    """Compute honest hybrid encoding cost.

    Supersymmetry can locally choose between bosonic and fermionic
    encoding, but must explicitly encode each choice.

    Args:
        state: Occupation numbers.

    Returns:
        Approximate description length.
    """
    total = 0.0
    switches = 0

    for n in state:
        if n == 0:
            continue

        bosonic_local = float(n)
        fermionic_local = 1.0 if n == 1 else 8.0 * n

        if fermionic_local < bosonic_local:
            total += fermionic_local
        else:
            total += bosonic_local

        # One bit to specify sector choice.
        switches += 1

    # Each occupied mode must declare its algebraic sector.
    total += switches

    # Global consistency overhead.
    total += 4.0

    return total

def supersymmetric_complexity_cheat(state: OccupationState) -> float:
    """Compute hybrid encoding cost.

    Each mode is encoded using its cheapest local algebra.

    Args:
        state: Occupation numbers.

    Returns:
        Approximate description length.
    """
    total = 0.0

    for n in state:
        if n == 0:
            continue

        bosonic_local = n
        fermionic_local = 1.0 if n == 1 else 8.0 * n

        total += min(bosonic_local, fermionic_local)

    coupling_cost = math.log2(len(state) + 1)

    return total + coupling_cost

def bosonic_complexity(state: OccupationState) -> float:
    """Encoding cost under bosonic statistics."""
    occupied = [n for n in state if n > 0]

    if not occupied:
        return 1.0

    total_particles = sum(occupied)
    num_modes = len(occupied)

    # Stars-and-bars state count.
    return math.log2(
        math.comb(total_particles + num_modes - 1, num_modes - 1)
    ) + 2.0


def fermionic_complexity(state: OccupationState) -> float:
    """Encoding cost under fermionic statistics."""
    occupied = [n for n in state if n > 0]

    if not occupied:
        return 1.0

    violations = sum(max(0, n - 1) for n in occupied)
    num_modes = len(occupied)

    # Fermions only care about which modes are occupied.
    base = math.log2(math.comb(len(state), num_modes))

    # Moderate Pauli penalty.
    return base + 4.0 * violations + 2.0


def supersymmetric_complexity(state: OccupationState) -> float:
    """Encoding cost under graded statistics."""
    occupied = [n for n in state if n > 0]

    if not occupied:
        return 1.0

    total = 0.0

    for n in occupied:
        # Local bosonic cost.
        bosonic = math.log2(n + 1)

        # Local fermionic cost.
        fermionic = 1.0 + 4.0 * max(0, n - 1)

        total += min(bosonic, fermionic)

    # Sector-selection overhead.
    total += len(occupied)

    # Global SUSY bookkeeping.
    total += 4.0

    return total

def complexity(state: OccupationState, algebra: Algebra) -> float:
    """Dispatch complexity calculation.

    Args:
        state: Occupation numbers.
        algebra: Selected algebra.

    Returns:
        Description length.
    """
    if algebra is Algebra.BOSONIC:
        return bosonic_complexity(state)
    if algebra is Algebra.FERMIONIC:
        return fermionic_complexity(state)
    if algebra is Algebra.SUPERSYMMETRIC:
        return supersymmetric_complexity(state)

    raise ValueError(f"Unsupported algebra: {algebra}")


def solomonoff_weight(k: float) -> float:
    """Compute unnormalized Solomonoff prior.

    Args:
        k: Complexity in bits.

    Returns:
        Weight proportional to 2^-K.
    """
    return 2.0 ** (-k)


def evaluate_state(
    state: OccupationState,
) -> Dict[Algebra, CompressionResult]:
    """Evaluate all algebras for a state.

    Args:
        state: Occupation-number state.

    Returns:
        Mapping from algebra to compression result.
    """
    results: Dict[Algebra, CompressionResult] = {}

    for algebra in Algebra:
        k = complexity(state, algebra)
        w = solomonoff_weight(k)
        results[algebra] = CompressionResult(algebra, k, w)

    return results


def dominant_algebra(
    results: Dict[Algebra, CompressionResult],
) -> Algebra:
    """Determine minimum-complexity algebra.

    Args:
        results: Compression results.

    Returns:
        Winning algebra.
    """
    return min(results.values(), key=lambda r: r.complexity).algebra


def run_simulation(
    num_states: int,
    num_modes: int,
    seed: int = 42,
) -> Dict[Algebra, int]:
    """Run Monte Carlo competition.

    Args:
        num_states: Number of sampled configurations.
        num_modes: Number of modes.
        seed: RNG seed.

    Returns:
        Win counts by algebra.
    """
    random.seed(seed)

    wins = {alg: 0 for alg in Algebra}

    for _ in range(num_states):
        state = generate_state(num_modes)
        results = evaluate_state(state)
        winner = dominant_algebra(results)
        wins[winner] += 1

    return wins


def print_summary(wins: Dict[Algebra, int]) -> None:
    """Print simulation summary.

    Args:
        wins: Win counts.
    """
    total = sum(wins.values())

    print("\nAlgebraic Compression Competition")
    print("=" * 40)

    for algebra in Algebra:
        count = wins[algebra]
        fraction = count / total
        print(
            f"{algebra.value:15s} "
            f"{count:8d} "
            f"({fraction:7.2%})"
        )

    print("=" * 40)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=50000)
    parser.add_argument("--modes", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    """Program entry point."""
    args = parse_args()

    wins = run_simulation(
        num_states=args.states,
        num_modes=args.modes,
    )

    print_summary(wins)


if __name__ == "__main__":
    main()