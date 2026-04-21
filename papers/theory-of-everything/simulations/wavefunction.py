import math
from collections import defaultdict
from typing import List, Tuple, Dict

Mode = Tuple[float, float, float]
# (amplitude, phase, frequency)


class Wavefunction:
    """
    Static spectral representation of a configuration.

    The wavefunction is evaluated via a description-length functional
    measuring the bit-cost of encoding spectral structure under finite resolution.
    """

    def __init__(self, num_bits: int):
        """
        Args:
            num_bits: Global resolution scale of representation space.
        """
        self.num_bits: int = num_bits
        self.modes: List[Mode] = []

        # resolution scales (can later be derived from num_bits)
        self.delta_A: float = 1e-3
        self.delta_phi: float = 1e-3
        self.delta_omega: float = 1e-3

    # ------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------

    def add_mode(self, amplitude: float, phase: float, omega: float) -> None:
        """
        Add spectral mode.

        Args:
            amplitude: A_i
            phase: φ_i in [0, 2π]
            omega: frequency ω_i
        """
        self.modes.append((amplitude, phase, omega))

    # ------------------------------------------------------------
    # Total complexity
    # ------------------------------------------------------------

    def spectral_complexity(self) -> float:
        """
        Total description length (bits) of the wavefunction.
        """
        return self._spectral_cost() + self._occupancy_penalty()

    # ------------------------------------------------------------
    # Spectral cost
    # ------------------------------------------------------------

    def _spectral_cost(self) -> float:
        """
        Cost of encoding individual modes.
        """
        cost: float = 0.0

        for amplitude, phase, omega in self.modes:
            cost += math.log2(abs(amplitude) / self.delta_A + 1.0)
            cost += math.log2(2.0 * math.pi / self.delta_phi)
            cost += math.log2(abs(omega) / self.delta_omega + 1.0)

        return cost

    # ------------------------------------------------------------
    # Occupancy / symmetry cost
    # ------------------------------------------------------------

    def _occupancy_penalty(self) -> float:
        """
        Information cost of representing indistinguishable mode occupancy.

        Key idea:
        - compression depends on how distinguishable modes remain under resolution
        - indistinguishable clusters require relational encoding overhead
        """
        buckets: Dict[tuple, List[Mode]] = defaultdict(list)

        for amplitude, phase, omega in self.modes:
            key = self._mode_class(amplitude, phase, omega)
            buckets[key].append((amplitude, phase, omega))

        cost: float = 0.0

        for modes in buckets.values():
            n = len(modes)
            if n <= 1:
                continue
            cost += self._cluster_cost(modes)

        return cost

    # ------------------------------------------------------------
    # Mode equivalence (coarse-graining)
    # ------------------------------------------------------------

    def _mode_class(self, amplitude: float, phase: float, omega: float) -> tuple:
        """
        Maps mode into equivalence class under resolution.
        """
        return (
            round(amplitude / self.delta_A),
            round(phase / self.delta_phi),
            round(omega / self.delta_omega),
        )

    # ------------------------------------------------------------
    # Cluster encoding cost (core upgrade)
    # ------------------------------------------------------------

    def _cluster_cost(self, modes: List[Mode]) -> float:
        """
        Cost of encoding a cluster of indistinguishable modes.

        This replaces naive factorial counting with
        resolution-aware representational cost.
        """
        n: int = len(modes)

        # measure internal spread of indistinguishability
        total_residual: float = 0.0
        count: int = 0

        for i in range(n):
            for j in range(i + 1, n):
                ai, pi, wi = modes[i]
                aj, pj, wj = modes[j]

                total_residual += (
                    abs(ai - aj) / self.delta_A +
                    abs(pi - pj) / self.delta_phi +
                    abs(wi - wj) / self.delta_omega
                )
                count += 1

        avg_residual: float = total_residual / max(count, 1)

        # compressibility measure (purely emergent, no regime flags)
        indistinguishability: float = math.exp(-avg_residual)

        # smooth transition between regimes (no hard switch)
        # bosonic-like: high compressibility
        # fermionic-like: low compressibility (representation strain)
        return math.log2(n + 1) + (1.0 - indistinguishability) * n * math.log2(n + 1)
