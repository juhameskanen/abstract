"""Observer-conditioned compressed quantum-history backend.

This module implements a third cosmological-model backend alongside the
classical statistical bitstring model and the fixed-sector Dicke model.

The construction has three layers:

1.  A finite bitstring follows the exact discrete Ehrenfest rule: exactly one
    uniformly selected bit is toggled per internal tick.
2.  The *complete history* is conditioned on a finite observer record at one
    internal tick.  Conditioning is computed exactly by a forward/backward
    bridge on three exchangeable groups of bits.
3.  Every conditional frame is lifted to a complex wavefunction

        psi_t(x | O) = sqrt(P_t(x | O)) exp(i Phi_t(x)),

    where Phi is a low-description DCT-like phase codec.  The phase codec can
    break permutation symmetry and generate interference/entanglement, while
    computational-basis Born sampling remains exactly P_t(x | O).

The module is deliberately self-contained (NumPy only) so it can be dropped
into papers/cosmological-model/simulations without adding dependencies.

Important scope note
--------------------
This is a first computable quantum-history lift.  The finite spectral-code
length below is an MDL proxy, not an implementation of uncomputable Solomonoff
induction.  It is intended as a test bed for a future universal mixture over
short generators.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log2
from typing import Iterable, Literal, Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

TRUE_K_RATE = 2.0


def _validate_probability_vector(prob: FloatArray, *, name: str = "probability") -> None:
    if prob.ndim == 0:
        raise ValueError(f"{name} must be an array, not a scalar")
    if not np.all(np.isfinite(prob)):
        raise ValueError(f"{name} contains NaN or infinity")
    if np.any(prob < -1e-13):
        raise ValueError(f"{name} contains negative entries")
    total = float(np.sum(prob))
    if not np.isclose(total, 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} is not normalized: sum={total:.16g}")


def binary_entropy(prob: FloatArray | float) -> FloatArray | float:
    """Binary Shannon entropy H_2(p), in bits."""

    p = np.asarray(prob, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    out = np.zeros_like(p)
    middle = (p > 0.0) & (p < 1.0)
    pm = p[middle]
    out[middle] = -pm * np.log2(pm) - (1.0 - pm) * np.log2(1.0 - pm)
    if np.ndim(prob) == 0:
        return float(out)
    return out


def log2_binomial_row(size: int) -> FloatArray:
    """Return log2(C(size, k)) for k=0..size without SciPy."""

    if size < 0:
        raise ValueError("size must be non-negative")
    values = np.empty(size + 1, dtype=float)
    ln2 = np.log(2.0)
    for k in range(size + 1):
        values[k] = (lgamma(size + 1) - lgamma(k + 1) - lgamma(size - k + 1)) / ln2
    return values


def hypergeom_pmf(population: int, successes: IntArray | int, draws: int, observed: int) -> FloatArray:
    """Vectorized hypergeometric PMF, implemented in log space.

    P(J=observed | population, successes, draws)
    """

    if population < 0:
        raise ValueError("population must be non-negative")
    if draws < 0 or draws > population:
        raise ValueError("draws must be in [0, population]")
    if observed < 0 or observed > draws:
        return np.zeros_like(np.asarray(successes, dtype=float), dtype=float)

    k = np.asarray(successes, dtype=int)
    out = np.zeros_like(k, dtype=float)
    valid = (
        (k >= 0)
        & (k <= population)
        & (observed <= k)
        & ((draws - observed) <= (population - k))
    )
    if not np.any(valid):
        return out

    log_c_population_draws = (
        lgamma(population + 1)
        - lgamma(draws + 1)
        - lgamma(population - draws + 1)
    ) / np.log(2.0)

    kv = k[valid]
    # C(k, observed)
    log_left = np.array(
        [
            (lgamma(int(value) + 1) - lgamma(observed + 1) - lgamma(int(value) - observed + 1))
            / np.log(2.0)
            for value in kv
        ],
        dtype=float,
    )
    remaining_draws = draws - observed
    log_right = np.array(
        [
            (
                lgamma(population - int(value) + 1)
                - lgamma(remaining_draws + 1)
                - lgamma(population - int(value) - remaining_draws + 1)
            )
            / np.log(2.0)
            for value in kv
        ],
        dtype=float,
    )
    out[valid] = np.exp2(log_left + log_right - log_c_population_draws)
    return out


def exact_ehrenfest_weight_distributions(n_bits: int, max_tick: int) -> FloatArray:
    """Exact P(K_t=k) for the one-flip-per-tick Ehrenfest chain.

    Returns an array of shape (max_tick + 1, n_bits + 1).
    """

    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if max_tick < 0:
        raise ValueError("max_tick must be non-negative")

    history = np.zeros((max_tick + 1, n_bits + 1), dtype=float)
    history[0, 0] = 1.0
    up_prob = (n_bits - np.arange(n_bits, dtype=float)) / n_bits
    down_prob = np.arange(1, n_bits + 1, dtype=float) / n_bits

    for tick in range(max_tick):
        current = history[tick]
        nxt = np.zeros_like(current)
        nxt[1:] += current[:-1] * up_prob
        nxt[:-1] += current[1:] * down_prob
        history[tick + 1] = nxt

    return history


@dataclass(frozen=True)
class ObserverRecord:
    """A finite pixel record used to condition the complete history.

    ``indices`` and ``pattern`` have the same length.  At ``tick`` the bits at
    those indices are required to equal the pattern exactly.
    """

    indices: tuple[int, ...]
    pattern: tuple[int, ...]
    tick: int
    label: str = "observer"

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError("observer tick must be non-negative")
        if len(self.indices) != len(self.pattern):
            raise ValueError("observer indices and pattern must have equal length")
        if len(set(self.indices)) != len(self.indices):
            raise ValueError("observer indices must be unique")
        if any(bit not in (0, 1) for bit in self.pattern):
            raise ValueError("observer pattern must contain only 0 and 1")

    @property
    def width(self) -> int:
        return len(self.pattern)

    @property
    def one_indices(self) -> tuple[int, ...]:
        return tuple(index for index, bit in zip(self.indices, self.pattern) if bit == 1)

    @property
    def zero_indices(self) -> tuple[int, ...]:
        return tuple(index for index, bit in zip(self.indices, self.pattern) if bit == 0)

    @staticmethod
    def none(tick: int = 0) -> "ObserverRecord":
        return ObserverRecord(indices=(), pattern=(), tick=tick, label="identity")

    @staticmethod
    def contiguous(pattern: str, tick: int, start: int = 0, label: str = "observer") -> "ObserverRecord":
        cleaned = pattern.strip().replace("_", "")
        if any(char not in "01" for char in cleaned):
            raise ValueError("pattern must be a binary string")
        indices = tuple(range(start, start + len(cleaned)))
        return ObserverRecord(indices, tuple(int(char) for char in cleaned), tick, label)

    @staticmethod
    def gaussian_blob(
        width: int,
        tick: int,
        *,
        start: int = 0,
        sigma: float | None = None,
        threshold: float = 0.45,
        label: str = "gaussian-observer",
    ) -> "ObserverRecord":
        """Create a thresholded one-dimensional Gaussian pixel record."""

        if width <= 0:
            raise ValueError("Gaussian observer width must be positive")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie strictly between 0 and 1")
        resolved_sigma = sigma if sigma is not None else max(width / 5.0, 0.75)
        x = np.arange(width, dtype=float)
        center = 0.5 * (width - 1)
        profile = np.exp(-0.5 * ((x - center) / resolved_sigma) ** 2)
        pattern = tuple(int(value >= threshold) for value in profile)
        return ObserverRecord(tuple(range(start, start + width)), pattern, tick, label)


@dataclass(frozen=True)
class SpectralPhaseCodecConfig:
    """Low-description DCT-like phase codec configuration."""

    temporal_modes: int = 3
    spatial_modes: int = 3
    phase_strength: float = 1.0
    topology: Literal["ring", "open-chain", "seeded-sparse", "none"] = "ring"
    sparse_edges: int | None = None
    seed: int = 7

    def __post_init__(self) -> None:
        if self.temporal_modes < 0 or self.spatial_modes < 0:
            raise ValueError("mode counts must be non-negative")
        if self.phase_strength < 0:
            raise ValueError("phase_strength must be non-negative")
        if self.sparse_edges is not None and self.sparse_edges < 0:
            raise ValueError("sparse_edges must be non-negative")


class SpectralPhaseCodec:
    """A compact complex phase residual over bit configurations.

    The phase is a low-rank spatial/temporal cosine expansion plus pairwise
    controlled-phase terms on a simple graph.  Because it is diagonal in the
    pixel basis, it leaves all computational-basis Born probabilities exactly
    unchanged.
    """

    def __init__(self, n_bits: int, config: SpectralPhaseCodecConfig) -> None:
        if n_bits <= 0:
            raise ValueError("n_bits must be positive")
        self.n_bits = int(n_bits)
        self.config = config
        self.edges = self._build_edges()
        self._node_coeff, self._edge_coeff = self._build_coefficients()

    def _build_edges(self) -> IntArray:
        topology = self.config.topology
        n = self.n_bits
        if topology == "none" or n == 1:
            return np.empty((0, 2), dtype=np.int64)
        if topology == "ring":
            return np.array([(i, (i + 1) % n) for i in range(n)], dtype=np.int64)
        if topology == "open-chain":
            return np.array([(i, i + 1) for i in range(n - 1)], dtype=np.int64)
        if topology == "seeded-sparse":
            max_edges = n * (n - 1) // 2
            requested = self.config.sparse_edges
            edge_count = min(max_edges, requested if requested is not None else max(n, 1))
            rng = np.random.default_rng(self.config.seed)
            all_pairs = np.array([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64)
            if edge_count == 0:
                return np.empty((0, 2), dtype=np.int64)
            chosen = rng.choice(len(all_pairs), size=edge_count, replace=False)
            return all_pairs[np.sort(chosen)]
        raise ValueError(f"unsupported topology: {topology}")

    def _build_coefficients(self) -> tuple[FloatArray, FloatArray]:
        spatial = self.config.spatial_modes
        temporal = self.config.temporal_modes
        if spatial == 0 or temporal == 0 or self.config.phase_strength == 0.0:
            return (
                np.zeros((spatial, temporal), dtype=float),
                np.zeros((spatial, temporal), dtype=float),
            )

        rng = np.random.default_rng(self.config.seed)
        damping = 1.0 / (
            (1.0 + np.arange(spatial, dtype=float))[:, None]
            * (1.0 + np.arange(temporal, dtype=float))[None, :]
        ) ** 1.5
        node = rng.normal(size=(spatial, temporal)) * damping
        edge = rng.normal(size=(spatial, temporal)) * damping
        return node, edge

    @property
    def description_length_proxy_bits(self) -> float:
        """Finite MDL proxy for the codec, not Kolmogorov complexity."""

        cfg = self.config
        if cfg.topology == "none" or cfg.phase_strength == 0.0:
            return 1.0
        # Opcode + seed + integer parameters + quantized low-rank coefficients.
        coefficient_count = 2 * cfg.spatial_modes * cfg.temporal_modes
        topology_cost = 2.0
        if cfg.topology == "seeded-sparse":
            topology_cost += log2(max(2, len(self.edges) + 1))
        return 32.0 + 32.0 + 4.0 * 16.0 + 12.0 * coefficient_count + topology_cost

    def _temporal_basis(self, normalized_time: float) -> FloatArray:
        modes = np.arange(1, self.config.temporal_modes + 1, dtype=float)
        return np.cos(np.pi * modes * np.clip(normalized_time, 0.0, 1.0))

    def _spatial_basis(self, positions: FloatArray) -> FloatArray:
        modes = np.arange(1, self.config.spatial_modes + 1, dtype=float)[:, None]
        return np.cos(np.pi * modes * (positions[None, :] + 0.5) / self.n_bits)

    def phases(self, normalized_time: float) -> tuple[FloatArray, FloatArray]:
        """Return node phases alpha_j and edge phases beta_e."""

        if self.config.temporal_modes == 0 or self.config.spatial_modes == 0:
            return np.zeros(self.n_bits), np.zeros(len(self.edges))
        temporal = self._temporal_basis(normalized_time)
        node_mode_amplitudes = self._node_coeff @ temporal
        edge_mode_amplitudes = self._edge_coeff @ temporal

        node_basis = self._spatial_basis(np.arange(self.n_bits, dtype=float))
        alpha = self.config.phase_strength * (node_mode_amplitudes @ node_basis)

        if len(self.edges) == 0:
            beta = np.zeros(0, dtype=float)
        else:
            edge_positions = 0.5 * (self.edges[:, 0] + self.edges[:, 1]).astype(float)
            edge_basis = self._spatial_basis(edge_positions)
            beta = self.config.phase_strength * (edge_mode_amplitudes @ edge_basis)
        return alpha.astype(float), beta.astype(float)

    def phase_of_bitstrings(self, bitstrings: BoolArray, normalized_time: float) -> FloatArray:
        """Evaluate Phi_t(x) for a batch of bitstrings."""

        bits = np.asarray(bitstrings, dtype=float)
        if bits.ndim != 2 or bits.shape[1] != self.n_bits:
            raise ValueError(f"bitstrings must have shape (samples, {self.n_bits})")
        alpha, beta = self.phases(normalized_time)
        phase = bits @ alpha
        if len(self.edges):
            pair_products = bits[:, self.edges[:, 0]] * bits[:, self.edges[:, 1]]
            phase = phase + pair_products @ beta
        return np.asarray(phase, dtype=float)

    def residual_rms(self, normalized_time: float) -> float:
        alpha, beta = self.phases(normalized_time)
        values = np.concatenate([alpha, beta])
        return float(np.sqrt(np.mean(values**2))) if values.size else 0.0


@dataclass(frozen=True)
class CountGroups:
    """Partition induced by an exact observer pattern."""

    target_one_positions: tuple[int, ...]
    target_zero_positions: tuple[int, ...]
    rest_positions: tuple[int, ...]

    @property
    def target_ones(self) -> int:
        return len(self.target_one_positions)

    @property
    def target_zeros(self) -> int:
        return len(self.target_zero_positions)

    @property
    def rest(self) -> int:
        return len(self.rest_positions)


@dataclass
class ConditionalFrame:
    """Observer-conditioned Born distribution at one internal tick."""

    tick: int
    probability_counts: FloatArray
    groups: CountGroups
    phase_codec: SpectralPhaseCodec
    max_tick: int
    _log2_combinations: tuple[FloatArray, FloatArray, FloatArray]

    def __post_init__(self) -> None:
        _validate_probability_vector(self.probability_counts, name="conditional count distribution")
        expected_shape = (
            self.groups.target_ones + 1,
            self.groups.target_zeros + 1,
            self.groups.rest + 1,
        )
        if self.probability_counts.shape != expected_shape:
            raise ValueError(
                f"conditional count distribution has shape {self.probability_counts.shape}, "
                f"expected {expected_shape}"
            )

    @property
    def n_bits(self) -> int:
        return self.groups.target_ones + self.groups.target_zeros + self.groups.rest

    @property
    def normalized_time(self) -> float:
        if self.max_tick <= 0:
            return 0.0
        return self.tick / self.max_tick

    def count_expectations(self) -> tuple[float, float, float]:
        a_idx, b_idx, l_idx = np.indices(self.probability_counts.shape)
        p = self.probability_counts
        return (
            float(np.sum(p * a_idx)),
            float(np.sum(p * b_idx)),
            float(np.sum(p * l_idx)),
        )

    @property
    def mean_ones(self) -> float:
        return float(sum(self.count_expectations()))

    @property
    def mean_one_fraction(self) -> float:
        return self.mean_ones / self.n_bits

    def total_weight_distribution(self) -> FloatArray:
        """Return P(K_t=k | observer), k=0..n."""

        a_idx, b_idx, l_idx = np.indices(self.probability_counts.shape)
        total = (a_idx + b_idx + l_idx).ravel()
        values = np.bincount(
            total,
            weights=self.probability_counts.ravel(),
            minlength=self.n_bits + 1,
        )
        values = np.asarray(values, dtype=float)
        values /= np.sum(values)
        return values

    def rest_weight_distribution(self) -> FloatArray:
        """Return the exact marginal P(L_t=l | observer) for non-observer bits."""

        values = np.sum(self.probability_counts, axis=(0, 1))
        values = np.asarray(values, dtype=float)
        values /= np.sum(values)
        return values

    @property
    def born_entropy_bits(self) -> float:
        """Shannon entropy of computational-basis Born outcomes."""

        p = self.probability_counts
        positive = p > 0.0
        coarse_entropy = float(-np.sum(p[positive] * np.log2(p[positive])))

        log_c_a, log_c_b, log_c_l = self._log2_combinations
        a_idx, b_idx, l_idx = np.indices(p.shape)
        degeneracy_entropy = float(
            np.sum(p * (log_c_a[a_idx] + log_c_b[b_idx] + log_c_l[l_idx]))
        )
        return coarse_entropy + degeneracy_entropy

    def bulk_composition_probability(self, width: int, ones: int) -> float:
        """Probability that a width-w window in the non-observer bulk has `ones` ones."""

        if width <= 0:
            raise ValueError("width must be positive")
        if width > self.groups.rest:
            raise ValueError(
                f"width={width} exceeds non-observer bulk size {self.groups.rest}; "
                "use a smaller observer or a smaller window"
            )
        if ones < 0 or ones > width:
            return 0.0
        l_values = np.arange(self.groups.rest + 1, dtype=np.int64)
        pmf = hypergeom_pmf(self.groups.rest, l_values, width, ones)
        return float(np.dot(self.rest_weight_distribution(), pmf))

    def _sample_group_configurations(
        self,
        count: int,
        positions: tuple[int, ...],
        rng: np.random.Generator,
    ) -> NDArray[np.bool_]:
        chosen = np.zeros(self.n_bits, dtype=bool)
        if count:
            selected = rng.choice(np.asarray(positions, dtype=int), size=count, replace=False)
            chosen[selected] = True
        return chosen

    def sample_bitstrings(self, samples: int, rng: np.random.Generator) -> BoolArray:
        """Draw exact computational-basis Born samples from this frame."""

        if samples <= 0:
            raise ValueError("samples must be positive")
        flat = self.probability_counts.ravel()
        states = rng.choice(flat.size, size=samples, p=flat)
        counts = np.array(np.unravel_index(states, self.probability_counts.shape)).T
        output = np.zeros((samples, self.n_bits), dtype=bool)
        for row, (a_count, b_count, l_count) in enumerate(counts):
            output[row] |= self._sample_group_configurations(
                int(a_count), self.groups.target_one_positions, rng
            )
            output[row] |= self._sample_group_configurations(
                int(b_count), self.groups.target_zero_positions, rng
            )
            output[row] |= self._sample_group_configurations(
                int(l_count), self.groups.rest_positions, rng
            )
        return output

    def configuration_probability(self, bitstrings: BoolArray) -> FloatArray:
        """Return P_t(x | observer) for a batch of bitstrings."""

        bits = np.asarray(bitstrings, dtype=bool)
        if bits.ndim != 2 or bits.shape[1] != self.n_bits:
            raise ValueError(f"bitstrings must have shape (samples, {self.n_bits})")
        one_pos = np.asarray(self.groups.target_one_positions, dtype=int)
        zero_pos = np.asarray(self.groups.target_zero_positions, dtype=int)
        rest_pos = np.asarray(self.groups.rest_positions, dtype=int)
        a = np.sum(bits[:, one_pos], axis=1) if one_pos.size else np.zeros(len(bits), dtype=int)
        b = np.sum(bits[:, zero_pos], axis=1) if zero_pos.size else np.zeros(len(bits), dtype=int)
        l = np.sum(bits[:, rest_pos], axis=1) if rest_pos.size else np.zeros(len(bits), dtype=int)

        log_c_a, log_c_b, log_c_l = self._log2_combinations
        coarse = self.probability_counts[a, b, l]
        degeneracy = np.exp2(log_c_a[a] + log_c_b[b] + log_c_l[l])
        return coarse / degeneracy

    def amplitudes(self, bitstrings: BoolArray) -> ComplexArray:
        probabilities = self.configuration_probability(bitstrings)
        phases = self.phase_codec.phase_of_bitstrings(bitstrings, self.normalized_time)
        return np.sqrt(probabilities).astype(complex) * np.exp(1j * phases)

    def exact_statevector(self, max_qubits: int = 20) -> ComplexArray:
        """Construct the full 2^n statevector for validation at small n."""

        if self.n_bits > max_qubits:
            raise ValueError(
                f"n_bits={self.n_bits} exceeds max_qubits={max_qubits}; "
                "use sample_bitstrings for the scalable representation"
            )
        basis = np.arange(1 << self.n_bits, dtype=np.uint64)
        bit_positions = np.arange(self.n_bits, dtype=np.uint64)
        bits = ((basis[:, None] >> bit_positions[None, :]) & 1).astype(bool)
        psi = self.amplitudes(bits)
        norm = float(np.vdot(psi, psi).real)
        if not np.isclose(norm, 1.0, atol=1e-10):
            raise RuntimeError(f"constructed statevector is not normalized: norm={norm}")
        return psi

    def half_chain_entanglement_entropy(self, max_qubits: int = 16) -> float:
        """Exact bipartite von Neumann entropy for small validation systems."""

        psi = self.exact_statevector(max_qubits=max_qubits)
        left = self.n_bits // 2
        right = self.n_bits - left
        matrix = psi.reshape((1 << right, 1 << left))
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        probabilities = singular_values**2
        probabilities = probabilities[probabilities > 1e-15]
        return float(-np.sum(probabilities * np.log2(probabilities)))


class ObserverConditionedEhrenfestHistory:
    """Exact observer-conditioned discrete Ehrenfest history.

    The observer pattern partitions the bits into three exchangeable groups:
    positions required to be one, positions required to be zero, and all
    remaining bits.  The bridge therefore needs only O((s+1)(z+1)(m+1))
    states instead of 2^n bit configurations.
    """

    def __init__(
        self,
        n_bits: int,
        max_tick: int,
        observer: ObserverRecord | None,
        phase_codec: SpectralPhaseCodec | None = None,
    ) -> None:
        if n_bits <= 0:
            raise ValueError("n_bits must be positive")
        if max_tick < 0:
            raise ValueError("max_tick must be non-negative")
        self.n_bits = int(n_bits)
        self.max_tick = int(max_tick)
        self.observer = observer if observer is not None else ObserverRecord.none()
        if self.observer.tick > self.max_tick:
            raise ValueError("observer tick must not exceed max_tick")
        if any(index < 0 or index >= self.n_bits for index in self.observer.indices):
            raise ValueError("observer index outside the bitstring")

        one_positions = self.observer.one_indices
        zero_positions = self.observer.zero_indices
        used = set(one_positions) | set(zero_positions)
        rest_positions = tuple(index for index in range(self.n_bits) if index not in used)
        self.groups = CountGroups(one_positions, zero_positions, rest_positions)
        self.phase_codec = phase_codec or SpectralPhaseCodec(
            self.n_bits,
            SpectralPhaseCodecConfig(topology="none", phase_strength=0.0),
        )
        self._shape = (
            self.groups.target_ones + 1,
            self.groups.target_zeros + 1,
            self.groups.rest + 1,
        )
        self._log2_combinations = (
            log2_binomial_row(self.groups.target_ones),
            log2_binomial_row(self.groups.target_zeros),
            log2_binomial_row(self.groups.rest),
        )
        self._conditional_history, self.observer_evidence = self._build_bridge()

    def _forward_step(self, current: FloatArray) -> FloatArray:
        s = self.groups.target_ones
        z = self.groups.target_zeros
        m = self.groups.rest
        n = self.n_bits
        nxt = np.zeros_like(current)

        if s:
            up = (s - np.arange(s, dtype=float)) / n
            down = np.arange(1, s + 1, dtype=float) / n
            nxt[1:, :, :] += current[:-1, :, :] * up[:, None, None]
            nxt[:-1, :, :] += current[1:, :, :] * down[:, None, None]
        if z:
            up = (z - np.arange(z, dtype=float)) / n
            down = np.arange(1, z + 1, dtype=float) / n
            nxt[:, 1:, :] += current[:, :-1, :] * up[None, :, None]
            nxt[:, :-1, :] += current[:, 1:, :] * down[None, :, None]
        if m:
            up = (m - np.arange(m, dtype=float)) / n
            down = np.arange(1, m + 1, dtype=float) / n
            nxt[:, :, 1:] += current[:, :, :-1] * up[None, None, :]
            nxt[:, :, :-1] += current[:, :, 1:] * down[None, None, :]

        total = float(np.sum(nxt))
        if not np.isclose(total, float(np.sum(current)), atol=1e-12):
            raise RuntimeError(
                f"forward transition lost probability: before={np.sum(current)}, after={total}"
            )
        return nxt

    def _backward_step(self, beta_next: FloatArray) -> FloatArray:
        s = self.groups.target_ones
        z = self.groups.target_zeros
        m = self.groups.rest
        n = self.n_bits
        beta = np.zeros_like(beta_next)

        if s:
            up = (s - np.arange(s, dtype=float)) / n
            down = np.arange(1, s + 1, dtype=float) / n
            beta[:-1, :, :] += beta_next[1:, :, :] * up[:, None, None]
            beta[1:, :, :] += beta_next[:-1, :, :] * down[:, None, None]
        if z:
            up = (z - np.arange(z, dtype=float)) / n
            down = np.arange(1, z + 1, dtype=float) / n
            beta[:, :-1, :] += beta_next[:, 1:, :] * up[None, :, None]
            beta[:, 1:, :] += beta_next[:, :-1, :] * down[None, :, None]
        if m:
            up = (m - np.arange(m, dtype=float)) / n
            down = np.arange(1, m + 1, dtype=float) / n
            beta[:, :, :-1] += beta_next[:, :, 1:] * up[None, None, :]
            beta[:, :, 1:] += beta_next[:, :, :-1] * down[None, None, :]
        return beta

    def _build_bridge(self) -> tuple[FloatArray, float]:
        observer_tick = self.observer.tick
        alpha_history = np.zeros((observer_tick + 1,) + self._shape, dtype=float)
        alpha_history[0, 0, 0, 0] = 1.0
        for tick in range(observer_tick):
            alpha_history[tick + 1] = self._forward_step(alpha_history[tick])

        if self.observer.width == 0:
            event_mask = np.ones(self._shape, dtype=float)
        else:
            event_mask = np.zeros(self._shape, dtype=float)
            event_mask[self.groups.target_ones, 0, :] = 1.0

        evidence = float(np.sum(alpha_history[observer_tick] * event_mask))
        if evidence <= 0.0:
            raise ValueError(
                "observer record has zero probability at the selected tick; "
                "increase observer_tick or choose a reachable record"
            )

        conditional = np.zeros((self.max_tick + 1,) + self._shape, dtype=float)
        beta = event_mask
        for tick in range(observer_tick, -1, -1):
            posterior = alpha_history[tick] * beta / evidence
            posterior /= np.sum(posterior)
            conditional[tick] = posterior
            if tick:
                beta = self._backward_step(beta)

        posterior = conditional[observer_tick]
        for tick in range(observer_tick + 1, self.max_tick + 1):
            posterior = self._forward_step(posterior)
            posterior /= np.sum(posterior)
            conditional[tick] = posterior

        return conditional, evidence

    def frame(self, tick: int) -> ConditionalFrame:
        resolved = int(tick)
        if resolved < 0 or resolved > self.max_tick:
            raise ValueError(f"tick must lie in [0, {self.max_tick}]")
        return ConditionalFrame(
            tick=resolved,
            probability_counts=self._conditional_history[resolved],
            groups=self.groups,
            phase_codec=self.phase_codec,
            max_tick=self.max_tick,
            _log2_combinations=self._log2_combinations,
        )

    def frames(self, ticks: Iterable[int]) -> list[ConditionalFrame]:
        return [self.frame(int(tick)) for tick in ticks]


@dataclass(frozen=True)
class MatterLevelSpec:
    width: int
    ones: int
    label: str | None = None

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("matter width must be positive")
        if self.ones < 0 or self.ones > self.width:
            raise ValueError("matter ones count must be in [0, width]")

    @property
    def zeros(self) -> int:
        return self.width - self.ones

    @property
    def resolved_label(self) -> str:
        return self.label or f"w={self.width}, a={self.ones}, b={self.zeros}"


@dataclass
class QuantumLevelSeries:
    spec: MatterLevelSpec
    match_probability: FloatArray
    survival_probability: float
    persistent_probability: FloatArray
    matter_bits: FloatArray


@dataclass
class QuantumHistorySimulationResult:
    n_bits: int
    max_tick: int
    ticks: IntArray
    observer: ObserverRecord
    observer_evidence: float
    born_entropy_bits: FloatArray
    mean_one_fraction: FloatArray
    phase_residual_rms: FloatArray
    codec_description_bits: float
    levels: list[QuantumLevelSeries]
    total_matter_bits: FloatArray
    size_measure: FloatArray
    total_weight_snapshots: dict[int, FloatArray]


def default_composition(width: int) -> tuple[int, int]:
    """Match the current Dicke backend's explicitly flagged default."""

    if width <= 0:
        raise ValueError("width must be positive")
    ones = max(1, width // 3)
    zeros = width - ones
    if ones >= zeros:
        ones, zeros = 1, width - 1
    return ones, zeros


def literal_survival_probability(n_bits: int, width: int) -> float:
    """Probability that none of a structure's bits is selected for width ticks."""

    if width > n_bits:
        return 0.0
    return float(((n_bits - width) / n_bits) ** width)


def run_quantum_history_simulation(
    *,
    n_bits: int,
    max_tick: int,
    steps: int,
    observer: ObserverRecord | None,
    levels: Sequence[MatterLevelSpec],
    phase_config: SpectralPhaseCodecConfig,
    matter_power: float = 1.0,
    hierarchy: Literal["parallel", "cascade"] = "parallel",
    survival: Literal["literal", "none"] = "literal",
    snapshot_ticks: Sequence[int] | None = None,
) -> QuantumHistorySimulationResult:
    """Run the observer-conditioned complex-wavefunction backend."""

    if steps <= 1:
        raise ValueError("steps must be greater than one")
    if matter_power < 0:
        raise ValueError("matter_power must be non-negative")
    if hierarchy not in ("parallel", "cascade"):
        raise ValueError("hierarchy must be 'parallel' or 'cascade'")

    codec = SpectralPhaseCodec(n_bits, phase_config)
    history = ObserverConditionedEhrenfestHistory(n_bits, max_tick, observer, codec)
    ticks = np.unique(np.rint(np.linspace(0, max_tick, steps)).astype(np.int64))
    frame_cache = {int(tick): history.frame(int(tick)) for tick in ticks}

    born_entropy = np.array([frame_cache[int(t)].born_entropy_bits for t in ticks], dtype=float)
    mean_one_fraction = np.array(
        [frame_cache[int(t)].mean_one_fraction for t in ticks], dtype=float
    )
    phase_rms = np.array(
        [codec.residual_rms(int(t) / max(max_tick, 1)) for t in ticks], dtype=float
    )

    # The order parameter is defined by the observer-conditioned mean one fraction.
    eta = np.clip(1.0 - 2.0 * mean_one_fraction, 0.0, 1.0)

    rest_distributions = np.stack(
        [frame_cache[int(t)].rest_weight_distribution() for t in ticks], axis=0
    )
    rest_counts = np.arange(history.groups.rest + 1, dtype=np.int64)

    level_series: list[QuantumLevelSeries] = []
    cumulative = np.ones(len(ticks), dtype=float)
    for spec in levels:
        if spec.width > history.groups.rest:
            raise ValueError(
                f"matter width {spec.width} exceeds the non-observer bulk size "
                f"{history.groups.rest}"
            )
        window_pmf = hypergeom_pmf(
            history.groups.rest, rest_counts, spec.width, spec.ones
        )
        match = rest_distributions @ window_pmf
        survive = literal_survival_probability(n_bits, spec.width) if survival == "literal" else 1.0
        if hierarchy == "cascade":
            cumulative = cumulative * match * survive
            persistent = cumulative.copy()
        else:
            persistent = match * survive

        # Non-overlapping bulk windows.  This keeps the bit budget finite and
        # avoids counting the same fabric bit in every sliding window.
        windows = history.groups.rest / spec.width
        raw_matter_bits = windows * spec.width * persistent
        matter_bits = raw_matter_bits * eta**matter_power
        level_series.append(
            QuantumLevelSeries(
                spec=spec,
                match_probability=match,
                survival_probability=survive,
                persistent_probability=persistent,
                matter_bits=matter_bits,
            )
        )

    total_matter = (
        np.sum([level.matter_bits for level in level_series], axis=0)
        if level_series
        else np.zeros(len(ticks), dtype=float)
    )
    size_measure = np.clip((born_entropy - total_matter) / n_bits, 0.0, None)

    if snapshot_ticks is None:
        resolved_snapshots = sorted(
            {
                0,
                min(history.observer.tick, max_tick),
                max_tick // 2,
                max_tick,
            }
        )
    else:
        resolved_snapshots = sorted({int(np.clip(value, 0, max_tick)) for value in snapshot_ticks})
    snapshots = {
        tick: history.frame(tick).total_weight_distribution() for tick in resolved_snapshots
    }

    return QuantumHistorySimulationResult(
        n_bits=n_bits,
        max_tick=max_tick,
        ticks=ticks,
        observer=history.observer,
        observer_evidence=history.observer_evidence,
        born_entropy_bits=born_entropy,
        mean_one_fraction=mean_one_fraction,
        phase_residual_rms=phase_rms,
        codec_description_bits=codec.description_length_proxy_bits,
        levels=level_series,
        total_matter_bits=total_matter,
        size_measure=size_measure,
        total_weight_snapshots=snapshots,
    )


def parse_compositions(
    scales: Sequence[int],
    raw: str | None,
) -> list[MatterLevelSpec]:
    """Parse the same `a:b,a:b` convention used by cosmic_psi.py."""

    widths = [int(value) for value in scales]
    if raw is None:
        compositions = [default_composition(width) for width in widths]
    else:
        pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
        if len(pieces) != len(widths):
            raise ValueError("number of compositions must match number of scales")
        compositions = []
        for piece in pieces:
            left, right = piece.split(":", maxsplit=1)
            compositions.append((int(left), int(right)))

    levels: list[MatterLevelSpec] = []
    for width, (ones, zeros) in zip(widths, compositions):
        if ones + zeros != width:
            raise ValueError(
                f"composition {ones}:{zeros} does not match width {width}"
            )
        levels.append(MatterLevelSpec(width=width, ones=ones))
    return levels
