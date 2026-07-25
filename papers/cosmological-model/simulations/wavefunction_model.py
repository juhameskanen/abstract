"""General complex-wavefunction backend for the cosmological toy model.

The state is represented without allocating a 2**n state vector::

    psi_tau(x) = sqrt(P_tau(x)) * exp(i Phi_tau(x))

where ``P_tau`` is the independent Bernoulli statistical shadow and
``Phi_tau`` is a short, position-dependent spectral phase codec.  The phase
contains pair terms, so the state can be entangled and need not be
permutation-symmetric, while a fabric-basis Born measurement still produces
exactly the intended statistical bitstring distribution.

The cosmological ledger is the quantum analogue of ``multiclock.py``:

* Born (screen) entropy supplies unfolded spacetime resolution;
* hump-family window probabilities supply elementary microstructures;
* those structures consume the same finite information budget;
* the resulting size curve displays rapid growth, matter-loading slowdown,
  and late recovery as the hump populations decline.

This module deliberately contains no observer bitmap or observer tick.
Observers are higher-order structures assembled from the same scale hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binom

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]

TRUE_K_RATE: float = 2.0


def binary_entropy(p: FloatArray | float) -> FloatArray | float:
    """Binary Shannon entropy H_2(p), with the endpoints handled exactly."""
    arr = np.asarray(p, dtype=float)
    out = np.zeros_like(arr)
    mask = (arr > 0.0) & (arr < 1.0)
    q = arr[mask]
    out[mask] = -(q * np.log2(q) + (1.0 - q) * np.log2(1.0 - q))
    return float(out) if np.ndim(p) == 0 else out


def relaxation_probability(
    tau_raw: FloatArray | float,
    n_bits: int,
    k_rate: float = TRUE_K_RATE,
) -> FloatArray | float:
    """Continuum Ehrenfest marginal used by the existing statistical model."""
    arr = np.asarray(tau_raw, dtype=float)
    p = 0.5 * (1.0 - np.exp(-k_rate * arr / float(n_bits)))
    return float(p) if np.ndim(tau_raw) == 0 else p


def order_parameter(
    tau_raw: FloatArray | float,
    n_bits: int,
    k_rate: float = TRUE_K_RATE,
) -> FloatArray | float:
    """Distance from equilibrium, eta = 1 - 2p."""
    arr = np.asarray(tau_raw, dtype=float)
    eta = np.exp(-k_rate * arr / float(n_bits))
    return float(eta) if np.ndim(tau_raw) == 0 else eta


def born_entropy_bits(
    tau_raw: FloatArray | float,
    n_bits: int,
    k_rate: float = TRUE_K_RATE,
) -> FloatArray | float:
    """Fabric-basis Born entropy of the compressed product-magnitude state."""
    p = relaxation_probability(tau_raw, n_bits, k_rate)
    value = n_bits * binary_entropy(p)
    return float(value) if np.ndim(tau_raw) == 0 else np.asarray(value, dtype=float)


def local_clock(tau_raw: FloatArray, width: int, mode: str) -> FloatArray:
    """Return the scale-local coordinate.

    ``block`` reproduces the existing multi-clock sample-and-hold convention:
    a width-w structure updates only after another block of w elementary flips.
    ``shared`` disables the backlog and is useful as an ablation.
    """
    if mode == "block":
        return width * np.floor(tau_raw / width)
    if mode == "shared":
        return np.asarray(tau_raw, dtype=float)
    raise ValueError(f"unknown clock mode {mode!r}; expected 'block' or 'shared'")


@dataclass(frozen=True)
class PhaseCodecConfig:
    """Short spectral description of the wavefunction phase residual."""

    strength: float = 0.9
    spatial_modes: int = 3
    temporal_modes: int = 3
    pair_range: int = 1
    seed: int = 0
    topology: str = "ring"

    def __post_init__(self) -> None:
        if self.spatial_modes < 1 or self.temporal_modes < 1:
            raise ValueError("spatial_modes and temporal_modes must be positive")
        if self.pair_range < 0:
            raise ValueError("pair_range must be non-negative")
        if self.topology not in {"ring", "line"}:
            raise ValueError("topology must be 'ring' or 'line'")

    @property
    def term_count(self) -> int:
        """Number of stored spectral coefficients (an MDL proxy, not K)."""
        return self.spatial_modes * self.temporal_modes * (1 + self.pair_range)


class SpectralPhaseCodec:
    """Deterministic DCT-like phase field with diagonal pair interactions."""

    def __init__(self, n_bits: int, config: PhaseCodecConfig) -> None:
        if n_bits < 2:
            raise ValueError("n_bits must be at least 2")
        if config.pair_range >= n_bits:
            raise ValueError("pair_range must be smaller than n_bits")
        self.n_bits = int(n_bits)
        self.config = config
        j = np.arange(self.n_bits, dtype=float)
        self._spatial_basis = np.stack(
            [
                np.cos(np.pi * r * (j + 0.5) / self.n_bits)
                for r in range(config.spatial_modes)
            ],
            axis=0,
        )

    def _coefficient(self, kind: int, r: int, m: int, distance: int = 0) -> float:
        # A short deterministic coefficient rule.  The seed selects another
        # member of the same compact family; no coefficient table is stored.
        phase = (
            (self.config.seed + 1)
            * (kind + 1)
            * (r + 1)
            * (m + 1)
            * (distance + 1)
            * np.sqrt(2.0)
        )
        return float(
            self.config.strength
            * np.sin(phase)
            / ((r + 1) * (m + 1) * (distance + 1))
        )

    def fields(self, tau_raw: float, tau_max: float) -> tuple[FloatArray, FloatArray]:
        """Return one-body ``alpha[j]`` and pair ``beta[d-1,j]`` fields."""
        if tau_max <= 0:
            raise ValueError("tau_max must be positive")
        u = float(np.clip(tau_raw / tau_max, 0.0, 1.0))
        temporal = np.array(
            [np.cos(np.pi * m * u) for m in range(self.config.temporal_modes)],
            dtype=float,
        )
        alpha = np.zeros(self.n_bits, dtype=float)
        beta = np.zeros((self.config.pair_range, self.n_bits), dtype=float)
        for r in range(self.config.spatial_modes):
            basis = self._spatial_basis[r]
            for m in range(self.config.temporal_modes):
                tm = temporal[m]
                alpha += self._coefficient(0, r, m) * tm * basis
                for d in range(1, self.config.pair_range + 1):
                    beta[d - 1] += self._coefficient(1, r, m, d) * tm * basis
        return alpha, beta

    def phase(
        self,
        bitstrings: NDArray[np.integer] | BoolArray,
        tau_raw: float,
        tau_max: float,
    ) -> FloatArray | float:
        """Evaluate Phi_tau(x) for one bitstring or a batch."""
        bits = np.asarray(bitstrings, dtype=float)
        single = bits.ndim == 1
        if single:
            bits = bits[None, :]
        if bits.ndim != 2 or bits.shape[1] != self.n_bits:
            raise ValueError(f"bitstrings must have shape ({self.n_bits},) or (m,{self.n_bits})")
        alpha, beta = self.fields(tau_raw, tau_max)
        phase = bits @ alpha
        for d in range(1, self.config.pair_range + 1):
            coeff = beta[d - 1]
            if self.config.topology == "ring":
                partner = np.roll(bits, -d, axis=1)
                phase += np.sum(bits * partner * coeff[None, :], axis=1)
            else:
                if d < self.n_bits:
                    phase += np.sum(
                        bits[:, :-d] * bits[:, d:] * coeff[None, :-d], axis=1
                    )
        return float(phase[0]) if single else phase

    def field_rms(self, tau_raw: float, tau_max: float) -> float:
        alpha, beta = self.fields(tau_raw, tau_max)
        values = alpha if beta.size == 0 else np.concatenate([alpha, beta.ravel()])
        return float(np.sqrt(np.mean(values * values)))

    def single_qubit_entanglement_entropy(
        self,
        p: float,
        tau_raw: float,
        tau_max: float,
        qubit: int = 0,
    ) -> float:
        """Exact one-qubit entropy for the phase-entangled product-magnitude state.

        Pair phases dephase the selected qubit when the other bits are traced
        out.  This scalable formula is an exact entanglement witness for the
        full n-bit state; no small-state approximation is used.
        """
        if not 0 <= qubit < self.n_bits:
            raise ValueError("qubit index out of range")
        if p <= 0.0 or p >= 1.0 or self.config.pair_range == 0:
            return 0.0
        alpha, beta = self.fields(tau_raw, tau_max)
        coherence = np.sqrt(p * (1.0 - p)) * np.exp(-1j * alpha[qubit])
        for d in range(1, self.config.pair_range + 1):
            # outgoing edge q -> q+d
            out_j = qubit
            out_k = qubit + d
            if self.config.topology == "ring":
                out_k %= self.n_bits
                coherence *= (1.0 - p) + p * np.exp(-1j * beta[d - 1, out_j])
            elif out_k < self.n_bits:
                coherence *= (1.0 - p) + p * np.exp(-1j * beta[d - 1, out_j])

            # incoming edge q-d -> q
            in_j = qubit - d
            if self.config.topology == "ring":
                in_j %= self.n_bits
                # The phase codec stores directed coefficients beta[d,j].
                # Even when incoming and outgoing neighbors coincide (for
                # example n=2, d=1), both directed terms are present in Phi.
                coherence *= (1.0 - p) + p * np.exp(-1j * beta[d - 1, in_j])
            elif in_j >= 0:
                coherence *= (1.0 - p) + p * np.exp(-1j * beta[d - 1, in_j])

        bloch_length = np.sqrt((1.0 - 2.0 * p) ** 2 + 4.0 * abs(coherence) ** 2)
        bloch_length = float(np.clip(bloch_length, 0.0, 1.0))
        eigenvalues = np.array(
            [(1.0 + bloch_length) / 2.0, (1.0 - bloch_length) / 2.0]
        )
        return float(binary_entropy(eigenvalues[0]))


@dataclass
class CompressedComplexWavefunction:
    """Implicit complex state at one internal coordinate."""

    n_bits: int
    tau_raw: float
    tau_max: float
    codec: SpectralPhaseCodec
    k_rate: float = TRUE_K_RATE

    @property
    def p(self) -> float:
        return float(relaxation_probability(self.tau_raw, self.n_bits, self.k_rate))

    @property
    def screen_entropy_bits(self) -> float:
        return float(self.n_bits * binary_entropy(self.p))

    def log_born_probability(self, bitstrings: NDArray[np.integer] | BoolArray) -> FloatArray | float:
        bits = np.asarray(bitstrings, dtype=int)
        single = bits.ndim == 1
        if single:
            bits = bits[None, :]
        if bits.ndim != 2 or bits.shape[1] != self.n_bits:
            raise ValueError(f"bitstrings must have shape ({self.n_bits},) or (m,{self.n_bits})")
        if np.any((bits != 0) & (bits != 1)):
            raise ValueError("bitstrings must contain only 0 and 1")
        ones = np.sum(bits, axis=1)
        p = self.p
        if p == 0.0:
            values = np.where(ones == 0, 0.0, -np.inf)
        elif p == 1.0:
            values = np.where(ones == self.n_bits, 0.0, -np.inf)
        else:
            values = ones * np.log(p) + (self.n_bits - ones) * np.log1p(-p)
        return float(values[0]) if single else values

    def born_probability(self, bitstrings: NDArray[np.integer] | BoolArray) -> FloatArray | float:
        values = np.exp(self.log_born_probability(bitstrings))
        return float(values) if np.ndim(values) == 0 else values

    def amplitude(self, bitstrings: NDArray[np.integer] | BoolArray) -> ComplexArray | complex:
        prob = np.asarray(self.born_probability(bitstrings), dtype=float)
        phase = np.asarray(self.codec.phase(bitstrings, self.tau_raw, self.tau_max), dtype=float)
        amp = np.sqrt(prob) * np.exp(1j * phase)
        return complex(amp) if amp.ndim == 0 else amp.astype(np.complex128)

    def sample(self, shots: int, rng: np.random.Generator | None = None) -> BoolArray:
        if shots <= 0:
            raise ValueError("shots must be positive")
        generator = np.random.default_rng() if rng is None else rng
        return generator.random((shots, self.n_bits)) < self.p

    def statevector(self, max_bits: int = 18) -> ComplexArray:
        """Construct the full state vector for tests or small demonstrations."""
        if self.n_bits > max_bits:
            raise ValueError(
                f"n_bits={self.n_bits} exceeds statevector safety limit {max_bits}"
            )
        indices = np.arange(1 << self.n_bits, dtype=np.uint64)
        shifts = np.arange(self.n_bits, dtype=np.uint64)
        bits = ((indices[:, None] >> shifts[None, :]) & 1).astype(np.int8)
        psi = np.asarray(self.amplitude(bits), dtype=np.complex128)
        norm = np.linalg.norm(psi)
        if not np.isfinite(norm) or norm == 0.0:
            raise FloatingPointError("invalid wavefunction norm")
        return psi / norm


@dataclass
class QuantumScaleResult:
    width: int
    lapse: float
    tau_local: FloatArray
    pool_true: FloatArray
    pool_eff: FloatArray
    f_fall: FloatArray
    f_bump: FloatArray
    f_rise: FloatArray
    fabric: FloatArray
    structure: FloatArray
    promoted: FloatArray
    pending: FloatArray
    matter_bits: FloatArray = field(default_factory=lambda: np.empty(0))

    @property
    def structure_count(self) -> FloatArray:
        return self.structure / self.width

    @property
    def matter_count(self) -> FloatArray:
        return self.matter_bits / self.width


def window_family_fractions(p: FloatArray, width: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Born probabilities of falling, hump, and rising composition families."""
    if width < 2:
        raise ValueError("microstructure widths must be at least 2")
    threshold = int(np.ceil(width / 2.0))
    fall = binom.pmf(0, width, p)
    rise = binom.sf(threshold - 1, width, p)
    bump = np.clip(1.0 - fall - rise, 0.0, 1.0)
    return np.asarray(fall), np.asarray(bump), np.asarray(rise)


def _level_state(
    tau_raw: FloatArray,
    level_idx: int,
    widths: tuple[int, ...],
    n_bits: int,
    k_rate: float,
    clock_mode: str,
    cache: dict[tuple[int, bytes], QuantumScaleResult],
) -> QuantumScaleResult:
    key = (level_idx, np.asarray(tau_raw, dtype=float).tobytes())
    if key in cache:
        return cache[key]

    width = widths[level_idx]
    tau_local = local_clock(tau_raw, width, clock_mode)
    if level_idx == 0:
        pool_true = np.asarray(born_entropy_bits(tau_raw, n_bits, k_rate), dtype=float)
        pool_eff = np.asarray(born_entropy_bits(tau_local, n_bits, k_rate), dtype=float)
    else:
        prev_true = _level_state(
            tau_raw, level_idx - 1, widths, n_bits, k_rate, clock_mode, cache
        )
        prev_eff = _level_state(
            tau_local, level_idx - 1, widths, n_bits, k_rate, clock_mode, cache
        )
        pool_true = prev_true.promoted
        pool_eff = prev_eff.promoted

    p_local = np.asarray(relaxation_probability(tau_local, n_bits, k_rate), dtype=float)
    f_fall, f_bump, f_rise = window_family_fractions(p_local, width)
    result = QuantumScaleResult(
        width=width,
        lapse=1.0 / width,
        tau_local=tau_local,
        pool_true=pool_true,
        pool_eff=pool_eff,
        f_fall=f_fall,
        f_bump=f_bump,
        f_rise=f_rise,
        fabric=pool_eff * f_fall,
        structure=pool_eff * f_bump,
        promoted=pool_eff * f_rise,
        pending=pool_true - pool_eff,
    )
    cache[key] = result
    return result


def build_quantum_hierarchy(
    tau_raw: FloatArray,
    scales: Sequence[int],
    n_bits: int,
    k_rate: float = TRUE_K_RATE,
    clock_mode: str = "block",
) -> list[QuantumScaleResult]:
    widths = tuple(int(w) for w in scales)
    if not widths:
        raise ValueError("at least one scale is required")
    if any(w < 2 or w > n_bits for w in widths):
        raise ValueError("each scale must satisfy 2 <= width <= n_bits")
    cache: dict[tuple[int, bytes], QuantumScaleResult] = {}
    return [
        _level_state(tau_raw, i, widths, n_bits, k_rate, clock_mode, cache)
        for i in range(len(widths))
    ]


@dataclass(frozen=True)
class ProfileDiagnostics:
    matter_peak_time: float
    early_growth_rate: float
    matter_loading_rate: float
    recovery_rate: float
    slowdown_strength: float
    recovery_hubble_proxy: float
    peak_suppression: float
    end_suppression: float
    three_stage_detected: bool


@dataclass
class QuantumCosmologyResult:
    t_bf: FloatArray
    tau_norm: FloatArray
    n_bits: int
    p: FloatArray
    entropy_bits: FloatArray
    entropy_fraction: FloatArray
    levels: list[QuantumScaleResult]
    pending_total: FloatArray
    total_matter_bits: FloatArray
    total_matter_count: FloatArray
    no_matter_size: FloatArray
    size_measure: FloatArray
    size_rate: FloatArray
    hubble_proxy: FloatArray
    phase_field_rms: FloatArray
    qubit_entanglement: FloatArray
    conservation_max_error: float
    codec: PhaseCodecConfig
    diagnostics: ProfileDiagnostics
    matter_power: float
    clock_mode: str


def _safe_median(values: FloatArray, fallback: float = 0.0) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.median(finite)) if finite.size else float(fallback)


def _profile_diagnostics(
    t: FloatArray,
    entropy_fraction: FloatArray,
    no_matter_size: FloatArray,
    size: FloatArray,
    matter_bits: FloatArray,
) -> tuple[FloatArray, FloatArray, ProfileDiagnostics]:
    sigma = max(2.0, len(t) / 300.0)
    smooth_size = gaussian_filter1d(size, sigma=sigma, mode="nearest")
    smooth_base = gaussian_filter1d(no_matter_size, sigma=sigma, mode="nearest")
    rate = np.gradient(smooth_size, t)
    base_rate = np.gradient(smooth_base, t)
    hubble = rate / np.maximum(smooth_size, 1e-9)

    peak_idx = int(np.argmax(matter_bits))
    peak = float(matter_bits[peak_idx])
    indices = np.arange(len(t))
    early_mask = (
        (entropy_fraction > 0.05)
        & (entropy_fraction < 0.30)
        & (indices < peak_idx)
    )
    rise_mask = (
        (indices < peak_idx)
        & (matter_bits > 0.20 * peak)
        & (matter_bits < 0.90 * peak)
    )
    recovery_mask = (
        (indices > peak_idx)
        & (matter_bits < 0.50 * peak)
        & (matter_bits > 0.05 * peak)
    )
    if not np.any(recovery_mask):
        recovery_mask = indices > peak_idx

    early_rate = _safe_median(rate[early_mask], fallback=rate[min(10, len(rate) - 1)])
    loading_rate = _safe_median(rate[rise_mask], fallback=rate[peak_idx])
    recovery_rate = _safe_median(rate[recovery_mask], fallback=rate[-1])
    slowdown = _safe_median(base_rate[rise_mask] - rate[rise_mask], fallback=0.0)
    recovery_h = _safe_median(hubble[recovery_mask], fallback=hubble[-1])
    suppression = no_matter_size - size
    peak_suppression = float(np.max(suppression))
    end_suppression = float(max(suppression[-1], 0.0))
    detected = bool(
        0 < peak_idx < len(t) - 1
        and early_rate > loading_rate
        and slowdown > 0.0
        and recovery_rate > 0.0
        and size[-1] > size[peak_idx]
        and end_suppression < 0.10 * max(peak_suppression, 1e-12)
    )
    diagnostics = ProfileDiagnostics(
        matter_peak_time=float(t[peak_idx]),
        early_growth_rate=early_rate,
        matter_loading_rate=loading_rate,
        recovery_rate=recovery_rate,
        slowdown_strength=slowdown,
        recovery_hubble_proxy=recovery_h,
        peak_suppression=peak_suppression,
        end_suppression=end_suppression,
        three_stage_detected=detected,
    )
    return rate, hubble, diagnostics


def run_wavefunction_cosmology(
    n_bits: int = 184,
    scales: Sequence[int] = (6, 12, 20),
    steps: int = 3000,
    t_bf_max: float | None = None,
    matter_power: float = 1.0,
    clock_mode: str = "block",
    phase_config: PhaseCodecConfig | None = None,
    k_rate: float = TRUE_K_RATE,
) -> QuantumCosmologyResult:
    """Run the third, general-complex-wavefunction backend."""
    n_bits = int(n_bits)
    if n_bits < 2:
        raise ValueError("n_bits must be at least 2")
    if steps < 50:
        raise ValueError("steps must be at least 50")
    if matter_power < 0:
        raise ValueError("matter_power must be non-negative")
    resolved_max = n_bits * np.log(n_bits) if t_bf_max is None else n_bits * float(t_bf_max)
    if resolved_max <= 0:
        raise ValueError("t_bf_max must be positive")

    t = np.linspace(1e-9, resolved_max, steps)
    p = np.asarray(relaxation_probability(t, n_bits, k_rate), dtype=float)
    entropy_bits = np.asarray(n_bits * binary_entropy(p), dtype=float)
    entropy_fraction = entropy_bits / n_bits
    levels = build_quantum_hierarchy(t, scales, n_bits, k_rate, clock_mode)

    pending_total = sum((lvl.pending for lvl in levels), np.zeros_like(t))
    fabric_total = sum((lvl.fabric for lvl in levels), np.zeros_like(t))
    structure_total = sum((lvl.structure for lvl in levels), np.zeros_like(t))
    accounted = pending_total + fabric_total + structure_total + levels[-1].promoted
    conservation_error = float(np.max(np.abs(accounted - entropy_bits)))

    eta = np.asarray(order_parameter(t, n_bits, k_rate), dtype=float)
    weight = eta ** matter_power
    for level in levels:
        level.matter_bits = level.structure * weight
    total_matter_bits = sum((lvl.matter_bits for lvl in levels), np.zeros_like(t))
    total_matter_count = sum((lvl.matter_count for lvl in levels), np.zeros_like(t))

    no_matter_size = (entropy_bits - pending_total) / n_bits
    size_measure = (entropy_bits - pending_total - total_matter_bits) / n_bits

    codec_config = phase_config or PhaseCodecConfig()
    codec = SpectralPhaseCodec(n_bits, codec_config)
    phase_rms = np.array([codec.field_rms(float(tt), resolved_max) for tt in t])
    qubit_ent = np.array(
        [
            codec.single_qubit_entanglement_entropy(
                float(pp), float(tt), resolved_max, qubit=0
            )
            for pp, tt in zip(p, t)
        ],
        dtype=float,
    )

    size_rate, hubble, diagnostics = _profile_diagnostics(
        t, entropy_fraction, no_matter_size, size_measure, total_matter_bits
    )

    return QuantumCosmologyResult(
        t_bf=t,
        tau_norm=t / n_bits,
        n_bits=n_bits,
        p=p,
        entropy_bits=entropy_bits,
        entropy_fraction=entropy_fraction,
        levels=levels,
        pending_total=pending_total,
        total_matter_bits=total_matter_bits,
        total_matter_count=total_matter_count,
        no_matter_size=no_matter_size,
        size_measure=size_measure,
        size_rate=size_rate,
        hubble_proxy=hubble,
        phase_field_rms=phase_rms,
        qubit_entanglement=qubit_ent,
        conservation_max_error=conservation_error,
        codec=codec_config,
        diagnostics=diagnostics,
        matter_power=matter_power,
        clock_mode=clock_mode,
    )
