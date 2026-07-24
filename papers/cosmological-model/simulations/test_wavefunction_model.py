from __future__ import annotations

import unittest

import numpy as np

from wavefunction_model import (
    CompressedComplexWavefunction,
    PhaseCodecConfig,
    SpectralPhaseCodec,
    binary_entropy,
    run_wavefunction_cosmology,
    window_family_fractions,
)


class WavefunctionModelTests(unittest.TestCase):
    def test_born_distribution_is_exact_statistical_shadow(self) -> None:
        n = 8
        config = PhaseCodecConfig(strength=1.1, spatial_modes=3, temporal_modes=2, pair_range=1)
        codec = SpectralPhaseCodec(n, config)
        state = CompressedComplexWavefunction(n, tau_raw=4.7, tau_max=20.0, codec=codec)
        psi = state.statevector(max_bits=10)
        probs = np.abs(psi) ** 2
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=12)

        indices = np.arange(1 << n, dtype=np.uint64)
        bits = ((indices[:, None] >> np.arange(n, dtype=np.uint64)[None, :]) & 1).astype(int)
        expected = np.asarray(state.born_probability(bits))
        np.testing.assert_allclose(probs, expected, rtol=1e-12, atol=1e-14)

        entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
        self.assertAlmostEqual(float(entropy), n * float(binary_entropy(state.p)), places=10)

    def test_state_is_not_restricted_to_a_dicke_sector(self) -> None:
        n = 8
        config = PhaseCodecConfig(
            strength=1.0, spatial_modes=4, temporal_modes=3, pair_range=1, seed=2
        )
        codec = SpectralPhaseCodec(n, config)
        state = CompressedComplexWavefunction(n, tau_raw=6.0, tau_max=20.0, codec=codec)
        zero = np.zeros(n, dtype=int)
        one_a = zero.copy(); one_a[0] = 1
        one_b = zero.copy(); one_b[3] = 1
        two = zero.copy(); two[:2] = 1
        self.assertGreater(float(state.born_probability(zero)), 0.0)
        self.assertGreater(float(state.born_probability(one_a)), 0.0)
        self.assertGreater(float(state.born_probability(two)), 0.0)
        self.assertAlmostEqual(
            float(state.born_probability(one_a)), float(state.born_probability(one_b)), places=14
        )
        self.assertGreater(abs(complex(state.amplitude(one_a)) - complex(state.amplitude(one_b))), 1e-8)

    def test_phase_changes_state_but_not_cosmology(self) -> None:
        plain = run_wavefunction_cosmology(
            n_bits=64,
            scales=(4, 8, 12),
            steps=500,
            phase_config=PhaseCodecConfig(strength=0.0, pair_range=0),
        )
        complex_state = run_wavefunction_cosmology(
            n_bits=64,
            scales=(4, 8, 12),
            steps=500,
            phase_config=PhaseCodecConfig(strength=1.2, spatial_modes=4, temporal_modes=3, pair_range=2),
        )
        np.testing.assert_allclose(plain.size_measure, complex_state.size_measure, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(plain.total_matter_bits, complex_state.total_matter_bits, atol=0.0, rtol=0.0)
        self.assertLess(float(np.max(plain.qubit_entanglement)), 1e-12)
        self.assertGreater(float(np.max(complex_state.qubit_entanglement)), 1e-4)

    def test_analytic_entanglement_matches_small_statevector(self) -> None:
        n = 6
        config = PhaseCodecConfig(
            strength=0.8,
            spatial_modes=3,
            temporal_modes=3,
            pair_range=1,
            topology="line",
        )
        codec = SpectralPhaseCodec(n, config)
        state = CompressedComplexWavefunction(n, tau_raw=7.0, tau_max=20.0, codec=codec)
        psi = state.statevector(max_bits=10)

        # Axis -1 is the least-significant bit (qubit 0) in C-order reshape.
        tensor = psi.reshape([2] * n)
        matrix = np.moveaxis(tensor, -1, 0).reshape(2, -1)
        rho = matrix @ matrix.conj().T
        eigs = np.linalg.eigvalsh(rho)
        eigs = eigs[eigs > 1e-15]
        exact = float(-np.sum(eigs * np.log2(eigs)))
        analytic = codec.single_qubit_entanglement_entropy(
            state.p, state.tau_raw, state.tau_max, qubit=0
        )
        self.assertAlmostEqual(exact, analytic, places=10)

    def test_ring_entanglement_matches_small_statevector(self) -> None:
        n = 6
        config = PhaseCodecConfig(
            strength=0.65, spatial_modes=2, temporal_modes=2, pair_range=1, topology="ring"
        )
        codec = SpectralPhaseCodec(n, config)
        state = CompressedComplexWavefunction(n, tau_raw=5.0, tau_max=17.0, codec=codec)
        psi = state.statevector(max_bits=10)
        tensor = psi.reshape([2] * n)
        matrix = np.moveaxis(tensor, -1, 0).reshape(2, -1)
        rho = matrix @ matrix.conj().T
        eigs = np.linalg.eigvalsh(rho)
        eigs = eigs[eigs > 1e-15]
        exact = float(-np.sum(eigs * np.log2(eigs)))
        analytic = codec.single_qubit_entanglement_entropy(
            state.p, state.tau_raw, state.tau_max, qubit=0
        )
        self.assertAlmostEqual(exact, analytic, places=10)

    def test_hump_family_is_interior_and_partitions_probability(self) -> None:
        p = np.linspace(0.0, 0.5, 1001)
        for width in (6, 12, 20):
            fall, bump, rise = window_family_fractions(p, width)
            np.testing.assert_allclose(fall + bump + rise, 1.0, atol=2e-14)
            peak = int(np.argmax(bump))
            self.assertGreater(peak, 0)
            self.assertLess(peak, len(p) - 1)
            self.assertGreater(float(bump[peak]), float(bump[0]))
            self.assertGreater(float(bump[peak]), float(bump[-1]))

    def test_default_run_has_matter_driven_three_stage_profile(self) -> None:
        result = run_wavefunction_cosmology(
            n_bits=184,
            scales=(6, 12, 20),
            steps=1400,
            matter_power=1.0,
            phase_config=PhaseCodecConfig(),
        )
        d = result.diagnostics
        self.assertTrue(d.three_stage_detected)
        self.assertGreater(d.early_growth_rate, d.matter_loading_rate)
        self.assertGreater(d.slowdown_strength, 0.0)
        self.assertGreater(d.recovery_rate, 0.0)
        self.assertGreater(d.recovery_hubble_proxy, 0.0)
        self.assertLess(d.end_suppression, 0.1 * d.peak_suppression)
        peak_idx = int(np.argmax(result.total_matter_bits))
        self.assertGreater(result.size_measure[-1], result.size_measure[peak_idx])
        self.assertLess(result.conservation_max_error, 1e-9)

    def test_born_sampling_matches_probability(self) -> None:
        n = 32
        codec = SpectralPhaseCodec(n, PhaseCodecConfig(strength=0.7, pair_range=1))
        state = CompressedComplexWavefunction(n, tau_raw=18.0, tau_max=100.0, codec=codec)
        samples = state.sample(20000, np.random.default_rng(12345))
        observed = float(np.mean(samples))
        standard_error = np.sqrt(state.p * (1.0 - state.p) / samples.size)
        self.assertLess(abs(observed - state.p), 5.0 * standard_error)


if __name__ == "__main__":
    unittest.main()
