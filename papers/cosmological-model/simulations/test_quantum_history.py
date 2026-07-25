from __future__ import annotations

import unittest

import numpy as np

from quantum_history import (
    MatterLevelSpec,
    ObserverConditionedEhrenfestHistory,
    ObserverRecord,
    SpectralPhaseCodec,
    SpectralPhaseCodecConfig,
    exact_ehrenfest_weight_distributions,
    run_quantum_history_simulation,
)


class QuantumHistoryTests(unittest.TestCase):
    def test_identity_observer_matches_exact_ehrenfest_weight_chain(self) -> None:
        n_bits = 7
        max_tick = 10
        history = ObserverConditionedEhrenfestHistory(
            n_bits,
            max_tick,
            ObserverRecord.none(),
        )
        expected = exact_ehrenfest_weight_distributions(n_bits, max_tick)
        for tick in range(max_tick + 1):
            actual = history.frame(tick).total_weight_distribution()
            np.testing.assert_allclose(actual, expected[tick], atol=1e-12, rtol=1e-12)

    def test_observer_record_is_certain_at_observer_tick(self) -> None:
        observer = ObserverRecord.contiguous("1010", tick=7)
        history = ObserverConditionedEhrenfestHistory(8, 12, observer)
        frame = history.frame(observer.tick)
        probability = frame.probability_counts
        self.assertAlmostEqual(float(np.sum(probability[-1, 0, :])), 1.0, places=12)
        self.assertAlmostEqual(float(np.sum(probability[:-1, :, :])), 0.0, places=12)
        self.assertAlmostEqual(float(np.sum(probability[:, 1:, :])), 0.0, places=12)


    def test_conditioned_bridge_matches_bruteforce_paths(self) -> None:
        n_bits = 3
        max_tick = 4
        observer = ObserverRecord.contiguous("10", tick=2)
        history = ObserverConditionedEhrenfestHistory(n_bits, max_tick, observer)

        # Enumerate all n**max_tick flip-label histories with equal probability.
        paths = []
        for labels in np.ndindex(*(n_bits for _ in range(max_tick))):
            state = np.zeros(n_bits, dtype=np.int8)
            frames = [state.copy()]
            for label in labels:
                state = state.copy()
                state[label] ^= 1
                frames.append(state)
            if tuple(frames[observer.tick][:2]) == observer.pattern:
                paths.append(frames)
        self.assertGreater(len(paths), 0)

        one_positions = np.asarray(observer.one_indices, dtype=int)
        zero_positions = np.asarray(observer.zero_indices, dtype=int)
        rest_positions = np.asarray([2], dtype=int)
        for tick in range(max_tick + 1):
            brute = np.zeros((2, 2, 2), dtype=float)
            for frames in paths:
                state = frames[tick]
                a = int(np.sum(state[one_positions]))
                b = int(np.sum(state[zero_positions]))
                l = int(np.sum(state[rest_positions]))
                brute[a, b, l] += 1.0
            brute /= np.sum(brute)
            np.testing.assert_allclose(
                history.frame(tick).probability_counts,
                brute,
                atol=1e-12,
                rtol=1e-12,
            )

    def test_exact_statevector_born_probabilities_match_frame(self) -> None:
        n_bits = 6
        observer = ObserverRecord.contiguous("10", tick=4)
        codec = SpectralPhaseCodec(
            n_bits,
            SpectralPhaseCodecConfig(
                temporal_modes=2,
                spatial_modes=2,
                phase_strength=0.8,
                topology="ring",
                seed=3,
            ),
        )
        history = ObserverConditionedEhrenfestHistory(n_bits, 8, observer, codec)
        frame = history.frame(3)
        psi = frame.exact_statevector(max_qubits=10)
        probabilities = np.abs(psi) ** 2

        basis = np.arange(1 << n_bits, dtype=np.uint64)
        positions = np.arange(n_bits, dtype=np.uint64)
        bits = ((basis[:, None] >> positions[None, :]) & 1).astype(bool)
        expected = frame.configuration_probability(bits)
        np.testing.assert_allclose(probabilities, expected, atol=1e-12, rtol=1e-12)
        self.assertAlmostEqual(float(np.sum(probabilities)), 1.0, places=12)

    def test_phase_codec_changes_phase_not_pixel_probabilities(self) -> None:
        n_bits = 6
        observer = ObserverRecord.contiguous("10", tick=4)
        zero_codec = SpectralPhaseCodec(
            n_bits,
            SpectralPhaseCodecConfig(topology="none", phase_strength=0.0),
        )
        phase_codec = SpectralPhaseCodec(
            n_bits,
            SpectralPhaseCodecConfig(
                temporal_modes=3,
                spatial_modes=2,
                phase_strength=1.1,
                topology="ring",
                seed=11,
            ),
        )
        frame_zero = ObserverConditionedEhrenfestHistory(
            n_bits, 8, observer, zero_codec
        ).frame(5)
        frame_phase = ObserverConditionedEhrenfestHistory(
            n_bits, 8, observer, phase_codec
        ).frame(5)
        psi_zero = frame_zero.exact_statevector(max_qubits=10)
        psi_phase = frame_phase.exact_statevector(max_qubits=10)
        np.testing.assert_allclose(
            np.abs(psi_zero) ** 2,
            np.abs(psi_phase) ** 2,
            atol=1e-12,
            rtol=1e-12,
        )
        self.assertGreater(float(np.max(np.abs(psi_zero - psi_phase))), 1e-6)

    def test_sampling_matches_analytic_mean(self) -> None:
        n_bits = 10
        observer = ObserverRecord.contiguous("101", tick=8)
        history = ObserverConditionedEhrenfestHistory(n_bits, 12, observer)
        frame = history.frame(8)
        samples = frame.sample_bitstrings(20000, np.random.default_rng(123))
        sampled = float(np.mean(samples))
        self.assertLess(abs(sampled - frame.mean_one_fraction), 0.01)

    def test_end_to_end_simulation_is_finite(self) -> None:
        result = run_quantum_history_simulation(
            n_bits=12,
            max_tick=24,
            steps=25,
            observer=ObserverRecord.gaussian_blob(5, tick=12),
            levels=[MatterLevelSpec(3, 1), MatterLevelSpec(4, 1)],
            phase_config=SpectralPhaseCodecConfig(
                temporal_modes=2,
                spatial_modes=2,
                phase_strength=0.5,
                topology="ring",
            ),
            matter_power=1.0,
            hierarchy="parallel",
            survival="literal",
        )
        self.assertTrue(np.all(np.isfinite(result.born_entropy_bits)))
        self.assertTrue(np.all(np.isfinite(result.size_measure)))
        self.assertTrue(np.all(result.size_measure >= 0.0))
        self.assertGreater(result.observer_evidence, 0.0)


if __name__ == "__main__":
    unittest.main()
