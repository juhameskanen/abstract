"""
Wavefunction
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class SpectralMode:
    frequency: float
    amplitude: float
    phase: float

    def __post_init__(self) -> None:
        self.amplitude = float(abs(self.amplitude))
        self.phase = float(self.phase % (2.0 * np.pi))


class Wavefunction:
    def __init__(self, psi, dx=1.0, delta_omega=None, fidelity_target=0.999, phase_resolution=1.0):
        self._psi = self._normalise(np.asarray(psi, dtype=complex))
        self.dx = float(dx)
        self.fidelity_target = float(fidelity_target)
        self.phase_resolution = float(phase_resolution)
        N = len(self._psi)
        self.delta_omega = float(delta_omega) if delta_omega is not None else 2.0 * np.pi / (N * self.dx)
        self.hbar_identified = self.delta_omega / np.log(2.0)

    @property
    def psi(self): return self._psi
    @property
    def N(self): return len(self._psi)
    @property
    def x(self): return np.arange(self.N) * self.dx
    @property
    def probability_density(self): return np.abs(self._psi) ** 2

    def retained_modes(self):
        N = self.N
        fft_coeffs = np.fft.fft(self._psi)
        freqs = 2.0 * np.pi * np.fft.fftfreq(N, d=self.dx)
        power = np.abs(fft_coeffs) ** 2
        total_power = float(power.sum())
        if total_power == 0.0: return []
        sorted_idx = np.argsort(power)[::-1]
        accumulated = 0.0
        kept = []
        for idx in sorted_idx:
            accumulated += float(power[idx])
            kept.append(int(idx))
            if accumulated / total_power >= self.fidelity_target: break
        modes = [SpectralMode(frequency=float(freqs[k]), amplitude=float(np.abs(fft_coeffs[k]))/N, phase=float(np.angle(fft_coeffs[k]) % (2.0*np.pi))) for k in kept]
        modes.sort(key=lambda m: abs(m.frequency))
        return modes

    def spectral_complexity(self, verbose=False):
        modes = self.retained_modes()
        if not modes: return 0.0
        ref_amplitude = max(m.amplitude for m in modes)
        ref_freq_abs = min(abs(m.frequency) for m in modes if m.amplitude == ref_amplitude)
        total = 0.0
        for i, mode in enumerate(modes):
            freq_cost = abs(mode.frequency) / self.delta_omega
            is_reference = (mode.amplitude == ref_amplitude and abs(mode.frequency) == ref_freq_abs)
            phase_cost = 0.0 if is_reference else self.phase_resolution
            total += freq_cost + phase_cost
        return total

    def solomonoff_weight(self): return 2.0 ** (-self.spectral_complexity())

    def __add__(self, other):
        if self.N != other.N: raise ValueError(f"Grid size mismatch: {self.N} vs {other.N}")
        return Wavefunction(self._psi + other._psi, dx=self.dx, delta_omega=self.delta_omega, fidelity_target=self.fidelity_target, phase_resolution=self.phase_resolution)

    def __mul__(self, scalar):
        return Wavefunction(self._psi * scalar, dx=self.dx, delta_omega=self.delta_omega, fidelity_target=self.fidelity_target, phase_resolution=self.phase_resolution)

    def __rmul__(self, scalar): return self.__mul__(scalar)

    def inner_product(self, other):
        return complex(np.sum(self._psi.conj() * other._psi) * self.dx)

    @staticmethod
    def _normalise(psi):
        norm = float(np.sqrt(np.sum(np.abs(psi) ** 2)))
        if norm == 0.0: raise ValueError("Cannot normalise a zero wavefunction.")
        return psi / norm

    def __repr__(self):
        cs = self.spectral_complexity()
        return f"Wavefunction(N={self.N}, dx={self.dx}, C_s={cs:.3f}, modes={len(self.retained_modes())})"

