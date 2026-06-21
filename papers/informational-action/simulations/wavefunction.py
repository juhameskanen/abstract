"""
Wavefunction class with Spectral Complexity measure.

Based on: "The Wavefunction as Compression: Spectral Complexity, Emergent
Quantum Behaviour, and the Informational Action Principle" (Meskanen, 2026)
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Spectral mode data container
# ---------------------------------------------------------------------------

@dataclass
class SpectralMode:
    """A single mode in the spectral decomposition of a wavefunction."""
    frequency: float
    amplitude: float
    phase: float

    def __post_init__(self):
        self.amplitude = abs(self.amplitude)
        self.phase = self.phase % (2 * np.pi)


# ---------------------------------------------------------------------------
# Main wavefunction class
# ---------------------------------------------------------------------------

class Wavefunction:
    """A discrete, complex-valued wavefunction on a 1-D spatial grid.

    Calculates the Spectral Complexity C_s using a global fidelity compression engine.
    """

    def __init__(
        self,
        psi: np.ndarray,
        dx: float = 1.0,
        phase_bits: int = 8,
        amplitude_bits: int = 8,
        delta_omega: Optional[float] = None,
        c_base: float = 16.0,
        fidelity_target: float = 0.999  # Captures 99.9% of the wavefunction's total power
    ):
        self.dx = float(dx)
        # Always normalize incoming psi to ensure stable, scale-invariant compression metrics
        self._psi = np.asarray(psi, dtype=complex)
        self._psi = self._psi / np.sqrt(np.sum(np.abs(self._psi) ** 2))
        
        self.phase_bits = int(phase_bits)
        self.amplitude_bits = int(amplitude_bits)
        self.c_base = float(c_base)
        self.fidelity_target = float(fidelity_target)
        
        N = len(self._psi)
        self.delta_omega = float(delta_omega) if delta_omega is not None else 2.0 * np.pi / (N * self.dx)
        self.hbar_identified = self.delta_omega / np.log(2.0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def psi(self) -> np.ndarray:
        return self._psi

    @property
    def N(self) -> int:
        return len(self._psi)

    @property
    def x(self) -> np.ndarray:
        return np.arange(self.N) * self.dx

    @property
    def probability_density(self) -> np.ndarray:
        return np.abs(self._psi) ** 2

    # ------------------------------------------------------------------
    # Spectral decomposition & Compression Engine
    # ------------------------------------------------------------------

    def compute_compressed_modes(self) -> list[SpectralMode]:
        """
        Compresses the wavefunction by sorting Fourier coefficients by power
        and keeping only the dominant modes required to hit the fidelity target.
        """
        N = self.N
        fft_coeffs = np.fft.fft(self._psi)
        freqs = 2.0 * np.pi * np.fft.fftfreq(N, d=self.dx)
        
        power = np.abs(fft_coeffs) ** 2
        total_power = np.sum(power)
        
        sorted_indices = np.argsort(power)[::-1]
        
        accumulated_power = 0.0
        compressed_indices = []
        
        for idx in sorted_indices:
            accumulated_power += power[idx]
            compressed_indices.append(idx)
            if (accumulated_power / total_power) >= self.fidelity_target:
                break
                
        modes = []
        for idx in compressed_indices:
            amp = np.abs(fft_coeffs[idx]) / N
            phase = np.angle(fft_coeffs[idx]) % (2.0 * np.pi)
            modes.append(SpectralMode(frequency=freqs[idx], amplitude=amp, phase=phase))
            
        modes.sort(key=lambda m: abs(m.frequency))
        return modes

    def spectral_modes(self) -> list[SpectralMode]:
        """Helper to expose the underlying compressed modes for backward compatibility."""
        return self.compute_compressed_modes()

    def spectral_complexity(self, verbose: bool = False) -> float:
        """Compute the Spectral Complexity C_s(Ψ) of the wavefunction."""
        modes = self.compute_compressed_modes()
        if not modes:
            return self.c_base
            
        total_cost = self.c_base
        
        if verbose:
            print(f"{'Mode':>5}  {'ω':>10}  {'A':>8}  {'φ/2π':>6}  "
                  f"{'C(φ)':>6}  {'C(A)':>6}  {'ω/Δω':>10}  {'mode cost':>10}")
            print("-" * 72)

        for i, mode in enumerate(modes):
            c_amp = float(self.amplitude_bits)
            c_freq = abs(mode.frequency) / self.delta_omega
            c_phase = float(self.phase_bits) if i > 0 else 0.0
            
            mode_cost = c_amp + c_freq + c_phase
            total_cost += mode_cost

            if verbose:
                print(f"{i:>5}  {mode.frequency:>10.4f}  {mode.amplitude:>8.4f}  "
                      f"{mode.phase / (2*np.pi):>6.3f}  "
                      f"{c_phase:>6.1f}  {c_amp:>6.1f}  {c_freq:>10.2f}  {mode_cost:>10.2f}")
            
        if verbose:
            print("-" * 72)
            print(f"  C_base = {self.c_base:.2f} bits")
            print(f"  C_s    = {total_cost:.4f} bits ({len(modes)} modes tracking fidelity)")

        return total_cost

    def solomonoff_weight(self) -> float:
        return 2.0 ** (-self.spectral_complexity())

    def boltzmann_mode_probabilities(self) -> dict[float, float]:
        modes = self.compute_compressed_modes()
        if not modes:
            return {}

        raw = {m.frequency: 2.0 ** (-abs(m.frequency) / self.delta_omega) for m in modes}
        Z = sum(raw.values())
        return {omega: w / Z for omega, w in raw.items()}

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def gaussian_packet(cls, N: int = 256, x0: float = 0.0, sigma: float = 1.0, k0: float = 1.0, dx: float = 0.1, **kwargs) -> Wavefunction:
        x = np.arange(N) * dx
        psi = np.exp(-((x - x0) ** 2) / (4 * sigma ** 2)) * np.exp(1j * k0 * x)
        return cls(psi, dx=dx, **kwargs)

    @classmethod
    def superposition(cls, N: int = 256, amplitudes: list[float] = (0.6, 0.8), wavenumbers: list[float] = (1.0, 3.0), phases: list[float] = (0.0, 0.0), dx: float = 0.1, **kwargs) -> Wavefunction:
        x = np.arange(N) * dx
        psi = np.zeros(N, dtype=complex)
        for a, k, phi in zip(amplitudes, wavenumbers, phases):
            psi += a * np.exp(1j * (k * x + phi))
        return cls(psi, dx=dx, **kwargs)

    @classmethod
    def random_state(cls, N: int = 256, seed: int = 42, dx: float = 0.1, **kwargs) -> Wavefunction:
        rng = np.random.default_rng(seed)
        psi = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        return cls(psi, dx=dx, **kwargs)

    # ------------------------------------------------------------------
    # Arithmetic (Fixed to pass fidelity_target cleanly)
    # ------------------------------------------------------------------

    def __add__(self, other: Wavefunction) -> Wavefunction:
        if self.N != other.N:
            raise ValueError("Wavefunctions must have the same grid size.")
        return Wavefunction(self._psi + other._psi, dx=self.dx,
                            phase_bits=self.phase_bits,
                            amplitude_bits=self.amplitude_bits,
                            delta_omega=self.delta_omega,
                            c_base=self.c_base,
                            fidelity_target=self.fidelity_target)

    def __mul__(self, scalar: complex) -> Wavefunction:
        return Wavefunction(self._psi * scalar, dx=self.dx,
                            phase_bits=self.phase_bits,
                            amplitude_bits=self.amplitude_bits,
                            delta_omega=self.delta_omega,
                            c_base=self.c_base,
                            fidelity_target=self.fidelity_target)

    def __rmul__(self, scalar: complex) -> Wavefunction:
        return self.__mul__(scalar)

    def inner_product(self, other: Wavefunction) -> complex:
        return np.sum(self._psi.conj() * other._psi) * self.dx

    def __repr__(self) -> str:
        cs = self.spectral_complexity()
        return f"Wavefunction(N={self.N}, dx={self.dx}, C_s={cs:.2f} bits, ℏ_id={self.hbar_identified:.4e})"


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Spectral Complexity Codec Demo (Meskanen 2026)")
    print("=" * 60)

    gp = Wavefunction.gaussian_packet(N=256, x0=12.8, sigma=2.0, k0=2.0, dx=0.1)
    print("\n[1] Gaussian wave packet")
    gp.spectral_complexity(verbose=True)

    sp = Wavefunction.superposition(N=256, amplitudes=[0.6, 0.8], wavenumbers=[1.0, 5.0], phases=[0.0, np.pi / 4], dx=0.1)
    print("\n[2] Two-component superposition")
    sp.spectral_complexity(verbose=True)
