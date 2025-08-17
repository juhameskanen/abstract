import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time
import math


@njit
def geodesic_rhs_newton(r, v, M):
    """Newtonian-style approximation of dv/dτ for radial infall."""
    if r <= 1e-8:
        return 0.0
    return -M / r**2


@njit
def geodesic_rhs(r, v, M):
    """
    Radial acceleration dv/dτ in Painleve-Gullstrand coordinates for 
    massive test particle falling radially (no angular momentum).
    Inputs:
        r : radius
        v : radial velocity dr/dτ
        M : mass of central object
    Returns:
        acceleration dv/dτ
    """
    if r <= 1e-8:
        return 0.0

    sqrt_2M_r = math.sqrt(2.0 * M / r)
    accel = -M / (r * r) + 1.5 * sqrt_2M_r * v / r
    return accel


@njit
def integrate_particle(r0, v0, M, max_tau, dt, tolerance):
    n_steps = int(max_tau / dt)
    r_values = np.zeros(n_steps)
    tau_values = np.zeros(n_steps)
    r, v = r0, v0
    for i in range(n_steps):
        if r <= tolerance:
            r_values[i:] = 0.0
            tau_values[i:] = tau_values[i - 1]
            break
        r_values[i] = r
        tau_values[i] = i * dt
        a = geodesic_rhs(r, v, M)
        v += a * dt
        r += v * dt
    return tau_values, r_values


@njit(parallel=True)
def simulate_all_particles(r_starts, v_starts, M, max_tau, dt, tolerance):
    num_particles = len(r_starts)
    n_steps = int(max_tau / dt)
    all_r = np.zeros((num_particles, n_steps))
    all_tau = np.zeros((num_particles, n_steps))
    for i in prange(num_particles):
        tau_vals, r_vals = integrate_particle(r_starts[i], v_starts[i], M, max_tau, dt, tolerance)
        all_r[i, :] = r_vals
        all_tau[i, :] = tau_vals
    return all_tau, all_r


def compute_bit_entropy(positions: np.ndarray, scale: float = 1000.0) -> float:
    """
    Convert positions to bitstring and compute Shannon entropy over bits.
    
    Steps:
    1. abs + scale to make positive integers
    2. convert to 32-bit unsigned ints
    3. convert to flat array of bits
    4. compute bitwise Shannon entropy

    Returns:
    - Entropy in bits (max = 1.0 if perfectly random 0/1)
    """
    if positions.size == 0:
        return 0.0

    # Step 1: ensure positive, scale, and quantize to integers
    ints = np.floor(np.abs(positions) * scale).astype(np.uint32)

    # Step 2: convert each 32-bit int into a list of bits
    bits = np.unpackbits(ints.view(np.uint8)).reshape(-1)

    # Step 3: count 0s and 1s
    p0 = np.count_nonzero(bits == 0) / bits.size
    p1 = 1.0 - p0

    # Step 4: Shannon entropy over bits
    entropy = 0.0
    if p0 > 0:
        entropy -= p0 * np.log2(p0)
    if p1 > 0:
        entropy -= p1 * np.log2(p1)

    return entropy


def compute_entropy_of_positive_integers(positions: np.ndarray, num_bins: int = 50, scale: float = 1000.0) -> float:
    """
    Compute Shannon entropy of discretized positive integer positions.
    
    Parameters:
    - positions: float array of positions (e.g. radii)
    - num_bins: number of histogram bins (ignored here, since we bin by integers)
    - scale: scaling factor to convert float positions to integers (controls discretization)
    
    Returns:
    - Shannon entropy in bits
    """

    # positive positions only to eliminate noise due to two's complement
    pos = np.abs(positions)

    if pos.size == 0:
        return 0.0

    # Scale and convert to int
    ints = np.floor(pos * scale).astype(np.uint32)

    # Count frequency of each integer position
    values, counts = np.unique(ints, return_counts=True)

    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


def run_simulation(
    num_particles=100,
    r_start=3.1,
    spacing=0.05,
    mass=1.0,
    max_tau=20.0,
    dt=0.001,
    tolerance=1e-8,
    num_entropy_steps=500
):
    r_starts = np.array([r_start + i * spacing for i in range(num_particles)])
    v_starts = -np.sqrt(2 * mass / r_starts)  # initial infall velocities
    all_tau, all_r = simulate_all_particles(r_starts, v_starts, mass, max_tau, dt, tolerance)

    # Entropy over evenly spaced τ
    entropy_taus = np.linspace(0, max_tau, num_entropy_steps)
    entropies = []

    for tau in entropy_taus:
        idx = int(tau / dt)
        if idx >= all_r.shape[1]:
            break
        rs_at_tau = all_r[:, idx]
        entropy = compute_bit_entropy(rs_at_tau)
        entropies.append(entropy)

    return entropy_taus[:len(entropies)], entropies, all_tau, all_r

def visualize(entropy_taus, entropies, all_tau, all_r):
    fig, axes = plt.subplots(2, 1, figsize=(14, 20))

    # Entropy plot
    axes[0].plot(entropy_taus, entropies, label="Shannon Entropy")
    axes[0].set_ylabel("Entropy (bits)")
    axes[0].set_xlabel("Proper Time τ")
    axes[0].set_title("Total Shannon Entropy of Particle Positions")
    axes[0].legend()
    axes[0].grid(True)

    # Particle trajectories
    for i in range(all_r.shape[0]):
        axes[1].plot(all_tau[i], all_r[i], alpha=0.4)
    axes[1].axhline(y=2.0, color="red", linestyle="--", label="Event Horizon (r=2)")
    axes[1].set_xlabel("Proper Time τ")
    axes[1].set_ylabel("Radius r")
    axes[1].set_title("Radial Infall of Particles")
    axes[1].legend()
    axes[1].grid(True)

    # Sync x-axis limits
    # Find common min and max τ from both datasets
    xmin = min(entropy_taus.min(), min(np.min(t) for t in all_tau))
    xmax = max(entropy_taus.max(), max(np.max(t) for t in all_tau))

    axes[0].set_xlim(xmin, xmax)
    axes[1].set_xlim(xmin, xmax)

    plt.tight_layout()
    plt.savefig("schwarzschild_dust.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    start = time.time()
    entropy_taus, entropies, all_tau, all_r = run_simulation(
        num_particles=100,
        r_start=3.1,
        spacing=0.05,
        mass=1.0,
        max_tau=10.0,
        dt=0.001,
        tolerance=1e-8,
        num_entropy_steps=500
    )
    print(f"Simulation finished in {time.time() - start:.2f} seconds.")
    visualize(entropy_taus, entropies, all_tau, all_r)
    print(f"Stay safe!")
