import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import time
import math


@njit
def acceleration_newton(r, M):
    """Newtonian radial acceleration: a = -GM / r^2."""
    if r <= 1e-8:
        return 0.0
    return -M / (r**2)


@njit
def integrate_particle_newton(r0, v0, M, max_t, dt, tolerance):
    """Integrate radial infall under Newtonian gravity."""
    n_steps = int(max_t / dt)
    r_values = np.zeros(n_steps)
    t_values = np.zeros(n_steps)
    r, v = r0, v0
    for i in range(n_steps):
        if r <= tolerance:
            r_values[i:] = 0.0
            t_values[i:] = t_values[i - 1]
            break
        r_values[i] = r
        t_values[i] = i * dt
        a = acceleration_newton(r, M)
        v += a * dt
        r += v * dt
    return t_values, r_values


@njit(parallel=True)
def simulate_all_particles_newton(r_starts, v_starts, M, max_t, dt, tolerance):
    num_particles = len(r_starts)
    n_steps = int(max_t / dt)
    all_r = np.zeros((num_particles, n_steps))
    all_t = np.zeros((num_particles, n_steps))
    for i in prange(num_particles):
        t_vals, r_vals = integrate_particle_newton(r_starts[i], v_starts[i], M, max_t, dt, tolerance)
        all_r[i, :] = r_vals
        all_t[i, :] = t_vals
    return all_t, all_r


def compute_bit_entropy(positions: np.ndarray, scale: float = 1000.0) -> float:
    """Bitwise Shannon entropy of scaled integer positions."""
    if positions.size == 0:
        return 0.0
    ints = np.floor(np.abs(positions) * scale).astype(np.uint32)
    bits = np.unpackbits(ints.view(np.uint8)).reshape(-1)
    p0 = np.count_nonzero(bits == 0) / bits.size
    p1 = 1.0 - p0
    entropy = 0.0
    if p0 > 0:
        entropy -= p0 * np.log2(p0)
    if p1 > 0:
        entropy -= p1 * np.log2(p1)
    return entropy


def run_simulation_newton(
    num_particles=100,
    r_start=3.1,
    spacing=0.05,
    mass=1.0,
    max_t=20.0,
    dt=0.001,
    tolerance=1e-8,
    num_entropy_steps=500
):
    r_starts = np.array([r_start + i * spacing for i in range(num_particles)])
    v_starts = -np.sqrt(2 * mass / r_starts)  # Newtonian escape speed
    all_t, all_r = simulate_all_particles_newton(r_starts, v_starts, mass, max_t, dt, tolerance)

    entropy_times = np.linspace(0, max_t, num_entropy_steps)
    entropies = []

    for t in entropy_times:
        idx = int(t / dt)
        if idx >= all_r.shape[1]:
            break
        rs_at_t = all_r[:, idx]
        entropy = compute_bit_entropy(rs_at_t)
        entropies.append(entropy)

    return entropy_times[:len(entropies)], entropies, all_t, all_r


def visualize(entropy_times, entropies, all_t, all_r):
    fig, axes = plt.subplots(2, 1, figsize=(14, 20))

    axes[0].plot(entropy_times, entropies, label="Shannon Entropy (Newton)")
    axes[0].set_ylabel("Entropy (bits)")
    axes[0].set_xlabel("Time t")
    axes[0].set_title("Total Shannon Entropy of Particle Positions (Newtonian)")
    axes[0].legend()
    axes[0].grid(True)

    for i in range(all_r.shape[0]):
        axes[1].plot(all_t[i], all_r[i], alpha=0.4)
    axes[1].set_xlabel("Time t")
    axes[1].set_ylabel("Radius r")
    axes[1].set_title("Radial Infall of Particles (Newtonian)")
    axes[1].legend()
    axes[1].grid(True)

    xmin = min(entropy_times.min(), min(np.min(t) for t in all_t))
    xmax = max(entropy_times.max(), max(np.max(t) for t in all_t))
    axes[0].set_xlim(xmin, xmax)
    axes[1].set_xlim(xmin, xmax)

    plt.tight_layout()
    plt.savefig("newton_dust.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    start = time.time()
    entropy_times, entropies, all_t, all_r = run_simulation_newton(
        num_particles=100,
        r_start=3.1,
        spacing=0.05,
        mass=1.0,
        max_t=10.0,
        dt=0.001,
        tolerance=1e-8,
        num_entropy_steps=500
    )
    print(f"Simulation finished in {time.time() - start:.2f} seconds.")
    visualize(entropy_times, entropies, all_t, all_r)
    print("Newtonian run complete.")
