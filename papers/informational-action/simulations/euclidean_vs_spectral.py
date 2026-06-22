"""
poc_spectral_vs_euclidean.py
Main Proof-of-Concept for the paper:
"Classical Geodesics as Minimal-Spectral-Complexity Trajectories in Informational Space"

Demonstrates strong alignment between Euclidean action minimization and spectral complexity
minimization in Wheeler-DeWitt minisuperspace with gravity + scalar field.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from wavefunction import Wavefunction

def compute_euclidean_action(a_path, phi_path, dtau, Lambda=1.0):
    a = np.clip(a_path, 1e-4, None)
    da = np.gradient(a, dtau)
    dphi = np.gradient(phi_path, dtau)
    
    integrand = (-a + 0.15 * (da**2) / a +
                 0.5 * a * (dphi**2) +
                 0.5 * a**3 * (phi_path**2)) + (Lambda / 3.0) * a**3
    boundary = 2.0 * a[-1]
    return float(np.sum(integrand) * dtau + boundary)


def analytic_hawking_trajectory(tau_grid, Lambda=1.0):
    omega = np.sqrt(Lambda / 3.0)
    a_max = 1.0 / omega
    return a_max * np.sin(omega * tau_grid)


if __name__ == "__main__":
    N_steps = 2000
    Lambda = 1.0
    tau_max = np.pi / (2.0 * np.sqrt(Lambda / 3.0))
    tau_grid = np.linspace(1e-4, tau_max, N_steps)
    dtau = tau_grid[1] - tau_grid[0]

    # Classical background with non-trivial scalar
    a_class = analytic_hawking_trajectory(tau_grid, Lambda)
    phi0 = 0.8
    phi_class = phi0 * np.sin(np.sqrt(Lambda / 3.0) * tau_grid)

    window = np.sin(np.pi * np.arange(N_steps) / (N_steps - 1))**2

    # Classical
    signal_class = a_class + 1j * phi_class
    wf_class = Wavefunction(signal_class * window, dx=dtau, fidelity_target=0.99)
    cs_class = wf_class.spectral_complexity()
    action_class = compute_euclidean_action(a_class, phi_class, dtau, Lambda)

    print("Classical retained modes:")
    wf_class.spectral_complexity(verbose=True)

    # Generate ensemble
    np.random.seed(42)
    num_paths = 5000
    results = []

    for i in range(-1, num_paths):   # -1 = classical
        if i == -1:
            a_path = a_class.copy()
            phi_path = phi_class.copy()
            label = "Classical"
        else:
            noise_a = np.sin(np.pi * tau_grid / tau_max) * np.random.normal(0, 1.0, N_steps)
            noise_phi = np.sin(np.pi * tau_grid / tau_max) * np.random.normal(0, 0.8, N_steps)
            a_path = np.clip(a_class + noise_a * np.random.uniform(0.008, 0.18), 1e-3, None)
            phi_path = phi_class + noise_phi * np.random.uniform(0.06, 0.40)
            label = f"Pert_{i}"

        action = compute_euclidean_action(a_path, phi_path, dtau, Lambda)
        signal = a_path + 1j * phi_path
        wf = Wavefunction(signal * window, dx=dtau, fidelity_target=0.99)
        cs = wf.spectral_complexity()

        results.append({
            "label": label,
            "action": action,
            "complexity": cs,
            "a": a_path,
            "phi": phi_path
        })

    # Analysis
    actions = np.array([r["action"] for r in results])
    complexities = np.array([r["complexity"] for r in results])
    rho, _ = spearmanr(complexities, actions)
    min_idx = np.argmin(complexities)

    print("\n=== FINAL RESULTS ===")
    print(f"Spearman ρ                    : {rho:.5f}")
    print(f"Classical Action / C_s        : {action_class:.4f} / {cs_class:.2f}")
    print(f"Min-C_s Action                : {results[min_idx]['action']:.4f} ({results[min_idx]['label']})")

    weights = 2.0 ** (-complexities)
    weights /= weights.sum()
    print(f"Classical Solomonoff fraction : {weights[0]*100:.5f}%")

    # === Save figures for the paper ===
    plt.figure(figsize=(12, 5))
    plt.scatter(actions, complexities, c=complexities, cmap='viridis', alpha=0.6, s=6)
    plt.plot(action_class, cs_class, 'ro', markersize=12, markeredgecolor='k', label='Classical')
    plt.xscale('log')
    plt.xlabel('Euclidean Action $S_E$')
    plt.ylabel('Spectral Complexity $C_s$')
    plt.title(f'Spectral vs Euclidean — Spearman $\\rho = {rho:.5f}$')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.colorbar(label='$C_s$')
    plt.tight_layout()
    plt.savefig('../figures/variational_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Second figure: example histories
    plt.figure(figsize=(12, 6))
    idx = 300
    ax1 = plt.gca()
    ax1.plot(tau_grid, a_class, 'r-', lw=2.5, label='a(τ) Classical')
    ax1.plot(tau_grid, results[idx]["a"], 'b-', alpha=0.75, label='a(τ) Perturbed')
    ax1.set_ylabel('Scale factor a(τ)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')

    ax2 = ax1.twinx()
    ax2.plot(tau_grid, phi_class, 'm-', lw=2.0, label='φ(τ) Classical')
    ax2.plot(tau_grid, results[idx]["phi"], 'g--', alpha=0.8, label='φ(τ) Perturbed')
    ax2.set_ylabel('Scalar field φ(τ)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')

    ax1.set_xlabel('$\\tau$')
    ax1.set_title('Classical and Perturbed Histories')
    ax1.grid(True, alpha=0.4)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.tight_layout()
    plt.savefig('../figures/example_histories.png', dpi=300, bbox_inches='tight')
    plt.show()
