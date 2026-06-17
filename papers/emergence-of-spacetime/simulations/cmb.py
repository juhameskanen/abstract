"""
Entropic Informational Cosmology
===================================================================
Integrated Engine — Even Spatial Distribution & Relational Scaling Modes
Copyright 2026 - Juha Meskanen, The Abstract Universe Project
"""

import argparse
import json
import numpy as np

import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# ===========================================================================
# HYBRID SPECTRUM MAPPER — Handles 38 orders of magnitude via log-density
# ===========================================================================

class HybridSpectrumMapper:
    T_MIN_YEARS = 1e-4

    def __init__(self, json_path: str, age_limit_years: float):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.t_max_years = age_limit_years
        self._build_interpolators()

    def _time_to_lookup_fraction(self, t_years: float) -> float:
        log_t   = np.log10(max(t_years, self.T_MIN_YEARS))
        log_min = np.log10(self.T_MIN_YEARS)
        log_max = np.log10(self.t_max_years)
        return np.clip((log_t - log_min) / (log_max - log_min), 0.0, 1.0)

    def _build_interpolators(self):
        self.interpolators = {}
        self.earliest_true_fraction = {}
        
        for level_key, level_data in self.data["levels"].items():
            if level_key == "L0" or level_key == "L3":
                continue # Skip L0 and explicitly omit Atoms (L3)
                
            pts = level_data["abundances"]
            raw_years = np.array([p["time_years"] for p in pts], dtype=np.float64)
            raw_steps = np.array([self._time_to_lookup_fraction(y) for y in raw_years], dtype=np.float64)
            raw_densities = np.array([p["density_per_m3"] for p in pts], dtype=np.float64)
            
            if len(raw_steps) > 0:
                self.earliest_true_fraction[level_key] = float(np.min(raw_steps))
            else:
                self.earliest_true_fraction[level_key] = 1.0

            # Interpolate in log-space to safely handle huge density transitions
            log_densities = np.log10(raw_densities + 1e-30)
            max_log = np.max(log_densities)
            min_log = np.min(log_densities)
            
            if max_log != min_log:
                norm_matter = 1.0 - ((max_log - log_densities) / (max_log - min_log))
            else:
                norm_matter = np.ones_like(log_densities)

            # Enforce Present-Day boundary conditions
            steps = np.concatenate((np.array([0.0]), raw_steps, np.array([1.0])))
            counts = np.concatenate((np.array([0.0]), norm_matter, np.array([norm_matter[-1]])))
            
            order = np.argsort(steps)
            unique_steps, unique_indices = np.unique(steps[order], return_index=True)
            self.interpolators[level_key] = PchipInterpolator(unique_steps, counts[order][unique_indices])
            
    def get_empirical_fractions(self, step_fraction: float) -> dict:
        result = {"L1": 0.0, "L2": 0.0}
        for level_key, spline in self.interpolators.items():
            if step_fraction < self.earliest_true_fraction[level_key]:
                result[level_key] = 0.0
            else:
                result[level_key] = np.clip(float(spline(step_fraction)), 0.0, 1.0)
        return result

# ===========================================================================
# ANALYTICAL ENTROPY FABRIC GENERATOR 
# ===========================================================================

def compute_analytical_entropy(loop_fraction: float) -> float:
    """
    Evaluates the unconstrained informational potential curve of the universe.
    """
    steepness = 4.0
    return 1.0 / (1.0 + np.exp(-steepness * (loop_fraction - 0.15)))


# ===========================================================================
# MAIN PROGRAM EXECUTION
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrated Relational Even Distribution Engine.")
    parser.add_argument("--max_time_steps", type=int, default=400)
    parser.add_argument("--spectrum", type=str, default=None)
    parser.add_argument("--age", type=float, default=13.8e9)
    parser.add_argument("--time_scale_mode", type=str, default="log", choices=["log", "linear"])
    args = parser.parse_args()

    max_hardware_scale = 10000.0
    mapper = HybridSpectrumMapper(args.spectrum, args.age) if args.spectrum else None

    # History tracking arrays
    history_timestamps = []
    history_entropy_track = []
    history_g_counts = []
    history_l1_counts = []
    history_l2_counts = []
    history_total_scale = []
    history_h_emergent = []

    plt.ion()
    fig, (ax_spacetime, ax_metrics) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"IAME Cosmology Engine [EVEN RELATIONAL Canvas MODE]", fontsize=14, fontweight='bold')

    ax_spacetime.set_facecolor('#020205')
    ax_spacetime.set_xlabel("Cosmic Time (Years)")
    ax_spacetime.set_ylabel("Minkowski Spatial Horizon Envelope")
    ax_metrics.set_ylabel("Normalized Space/Matter Quantities")
    ax_metrics.set_xlabel("Cosmic Time (Years)")

    if mapper:
        ax_spacetime.set_xscale('log')
        ax_metrics.set_xscale('log')
        ax_spacetime.set_xlim(mapper.T_MIN_YEARS, mapper.t_max_years)
        ax_metrics.set_xlim(mapper.T_MIN_YEARS, mapper.t_max_years)
    else:
        ax_spacetime.set_xlim(0, args.max_time_steps)
        ax_metrics.set_xlim(0, args.max_time_steps)

    l0_scatter = ax_spacetime.scatter([], [], s=2.0, color='gray', alpha=0.5, label="L0 Horizon Edge")
    l1_scatter = ax_spacetime.scatter([], [], s=1.5, color='cyan', alpha=0.3, label="L1 Neutrinos")
    l2_scatter = ax_spacetime.scatter([], [], s=2.5, color='magenta', alpha=0.5, label="L2 Hadrons")
    
    all_time_l0_t, all_time_l0_r = [], []
    all_time_l1_t, all_time_l1_r = [], []
    all_time_l2_t, all_time_l2_r = [], []

    line_g,       = ax_metrics.plot([], [], label="L0 Free Space Fabric", color='gray', lw=1.5, linestyle=':')
    line_l1,      = ax_metrics.plot([], [], label="L1 Neutrinos",   color='cyan',    lw=2)
    line_l2,      = ax_metrics.plot([], [], label="L2 Hadrons",     color='magenta', lw=2)
    
    ax_entropy = ax_metrics.twinx()
    line_entropy, = ax_entropy.plot([], [], label="Potential Entropy Curve", color='red', linestyle='--', lw=2)
    line_h_emerge, = ax_entropy.plot([], [], label="H(t) Emergent Profile", color='orange', linestyle='-', lw=2.5)
    ax_entropy.set_ylim(-0.05, 1.5)
    ax_entropy.set_ylabel("Entropy Saturation / Operational H(t)")

    ax_spacetime.legend(loc="upper left", facecolor='#111115', edgecolor='gray', labelcolor='white')
    lines_all = [line_g, line_l1, line_l2, line_entropy, line_h_emerge]
    labels_all = [l.get_label() for l in lines_all]
    ax_metrics.legend(lines_all, labels_all, loc="upper left", facecolor='#111115', edgecolor='gray', labelcolor='white')

    current_time_tick = 0

    # -----------------------------------------------------------------------
    # CLOSED-SYSTEM DETERMINISTIC ENGINE LOOP
    # -----------------------------------------------------------------------
    while current_time_tick < args.max_time_steps:
        if not plt.fignum_exists(fig.number): break

        current_time_tick += 1
        loop_fraction = current_time_tick / args.max_time_steps

        # Evaluate background analytical potential
        true_mathematical_entropy = compute_analytical_entropy(loop_fraction)

        if mapper:
            if args.time_scale_mode == "log":
                log_min, log_max = np.log10(mapper.T_MIN_YEARS), np.log10(mapper.t_max_years)
                current_real_time = 10 ** (log_min + loop_fraction * (log_max - log_min))
                step_fraction = loop_fraction
            else:
                current_real_time = mapper.T_MIN_YEARS + loop_fraction * (mapper.t_max_years - mapper.T_MIN_YEARS)
                step_fraction = np.clip((np.log10(current_real_time) - np.log10(mapper.T_MIN_YEARS)) / (np.log10(mapper.t_max_years) - np.log10(mapper.T_MIN_YEARS)), 0.0, 1.0)
            
            emp_fractions = mapper.get_empirical_fractions(step_fraction)
            
            # Matter budget scaling derived proportional to current total informational voxels
            total_voxels = int(true_mathematical_entropy * max_hardware_scale)
            matter_weight = (emp_fractions["L1"] * 0.15 + emp_fractions["L2"] * 0.35)
            
            net_l0 = max(10, int(total_voxels * (1.0 - matter_weight)))
            net_l1 = int(total_voxels * 0.15 * emp_fractions["L1"])
            net_l2 = int(total_voxels * 0.35 * emp_fractions["L2"])
        else:
            current_real_time = float(current_time_tick)
            net_l2 = int(100 * np.sin(loop_fraction * np.pi))
            net_l1 = int(200 * np.sin(loop_fraction * np.pi))
            net_l0 = max(10, int(true_mathematical_entropy * max_hardware_scale))

        # ===================================================================
        # LOCAL RELATIONAL CANVAS (Even Spatial Distribution Mode)
        # ===================================================================
        scale_factor = 10.0 / max_hardware_scale
        
        # Total unconstrained geometric expansion envelope
        base_scale = (net_l0 * scale_factor) * 1.0
        
        # Relational Compression: Active matter density locally curves/contracts the scale
        matter_compression = 1.0 + (0.15 * (net_l1 / max_hardware_scale) + 0.35 * (net_l2 / max_hardware_scale))
        total_scale = base_scale / matter_compression
        
        min_spatial_boundary = -0.5 * total_scale
        max_spatial_boundary =  0.5 * total_scale

        history_timestamps.append(current_real_time)
        history_entropy_track.append(true_mathematical_entropy)
        history_g_counts.append(net_l0)
        history_l1_counts.append(net_l1)
        history_l2_counts.append(net_l2)
        history_total_scale.append(total_scale)

        # Operational Hubble calculation across dimensionless logarithmic updates
        if len(history_total_scale) > 2:
            d_ln_a = np.log(history_total_scale[-1] + 1e-15) - np.log(history_total_scale[-2] + 1e-15)
            d_ln_t = np.log(history_timestamps[-1]) - np.log(history_timestamps[-2])
            latest_H = d_ln_a / d_ln_t if d_ln_t > 0 else 0.0
        else:
            latest_H = 0.0
        history_h_emergent.append(latest_H)

        # Map current boundary lines to the timeline
        all_time_l0_t.append(current_real_time)
        all_time_l0_r.append(min_spatial_boundary)
        all_time_l0_t.append(current_real_time)
        all_time_l0_r.append(max_spatial_boundary)

        # Particle Mapping: Distributed evenly across the current metric bounds
        if net_l2 > 0:
            for x_coord in np.linspace(min_spatial_boundary, max_spatial_boundary, max(1, net_l2 // 20)):
                all_time_l2_t.append(current_real_time)
                all_time_l2_r.append(x_coord)

        if net_l1 > 0:
            for x_coord in np.linspace(min_spatial_boundary, max_spatial_boundary, max(1, net_l1 // 20)):
                all_time_l1_t.append(current_real_time)
                all_time_l1_r.append(x_coord)

        # Render visual steps
        if current_time_tick % 4 == 0 or current_time_tick == args.max_time_steps:
            if all_time_l0_t: l0_scatter.set_offsets(np.column_stack((all_time_l0_t, all_time_l0_r)))
            if all_time_l1_t: l1_scatter.set_offsets(np.column_stack((all_time_l1_t, all_time_l1_r)))
            if all_time_l2_t: l2_scatter.set_offsets(np.column_stack((all_time_l2_t, all_time_l2_r)))

            ax_spacetime.set_ylim(-max(0.1, total_scale * 0.6), max(0.1, total_scale * 0.6))

            line_g.set_data(history_timestamps,  history_g_counts)
            line_l1.set_data(history_timestamps, history_l1_counts)
            line_l2.set_data(history_timestamps, history_l2_counts)
            line_entropy.set_data(history_timestamps, history_entropy_track)
            line_h_emerge.set_data(history_timestamps, history_h_emergent)

            ax_metrics.relim()
            ax_metrics.autoscale(enable=True, axis='y', tight=False)
            ax_entropy.relim()

            fig.canvas.draw_idle()
            plt.pause(0.002)

    plt.ioff()
    plt.show()
