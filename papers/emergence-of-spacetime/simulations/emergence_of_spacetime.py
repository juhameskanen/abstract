"""
Time-Reversed Entropic Cosmology: Pure Empirical Edition
===================================================================
An informational ontology simulation utilizing a hidden background fabric
of Fabric Quanta to derive spatial coordinates relationally. 

When a JSON spectrum is provided, this engine operates in a PURE ANALYTICAL 
MODE: it bypasses the discrete bitfield simulation entirely, extracting both 
the unconstrained vacuum baseline (L0) and bound matter hierarchies (L1-L3) 
directly from explicit empirical data curves.

Copyright 2026 - Juha Meskanen, The Abstract Universe Project
"""

import random
import argparse
import json
import numpy as np

import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
from numba import njit


# ===========================================================================
# SPECTRUM LOADER — Converts Comoving Densities to Relational Physical Trajectories
# ===========================================================================

class SpectrumLoader:
    """
    Loads comoving abundance data, converts it to a dynamic physical fraction,
    and scales each structural layer appropriately into the simulation's dynamic range.
    """

    T_MAX_YEARS = 13.8e9
    T_MIN_YEARS = 1e-4

    def __init__(self, json_path: str, n_max: float):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.n_max = n_max
        self._build_interpolators()

    def _time_to_lookup_fraction(self, t_years: float) -> float:
        # Maps historical timelines to a normalized 0.0 - 1.0 step domain
        log_t   = np.log10(max(t_years, self.T_MIN_YEARS))
        log_min = np.log10(self.T_MIN_YEARS)
        log_max = np.log10(self.T_MAX_YEARS)
        return np.clip((log_t - log_min) / (log_max - log_min), 0.0, 1.0)

    def _build_interpolators(self):
        self.interpolators = {}
        
        for level_key, level_data in self.data["levels"].items():
            pts = level_data["abundances"]
            
            # 1. Parse raw data streams explicitly into solid float64 arrays
            raw_steps = np.array([self._time_to_lookup_fraction(p["time_years"]) for p in pts], dtype=np.float64)
            raw_densities = np.array([p["density_per_m3"] for p in pts], dtype=np.float64)
            
            # 2. Inject the absolute zero boundary condition at the singularity using native NumPy arrays
            steps = np.concatenate((np.array([0.0]), raw_steps))
            
            if level_key == "L0":
                # Space Fabric uses Log-Space to handle the massive volume scaling range
                log_densities = np.log10(raw_densities + 1e-15)
                max_log = np.max(log_densities)
                min_log = np.min(log_densities)
                
                norm_volume = (max_log - log_densities) / (max_log - min_log + 1e-15)
                norm_counts = norm_volume * self.n_max
                counts = np.concatenate((np.array([0.0]), norm_counts))
            else:
                # Matter layers use Linear-Space relative to their localized historical peaks,
                # ensuring they remain at EXACT zero during the zero-entropy state.
                peak_val = np.max(raw_densities) if np.max(raw_densities) > 0 else 1.0
                norm_matter = raw_densities / peak_val
                norm_counts = norm_matter * (self.n_max * 0.25)
                
                # Force the absolute 0-count baseline at the singularity point
                counts = np.concatenate((np.array([0.0]), norm_counts))
            
            # 3. Enforce strict monotonic sorting across the unified domain
            order = np.argsort(steps)
            self.interpolators[level_key] = (steps[order], counts[order])
    
    def get_counts(self, step_fraction: float) -> dict:
        result = {}
        for level_key, (step_arr, cnt_arr) in self.interpolators.items():
            result[level_key] = float(np.interp(step_fraction, step_arr, cnt_arr))
        return result

    def print_summary(self):
        print(f"\nPure Empirical Mode Activated: {self.data['name']}")
        print(f"   Dynamic Range Ceiling (N_max): {self.n_max:.0f}")
        for k, v in self.data["levels"].items():
            print(f"   {k} ({v['name']}): Bound to explicit observation curves.")
        print()


# ===========================================================================
# BITSTRING ENGINE (Bypassed in Pure Empirical Mode)
# ===========================================================================

@njit(fastmath=True)
def compute_entropy(bitstring: np.ndarray) -> float:
    n = bitstring.size
    if n == 0: return 0.0
    count1 = np.sum(bitstring)
    count0 = n - count1
    p0 = count0 / n
    p1 = count1 / n
    entropy = 0.0
    if p0 > 0.0: entropy -= p0 * np.log2(p0)
    if p1 > 0.0: entropy -= p1 * np.log2(p1)
    return entropy


@njit(fastmath=True)
def run_graviton_filter(bitfield: np.ndarray, w: int, target_val: int) -> np.ndarray:
    num_fragments = bitfield.size // w
    success_string = np.zeros(num_fragments, dtype=np.uint8)
    for m in range(num_fragments):
        val = 0
        start_idx = m * w
        for j in range(w):
            if bitfield[start_idx + j]:
                val |= (1 << (w - 1 - j))
        if val == target_val:
            success_string[m] = 1
    return success_string


@njit(fastmath=True)
def run_recursive_density_filter(input_string: np.ndarray, window_size: int, threshold: int) -> np.ndarray:
    if input_string.size < window_size:
        return np.zeros(0, dtype=np.uint8)
    num_fragments = input_string.size // window_size
    success_string = np.zeros(num_fragments, dtype=np.uint8)
    for m in range(num_fragments):
        density = 0
        start_idx = m * window_size
        for j in range(window_size):
            density += input_string[start_idx + j]
        if density >= threshold:
            success_string[m] = 1
    return success_string


class EntropicCosmologyEngine:
    def __init__(self, total_bits, pattern_str, l2_w, l2_t, l3_w, l3_t, l4_w, l4_t):
        self.total_bits = total_bits
        self.w = len(pattern_str)
        self.l2_w, self.l2_t = l2_w, l2_t
        self.l3_w, self.l3_t = l3_w, l3_t
        self.l4_w, self.l4_t = l4_w, l4_t

        self.target_val = 0
        for i, char in enumerate(pattern_str):
            if char == '1':
                self.target_val |= (1 << (self.w - 1 - i))

        self.bitfield = np.zeros(total_bits, dtype=np.uint8)
        self.step = 0

        self.history_timestamps = []
        self.history_entropy    = []
        self.history_g_counts   = []
        self.history_l1_counts  = []
        self.history_l2_counts  = []
        self.history_l3_counts  = []

    def get_layer_data(self):
        s_g  = run_graviton_filter(self.bitfield, self.w, self.target_val)
        g_indices  = np.where(s_g == 1)[0].astype(np.float64)

        s_l1 = run_recursive_density_filter(s_g,  self.l2_w, self.l2_t)
        l1_indices = np.where(s_l1 == 1)[0].astype(np.float64)

        s_l2 = run_recursive_density_filter(s_l1, self.l3_w, self.l3_t)
        l2_indices = np.where(s_l2 == 1)[0].astype(np.float64)

        s_l3 = run_recursive_density_filter(s_l2, self.l4_w, self.l4_t)
        l3_indices = np.where(s_l3 == 1)[0].astype(np.float64)

        if g_indices.size > 0:
            l0_r = g_indices
            l1_r = l1_indices * self.l2_w
            l2_r = l2_indices * (self.l2_w * self.l3_w)
            l3_r = l3_indices * (self.l2_w * self.l3_w * self.l4_w)
        else:
            l0_r = l1_r = l2_r = l3_r = np.zeros(0)

        return l0_r, l1_r, l2_r, l3_r

    def mutate_state(self, mutations_per_step: int):
        for _ in range(mutations_per_step):
            idx = random.randint(0, self.total_bits - 1)
            self.bitfield[idx] ^= 1
        self.step += 1


# ===========================================================================
# HUBBLE PARAMETER COMPUTATION
# ===========================================================================

def compute_hubble(history_steps, history_active):
    if len(history_active) < 2:
        return np.array([]), np.array([])

    steps = np.array(history_steps, dtype=float)
    act   = np.array(history_active, dtype=float)

    dact   = np.diff(act)
    dsteps = np.diff(steps)

    valid = (np.abs(dsteps) > 1e-12) & (act[:-1] > 1.0)
    H = np.zeros(len(dact))
    H[valid] = (dact[valid] / dsteps[valid]) / act[:-1][valid]

    return steps[:-1], H


def generate_even_coordinates(count, scale):
    if count <= 0:  return np.zeros(0)
    if count == 1:  return np.array([0.0])
    return np.linspace(-0.5, 0.5, count) * scale


# ===========================================================================
# MAIN PROGRAM EXECUTION
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified IAME Cosmology Engine: Maps relational topological coordinates "
                    "and informational entropy milestones across both algorithmic bitstring modes "
                    "and empirical database timeline spectrums.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Fundamental Engine Configurations
    parser.add_argument("--bits", type=int, default=65535,
                        help="Total capacity depth of the fundamental bitfield matrix. Sets the baseline resolution.")
    parser.add_argument("--pattern", type=str, default="01",
                        help="The core target token sequence representing structural creation rules.")
    parser.add_argument("--quantum_mutations", type=int, default=1000,
                        help="Total number of random bitwise flips executed per step during standard simulation loops.")
    parser.add_argument("--max_time_steps", type=int, default=1000,
                        help="Total timeframe steps to run the simulation loop before termination.")
    parser.add_argument("--recursion", type=int, default=3, choices=[0,1,2,3],
                        help="Depth tier of structural nesting allowed during standard visualization draws.")
    parser.add_argument("--feynman", action="store_true",
                        help="Force topological worldline tracing vector paths instead of standard localized dot clouds.")
    parser.add_argument("--spectrum", type=str, default="spectrum.json",
                        help="Path to an observational JSON dataset. Activating this instantly forces Pure Analytical Mode, "
                             "mapping real comoving densities directly onto the relational space clock.")
    
    # Rendering Optimization
    parser.add_argument("--max_visual_lines", type=int, default=25,
                        help="Informational Decimation Filter Limit: Sets the maximum visual lines rendered per level "
                             "in Feynman Space. Drastically improves render frame-rates without impacting calculation arrays.")

    # Matter Hierarchical Processing Windows
    parser.add_argument("--l2_window", type=int, default=8,
                        help="Bit window tracking range used when filtering Layer 1 Hadrons out of Layer 0 Fabric fields.")
    parser.add_argument("--l2_threshold", type=int, default=5,
                        help="Density activation limit required to confirm Layer 1 Hadron structures inside the target tracking window.")
    parser.add_argument("--l3_window", type=int, default=6,
                        help="Bit window tracking range used when filtering Layer 2 Atoms out of Layer 1 Hadron pools.")
    parser.add_argument("--l3_threshold", type=int, default=3,
                        help="Density activation limit required to confirm Layer 2 Atom structures inside the target tracking window.")
    parser.add_argument("--l4_window", type=int, default=8,
                        help="Bit window tracking range used when filtering Layer 3 Compounds out of Layer 2 Atom structures.")
    parser.add_argument("--l4_threshold", type=int, default=3,
                        help="Density activation limit required to confirm Layer 3 Compound structures inside the target tracking window.")
    args = parser.parse_args()

    engine = EntropicCosmologyEngine(
        args.bits, args.pattern,
        args.l2_window, args.l2_threshold,
        args.l3_window, args.l3_threshold,
        args.l4_window, args.l4_threshold
    )

    spectrum = None
    n_max = max(1.0, float(args.bits // engine.w))
    if args.spectrum:
        spectrum = SpectrumLoader(args.spectrum, n_max)
        spectrum.print_summary()

    history_active = []
    history_vacuum = []  
    history_timestamps = []
    history_entropy_track = []
    history_g_counts = []
    history_l1_counts = []
    history_l2_counts = []
    history_l3_counts = []

    plt.ion()
    fig, (ax_spacetime, ax_metrics) = plt.subplots(1, 2, figsize=(16, 7))

    mode_label = "Pure Empirical Spectrum" if spectrum else ("Feynman" if args.feynman else "Scatter")
    fig.suptitle(f"IAME Cosmology Engine [{mode_label}]", fontsize=16, fontweight='bold')

    # Convert graph x-axes to handle true log scales
    ax_spacetime.set_facecolor('#020205')
    ax_spacetime.set_xlabel("Cosmic Time (Years)")
    ax_spacetime.set_ylabel("Relational Spatial Coordinate")
    
    ax_metrics.set_ylabel("Structural Count / Baseline Scale")
    ax_metrics.set_xlabel("Cosmic Time (Years)")
    ax_metrics.set_ylim(bottom=0)

    if spectrum:
        ax_spacetime.set_xscale('log')
        ax_metrics.set_xscale('log')
        ax_spacetime.set_xlim(spectrum.T_MIN_YEARS, spectrum.T_MAX_YEARS)
        ax_metrics.set_xlim(spectrum.T_MIN_YEARS, spectrum.T_MAX_YEARS)
    else:
        ax_spacetime.set_xlim(0, args.max_time_steps)
        ax_metrics.set_xlim(0, args.max_time_steps)

    if args.feynman or spectrum:
        ax_spacetime.set_title("Emergent Topological Feynman Space")
        ax_spacetime.plot([], [], color='gray',    alpha=0.3, label="L0 Fabric Baseline")
        ax_spacetime.plot([], [], color='cyan',    alpha=0.6, label="L1 Hadrons")
        ax_spacetime.plot([], [], color='magenta', alpha=0.7, label="L2 Atoms")
        ax_spacetime.plot([], [], color='lime',    alpha=0.8, label="L3 Compounds")
    else:
        ax_spacetime.set_title("Relational Spacetime Flow")
        l0_scatter = ax_spacetime.scatter([], [], s=1, color='gray',    marker='.',  alpha=0.10, label="L0 Fabric Baseline")
        l1_scatter = ax_spacetime.scatter([], [], s=1, color='cyan',    marker='o',  alpha=0.30, label="L1 Hadrons")
        l2_scatter = ax_spacetime.scatter([], [], s=2, color='magenta', marker='*',  alpha=0.50, label="L2 Atoms")
        l3_scatter = ax_spacetime.scatter([], [], s=4, color='lime',    marker='^',  alpha=0.70, label="L3 Compounds")
        all_time_l0_t, all_time_l0_r = [], []
        all_time_l1_t, all_time_l1_r = [], []
        all_time_l2_t, all_time_l2_r = [], []
        all_time_l3_t, all_time_l3_r = [], []

    ax_spacetime.legend(loc="upper left")

    # Metrics panel setup
    line_g,       = ax_metrics.plot([], [], label="L0 Free Fabric (Relational Pool)", color='gray', lw=1, linestyle=':')
    line_l1,      = ax_metrics.plot([], [], label="L1 Hadrons",       color='cyan',    lw=2)
    line_l2,      = ax_metrics.plot([], [], label="L2 Atoms",          color='magenta', lw=2)
    line_l3,      = ax_metrics.plot([], [], label="L3 Compounds",      color='lime',    lw=2.5)
    line_vacuum,  = ax_metrics.plot([], [], label="Explicit L0 Vacuum Trend (Unconstrained)", color='blue', lw=1.5, linestyle='-.')

    ax_entropy = ax_metrics.twinx()
    line_entropy, = ax_entropy.plot([], [], label="Shannon Entropy (Structural)", color='red',   linestyle='--', lw=1.5)
    line_hubble,  = ax_entropy.plot([], [], label="H(t) Relational", color='yellow', linestyle='-',  lw=1.5)
    ax_entropy.set_ylabel("Structural Information Entropy / H(t)", color='red')
    ax_entropy.set_ylim(-0.05, 1.05)

    lines  = [line_g, line_l1, line_l2, line_l3, line_vacuum, line_entropy, line_hubble]
    labels = [l.get_label() for l in lines]
    ax_metrics.legend(lines, labels, loc="upper left")

    plt.tight_layout()

    current_time_tick = 0
    prev_t = spectrum.T_MIN_YEARS if spectrum else 0
    prev_l0_x = prev_l1_x = prev_l2_x = prev_l3_x = None

    # -----------------------------------------------------------------------
    # ANALYTICAL SIMULATION LOOP
    # -----------------------------------------------------------------------
    while current_time_tick < args.max_time_steps:
        if not plt.fignum_exists(fig.number):
            break

        if spectrum:
            current_time_tick += 1
            step_fraction = current_time_tick / args.max_time_steps
            
            # Map structural execution back to actual timeline values
            log_min = np.log10(spectrum.T_MIN_YEARS)
            log_max = np.log10(spectrum.T_MAX_YEARS)
            current_real_years = 10 ** (log_min + step_fraction * (log_max - log_min))
            
            # PURE EXPLICIT DATA TRAJECTORIES - Fetched directly from JSON fields
            counts   = spectrum.get_counts(step_fraction)
            raw_l0   = int(counts.get("L0", 0))
            raw_l1   = int(counts.get("L1", 0))
            raw_l2   = int(counts.get("L2", 0))
            raw_l3   = int(counts.get("L3", 0))
            
            # Matter elements withdraw resolution directly from the explicit L0 boundary trend
            vacuum_elements = raw_l0
            net_l3 = raw_l3
            net_l2 = raw_l2
            net_l1 = raw_l1
            net_l0 = max(1, raw_l0 - (net_l1 + net_l2 + net_l3))
            
            # CALCULATE STRUCTURAL SHANNON ENTROPY FROM EMERGENCE PROBABILITIES
            total_tokens = net_l0 + net_l1 + net_l2 + net_l3
            if total_tokens > 0:
                p_arr = np.array([net_l0, net_l1, net_l2, net_l3], dtype=np.float64) / total_tokens
                p_arr = p_arr[p_arr > 0]
                sim_indicator = -np.sum(p_arr * np.log2(p_arr)) / 2.0  # Normalize to display safely
            else:
                sim_indicator = 0.0
        else:
            # Traditional simulated algorithmic filter execution mode
            engine.mutate_state(args.quantum_mutations)
            current_time_tick = engine.step
            current_real_years = float(current_time_tick)
            
            l0_pos_raw, l1_pos_raw, l2_pos_raw, l3_pos_raw = engine.get_layer_data()
            raw_l0, raw_l1, raw_l2, raw_l3 = l0_pos_raw.size, l1_pos_raw.size, l2_pos_raw.size, l3_pos_raw.size

            net_l3 = raw_l3
            net_l2 = max(0, raw_l2 - (net_l3 * int(args.l4_window)))
            net_l1 = max(0, raw_l1 - (raw_l2  * int(args.l3_window)))
            net_l0 = max(0, raw_l0 - (raw_l1  * int(args.l2_window)))
            
            vacuum_elements = raw_l0
            sim_indicator = compute_entropy(engine.bitfield)

        # Calculate relational metrics based strictly on unbound spatial token capacities
        active_elements = max(1, net_l0)
        total_scale     = (active_elements / n_max) * 4.0

        history_active.append(active_elements)
        history_vacuum.append(vacuum_elements)
        history_timestamps.append(current_real_years)
        history_entropy_track.append(sim_indicator)
        history_g_counts.append(net_l0)
        history_l1_counts.append(net_l1)
        history_l2_counts.append(net_l2)
        history_l3_counts.append(net_l3)

        # Derive H(t) from explicit timeline variations
        _, H_vals = compute_hubble(history_timestamps, history_active)

        # Coordinate generator visualization routing with Informational Decimation Optimization
        if args.feynman or spectrum:
            # Generate absolute full calculation coordinate fields to maintain backend arithmetic fidelity
            full_l0_x = generate_even_coordinates(net_l0, total_scale)
            full_l1_x = generate_even_coordinates(net_l1, total_scale)
            full_l2_x = generate_even_coordinates(net_l2, total_scale)
            full_l3_x = generate_even_coordinates(net_l3, total_scale)

            # Slicing filters applied to cap graphics processing overhead down to custom argument limit
            curr_l0_x = full_l0_x[::max(1, full_l0_x.size // args.max_visual_lines)]
            curr_l1_x = full_l1_x[::max(1, full_l1_x.size // args.max_visual_lines)]
            curr_l2_x = full_l2_x[::max(1, full_l2_x.size // args.max_visual_lines)]
            curr_l3_x = full_l3_x[::max(1, full_l3_x.size // args.max_visual_lines)]

            if ((spectrum and current_time_tick > 1) or (not spectrum and current_time_tick > 1)) and prev_l0_x is not None:
                for arr_p, arr_c, col, al, lw in [
                    (prev_l0_x, curr_l0_x, 'gray',    0.05, 0.5),
                    (prev_l1_x, curr_l1_x, 'cyan',    0.35, 1.0),
                    (prev_l2_x, curr_l2_x, 'magenta', 0.55, 1.2),
                    (prev_l3_x, curr_l3_x, 'lime',    0.75, 1.5),
                ]:
                    if arr_p.size > 0 and arr_c.size > 0:
                        for px in arr_p:
                            cx = arr_c[np.argmin(np.abs(arr_c - px))]
                            ax_spacetime.plot([prev_t, current_real_years],
                                             [px, cx], color=col, alpha=al, lw=lw)

            prev_t    = current_real_years
            prev_l0_x = curr_l0_x
            prev_l1_x = curr_l1_x
            prev_l2_x = curr_l2_x
            prev_l3_x = curr_l3_x
        else:
            def _append(t, count, t_list, r_list):
                norm = np.linspace(-0.5, 0.5, count) if count > 0 else np.zeros(0)
                for v in norm:
                    t_list.append(t)
                    r_list.append(v * total_scale)

            _append(current_real_years, net_l0, all_time_l0_t, all_time_l0_r)
            _append(current_real_years, net_l1, all_time_l1_t, all_time_l1_r)
            _append(current_real_years, net_l2, all_time_l2_t, all_time_l2_r)
            _append(current_real_years, net_l3, all_time_l3_t, all_time_l3_r)

            if all_time_l0_t: l0_scatter.set_offsets(np.column_stack((all_time_l0_t, all_time_l0_r)))
            if all_time_l1_t: l1_scatter.set_offsets(np.column_stack((all_time_l1_t, all_time_l1_r)))
            if all_time_l2_t: l2_scatter.set_offsets(np.column_stack((all_time_l2_t, all_time_l2_r)))
            if all_time_l3_t: l3_scatter.set_offsets(np.column_stack((all_time_l3_t, all_time_l3_r)))

        ax_spacetime.set_ylim(-max(0.05, total_scale * 0.55), max(0.05, total_scale * 0.55))

        # Realtime graph updating
        line_g.set_data(history_timestamps,  history_g_counts)
        line_l1.set_data(history_timestamps, history_l1_counts)
        line_l2.set_data(history_timestamps, history_l2_counts)
        line_l3.set_data(history_timestamps, history_l3_counts)
        line_vacuum.set_data(history_timestamps, history_vacuum)
        line_entropy.set_data(history_timestamps, history_entropy_track)

        if len(H_vals) > 1:
            H_display = H_vals / (np.max(np.abs(H_vals)) + 1e-12) * 0.5
            line_hubble.set_data(history_timestamps[:-1], H_display)

        ax_metrics.relim()
        ax_metrics.autoscale(enable=True, axis='y', tight=False)
        ax_entropy.relim()

        if spectrum:
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.005)
        else:
            refresh_tick = 5
            if current_time_tick % refresh_tick == 0:
                fig.canvas.draw_idle()
                plt.pause(0.001)

    plt.ioff()
    plt.show()
