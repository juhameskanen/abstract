"""
Time-Reversed Entropic Cosmology: Unified Edition (Fixed Relational Scale)
===================================================================
An informational ontology simulation utilizing a hidden background fabric
of Fabric Quanta to derive spatial coordinates relationally. Maps hierarchical
matter spectrums symmetrically, featuring both fractional scatter tracing
and topological proximity-linked Feynman vertex plotting.

Features:
- Juha Meskanen's Observer Matter Release Principle
- CONSERVED TOTAL SCALE: Spacetime horizon diameter derived relationally via
  hierarchical structural token conservation logic.
- Dynamic instantaneous Y-axis scaling to stabilize relational worldline tracing.
- Dual-Mode Graphics Architecture: Choose between Scatter or Feynman Worldlines.
- Real Spectrum Mode: Load observed baryonic abundance data from JSON file
  to produce a first-principles prediction of H(t).

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
# SPECTRUM LOADER — maps real cosmological abundance data into simulation units
# ===========================================================================

class SpectrumLoader:
    """
    Loads observed baryonic particle abundance data from a JSON file and
    provides interpolated N_k(entropy) curves for use in the simulation.
    """

    T_MAX_YEARS = 13.8e9
    T_MIN_YEARS = 1e-10

    def __init__(self, json_path: str, n_max: float):
        with open(json_path, 'r') as f:
            self.data = json.load(f)

        self.n_max = n_max
        self._build_interpolators()

    def _time_to_entropy(self, t_years: float) -> float:
        log_t   = np.log10(max(t_years, self.T_MIN_YEARS))
        log_min = np.log10(self.T_MIN_YEARS)
        log_max = np.log10(self.T_MAX_YEARS)
        return np.clip((log_t - log_min) / (log_max - log_min), 0.0, 1.0)

    def _build_interpolators(self):
        l0_densities = [pt["density_per_m3"]
                        for pt in self.data["levels"]["L0"]["abundances"]]
        self.peak_l0_density = max(l0_densities)

        self.interpolators = {}
        for level_key, level_data in self.data["levels"].items():
            pts = level_data["abundances"]
            entropies = np.array([self._time_to_entropy(p["time_years"]) for p in pts])
            counts = np.array(
                [(p["density_per_m3"] / self.peak_l0_density) * self.n_max
                 for p in pts]
            )
            order = np.argsort(entropies)
            self.interpolators[level_key] = (entropies[order], counts[order])

    def get_counts(self, entropy: float) -> dict:
        result = {}
        for level_key, (ent_arr, cnt_arr) in self.interpolators.items():
            result[level_key] = float(np.interp(entropy, ent_arr, cnt_arr))
        return result

    def print_summary(self):
        print(f"\nSpectrum loaded: {self.data['name']}")
        print(f"  Time unit   : {self.data['time_unit']}")
        print(f"  Density unit: {self.data['density_unit']}")
        print(f"  Peak L0 density (normalization ref): {self.peak_l0_density:.3e} /m^3")
        print(f"  N_max (simulation ceiling): {self.n_max:.0f}")
        for k, v in self.data["levels"].items():
            n_pts = len(v["abundances"])
            print(f"  {k} ({v['name']}): {n_pts} data points")
        print()


# ===========================================================================
# BITSTRING ENGINE
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

        self.current_g_count = g_indices.size

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

def compute_hubble(history_entropy, history_active, n_max):
    """
    Computes the relational Hubble parameter H(t) from the history of
    active element counts.
    """
    if len(history_active) < 2:
        return np.array([]), np.array([])

    ent = np.array(history_entropy)
    act = np.array(history_active, dtype=float)

    dact = np.diff(act)
    dent = np.diff(ent)

    valid = (np.abs(dent) > 1e-12) & (act[:-1] > 1.0)
    H = np.zeros(len(dact))
    H[valid] = (dact[valid] / dent[valid]) / act[:-1][valid]

    return ent[:-1], H


# ===========================================================================
# COORDINATE GENERATOR
# ===========================================================================

def generate_even_coordinates(count, scale):
    if count <= 0:  return np.zeros(0)
    if count == 1:  return np.array([0.0])
    return np.linspace(-0.5, 0.5, count) * scale


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified IAME Cosmology Visualizer")
    parser.add_argument("--bits",              type=int,   default=32*1024)
    parser.add_argument("--pattern",          type=str,   default="10")
    parser.add_argument("--quantum_mutations",type=int,   default=100)
    parser.add_argument("--max_time_steps",   type=int,   default=1000)
    parser.add_argument("--recursion",        type=int,   default=3, choices=[0,1,2,3])
    parser.add_argument("--feynman",          action="store_true")
    parser.add_argument("--spectrum",         type=str,   default=None)

    parser.add_argument("--l2_window",    type=int, default=8)
    parser.add_argument("--l2_threshold", type=int, default=5)
    parser.add_argument("--l3_window",    type=int, default=6)
    parser.add_argument("--l3_threshold", type=int, default=3)
    parser.add_argument("--l4_window",    type=int, default=8)
    parser.add_argument("--l4_threshold", type=int, default=3)
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

    # History lists
    history_active = []
    history_vacuum = []  # Tracks unconstrained matter-free metric trajectory

    plt.ion()
    fig, (ax_spacetime, ax_metrics) = plt.subplots(1, 2, figsize=(16, 7))

    mode_label = "Real Spectrum" if spectrum else ("Feynman" if args.feynman else "Scatter")
    fig.suptitle(f"IAME Cosmology Engine [{mode_label}]", fontsize=16, fontweight='bold')

    ax_spacetime.set_xlim(0, args.max_time_steps)
    ax_spacetime.set_facecolor('#020205')
    ax_spacetime.set_xlabel("Entropic Time Steps")
    ax_spacetime.set_ylabel("Relational Spatial Coordinate")

    if args.feynman:
        ax_spacetime.set_title("Emergent Topological Feynman Space")
        ax_spacetime.plot([], [], color='gray',    alpha=0.3, label="L0 Fabric Quanta")
        ax_spacetime.plot([], [], color='cyan',    alpha=0.6, label="L1 Hadrons")
        ax_spacetime.plot([], [], color='magenta', alpha=0.7, label="L2 Atoms")
        ax_spacetime.plot([], [], color='lime',    alpha=0.8, label="L3 Compounds")
    else:
        ax_spacetime.set_title("Relational Spacetime Flow")
        l0_scatter = ax_spacetime.scatter([], [], s=1, color='gray',    marker='.',  alpha=0.10, label="L0 Fabric Quanta")
        l1_scatter = ax_spacetime.scatter([], [], s=1, color='cyan',    marker='o',  alpha=0.30, label="L1 Hadrons")
        l2_scatter = ax_spacetime.scatter([], [], s=2, color='magenta', marker='*',  alpha=0.50, label="L2 Atoms")
        l3_scatter = ax_spacetime.scatter([], [], s=4, color='lime',    marker='^',  alpha=0.70, label="L3 Compounds")
        all_time_l0_t, all_time_l0_r = [], []
        all_time_l1_t, all_time_l1_r = [], []
        all_time_l2_t, all_time_l2_r = [], []
        all_time_l3_t, all_time_l3_r = [], []

    ax_spacetime.legend(loc="upper left")

    # --- Metrics panel ---
    line_g,       = ax_metrics.plot([], [], label="L0 Fabric Quanta (Net Unbound)", color='gray', lw=1, linestyle=':')
    line_l1,      = ax_metrics.plot([], [], label="L1 Hadrons",       color='cyan',    lw=2)
    line_l2,      = ax_metrics.plot([], [], label="L2 Atoms",          color='magenta', lw=2)
    line_l3,      = ax_metrics.plot([], [], label="L3 Compounds",      color='lime',    lw=2.5)
    
    # NEW: Vacuum Baseline Curve (Scale factor if matter did not consume spatial resolution tokens)
    line_vacuum,  = ax_metrics.plot([], [], label="Vacuum Fabric baseline (Matter-Free)", color='blue', lw=1.5, linestyle='-.')

    ax_metrics.set_ylabel("Structural Count / Baseline Scale")
    ax_metrics.set_xlabel("Entropic Time Steps")
    ax_metrics.set_ylim(bottom=0)

    ax_entropy = ax_metrics.twinx()
    line_entropy, = ax_entropy.plot([], [], label="Shannon Entropy", color='red',    linestyle='--', lw=1.5)
    line_hubble,  = ax_entropy.plot([], [], label="H(t) relational", color='yellow', linestyle='-',  lw=1.5)
    ax_entropy.set_ylabel("Entropy / H(t)", color='red')
    ax_entropy.set_ylim(-0.5, 1.05)

    lines  = [line_g, line_l1, line_l2, line_l3, line_vacuum, line_entropy, line_hubble]
    labels = [l.get_label() for l in lines]
    ax_metrics.legend(lines, labels, loc="upper left")

    plt.tight_layout()

    current_time_tick = 0
    last_l1_count     = 0
    current_stride    = 1
    target_l1_growth  = 50

    prev_t = 0
    prev_l0_x = prev_l1_x = prev_l2_x = prev_l3_x = None

    # -----------------------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------------------
    while current_time_tick < args.max_time_steps:
        if not plt.fignum_exists(fig.number):
            break

        stride_limit = 1 if (args.feynman or spectrum) else current_stride

        for _ in range(stride_limit):
            engine.mutate_state(args.quantum_mutations)
            current_time_tick += 1
            if current_time_tick >= args.max_time_steps:
                break

        entropy = compute_entropy(engine.bitfield)

        if spectrum:
            counts   = spectrum.get_counts(entropy)
            raw_l0   = int(counts.get("L0", 0))
            raw_l1   = int(counts.get("L1", 0))
            raw_l2   = int(counts.get("L2", 0))
            raw_l3   = int(counts.get("L3", 0))
            l0_pos_raw = np.arange(raw_l0, dtype=np.float64)
            l1_pos_raw = np.arange(raw_l1, dtype=np.float64)
            l2_pos_raw = np.arange(raw_l2, dtype=np.float64)
            l3_pos_raw = np.arange(raw_l3, dtype=np.float64)
        else:
            l0_pos_raw, l1_pos_raw, l2_pos_raw, l3_pos_raw = engine.get_layer_data()
            raw_l0 = int(l0_pos_raw.size)
            raw_l1 = int(l1_pos_raw.size)
            raw_l2 = int(l2_pos_raw.size)
            raw_l3 = int(l3_pos_raw.size)

        # -------------------------------------------------------------------
        # CONSERVATION LOGIC — net unbound counts
        # -------------------------------------------------------------------
        net_l3 = raw_l3
        net_l2 = max(0, raw_l2 - (net_l3 * int(args.l4_window)))
        net_l1 = max(0, raw_l1 - (raw_l2  * int(args.l3_window)))
        net_l0 = max(0, raw_l0 - (raw_l1  * int(args.l2_window)))

        if not args.feynman and not spectrum:
            delta = raw_l1 - last_l1_count
            if delta > target_l1_growth:
                current_stride = max(1, int(current_stride * target_l1_growth / max(1, delta)))
            else:
                current_stride = min(500, int(current_stride * 1.1))
            last_l1_count = raw_l1

        # -------------------------------------------------------------------
        # MATHEMATICAL ALIGNMENT FIX: Relational Scale Factor
        # Tracking ONLY unconstrained space fabric pool vs. Matter-Free Background
        # -------------------------------------------------------------------
        active_elements = net_l0 
        total_scale     = max(0.001, (active_elements / n_max) * 2.0)

        # The pure GR vacuum baseline profile (unconstrained match count)
        vacuum_elements = raw_l0

        history_active.append(active_elements)
        history_vacuum.append(vacuum_elements)

        # -------------------------------------------------------------------
        # HUBBLE PARAMETER H(t)
        # -------------------------------------------------------------------
        ent_axis, H_vals = compute_hubble(
            engine.history_entropy + [entropy],
            history_active,
            n_max
        )

        # -------------------------------------------------------------------
        # VISUALIZATION
        # -------------------------------------------------------------------
        if args.feynman or spectrum:
            curr_l0_x = generate_even_coordinates(net_l0, total_scale)
            curr_l1_x = generate_even_coordinates(net_l1, total_scale)
            curr_l2_x = generate_even_coordinates(net_l2, total_scale)
            curr_l3_x = generate_even_coordinates(net_l3, total_scale)

            if current_time_tick > 1 and prev_l0_x is not None:
                for arr_p, arr_c, col, al, lw in [
                    (prev_l0_x, curr_l0_x, 'gray',    0.20, 0.8),
                    (prev_l1_x, curr_l1_x, 'cyan',    0.40, 1.0),
                    (prev_l2_x, curr_l2_x, 'magenta', 0.60, 1.2),
                    (prev_l3_x, curr_l3_x, 'lime',    0.80, 1.5),
                ]:
                    if arr_p.size > 0 and arr_c.size > 0:
                        for px in arr_p:
                            cx = arr_c[np.argmin(np.abs(arr_c - px))]
                            ax_spacetime.plot([prev_t, current_time_tick],
                                             [px, cx], color=col, alpha=al, lw=lw)

            prev_t    = current_time_tick
            prev_l0_x = curr_l0_x
            prev_l1_x = curr_l1_x
            prev_l2_x = curr_l2_x
            prev_l3_x = curr_l3_x

        else:
            def _append(t, pos_raw, t_list, r_list):
                norm = (pos_raw / n_max) - 0.5 if pos_raw.size > 0 else np.zeros(0)
                norm = np.abs(norm) * np.where(np.arange(norm.size) % 2 == 0, 1.0, -1.0)
                for v in norm:
                    t_list.append(t)
                    r_list.append((v + random.uniform(-0.5, 0.5) / n_max) * total_scale)

            _append(current_time_tick, l0_pos_raw, all_time_l0_t, all_time_l0_r)
            if args.recursion >= 1: _append(current_time_tick, l1_pos_raw, all_time_l1_t, all_time_l1_r)
            if args.recursion >= 2: _append(current_time_tick, l2_pos_raw, all_time_l2_t, all_time_l2_r)
            if args.recursion >= 3: _append(current_time_tick, l3_pos_raw, all_time_l3_t, all_time_l3_r)

            if all_time_l0_t: l0_scatter.set_offsets(np.column_stack((all_time_l0_t, all_time_l0_r)))
            if all_time_l1_t and args.recursion >= 1: l1_scatter.set_offsets(np.column_stack((all_time_l1_t, all_time_l1_r)))
            if all_time_l2_t and args.recursion >= 2: l2_scatter.set_offsets(np.column_stack((all_time_l2_t, all_time_l2_r)))
            if all_time_l3_t and args.recursion >= 3: l3_scatter.set_offsets(np.column_stack((all_time_l3_t, all_time_l3_r)))

        ax_spacetime.set_ylim(-max(0.05, total_scale * 0.55),
                               max(0.05, total_scale * 0.55))

        # Update metrics
        engine.history_timestamps.append(current_time_tick)
        engine.history_entropy.append(entropy)
        engine.history_g_counts.append(net_l0)
        engine.history_l1_counts.append(net_l1)
        engine.history_l2_counts.append(net_l2)
        engine.history_l3_counts.append(net_l3)

        t_axis = engine.history_timestamps
        line_g.set_data(t_axis,  engine.history_g_counts)
        line_l1.set_data(t_axis, engine.history_l1_counts)
        line_l2.set_data(t_axis, engine.history_l2_counts)
        line_l3.set_data(t_axis, engine.history_l3_counts)
        
        # Plot white vacuum baseline trend
        line_vacuum.set_data(t_axis, history_vacuum)
        
        line_entropy.set_data(t_axis, engine.history_entropy)

        if len(H_vals) > 1:
            H_display = H_vals / (np.max(np.abs(H_vals)) + 1e-12) * 0.5
            line_hubble.set_data(t_axis[:-1], H_display)

        ax_metrics.relim()
        ax_metrics.autoscale(enable=True, axis='both', tight=False)
        ax_entropy.relim()
        ax_entropy.autoscale_view(True, True, True)

        refresh_tick = 5 if (args.feynman or spectrum) else 1
        if current_time_tick % refresh_tick == 0:
            fig.canvas.draw_idle()
            plt.pause(0.001)

    plt.ioff()
    plt.show()
