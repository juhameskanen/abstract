"""
Time-Reversed Entropic Cosmology: Graviton Space Fabric Metric
===================================================================
An informational ontology simulation utilizing a hidden background fabric 
of Gravitons to derive spatial coordinates relationally, mapping 
hierarchical matter spectrums (Hadrons, Atoms, Compounds) symmetrically.

Features:
- Juha Meskanen's Observer Matter Release Principle (Subsection 5.2)
- Integrated fractional jittering for ultra-smooth sub-grid resolution
- FIXED INFLATIONARY GEOMETRY: Dynamic instantaneous Y-axis scaling completely
  eliminates historical scale trapping and nested cylinder artifacts.
- RECURSION FILTERING: Dynamic --recursion visibility limits (0 to 3).
- L0 FABRIC VISUALIZATION: Maps background space fabric coordinates.
- CONSERVED TOTAL SCALE: Spacetime horizon diameter derived relationally via 
  hierarchical structural token conservation logic.

Copyright 2026 - Juha Meskanen, The Abstract Universe Project
"""

import random
import argparse
import numpy as np

# Force TkAgg for clean window handling inside virtual environments
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass

import matplotlib.pyplot as plt
from numba import njit

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
    """Slices the raw universe to extract Level 0 Gravitons (Spacetime Fabric)."""
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
    """Scans the lower-level trace in non-overlapping blocks to prevent structural double-counting."""
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
    def __init__(self, total_bits: int, pattern_str: str, l2_w: int, l2_t: int, l3_w: int, l3_t: int, l4_w: int, l4_t: int):
        self.total_bits = total_bits
        self.w = len(pattern_str)
        
        # Configuration levers for the recursive multi-scale observer
        self.l2_w, self.l2_t = l2_w, l2_t  # Hadron scale
        self.l3_w, self.l3_t = l3_w, l3_t  # Atom scale
        self.l4_w, self.l4_t = l4_w, l4_t  # Molecule scale
        
        self.target_val = 0
        for i, char in enumerate(pattern_str):
            if char == '1':
                self.target_val |= (1 << (self.w - 1 - i))
                
        # H=0 Singularity state
        self.bitfield = np.zeros(total_bits, dtype=np.uint8)
        self.step = 0  
        
        # Metric timelines
        self.history_timestamps = []  
        self.history_entropy = []
        self.history_g_counts = []
        self.history_l1_counts = []
        self.history_l2_counts = []
        self.history_l3_counts = []


    def extract_graviton_metric_layers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Processes the bitfield relationally, pulling raw configuration arrays."""
        s_g = run_graviton_filter(self.bitfield, self.w, self.target_val)
        g_indices = np.where(s_g == 1)[0].astype(np.float64)
        
        s_l1 = run_recursive_density_filter(s_g, window_size=self.l2_w, threshold=self.l2_t)
        l1_indices = np.where(s_l1 == 1)[0].astype(np.float64)
        
        s_l2 = run_recursive_density_filter(s_l1, window_size=self.l3_w, threshold=self.l3_t)
        l2_indices = np.where(s_l2 == 1)[0].astype(np.float64)
        
        s_l3 = run_recursive_density_filter(s_l2, window_size=self.l4_w, threshold=self.l4_t)
        l3_indices = np.where(s_l3 == 1)[0].astype(np.float64)
        
        self.current_g_count = g_indices.size
        
        if g_indices.size > 0:
            l0_realigned = g_indices
            l1_realigned = l1_indices * self.l2_w
            l2_realigned = l2_indices * (self.l2_w * self.l3_w)
            l3_realigned = l3_indices * (self.l2_w * self.l3_w * self.l4_w)
        else:
            l0_realigned, l1_realigned, l2_realigned, l3_realigned = np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0)
        
        return l0_realigned, l1_realigned, l2_realigned, l3_realigned


    def mutate_state(self, mutations_per_step: int):
        for _ in range(mutations_per_step):
            idx = random.randint(0, self.total_bits - 1)
            self.bitfield[idx] ^= 1
        self.step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IAME Graviton Metric Space Simulation")
    parser.add_argument("--bits", type=int, default=10*1024, help="Total bitfield width")
    parser.add_argument("--pattern", type=str, default="10", help="Graviton target word")
    parser.add_argument("--quantum_mutations", type=int, default=10, help="Absolute time unit (bit flips per tick)")
    parser.add_argument("--max_time_steps", type=int, default=1000, help="Total chronological depth")
    parser.add_argument("--recursion", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Structural visibility horizon: 0=Fabric, 1=+Hadrons, 2=+Atoms, 3=+Compounds")
    
    # Cascade configuration settings
    parser.add_argument("--l2_window", type=int, default=4, help="Hadron window")
    parser.add_argument("--l2_threshold", type=int, default=2, help="Hadron density threshold")
    parser.add_argument("--l3_window", type=int, default=6, help="Atom window")
    parser.add_argument("--l3_threshold", type=int, default=2, help="Atom density threshold")
    parser.add_argument("--l4_window", type=int, default=10, help="Molecule window")
    parser.add_argument("--l4_threshold", type=int, default=3, help="Molecule density threshold")
    args = parser.parse_args()

    engine = EntropicCosmologyEngine(
        args.bits, args.pattern, 
        args.l2_window, args.l2_threshold, 
        args.l3_window, args.l3_threshold,
        args.l4_window, args.l4_threshold
    )
    
    plt.ion()
    fig, (ax_spacetime, ax_metrics) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("IAME Graviton Space Fabric (True Linear-Time Adaptive Simulation)", fontsize=16, fontweight='bold')
    
    # --- Left Panel: Minkowski Chart ---
    ax_spacetime.set_xlim(0, args.max_time_steps)
    ax_spacetime.set_ylim(-1.5, 1.5)  # Set a stable baseline spatial view bounds
    ax_spacetime.set_facecolor('#020205')
    ax_spacetime.set_title("Linear Relational Spacetime Flow")
    ax_spacetime.set_xlabel("Chronological Cosmic Time (Ticks)")
    ax_spacetime.set_ylabel("Spatial Position (Normalized Radius)")
    
    # Setup layers on the viewport conditionally based on visibility thresholds
    l0_scatter = ax_spacetime.scatter([], [], s=1, color='gray', marker='.', alpha=0.10, label="L0 Space Fabric")
    l1_scatter = ax_spacetime.scatter([], [], s=1, color='cyan', marker='o', alpha=0.3, label="Visible L1 Hadrons" if args.recursion >= 1 else "")
    l2_scatter = ax_spacetime.scatter([], [], s=2, color='magenta', marker='*', alpha=0.5, label="Visible L2 Atoms" if args.recursion >= 2 else "")
    l3_scatter = ax_spacetime.scatter([], [], s=4, color='lime', marker='^', alpha=0.7, label="Visible L3 Compounds" if args.recursion >= 3 else "")
    ax_spacetime.legend(loc="upper left")
    
    all_time_l0_t, all_time_l0_r = [], []
    all_time_l1_t, all_time_l1_r = [], []
    all_time_l2_t, all_time_l2_r = [], []
    all_time_l3_t, all_time_l3_r = [], []

    # --- Right Panel: Analytics Dash ---
    line_g,  = ax_metrics.plot([], [], label="Hidden Gravitons Space Fabric", color='gray', lw=1, linestyle=':')
    line_l1, = ax_metrics.plot([], [], label="L1 Hadrons", color='cyan', lw=2)
    line_l2, = ax_metrics.plot([], [], label="L2 Atoms", color='magenta', lw=2)
    line_l3, = ax_metrics.plot([], [], label="L3 Compounds", color='lime', lw=2.5)
    ax_metrics.set_ylabel("Structural Quantities")
    ax_metrics.set_xlabel("Chronological Cosmic Time (Ticks)")
    ax_metrics.set_ylim(bottom=0)
    
    ax_entropy = ax_metrics.twinx()
    line_entropy, = ax_entropy.plot([], [], label="Shannon Entropy", color='red', linestyle='--', lw=1.5)
    ax_entropy.set_ylabel("Entropy (bits)", color='red')
    ax_entropy.set_ylim(0, 1.05)
    
    lines = [line_g, line_l1, line_l2, line_l3, line_entropy]
    labels = [l.get_label() for l in lines]
    ax_metrics.legend(lines, labels, loc="upper left")
    
    plt.tight_layout()

    # --- CHRONOLOGICAL TIMESTEP VARIABLES ---
    current_time_tick = 0
    last_l1_count = 0

    current_stride = 1  
    target_l1_growth_per_sample = 50 

    # Capture baseline state metrics at Tick 0
    l0_pos_raw, l1_pos_raw, l2_pos_raw, l3_pos_raw = engine.extract_graviton_metric_layers()
    current_entropy = compute_entropy(engine.bitfield)
    
    engine.history_timestamps.append(0)
    engine.history_entropy.append(current_entropy)
    engine.history_g_counts.append(engine.current_g_count)
    engine.history_l1_counts.append(l1_pos_raw.size)
    engine.history_l2_counts.append(l2_pos_raw.size)
    engine.history_l3_counts.append(l3_pos_raw.size)
    
    # --- COSMOLOGICAL TEMPORAL EVENT LOOP ---
    while current_time_tick < args.max_time_steps:
        if not plt.fignum_exists(fig.number):
            break
            
        for _ in range(current_stride):
            engine.mutate_state(args.quantum_mutations)
            current_time_tick += 1
            if current_time_tick >= args.max_time_steps:
                break

        # Snapshot tracking at this specific timestamp
        l0_pos_raw, l1_pos_raw, l2_pos_raw, l3_pos_raw = engine.extract_graviton_metric_layers()
        current_entropy = compute_entropy(engine.bitfield)
        
        # --- ADAPTIVE SAMPLING VELOCITY CONTROLLER ---
        current_l1_count = l1_pos_raw.size
        growth_delta = current_l1_count - last_l1_count
        
        if growth_delta > target_l1_growth_per_sample:
            reduction_ratio = target_l1_growth_per_sample / max(1, growth_delta)
            current_stride = max(1, int(current_stride * reduction_ratio))
        else:
            current_stride = min(500, int(current_stride * 1.1))
            
        last_l1_count = current_l1_count
        
        # --- FIXED RELATIONAL DIAMETER ENGINE ---
        # 1. Capture raw, uncorrected entity numbers straight from digital filters
        raw_g  = int(l0_pos_raw.size)
        raw_l1 = int(l1_pos_raw.size)
        raw_l2 = int(l2_pos_raw.size)
        raw_l3 = int(l3_pos_raw.size)
        
        # 2. Compute true cascade conservation bounds (higher structures swallow background blocks)
        net_l3 = raw_l3
        net_l2 = max(0, raw_l2 - (net_l3 * int(args.l4_window)))
        net_l1 = max(0, raw_l1 - (raw_l2 * int(args.l3_window)))
        net_l0 = max(0, raw_g  - (raw_l1 * int(args.l2_window)))
        
        # 3. Aggregate unique observable positions conforming to active --recursion thresholds
        active_elements = net_l0
        if args.recursion >= 1: active_elements += net_l1
        if args.recursion >= 2: active_elements += net_l2
        if args.recursion >= 3: active_elements += net_l3
        
        # 4. Determine total scale based on addressable entities relative to base resolution ceiling
        max_fabric_width = max(1.0, float(args.bits // engine.w))
        total_scale = (active_elements / max_fabric_width) * 2.0
        if total_scale <= 0: total_scale = 0.001
        
        # Step 1: Calculate relative distance maps centering all tiers systematically
        norm_l0 = (l0_pos_raw / max_fabric_width) - 0.5 if l0_pos_raw.size > 0 else np.zeros(0)
        norm_l1 = (l1_pos_raw / max_fabric_width) - 0.5 if (l1_pos_raw.size > 0 and args.recursion >= 1) else np.zeros(0)
        norm_l2 = (l2_pos_raw / max_fabric_width) - 0.5 if (l2_pos_raw.size > 0 and args.recursion >= 2) else np.zeros(0)
        norm_l3 = (l3_pos_raw / max_fabric_width) - 0.5 if (l3_pos_raw.size > 0 and args.recursion >= 3) else np.zeros(0)
        
        # Step 2: Symmetric coordinate distribution via index parity polar flips
        if norm_l0.size > 0:
            signs_l0 = np.where(np.arange(norm_l0.size) % 2 == 0, 1.0, -1.0)
            norm_l0 = np.abs(norm_l0) * signs_l0
            
        if norm_l1.size > 0:
            signs_l1 = np.where(np.arange(norm_l1.size) % 2 == 0, 1.0, -1.0)
            norm_l1 = np.abs(norm_l1) * signs_l1
            
        if norm_l2.size > 0:
            signs_l2 = np.where(np.arange(norm_l2.size) % 2 == 0, 1.0, -1.0)
            norm_l2 = np.abs(norm_l2) * signs_l2
            
        if norm_l3.size > 0:
            signs_l3 = np.where(np.arange(norm_l3.size) % 2 == 0, 1.0, -1.0)
            norm_l3 = np.abs(norm_l3) * signs_l3
        
        # Append space fabric (L0) trace mappings
        for r_val in norm_l0:
            all_time_l0_t.append(current_time_tick)
            jitter = random.uniform(-0.5, 0.5) / max_fabric_width
            all_time_l0_r.append((r_val + jitter) * total_scale)

        # Append matter trajectories based on active visibility window settings
        if args.recursion >= 1:
            for r_val in norm_l1:
                all_time_l1_t.append(current_time_tick)
                jitter = random.uniform(-0.5, 0.5) / max_fabric_width
                all_time_l1_r.append((r_val + jitter) * total_scale)
                
        if args.recursion >= 2:
            for r_val in norm_l2:
                all_time_l2_t.append(current_time_tick)
                jitter = random.uniform(-0.5, 0.5) / max_fabric_width
                all_time_l2_r.append((r_val + jitter) * total_scale)
                
        if args.recursion >= 3:
            for r_val in norm_l3:
                all_time_l3_t.append(current_time_tick)
                jitter = random.uniform(-0.5, 0.5) / max_fabric_width
                all_time_l3_r.append((r_val + jitter) * total_scale)
                
        # Send updated offsets straight to the live viewports
        if len(all_time_l0_t) > 0:
            l0_scatter.set_offsets(np.column_stack((all_time_l0_t, all_time_l0_r)))
        if len(all_time_l1_t) > 0 and args.recursion >= 1:
            l1_scatter.set_offsets(np.column_stack((all_time_l1_t, all_time_l1_r)))
        if len(all_time_l2_t) > 0 and args.recursion >= 2:
            l2_scatter.set_offsets(np.column_stack((all_time_l2_t, all_time_l2_r)))
        if len(all_time_l3_t) > 0 and args.recursion >= 3:
            l3_scatter.set_offsets(np.column_stack((all_time_l3_t, all_time_l3_r)))
        
        # --- FIXED VIEWPORT DYNAMIC ADJUSTMENT ---
        current_max_bound = max(0.05, total_scale * 0.55)
        ax_spacetime.set_ylim(-current_max_bound, current_max_bound)

        # Track true metrics against timestamps
        engine.history_timestamps.append(current_time_tick)
        engine.history_entropy.append(current_entropy)
        engine.history_g_counts.append(engine.current_g_count)
        engine.history_l1_counts.append(current_l1_count)
        engine.history_l2_counts.append(l2_pos_raw.size)
        engine.history_l3_counts.append(l3_pos_raw.size)
        
        # Plot lines directly against non-uniform timestamp checkpoints
        t_axis = engine.history_timestamps
        line_g.set_data(t_axis, engine.history_g_counts)
        line_l1.set_data(t_axis, engine.history_l1_counts)
        line_l2.set_data(t_axis, engine.history_l2_counts)
        line_l3.set_data(t_axis, engine.history_l3_counts)
        line_entropy.set_data(t_axis, engine.history_entropy)
        
        ax_metrics.relim()
        ax_metrics.autoscale(enable=True, axis='y', tight=False)
        ax_metrics.autoscale(enable=True, axis='x', tight=True)
        ax_metrics.set_ylim(bottom=0)
        
        ax_entropy.relim()
        ax_entropy.autoscale_view(True, True, True)
        ax_entropy.set_ylim(0, 1.05)
        
        fig.canvas.draw_idle()
        plt.pause(0.001)

    plt.ioff()
    plt.show()
