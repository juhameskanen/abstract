"""
Time-Reversed Entropic Cosmology: Unified Edition
===================================================================
An informational ontology simulation utilizing a hidden background fabric 
of Gravitons to derive spatial coordinates relationally. Maps hierarchical 
matter spectrums symmetrically, featuring both fractional scatter tracing 
and topological proximity-linked Feynman vertex plotting.

Features:
- Juha Meskanen's Observer Matter Release Principle (Subsection 5.2)
- CONSERVED TOTAL SCALE: Spacetime horizon diameter derived relationally via 
  hierarchical structural token conservation logic.
- Dynamic instantaneous Y-axis scaling to stabilize relational worldline tracing.
- Dual-Mode Graphics Architecture: Choose between Scatter or Feynman Worldlines.

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
    """Slices raw universe to extract Level 0 space fabric elements."""
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
    def __init__(self, total_bits: int, pattern_str: str, l2_w: int, l2_t: int, l3_w: int, l3_t: int, l4_w: int, l4_t: int):
        self.total_bits = total_bits
        self.w = len(pattern_str)
        
        self.l2_w, self.l2_t = l2_w, l2_t  # L1 Hadrons
        self.l3_w, self.l3_t = l3_w, l3_t  # L2 Atoms
        self.l4_w, self.l4_t = l4_w, l4_t  # L3 Compounds
        
        self.target_val = 0
        for i, char in enumerate(pattern_str):
            if char == '1':
                self.target_val |= (1 << (self.w - 1 - i))
                
        self.bitfield = np.zeros(total_bits, dtype=np.uint8)
        self.step = 0  
        
        # Analytics History Tables
        self.history_timestamps = []  
        self.history_entropy = []
        self.history_g_counts = []
        self.history_l1_counts = []
        self.history_l2_counts = []
        self.history_l3_counts = []

    def get_layer_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Processes the bitfield relationally, pulling raw spatial index locations."""
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


def generate_even_coordinates(count, scale):
    """Distributes an exact count of indistinguishable particles evenly across the horizon."""
    if count <= 0:
        return np.zeros(0)
    if count == 1:
        return np.array([0.0])
    return np.linspace(-0.5, 0.5, count) * scale


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified IAME Cosmology Visualizer")
    parser.add_argument("--bits", type=int, default=10*1024, help="Total bitfield width")
    parser.add_argument("--pattern", type=str, default="10", help="Fabric target word")
    parser.add_argument("--quantum_mutations", type=int, default=10, help="Bit flips per tick")
    parser.add_argument("--max_time_steps", type=int, default=1000, help="Total timeline depth")
    parser.add_argument("--recursion", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Structural visibility horizon for scatter mode: 0=Fabric, 1=+Hadrons, 2=+Atoms, 3=+Compounds")
    
    # NEW FLAG: Controls visualization backend style
    parser.add_argument("--feynman", action="store_true", help="Enable proximity-linked Feynman worldline plotting")
    
    # Cascade configuration settings (Stably Fine-Tuned Defaults)
    parser.add_argument("--l2_window", type=int, default=8, help="Hadron window")
    parser.add_argument("--l2_threshold", type=int, default=5, help="Hadron threshold")
    parser.add_argument("--l3_window", type=int, default=6, help="Atom window")
    parser.add_argument("--l3_threshold", type=int, default=3, help="Atom threshold")
    parser.add_argument("--l4_window", type=int, default=8, help="Molecule window")
    parser.add_argument("--l4_threshold", type=int, default=3, help="Molecule threshold")
    args = parser.parse_args()

    engine = EntropicCosmologyEngine(
        args.bits, args.pattern, 
        args.l2_window, args.l2_threshold, 
        args.l3_window, args.l3_threshold,
        args.l4_window, args.l4_threshold
    )
    
    plt.ion()
    fig, (ax_spacetime, ax_metrics) = plt.subplots(1, 2, figsize=(16, 7))
    
    mode_title = "Proximity Feynman Lines" if args.feynman else "Fractional Jitter Scatter"
    fig.suptitle(f"IAME Cosmology Engine ({mode_title})", fontsize=16, fontweight='bold')
    
    # --- Left Panel Viewport Configuration ---
    ax_spacetime.set_xlim(0, args.max_time_steps)
    ax_spacetime.set_facecolor('#020205')
    ax_spacetime.set_xlabel("Chronological Cosmic Time (Ticks)")
    ax_spacetime.set_ylabel("Relational Spatial Coordinates")
    
    # Setup plotting assets based on chosen engine layout
    if args.feynman:
        ax_spacetime.set_title("Emergent Topological Feynman Space")
        # Dummy lines to create a clean, persistent legend for line plots
        ax_spacetime.plot([], [], color='gray', alpha=0.3, label="L0 Space Fabric")
        ax_spacetime.plot([], [], color='cyan', alpha=0.6, label="L1 Hadrons")
        ax_spacetime.plot([], [], color='magenta', alpha=0.7, label="L2 Atoms")
        ax_spacetime.plot([], [], color='lime', alpha=0.8, label="L3 Compounds")
    else:
        ax_spacetime.set_title("Linear Relational Spacetime Flow")
        l0_scatter = ax_spacetime.scatter([], [], s=1, color='gray', marker='.', alpha=0.10, label="L0 Space Fabric")
        l1_scatter = ax_spacetime.scatter([], [], s=1, color='cyan', marker='o', alpha=0.3, label="Visible L1 Hadrons" if args.recursion >= 1 else "")
        l2_scatter = ax_spacetime.scatter([], [], s=2, color='magenta', marker='*', alpha=0.5, label="Visible L2 Atoms" if args.recursion >= 2 else "")
        l3_scatter = ax_spacetime.scatter([], [], s=4, color='lime', marker='^', alpha=0.7, label="Visible L3 Compounds" if args.recursion >= 3 else "")
        
        all_time_l0_t, all_time_l0_r = [], []
        all_time_l1_t, all_time_l1_r = [], []
        all_time_l2_t, all_time_l2_r = [], []
        all_time_l3_t, all_time_l3_r = [], []
        
    ax_spacetime.legend(loc="upper left")
    
    # --- Right Panel Analytics Configuration ---
    line_g,  = ax_metrics.plot([], [], label="Space Fabric Ticks (L0)", color='gray', lw=1, linestyle=':')
    line_l1, = ax_metrics.plot([], [], label="Hadrons (L1)", color='cyan', lw=2)
    line_l2, = ax_metrics.plot([], [], label="Atoms (L2)", color='magenta', lw=2)
    line_l3, = ax_metrics.plot([], [], label="Compounds (L3)", color='lime', lw=2.5)
    ax_metrics.set_ylabel("Global Structural Tally")
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

    # Temporal execution pointers
    current_time_tick = 0
    last_l1_count = 0
    current_stride = 1  
    target_l1_growth_per_sample = 50 
    
    # Feynman tracking cache
    prev_t = 0
    prev_l0_x, prev_l1_x, prev_l2_x, prev_l3_x = None, None, None, None

    # --- COSMOLOGICAL TEMPORAL EVENT LOOP ---
    while current_time_tick < args.max_time_steps:
        if not plt.fignum_exists(fig.number):
            break
            
        # In Feynman mode we force stride=1 to capture perfectly continuous line segments
        stride_limit = 1 if args.feynman else current_stride
        
        for _ in range(stride_limit):
            engine.mutate_state(args.quantum_mutations)
            current_time_tick += 1
            if current_time_tick >= args.max_time_steps:
                break

        # Capture filtration mappings
        l0_pos_raw, l1_pos_raw, l2_pos_raw, l3_pos_raw = engine.get_layer_data()
        entropy = compute_entropy(engine.bitfield)
        
        # Safe integer cast to protect conservation calculations
        raw_l0 = int(l0_pos_raw.size)
        raw_l1 = int(l1_pos_raw.size)
        raw_l2 = int(l2_pos_raw.size)
        raw_l3 = int(l3_pos_raw.size)
        
        net_l3 = raw_l3
        net_l2 = max(0, raw_l2 - (net_l3 * int(args.l4_window)))
        net_l1 = max(0, raw_l1 - (raw_l2 * int(args.l3_window)))
        net_l0 = max(0, raw_l0 - (raw_l1 * int(args.l2_window)))
        
        # --- ADAPTIVE VELOCITY SAMPLING CONTROLLER (Scatter Mode Only) ---
        if not args.feynman:
            growth_delta = raw_l1 - last_l1_count
            if growth_delta > target_l1_growth_per_sample:
                reduction_ratio = target_l1_growth_per_sample / max(1, growth_delta)
                current_stride = max(1, int(current_stride * reduction_ratio))
            else:
                current_stride = min(500, int(current_stride * 1.1))
            last_l1_count = raw_l1
        
        # --- FIXED CONSERVED SCALE ENGINE ---
        active_elements = net_l0 + net_l1 + net_l2 + net_l3
        max_fabric_width = max(1.0, float(args.bits // engine.w))
        total_scale = (active_elements / max_fabric_width) * 2.0
        if total_scale <= 0: total_scale = 0.001
        
        # --- GRAPHICS RENDERING SWITCH PIPELINE ---
        if args.feynman:
            # Mode A: Distribute and plot linked line segments
            curr_l0_x = generate_even_coordinates(net_l0, total_scale)
            curr_l1_x = generate_even_coordinates(net_l1, total_scale)
            curr_l2_x = generate_even_coordinates(net_l2, total_scale)
            curr_l3_x = generate_even_coordinates(net_l3, total_scale)
            
            if current_time_tick > 1:
                if prev_l0_x.size > 0 and curr_l0_x.size > 0:
                    for px in prev_l0_x:
                        closest_cx = curr_l0_x[np.argmin(np.abs(curr_l0_x - px))]
                        ax_spacetime.plot([prev_t, current_time_tick], [px, closest_cx], color='gray', alpha=0.15, lw=0.5)
                if prev_l1_x.size > 0 and curr_l1_x.size > 0:
                    for px in prev_l1_x:
                        closest_cx = curr_l1_x[np.argmin(np.abs(curr_l1_x - px))]
                        ax_spacetime.plot([prev_t, current_time_tick], [px, closest_cx], color='cyan', alpha=0.4, lw=1.0)
                if prev_l2_x.size > 0 and curr_l2_x.size > 0:
                    for px in prev_l2_x:
                        closest_cx = curr_l2_x[np.argmin(np.abs(curr_l2_x - px))]
                        ax_spacetime.plot([prev_t, current_time_tick], [px, closest_cx], color='magenta', alpha=0.6, lw=1.2)
                if prev_l3_x.size > 0 and curr_l3_x.size > 0:
                    for px in prev_l3_x:
                        closest_cx = curr_l3_x[np.argmin(np.abs(curr_l3_x - px))]
                        ax_spacetime.plot([prev_t, current_time_tick], [px, closest_cx], color='lime', alpha=0.8, lw=1.5)
                        
            prev_t = current_time_tick
            prev_l0_x, prev_l1_x, prev_l2_x, prev_l3_x = curr_l0_x, curr_l1_x, curr_l2_x, curr_l3_x
            
        else:
            # Mode B: Normalize indices and append to historic scatter arrays
            norm_l0 = (l0_pos_raw / max_fabric_width) - 0.5 if l0_pos_raw.size > 0 else np.zeros(0)
            norm_l1 = (l1_pos_raw / max_fabric_width) - 0.5 if (l1_pos_raw.size > 0 and args.recursion >= 1) else np.zeros(0)
            norm_l2 = (l2_pos_raw / max_fabric_width) - 0.5 if (l2_pos_raw.size > 0 and args.recursion >= 2) else np.zeros(0)
            norm_l3 = (l3_pos_raw / max_fabric_width) - 0.5 if (l3_pos_raw.size > 0 and args.recursion >= 3) else np.zeros(0)
            
            if norm_l0.size > 0: norm_l0 = np.abs(norm_l0) * np.where(np.arange(norm_l0.size) % 2 == 0, 1.0, -1.0)
            if norm_l1.size > 0: norm_l1 = np.abs(norm_l1) * np.where(np.arange(norm_l1.size) % 2 == 0, 1.0, -1.0)
            if norm_l2.size > 0: norm_l2 = np.abs(norm_l2) * np.where(np.arange(norm_l2.size) % 2 == 0, 1.0, -1.0)
            if norm_l3.size > 0: norm_l3 = np.abs(norm_l3) * np.where(np.arange(norm_l3.size) % 2 == 0, 1.0, -1.0)
            
            for r_val in norm_l0:
                all_time_l0_t.append(current_time_tick)
                all_time_l0_r.append((r_val + random.uniform(-0.5, 0.5) / max_fabric_width) * total_scale)
            if args.recursion >= 1:
                for r_val in norm_l1:
                    all_time_l1_t.append(current_time_tick)
                    all_time_l1_r.append((r_val + random.uniform(-0.5, 0.5) / max_fabric_width) * total_scale)
            if args.recursion >= 2:
                for r_val in norm_l2:
                    all_time_l2_t.append(current_time_tick)
                    all_time_l2_r.append((r_val + random.uniform(-0.5, 0.5) / max_fabric_width) * total_scale)
            if args.recursion >= 3:
                for r_val in norm_l3:
                    all_time_l3_t.append(current_time_tick)
                    all_time_l3_r.append((r_val + random.uniform(-0.5, 0.5) / max_fabric_width) * total_scale)
                    
            if len(all_time_l0_t) > 0: l0_scatter.set_offsets(np.column_stack((all_time_l0_t, all_time_l0_r)))
            if len(all_time_l1_t) > 0 and args.recursion >= 1: l1_scatter.set_offsets(np.column_stack((all_time_l1_t, all_time_l1_r)))
            if len(all_time_l2_t) > 0 and args.recursion >= 2: l2_scatter.set_offsets(np.column_stack((all_time_l2_t, all_time_l2_r)))
            if len(all_time_l3_t) > 0 and args.recursion >= 3: l3_scatter.set_offsets(np.column_stack((all_time_l3_t, all_time_l3_r)))

        # --- VIEWPORT WINDOW DYNAMIC RE-BOUNDING ---
        current_max_bound = max(0.05, total_scale * 0.55)
        ax_spacetime.set_ylim(-current_max_bound, current_max_bound)

        # Map Analytics History logs
        engine.history_timestamps.append(current_time_tick)
        engine.history_entropy.append(entropy)
        engine.history_g_counts.append(net_l0)
        engine.history_l1_counts.append(net_l1)
        engine.history_l2_counts.append(net_l2)
        engine.history_l3_counts.append(net_l3)
        
        t_axis = engine.history_timestamps
        line_g.set_data(t_axis, engine.history_g_counts)
        line_l1.set_data(t_axis, engine.history_l1_counts)
        line_l2.set_data(t_axis, engine.history_l2_counts)
        line_l3.set_data(t_axis, engine.history_l3_counts)
        line_entropy.set_data(t_axis, engine.history_entropy)
        
        ax_metrics.relim()
        ax_metrics.autoscale(enable=True, axis='y', tight=False)
        ax_metrics.autoscale(enable=True, axis='x', tight=True)
        ax_entropy.relim()
        ax_entropy.autoscale_view(True, True, True)
        
        # Render update refresh rates based on active mode properties
        refresh_tick = 5 if args.feynman else 1
        if current_time_tick % refresh_tick == 0:
            fig.canvas.draw_idle()
            plt.pause(0.001)

    plt.ioff()
    plt.show()
