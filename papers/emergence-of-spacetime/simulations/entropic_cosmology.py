"""
Time-Reversed Entropic Cosmology: Graviton Space Fabric Metric
===================================================================
An informational ontology simulation utilizing a hidden background fabric 
of Gravitons to derive spatial coordinates relationally, mapping 
hierarchical matter spectrums (Hadrons, Atoms, Compounds) symmetrically.

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
    """Slices the raw universe to extract Level 1 Gravitons (Spacetime Fabric)."""
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
        
    # Step forward by the window_size, not 1
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
        
        self.history_entropy = []
        self.history_g_counts = []
        self.history_l1_counts = []
        self.history_l2_counts = []
        self.history_l3_counts = []


    def extract_graviton_metric_layers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Processes the bitfield and realigns compressed indices back to the Graviton metric."""
        # --- Filter passes ---
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
            # 1. Realignment: Scale compressed indices back up to Graviton coordinate equivalents
            l1_realigned = l1_indices * self.l2_w
            l2_realigned = l2_indices * (self.l2_w * self.l3_w)
            l3_realigned = l3_indices * (self.l2_w * self.l3_w * self.l4_w)
            
            # 2. Universal Horizon Normalization (Shared Scale Ceiling)
            spatial_ceiling = max(1.0, np.max(g_indices))
            
            norm_l1 = (l1_realigned / spatial_ceiling) - 0.5 if l1_indices.size > 0 else np.zeros(0)
            norm_l2 = (l2_realigned / spatial_ceiling) - 0.5 if l2_indices.size > 0 else np.zeros(0)
            norm_l3 = (l3_realigned / spatial_ceiling) - 0.5 if l3_indices.size > 0 else np.zeros(0)
        else:
            norm_l1, norm_l2, norm_l3 = np.zeros(0), np.zeros(0), np.zeros(0)
        
        return norm_l1, norm_l2, norm_l3


    def mutate_state(self, mutations_per_step: int):
        for _ in range(mutations_per_step):
            idx = random.randint(0, self.total_bits - 1)
            self.bitfield[idx] ^= 1
        self.step += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IAME Graviton Metric Space Simulation")
    parser.add_argument("--bits", type=int, default=131072, help="Total bitfield width")
    parser.add_argument("--pattern", type=str, default="0100", help="Graviton target word")
    parser.add_argument("--mutations", type=int, default=1024, help="Bit flips per frame")
    parser.add_argument("--frames", type=int, default=500, help="Total execution duration")
    
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
    fig.suptitle("IAME Graviton Space Fabric & Relational Matter Spectrum", fontsize=16, fontweight='bold')
    
    # --- Left Panel: Minkowski Chart ---
    ax_spacetime.set_xlim(-10, args.frames + 10)
    ax_spacetime.set_ylim(-0.55, 0.55)
    ax_spacetime.set_facecolor('#020205')
    ax_spacetime.set_title("Relational Spacetime Flow (Scaled to Hidden Gravitons)")
    ax_spacetime.set_xlabel("Temporal Axis (Steps)")
    ax_spacetime.set_ylabel("Spatial Position (Normalized to Graviton Horizon Ceiling)")
    
    l1_scatter = ax_spacetime.scatter([], [], s=12, color='cyan', marker='o', alpha=0.5, label="Visible L1 Hadrons")
    l2_scatter = ax_spacetime.scatter([], [], s=40, color='magenta', marker='*', alpha=0.7, label="Visible L2 Atoms")
    l3_scatter = ax_spacetime.scatter([], [], s=90, color='lime', marker='^', alpha=0.9, label="Visible L3 Compounds")
    ax_spacetime.legend(loc="upper left")
    
    all_time_l1_t, all_time_l1_r = [], []
    all_time_l2_t, all_time_l2_r = [], []
    all_time_l3_t, all_time_l3_r = [], []
    
    # --- Right Panel: Analytics Dash ---
    line_g,  = ax_metrics.plot([], [], label="Hidden Gravitons Space Fabric", color='gray', lw=1, linestyle=':')
    line_l1, = ax_metrics.plot([], [], label="L1 Hadrons", color='cyan', lw=2)
    line_l2, = ax_metrics.plot([], [], label="L2 Atoms", color='magenta', lw=2)
    line_l3, = ax_metrics.plot([], [], label="L3 Compounds", color='lime', lw=2.5)
    ax_metrics.set_ylabel("Structural Quantities")
    ax_metrics.set_xlabel("Evolutionary Step")
    ax_metrics.set_ylim(bottom=0)
    
    ax_entropy = ax_metrics.twinx()
    line_entropy, = ax_entropy.plot([], [], label="Shannon Entropy", color='red', linestyle='--', lw=1.5)
    ax_entropy.set_ylabel("Entropy (bits)", color='red')
    ax_entropy.set_ylim(0, 1.05)
    
    lines = [line_g, line_l1, line_l2, line_l3, line_entropy]
    labels = [l.get_label() for l in lines]
    ax_metrics.legend(lines, labels, loc="upper left")
    
    plt.tight_layout()

    # --- COSMOLOGICAL TEMPORAL EVENT LOOP ---
    for frame in range(args.frames):
        if not plt.fignum_exists(fig.number):
            break
            
        if frame > 0:
            engine.mutate_state(args.mutations)
            
        l1_pos, l2_pos, l3_pos = engine.extract_graviton_metric_layers()
        current_entropy = compute_entropy(engine.bitfield)
        density_scale = current_entropy if current_entropy > 0 else 0.001
        
        # Accumulate structural tracks over spacetime history
        for r_val in l1_pos:
            all_time_l1_t.append(frame)
            all_time_l1_r.append(r_val * density_scale)
            
        for r_val in l2_pos:
            all_time_l2_t.append(frame)
            all_time_l2_r.append(r_val * density_scale)
            
        for r_val in l3_pos:
            all_time_l3_t.append(frame)
            all_time_l3_r.append(r_val * density_scale)
            
        # Update render offsets cleanly
        if len(all_time_l1_t) > 0:
            l1_scatter.set_offsets(np.column_stack((all_time_l1_t, all_time_l1_r)))
        else:
            l1_scatter.set_offsets(np.empty((0, 2)))
            
        if len(all_time_l2_t) > 0:
            l2_scatter.set_offsets(np.column_stack((all_time_l2_t, all_time_l2_r)))
        else:
            l2_scatter.set_offsets(np.empty((0, 2)))
            
        if len(all_time_l3_t) > 0:
            l3_scatter.set_offsets(np.column_stack((all_time_l3_t, all_time_l3_r)))
        else:
            l3_scatter.set_offsets(np.empty((0, 2)))
        

        # 1. Append the current snapshot values to history first
        engine.history_entropy.append(current_entropy)
        engine.history_g_counts.append(engine.current_g_count)
        engine.history_l1_counts.append(l1_pos.size)
        engine.history_l2_counts.append(l2_pos.size)
        engine.history_l3_counts.append(l3_pos.size)
        
        # 2. Rebuild the X-axis timeline
        steps_axis = np.arange(len(engine.history_g_counts))
        
        # 3. Push the complete historical sequences to the Matplotlib lines
        line_g.set_data(steps_axis, engine.history_g_counts)
        line_l1.set_data(steps_axis, engine.history_l1_counts)
        line_l2.set_data(steps_axis, engine.history_l2_counts)
        line_l3.set_data(steps_axis, engine.history_l3_counts)
        line_entropy.set_data(steps_axis, engine.history_entropy) # <-- Fixed to sequence
        
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
