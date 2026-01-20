import random
import zlib
import matplotlib.pyplot as plt

def generate_moving_bit_universe(n):
    """
    Creates a 'smooth' universe where a single bit 
    moves from LSB to MSB repeatedly.
    """
    universe = []
    for _ in range(100): # 100 time steps
        for i in range(n):
            # Create a bitstring with only the i-th bit set
            bit_state = 1 << i
            # Format as binary string for visualization/compression
            universe.append(format(bit_state, f'0{n}b'))
    return universe

def generate_chaotic_universe(n):
    """
    Creates a 'chaotic' universe where bit configurations
    are chosen at random (High Kolmogorov Complexity).
    """
    universe = []
    all_possible_states = [format(i, f'0{n}b') for i in range(2**n)]
    for _ in range(100 * n):
        universe.append(random.choice(all_possible_states))
    return universe

def calculate_complexity(universe_list):
    """
    Uses zlib compression to estimate the information 
    density/description length of the universe.
    """
    data = "".join(universe_list).encode('utf-8')
    compressed = zlib.compress(data)
    return len(data), len(compressed)

# Parameters
n_bits = 8

# Run Simulation
ordered_u = generate_moving_bit_universe(n_bits)
chaotic_u = generate_chaotic_universe(n_bits)

# Analyze
raw_size_o, comp_size_o = calculate_complexity(ordered_u)
raw_size_c, comp_size_c = calculate_complexity(chaotic_u)

print(f"--- Universe Analysis (n={n_bits}) ---")
print(f"Ordered Universe (Moving Bit):")
print(f"  Raw size: {raw_size_o} bytes | Compressed: {comp_size_o} bytes")
print(f"  Compression Ratio: {raw_size_o / comp_size_o:.2f}x")
print(f"\nChaotic Universe (Random):")
print(f"  Raw size: {raw_size_c} bytes | Compressed: {comp_size_c} bytes")
print(f"  Compression Ratio: {raw_size_c / comp_size_c:.2f}x")

# Quick Visualization of the first 20 steps
print("\nVisualizing First 20 Time-Steps of Ordered Universe:")
print("\n".join(ordered_u[:20]))