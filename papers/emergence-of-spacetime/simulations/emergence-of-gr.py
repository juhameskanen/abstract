import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Planck constants for human-readable conversion
PLANCK_TIME_S = 5.391247e-44    # seconds
PLANCK_LENGTH_M = 1.616255e-35  # meters

def format_large_number(num):
    if num < 1e6:
        return f"{num:.2f}"
    elif num < 1e12:
        return f"{num/1e9:.2f} billion"
    else:
        return f"{num:.2e}"

def generate_entropy_curve(n, points=500):
    t_max = n * math.log(n) * 1.2
    t = np.linspace(0, t_max, points)
    # Sigmoid-like saturation modeling the random walk entropy growth
    entropy = n * (1 - np.exp(-2 * t / n))
    entropy = np.minimum(entropy, n)
    return t, entropy

def create_universe_figure(n):
    t_planck, entropy = generate_entropy_curve(n)
    
    t_max_planck = n * math.log(n)
    t_max_sec = t_max_planck * PLANCK_TIME_S
    t_max_years = t_max_sec / (3600 * 24 * 365.25)
    
    spatial_planck = 2 ** n
    spatial_meters = spatial_planck * PLANCK_LENGTH_M
    spatial_ly = spatial_meters / 9.46073e15  # light years
    
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=("Entropy Growth (Random Bit Flips)", 
                                       "Key Scales of the Informational Universe"),
                        vertical_spacing=0.18)
    
    # Entropy curve (barapola-like)
    fig.add_trace(go.Scatter(x=t_planck, y=entropy, mode='lines', 
                           name='Entropy S(t)', line=dict(color='royalblue', width=3)),
                  row=1, col=1)
    
    fig.update_xaxes(title="Time (bit flips)", row=1, col=1)
    fig.update_yaxes(title="Entropy (bits)", row=1, col=1)
    
    # Scales
    fig.add_annotation(
        text=f"<b>Maximum Time:</b><br>"
             f"{t_max_planck:.2e} Planck time units<br>"
             f"≈ {format_large_number(t_max_sec)} seconds<br>"
             f"≈ {t_max_years:.2e} years",
        xref="paper", yref="paper", x=0.5, y=0.78,
        showarrow=False, font=dict(size=15), align="center",
        row=2, col=1
    )
    
    fig.add_annotation(
        text=f"<b>Spatial Size:</b><br>"
             f"{spatial_planck:.2e} Planck lengths<br>"
             f"≈ {format_large_number(spatial_meters)} meters<br>"
             f"≈ {spatial_ly:.2e} light-years",
        xref="paper", yref="paper", x=0.5, y=0.35,
        showarrow=False, font=dict(size=15), align="center",
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"Informational Universe — n = {n:,} bits  |  c = 1 (naturally emergent)",
        height=720,
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

# =============== RUN THE VISUALIZER ===============
if __name__ == "__main__":
    print("=== Informational Universe Visualizer ===")
    # Try different n values (higher n = bigger numbers!)
    for n in [30, 60, 100, 140]:
        print(f"\n→ Generating for n = {n} bits...")
        fig = create_universe_figure(n)
        fig.show()
