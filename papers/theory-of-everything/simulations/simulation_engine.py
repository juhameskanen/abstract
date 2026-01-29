"""
simulations_engine.py

Base classes for observer-filtered Gibbs simulations.

This module defines a SimulationEngine class that encapsulates common methods:
- history initialization
- Metropolis-Hastings updates
- observer projections
- animation rendering
- command-line argument parsing

Derived classes implement dimension-specific behavior:
- ObserverGibbs1D: 1D bitstring histories
- ObserverGibbs2D: 2D lattice histories

All classes maintain docstrings and type annotations for clarity.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Any, Tuple


class SimulationEngine:
    """
    Base class for observer-filtered Gibbs simulations.
    """

    def __init__(self, time_steps: int, lambda_: float, res: int = 120, 
                 window: int = 1, alpha: float = 0.0) -> None:
        """
        Initialize simulation parameters.

        Args:
            time_steps: Number of time steps T
            lambda_: Gibbs parameter λ
            res: DPI for animation
            window: Observer coarse-graining window
            alpha: Optional spatial smoothness weight
        """
        self.T = time_steps
        self.lambda_ = lambda_
        self.res = res
        self.window = window
        self.alpha = alpha
        self.history: Any = None
        self.coarse_history: Any = None

    def initialize_history(self) -> None:
        """Initialize the simulation history. To be implemented in subclasses."""
        raise NotImplementedError

    def complexity(self, t: int) -> float:
        """
        Compute complexity for time step t. To be implemented in subclasses.
        """
        raise NotImplementedError

    def metropolis_step(self) -> None:
        """
        Perform one Metropolis-Hastings update.
        """
        raise NotImplementedError

    def observer_projection(self) -> Any:
        """
        Perform observer coarse-graining / projection.
        """
        raise NotImplementedError

    def animate(self, steps: int, filename: str, fmt: str = 'gif', fps: int = 12) -> None:
        """
        Animate the simulation and save as GIF or MP4.

        Args:
            steps: Number of frames to animate
            filename: Output file name (without extension)
            fmt: 'gif' or 'mp4'
            fps: Frames per second
        """
        if fmt == 'gif':
            sim_res = min(self.res, 60)
            writer_type = 'pillow'
            output_name = f"{filename}.gif"
        else:
            sim_res = self.res
            writer_type = 'ffmpeg'
            output_name = f"{filename}.mp4"

        fig, axes = self._setup_plot(dpi=sim_res)
        ani = animation.FuncAnimation(fig, self._update_plot, frames=steps,
                                      interval=1000//fps, blit=True)
        writer = animation.writers[writer_type](fps=fps)
        ani.save(output_name, writer=writer)
        print(f"Saved animation to {output_name}")

    def _setup_plot(self, dpi: int) -> Tuple[Any, Any]:
        """
        Setup matplotlib figure and axes. Returns figure and axes tuple.
        To be implemented in subclass.
        """
        raise NotImplementedError

    def _update_plot(self, frame: int) -> Any:
        """
        Update function for animation.FuncAnimation.
        To be implemented in subclass.
        """
        raise NotImplementedError

