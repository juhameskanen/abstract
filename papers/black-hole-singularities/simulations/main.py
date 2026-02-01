"""
Collapsing Dust Cloud Black Hole Simulation (Sequential Version)
================================================================

Runs Schwarzschild, Kerr, and AU dust cloud simulations sequentially.
Avoids multiprocessing to prevent numpy/fork issues.
"""

from __future__ import annotations
import argparse

from kerr import KerrIEFEquatorialGeodesicCloud
from schwarzschild import SchwarzschildEFGeodesicCloud
from schwarzschild_au import SchwarzschildAUGeodesicCloud
from blackhole import BlackHole, DustCloudSimulation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collapsing dust cloud black hole simulation "
                    "with entropy tracking and visualization."
    )
    parser.add_argument("--num_particles", type=int, default=100, help="Number of dust particles in the cloud")
    parser.add_argument("--mass", type=float, default=1.0, help="Black hole mass parameter")
    parser.add_argument("--spin", type=float, default=0.9, help="Black hole spin parameter a")
    parser.add_argument("--r0", type=float, default=6.0, help="Initial radial distance of cloud center")
    parser.add_argument("--spacing", type=float, default=0.02, help="Initial spacing between particles")
    parser.add_argument("--tangential_fraction", type=float, default=0.9, help="Fraction of initial velocity that is tangential")
    parser.add_argument("--radial_fraction", type=float, default=0.1, help="Fraction of initial velocity that is radial")
    parser.add_argument("--dt", type=float, default=1e-3, help="Time step for integration")
    parser.add_argument("--max_t", type=float, default=20.0, help="Maximum simulation time")
    parser.add_argument("--tolerance", type=float, default=1e-6, help="Integration tolerance")
    parser.add_argument("--no_visual", action="store_true", help="Disable visualization")
    parser.add_argument("--animate", action="store_true", help="Render MP4 animation of collapse")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for animation")
    parser.add_argument("--elev", type=float, default=25.0, help="Elevation angle for 3D animation")
    parser.add_argument("--azim_step", type=float, default=0.2, help="Azimuth step per frame for 3D rotation")
    parser.add_argument("--grid_size", type=int, default=64, help="Grid size for density animation")
    parser.add_argument("--sigma", type=float, default=3.0, help="Gaussian sigma for density animation")
    parser.add_argument("--every_n", type=int, default=2, help="Step between particles in trajectory animation")

    return parser.parse_args()


def main():
    args = parse_args()

    bh = BlackHole(mass=args.mass, spin=args.spin)
    print(f"Blackhole with mass {args.mass} and spin {args.spin}")

    simulation_configs = [
        (SchwarzschildAUGeodesicCloud, (args.num_particles, args.r0, args.spacing, bh, args.tangential_fraction, args.radial_fraction),
         args.dt, args.max_t, args.tolerance, "Schwarzschild AU (a=0.0)"),
        (SchwarzschildEFGeodesicCloud, (args.num_particles, args.r0, args.spacing, bh, args.tangential_fraction, args.radial_fraction),
         args.dt, args.max_t, args.tolerance, "Schwarzschild EF (a=0.0)"),
        (KerrIEFEquatorialGeodesicCloud, (args.num_particles, args.r0, args.spacing, bh, 0.4, 0.6),
         args.dt, args.max_t, args.tolerance, f"Kerr (a={args.spin})")
    ]

    results: list[DustCloudSimulation] = []

    # --- Run sequentially ---
    for cloud_class, cloud_args, dt, max_t, tolerance, label in simulation_configs:
        print(f"Running simulation: {label}")
        cloud = cloud_class(*cloud_args)
        sim = DustCloudSimulation(cloud, dt=dt, max_t=max_t, tolerance=tolerance, label=label)
        sim.run()
        results.append(sim)

    if not args.no_visual:
        for sim in results:
            sim.visualize(every_n=4)
            sim.visualize_3d(every_n=4)
            if args.animate:
                outfile = sim.label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
                print(f"Generating animation for {sim.label} -> {outfile}")
                sim.animate(save_path=f"{outfile}_traj.mp4", fps=args.fps,  elev=args.elev, azim = args.azim_step, every_n=args.every_n)
                sim.animate_density(save_path=f"{outfile}_density.mp4", fps=args.fps, grid_size=args.grid_size, sigma=args.sigma, every_n=args.every_n)
                sim.animate_geodesics(save_path=f"{outfile}_geodesics.mp4", fps=args.fps, every_n=args.every_n)

if __name__ == "__main__":
    main()
