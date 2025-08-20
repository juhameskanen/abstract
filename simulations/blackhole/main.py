from __future__ import annotations
from kerr import KerrEquatorialGeodesicCloud
from schwarzschild import SchwarzschildGeodesicCloud, SchwarzschildEFGeodesicCloud
from blackhole import BlackHole, DustCloudSimulation

if __name__ == "__main__":
    
    bh = BlackHole(mass=1.0, spin=0.9)


    # Use Eddington-Finkelstein coordinates to pass through the event horizon
    ef_cloud = SchwarzschildEFGeodesicCloud(50, r0=6.0, spacing=0.02, bh=bh,
                                            tangential_fraction=0.4, radial_fraction=0.6)
    sim_ef = DustCloudSimulation(ef_cloud, dt=1e-3, max_t=20.0, tolerance=1e-6)
    times3, radii3 = sim_ef.run()
    sim_ef.visualize(every_n=2, title="Schwarzschild EF (a=0.0)")
    sim_ef.visualize_3d(every_n=4, title="Schwarzschild EF (a=0.0)")

 
    # breaks at the event horizon singularity
    #sch_cloud = SchwarzschildGeodesicCloud(50, r0=6.0, spacing=0.02, bh=bh, tangential_fraction=0.4, radial_fraction=0.6)
    #sim_sch = DustCloudSimulation(sch_cloud, dt=1e-3, max_t=12.0, tolerance=1e-6)
    #times2, radii2 = sim_sch.run()
    #sim_sch.visualize(every_n=2, title="Schwarzschild (a=0.0)")
    #sim_sch.visualize_3d(every_n=4, title="Schwarzschild (a=0.0)")


    kerr_cloud = KerrEquatorialGeodesicCloud(50, r0=6.0, spacing=0.02, bh=bh, tangential_fraction=0.4, radial_fraction=0.6)
    sim_kerr = DustCloudSimulation(kerr_cloud, dt=1e-3, max_t=12.0, tolerance=1e-6)
    times, radii = sim_kerr.run()
    sim_kerr.visualize(every_n=2, title="Kerr (a=0.9)")
    sim_kerr.visualize_3d(every_n=4, title="Kerr (a=0.9)")



