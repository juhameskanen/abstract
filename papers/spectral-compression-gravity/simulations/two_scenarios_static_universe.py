"""
Static-universe (Einstein static) test of the two readings of the per-mode linear cost.
Geometry: S^3(r) x S^1(beta), beta >> 1/Lambda (Lambda=1), spinor modes. S_E = beta * [ ... ].
S^3 Dirac: omega_n = (n+3/2)/r, degeneracy (n+1)(n+2) per sign.   kappa = ln2 (cost->action, assumed, free).
Scenario 1 (bridge): geometry = full 4D spectral cost with smooth f=e^{-x}, + linear cost on N OCCUPIED modes, N*omega0/Delta_omega,
                     with the added identification beta = kappa/Delta_omega (ASSUMPTION).  omega0 = (3/2)/r (lowest Dirac mode).
                     per unit beta:  kappa*[e1(r) - m r^3] + N*omega0(r)      (m = hand-tuned cosmological counterterm)
   e1(r) = sum_n 4(n+1)(n+2) Int dw/2pi e^{-(w^2+omega_n^2)}            (spinor dim 4)
Scenario 2: linear cost on ALL modes, no separate geometry term:  per unit beta: E(r) - m r^3,   E = sum_n 2(n+1)(n+2) omega_n e^{-omega_n^2}.
Stationarity in beta and r  ->  (beta): e_tot=0 ;  (r): r e' - 3 e = source terms.
Einstein-static-universe prediction (G from the EH coefficient: 1/(16 pi G)*kappa^-1 ...): r^2 = 12 N/(kappa f2), f2=1.
"""
import numpy as np
from scipy.optimize import brentq
kap=np.log(2.0); n=np.arange(0,4000); u=n+1.5; deg4=4*(n+1)*(n+2.0); deg2=2*(n+1)*(n+2.0)
def e1(r):  return np.sum(deg4*np.exp(-u**2/r**2))/(2*np.sqrt(np.pi))
def e1p(r): return np.sum(deg4*np.exp(-u**2/r**2)*(2*u**2/r**3))/(2*np.sqrt(np.pi))
def E2(r):  return np.sum(deg2*(u/r)*np.exp(-u**2/r**2))
def E2p(r): return np.sum(deg2*u*(-1/r**2+2*u**2/r**4)*np.exp(-u**2/r**2))
rs=np.linspace(5,14,40)
for name,fn,ex in (("scenario 1 geometry e1(r)",e1,(0.5,-0.25)),("scenario 2 vacuum E2(r)",E2,(1.0,-0.25))):
    y=np.array([fn(r) for r in rs]); A=np.vstack([rs**3,rs,1/rs,1/rs**3]).T
    c,*_=np.linalg.lstsq(A,y,rcond=None)
    print(f"{name}: fit A r^3+B r+c/r+d/r^3 -> A={c[0]:.5f} (HK pred {ex[0]}), B={c[1]:.5f} (pred {ex[1]}), c={c[2]:.5f}, d={c[3]:.4f}")
print("   Euler-Maclaurin estimate of the Casimir-like 1/r coefficient for scenario 2: c=-1/48=%.5f"%(-1/48))
print("\n== Scenario 1: solve  kappa*(r e1' - 3 e1) = 4 N w = 6N/r  for r, vs Einstein r^2 = 12 N/(kappa*1) ==")
print("   N     r_exact   r^2_exact   r^2_Einstein   ratio")
for N in (5,10,20,40):
    F=lambda r: kap*(r*e1p(r)-3*e1(r))-6*N/r
    r=brentq(F,3,60); pr=12*N/kap
    print(f" {N:4d}  {r:8.3f}  {r*r:10.2f}  {pr:11.2f}    {r*r/pr:.4f}")
print("\n== controls inside scenario 1: matter cost ~ r^-s (s=1 radiation-like, 0 dust-like, 2 'omega^2'-like), N=10 ==")
N=10
for s_ in (0,1,2):
    w=lambda r: 1.5*r**(-s_)
    F=lambda r: kap*(r*e1p(r)-3*e1(r))-(3+s_)*N*w(r)          # spectral stationarity (beta and r equations combined)
    G=lambda r: r-2*(3+s_)*N*w(r)/kap                          # Einstein static universe with p=s*rho/3, G=3pi/(kappa f2)
    print(f"  s={s_}: spectral r={brentq(F,3,400):.3f}   Einstein-static r={brentq(G,3,400):.3f}")
print("\n== Scenario 2: stationarity in r with e_tot=0 imposed: F(r)=r E' - 3E  (asymptotically  -2B r - 4c/r, B=-1/4, c=-17/480) ==")
Fv=lambda r: r*E2p(r)-3*E2(r)
grid=np.linspace(0.2,20,400); vals=np.array([Fv(r) for r in grid])
roots=[brentq(Fv,grid[i],grid[i+1]) for i in range(len(grid)-1) if vals[i]*vals[i+1]<0]
print("  F(r) at r=0.3,1,5,10,20:",[round(float(Fv(r)),3) for r in (0.3,1,5,10,20)]," sign changes (roots) on [0.2,20]:",np.round(roots,3))
print("  asymptotic F = %.3f at r=5 (=0.5 r + 17/(120 r))"%(0.5*5+17/(120*5)))
