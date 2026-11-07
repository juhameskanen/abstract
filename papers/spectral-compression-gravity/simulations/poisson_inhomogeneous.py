"""
Inhomogeneous (planar) test of the Poisson/Newtonian limit of the spectral cost  (Scenario 1 geometry term).
4D static metric  ds^2 = N(z)^2 dtau^2 + e^{2 sigma(z)} (dx^2+dy^2+dz^2),  tau-circle long (beta>>1), transverse area A large.
Spectral cost (Lambda=1, f=e^{-x}, spinor multiplicity 4 mimicked by operator  L = -nabla^2_4 + xi R_4, xi=1/4 (Lichnerowicz D^2=nabla*nabla+R/4)):
  C/(beta A) = 4 * Int dw/2pi Int d^2k/(2pi)^2 Tr_z e^{-L(w,k)} ,   L on z:  -(N e^s u')'/(N e^{3s}) + [s_k e^{-2s} + w^2/N^2 + xi R4] u
  (Fourier-pseudospectral in z, periodic Lz=40.)  N = 1+d*cos(kz), sigma = e*cos(kz).
Local prediction from the heat kernel (f4=f2=1):  C/(beta A) ~ (1/4pi^2) Int dz N e^{3s} [ f4 + f2 (1/6-xi) R3 ]   (R4 total-derivative drops)
After 'tuning' away the volume term (hand-chosen cosmological counterterm, FREE PARAMETER), the Hessian in (e,d) at amplitude cos(kz):
   C_ed = (Lz/8pi^2) 4 f2 (1/6-xi) k^2 ,  C_ee = C_ed ,  C_dd = 0
GR (Newtonian gauge) predicts d = -e from  C_ee e + C_ed d = 0 ; the d-equation with matter coupling  N*rho  gives Poisson with G = 3 pi/(kappa f2).
"""
import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.laguerre import laggauss
n=97; Lz=40.0; z=np.arange(n)*Lz/n
kk=2*np.pi*np.fft.fftfreq(n,d=Lz/n)
D=np.real(np.fft.ifft(1j*kk[:,None]*np.fft.fft(np.eye(n),axis=0),axis=0))
xh,wh=hermgauss(14); xl,wl=laggauss(14)
def Ccost(e,d,k,xi):
    c=np.cos(k*z); s_=np.sin(k*z)
    N=1+d*c; Nz=-d*k*s_; Nzz=-d*k*k*c
    sg=e*c; sgz=-e*k*s_; sgzz=-e*k*k*c
    R3=-np.exp(-2*sg)*(4*sgzz+2*sgz**2)
    D3N=np.exp(-2*sg)*(sgz*Nz+Nzz)
    R4=R3-2*D3N/N
    W=N*np.exp(sg); M=N*np.exp(3*sg); Msi=1/np.sqrt(M)
    DWD=D.T@(W[:,None]*D)
    tot=0.0
    for om,a in zip(xh,wh):
        for s,b in zip(xl,wl):
            V=M*(s*np.exp(-2*sg)+om**2/N**2+xi*R4)
            A=Msi[:,None]*(DWD+np.diag(V))*Msi[None,:]
            lam=np.linalg.eigvalsh(A)
            tot+=a*b*np.sum(np.exp(-lam+om**2+s))
    return 4*tot/(2*np.pi*4*np.pi)
print("sanity C0 vs Lz/(4pi^2):",Ccost(0,0,2*np.pi/Lz,0.25),Lz/(4*np.pi**2))
eps=0.02
for xi,label in ((0.25,"xi=1/4 (Dirac-like)  [EH sign: right]"),(0.0,"xi=0  (minimal scalar) [control: wrong sign]"),(1/6,"xi=1/6 (conformal)   [control: no R term]")):
    print(f"\n=== {label} ===")
    print("   k      C_ed*8pi^2/Lz   pred(4(1/6-xi)k^2)   C_ee/C_ed(tuned)   C_dd*8pi^2/Lz")
    for m in (1,2,3,4):
        k=2*np.pi*m/Lz
        f=lambda e,d: Ccost(e,d,k,xi)
        ed=(f(eps,eps)-f(eps,-eps)-f(-eps,eps)+f(-eps,-eps))/(4*eps*eps)
        ee=(f(eps,0)-2*f(0,0)+f(-eps,0))/eps**2
        dd=(f(0,eps)-2*f(0,0)+f(0,-eps))/eps**2
        vol_ed=(1/(4*np.pi**2))*(Lz/2)*3; vol_ee=(1/(4*np.pi**2))*(Lz/2)*9   # analytic volume term (f4=1) to be tuned away
        ed_t=ed-vol_ed; ee_t=ee-vol_ee
        nrm=8*np.pi**2/Lz
        print(f" {k:6.3f}   {ed_t*nrm:12.5f}   {4*(1/6-xi)*k*k:14.5f}      {ee_t/ed_t if abs(ed_t)>1e-9 else float('nan'):10.4f}      {dd*nrm:10.5f}")
