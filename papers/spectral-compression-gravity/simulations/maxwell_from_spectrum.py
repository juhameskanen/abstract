"""
Does stationarity of the spectral cost w.r.t. A give Maxwell? Test the A-dependence directly.
Geometry: flat T^2(L) x T^2(L) (no curvature, so only the A-dependence is probed), Lambda=1, L=20.
Dirac operator D = D1 (x) 1 + gamma (x) D2[A], D^2 = D1^2 + D2[A]^2, charge-1 U(1) field depending on x1 (a coordinate of 2nd torus)
only:  A_2 = a cos(k x1)  -> field strength f_12 = -a k sin(k x1) (transverse).   A_1 = a cos(k x1) is pure gauge (control).
Cost f(x)=exp(-x) factorises: C = M0(T2_1) * M0(T2_2[A]),  M0 = Tr e^{-D^2}.
Control f(x)=x exp(-x) (f(0)=0): C = M1 M0' + M0 M1',  M1 = Tr D^2 e^{-D^2}.
Heat-kernel prediction (Gilkey a_4, my evaluation; validated against flux test earlier):
   dC = f0/(24 pi^2) * Int f_mu_nu f^mu_nu d^4x = f0 * L^4 a^2 k^2 /(24 pi^2)
"""
import numpy as np
L=20.0; M=21; ms=np.arange(-M,M+1); n=len(ms); q=2*np.pi*ms/L
s1=np.array([[0,1],[1,0]],complex); s2=np.array([[0,-1j],[1j,0]],complex)
def shift(mk):
    S=np.zeros((n,n)); 
    for i in range(n):
        for j in (i-mk,i+mk):
            if 0<=j<n: S[i,j]=0.5
    return S
def moments(a,mk,kind):
    """returns (M0,M1) = (Tr e^{-D^2}, Tr D^2 e^{-D^2}) on T^2 with the background"""
    S=shift(mk) if a!=0 else 0*np.eye(n); M0=M1=0.0
    for j in range(-M,M+1):
        p2=2*np.pi*j/L
        if kind=="transverse":   D1=np.diag(q);        D2=p2*np.eye(n)-a*S
        else:                    D1=np.diag(q)-a*S;    D2=p2*np.eye(n)
        D=np.kron(s1,D1)+np.kron(s2,D2)
        lam=np.linalg.eigvalsh(D)**2
        M0+=np.sum(np.exp(-lam)); M1+=np.sum(lam*np.exp(-lam))
    return M0,M1
# first torus exact: mult 2 per momentum
P=2*np.pi*np.arange(-M,M+1)/L; P2=(P[:,None]**2+P[None,:]**2)
N0=2*np.sum(np.exp(-P2)); N1=2*np.sum(P2*np.exp(-P2))
base=moments(0,1,"transverse")
print(f"sanity: Tr e^-D^2 on T^2: matrix {base[0]:.8f} vs exact {N0:.8f}")
print("\n   kind        m_k    k      a     dC_exp/pred   dC_ctrl/pred   (pred = L^4 a^2 k^2/(24 pi^2) )")
for kind in ("transverse","pure-gauge"):
    for mk in (1,2,3):
        k=2*np.pi*mk/L
        for a in (0.02,0.04):
            if kind=="pure-gauge" and a==0.04: continue
            m=moments(a,mk,"transverse" if kind=="transverse" else "gauge")
            dC=N0*(m[0]-base[0]); dCc=N1*(m[0]-base[0])+N0*(m[1]-base[1])
            pred=L**4*a*a*k*k/(24*np.pi**2)
            print(f" {kind:11s} {mk:3d} {k:6.3f} {a:6.2f}  {dC/pred:12.4f}  {dCc/pred:12.4f}   (dC={dC:.3e})")
