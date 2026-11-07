"""
Matter source test: Dirac cost on S^2(r1) x S^2(r2) with U(1) magnetic flux n through S^2(r2). x=r1^2, y=r2^2, Lambda=1.
Exact spectrum:
  S^2(r1), no flux : D^2 = (l+1)^2/x,  mult 4(l+1)
  S^2(r2), flux n  : D^2 = k(k+|n|)/y, k>=1, mult 2(2k+|n|);  plus |n| zero modes   [standard monopole-Dirac result; validated below by fit]
Heat-kernel predictions (Gilkey a_4, my own evaluation, validated below):
  C = 4 f4 xy - (2/3) f2 (x+y) + f0 [ (40 - 24(x/y+y/x))/360 + n^2 x/(3y) ]
Maxwell term appears at a_4, i.e. with coefficient f(0).
Einstein-Maxwell on S2xS2 (Euclid, flux on S2(r2)): 1/x = 1/y - 2 pi G n^2/(e^2 y^2); matching needs 2 pi G/e^2 = f0/f2.
"""
import numpy as np
from scipy.optimize import fsolve
K = 220; l = np.arange(K); k = np.arange(1,K, dtype=float)
e1, m1 = (l+1.0)**2, 4*(l+1.0)
def spec2(n):
    n=abs(n); return np.concatenate([[0.0],k*(k+n)]), np.concatenate([[float(n)],2*(2*k+n)])
def C(x,y,n,f):
    e2,m2 = spec2(n); E = e1[:,None]/x + e2[None,:]/y
    return np.sum(np.outer(m1,m2)*f(E))
fe  = lambda z: np.exp(-z)            # f0=1,f2=1,f4=1
fxe = lambda z: z*np.exp(-z)          # f0=0 control
pred = lambda x,y,n,f0,f2,f4: 4*f4*x*y-(2/3)*f2*(x+y)+f0*((40-24*(x/y+y/x))/360+n*n*x/(3*y))
print("== validation of the heat-kernel prediction, f=e^-x (f0=f2=f4=1) ==")
print("   x      y    n    C_exact    C_pred    diff")
for (x,y) in [(10,10),(16,10),(10,16),(24,14)]:
    for n in (0,1,2,3):
        ce = C(x,y,n,fe); cp = pred(x,y,n,1,1,1)
        print(f"{x:5.0f} {y:6.0f} {n:3d} {ce:10.4f} {cp:10.4f} {ce-cp:8.4f}")
print("\n== stationary points of C - mu*x*y, mu = 4 f4 (1-eps) ==")
def stat(f,n,eps,guess,f4):
    mu=4*f4*(1-eps); Ct=lambda x,y: C(x,y,n,f)-mu*x*y
    h=1e-3; g=lambda v:[(Ct(v[0]+h,v[1])-Ct(v[0]-h,v[1]))/(2*h),(Ct(v[0],v[1]+h)-Ct(v[0],v[1]-h))/(2*h)]
    s,info,ier,msg=fsolve(g,guess,full_output=True,xtol=1e-10)
    H=np.zeros((2,2)); d=0.05
    H[0,0]=(Ct(s[0]+d,s[1])-2*Ct(*s)+Ct(s[0]-d,s[1]))/d**2
    H[1,1]=(Ct(s[0],s[1]+d)-2*Ct(*s)+Ct(s[0],s[1]-d))/d**2
    H[0,1]=H[1,0]=(Ct(s[0]+d,s[1]+d)-Ct(s[0]+d,s[1]-d)-Ct(s[0]-d,s[1]+d)+Ct(s[0]-d,s[1]-d))/(4*d*d)
    return s,ier==1,np.linalg.eigvalsh(H)
eps=0.01
for n in (0,1,2):
    # leading-order analytic: 4 eps y^2 - (2/3) y + n^2/3 = 0 (large root), x = y^2/(y-n^2)
    a=4*eps; disc=(2/3)**2-4*a*n*n/3; y=((2/3)+np.sqrt(disc))/(2*a); x=y*y/(y-n*n)
    s,ok,ev=stat(fe,n,eps,[x,y],1.0)
    print(f" f=e^-x  n={n}: analytic (x,y)=({x:6.2f},{y:6.2f})  exact (x,y)=({s[0]:6.2f},{s[1]:6.2f}) ok={ok}  x/y={s[0]/s[1]:.3f}  Hess eig={np.round(ev,3)}")
print("\n== control f = x e^-x (f0=0: predicts NO leading-order flux effect), eps=0.01, f4=2 ==")
for n in (0,2):
    s,ok,ev=stat(fxe,n,eps,[20.,20.5],2.0)
    print(f" n={n}: exact (x,y)=({s[0]:6.2f},{s[1]:6.2f}) ok={ok}  x/y={s[0]/s[1]:.4f}")
