"""
d=4 anisotropic test: M = S^2(r1) x S^2(r2), exact spectra, Lambda=1.
 scalar: lam = l1(l1+1)/r1^2 + l2(l2+1)/r2^2,   mult (2l1+1)(2l2+1)
 Dirac : lam = (n1+1)^2/r1^2 + (n2+1)^2/r2^2,   mult 16 (n1+1)(n2+1)   (spinor dim 4)
Heat-kernel prediction (x=r1^2, y=r2^2; Vol=16pi^2 x y; R_scal = 2/x + 2/y):
 scalar: C ~ f4 xy + (f2/3)(x+y) + a4(f0)
 Dirac : C ~ 4 f4 xy - (2 f2/3)(x+y) + a4(f0)
Use f with f(0)=0 so no a4 term. Then (i) fit, (ii) stationary points & Hessian of
C_tot = C - mu*xy, where mu*xy is an ADDED, HAND-CHOSEN volume counterterm (free parameter,
= cosmological-constant tuning; NOT derived).
"""
import numpy as np
from scipy.optimize import fsolve
Lm = 140
l = np.arange(Lm)
S_e = l*(l+1.0); S_m = 2*l+1.0
D_e = (l+1.0)**2; D_m = 4*(l+1.0)      # per-S^2 Dirac D^2 (both signs), dim 2
def C(x, y, f, e, m):
    E = e[:,None]/x + e[None,:]/y
    return np.sum(np.outer(m, m)*f(E))
fl = {"x e^-x":(lambda z:z*np.exp(-z),1.0,2.0), "x^2 e^-x [wrong ctrl]":(lambda z:z*z*np.exp(-z),2.0,6.0)}
ops = {"scalar":(S_e,S_m,lambda f2,f4:(f4,f2/3)), "Dirac":(D_e,D_m*D_m[0]/D_m[0],lambda f2,f4:(4*f4,-2*f2/3))}
# Dirac multiplicity: product mult = D_m (x) D_m = 16 (n1+1)(n2+1)  -> outer(D_m,D_m)
rng = np.linspace(3,9,7)
for op,(e,m,pred) in ops.items():
    print(f"== {op}: fit C = a x y + b (x+y) + c  over r1,r2 in [3,9] ==")
    for name,(f,f2,f4) in fl.items():
        X,Y,Cv = [],[],[]
        for r1 in rng:
            for r2 in rng:
                X.append(r1**2); Y.append(r2**2); Cv.append(C(r1**2,r2**2,f,e,m))
        X,Y,Cv = map(np.array,(X,Y,Cv))
        A = np.vstack([X*Y, X+Y, np.ones_like(X)]).T
        (a,b,c),*_ = np.linalg.lstsq(A,Cv,rcond=None)
        pa,pb = pred(f2,f4)
        print(f"  {name:22s} a={a:8.4f} (pred {pa:7.3f})  b={b:8.4f} (pred {pb:7.3f})  c={c:7.4f}  maxresid={np.abs(Cv-A@[a,b,c]).max():.1e}")
print("\n== stationary points of C_tot = C - mu*x*y  (f = x e^-x) ==")
f,f2,f4 = fl["x e^-x"][0],1.0,2.0
for op,(e,m,pred) in ops.items():
    a,b = pred(f2,f4)
    print(f" {op}: heat-kernel says stationary iff b<0; predicted x=y=-b/(a-mu)")
    for eps in (0.01,0.004):
        mu = a*(1-eps)
        Ct = lambda x,y: C(x,y,f,e,m) - mu*x*y
        g = lambda v,h=1e-3:[(Ct(v[0]+h,v[1])-Ct(v[0]-h,v[1]))/(2*h),(Ct(v[0],v[1]+h)-Ct(v[0],v[1]-h))/(2*h)]
        x0 = (-b/(a*eps)) if b<0 else 10.0
        sol,info,ier,msg = fsolve(g,[x0,x0*1.05],full_output=True)
        h=0.05; H=np.zeros((2,2)); 
        H[0,0]=(Ct(sol[0]+h,sol[1])-2*Ct(*sol)+Ct(sol[0]-h,sol[1]))/h**2
        H[1,1]=(Ct(sol[0],sol[1]+h)-2*Ct(*sol)+Ct(sol[0],sol[1]-h))/h**2
        H[0,1]=H[1,0]=(Ct(sol[0]+h,sol[1]+h)-Ct(sol[0]+h,sol[1]-h)-Ct(sol[0]-h,sol[1]+h)+Ct(sol[0]-h,sol[1]-h))/(4*h*h)
        print(f"   eps={eps}: solver ok={ier==1}  (x,y)=({sol[0]:.2f},{sol[1]:.2f})  analytic x=y={(-b/(a*eps)):.2f}  Hessian eig={np.round(np.linalg.eigvalsh(H),3)}")
