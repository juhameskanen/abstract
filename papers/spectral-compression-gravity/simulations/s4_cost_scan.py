"""
d=4 test: C(R) = Tr f(L/Lambda^2) on round S^4 of radius R (Lambda=1), exact spectra.
 scalar  -nabla^2 : lambda=l(l+3)/R^2, mult (2l+3)(l+1)(l+2)/6
 Dirac   D^2      : lambda=(n+2)^2/R^2, mult 2*4*C(n+3,3)   (+/- branches)
Standard heat-kernel prediction (f2=int f dx, f4=int x f dx, f0=f(0)); S^4: Vol=8pi^2 R^4/3, R_scal=12/R^2
 scalar: C ~ (1/6) f4 R^4 + (1/3) f2 R^2 + f0*29/90
 Dirac : C ~ (2/3) f4 R^4 - (2/3) f2 R^2 + f0*a4_D      (a4_D not predicted here; fitted)
Only the SIGN/size of the R^2 term is the point: it is the Einstein-Hilbert term.
"""
import numpy as np
from math import comb
L = np.arange(0, 3000)
sc_e, sc_m = L*(L+3.0), (2*L+3)*(L+1)*(L+2)/6
di_e, di_m = (L+2.0)**2, 2*4*np.array([comb(int(n)+3,3) for n in L],dtype=float)
fs = {"exp(-x)":(lambda x:np.exp(-x),1.0,1.0,1.0),
      "x exp(-x) [f0=0]":(lambda x:x*np.exp(-x),1.0,1.0,0.0),   # f2=1, f4=int x^2 e^-x=2
      "x^2 exp(-x) [wrong ctrl]":(lambda x:x*x*np.exp(-x),2.0,6.0,0.0)}
# analytic moments: f2=int f, f4=int x f
mom = {"exp(-x)":(1.0,1.0),"x exp(-x) [f0=0]":(1.0,2.0),"x^2 exp(-x) [wrong ctrl]":(2.0,6.0)}
Rs = np.linspace(4,12,33)
for op,(e,m) in {"scalar":(sc_e,sc_m),"Dirac":(di_e,di_m)}.items():
    print(f"== {op} on S^4: fit C = a R^4 + b R^2 + c ==")
    for name,(f,_,_,f0) in fs.items():
        f2,f4 = mom[name]
        C = np.array([np.sum(m*f(e/R**2)) for R in Rs])
        A = np.vstack([Rs**4,Rs**2,np.ones_like(Rs)]).T
        (a,b,c),*_ = np.linalg.lstsq(A,C,rcond=None)
        pa,pb = ((1/6)*f4,(1/3)*f2) if op=="scalar" else ((2/3)*f4,-(2/3)*f2)
        print(f"  {name:26s} a={a:8.4f} (pred {pa:7.4f})  b={b:8.4f} (pred {pb:7.4f})  c={c:7.4f}  resid={np.abs(C-A@[a,b,c]).max():.1e}")
# exact C(R) for Dirac over small R: where would an extremum be?
print("\nDirac, f=exp(-x): exact C(R) near the asymptotic-formula minimum R^2=f2/(2 f4)=0.5")
e,m = di_e,di_m
for R in (0.5,0.7,1,1.5,2,3,4):
    print(f"  R={R:4.1f}  R^2Lambda^2={R*R:5.2f}  C={np.sum(m*np.exp(-e/R**2)):10.4f}  asymptotic(no c)={(2/3)*R**4-(2/3)*R**2:9.4f}")
