"""
Extension: C(R) = sum_i f(lambda_i / Lambda^2) on round S^2 of radius R (Lambda=1).
lambda_l = l(l+1)/R^2, multiplicity 2l+1  (exact spectrum, no mesh error).
Standard 2D heat-kernel/Weyl prediction for smooth f:
   C(R) ~ (Lambda^2 Vol/(4 pi)) * F2  +  (chi/6) * f(0),     F2 = int_0^inf f(x) dx, Vol=4 pi R^2, chi=2
i.e.  C ~ F2 * R^2  +  f(0)/3
Fit a*R^2 + b to the exact sums and compare with (F2, f(0)/3).
Controls: f = x^2 (cutoff) and sharp step (Weyl fluctuations). Nothing here is tuned.
"""
import numpy as np
l = np.arange(0, 4000); mult = 2*l+1; e = l*(l+1.0)
fs = {
 "exp(-x)":        (lambda x: np.exp(-x),                1.0,   1.0),
 "x exp(-x)":      (lambda x: x*np.exp(-x),              1.0,   0.0),
 "linear x*th(1-x)":(lambda x: np.where(x<1, x, 0.0),    0.5,   0.0),   # paper's linear cost + cutoff
 "x^2*th(1-x) [wrong ctrl]":(lambda x: np.where(x<1, x*x,0.0), 1/3, 0.0),
 "sharp step th(1-x)":(lambda x: np.where(x<1,1.0,0.0),  1.0,   1.0),
}
Rs = np.linspace(4, 14, 41)
print("%-26s %9s %9s | %9s %9s | max|C-fit|" % ("f","fit a","pred F2","fit b","pred f0/3"))
for name,(f,F2,f0) in fs.items():
    C = np.array([np.sum(mult*f(e/R**2)) for R in Rs])
    A = np.vstack([Rs**2, np.ones_like(Rs)]).T
    (a,b),*_ = np.linalg.lstsq(A, C, rcond=None)
    print("%-26s %9.4f %9.4f | %9.4f %9.4f | %.3f" % (name,a,F2,b,f0/3,np.abs(C-A@[a,b]).max()))
print("\ndC/dR > 0 everywhere in scan for every f above?",
      all(np.all(np.diff([np.sum(mult*f(e/R**2)) for R in Rs])>=-1e-9) for f,_,_ in fs.values()))
