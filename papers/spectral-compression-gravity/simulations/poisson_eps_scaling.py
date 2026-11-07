import numpy as np, importlib.util, sys, io, contextlib
spec=importlib.util.spec_from_file_location("p","poisson_inhomogeneous.py"); 
src=open("poisson_inhomogeneous.py").read().split('print("sanity')[0]
ns={}; exec(src,ns); Ccost=ns["Ccost"]; Lz=ns["Lz"]
nrm=8*np.pi**2/Lz; vol_ed=(1/(4*np.pi**2))*(Lz/2)*3
print("xi=1/4: C_ed(tuned)*8pi^2/Lz vs amplitude eps=delta; prediction -k^2/3")
rows=[]
for m in (2,4):
    k=2*np.pi*m/Lz
    for eps in (0.01,0.02,0.04):
        f=lambda e,d: Ccost(e,d,k,0.25)
        ed=(f(eps,eps)-f(eps,-eps)-f(-eps,eps)+f(-eps,-eps))/(4*eps*eps)-vol_ed
        print(f"  k={k:.3f} eps={eps:.2f}: {ed*nrm:9.5f}   pred {-k*k/3:9.5f}")

print("\nC_ee/C_ed (tuned) and C_dd, xi=1/4, amplitude scaling (Richardson in eps^2):")
vol_ee=(1/(4*np.pi**2))*(Lz/2)*9
for m in (2,3,4):
    k=2*np.pi*m/Lz; out={}
    for eps in (0.01,0.02):
        f=lambda e,d: Ccost(e,d,k,0.25)
        ed=(f(eps,eps)-f(eps,-eps)-f(-eps,eps)+f(-eps,-eps))/(4*eps*eps)-vol_ed
        ee=(f(eps,0)-2*f(0,0)+f(-eps,0))/eps**2-vol_ee
        dd=(f(0,eps)-2*f(0,0)+f(0,-eps))/eps**2
        out[eps]=(ed,ee,dd)
    ex=[out[0.01][i]-(out[0.02][i]-out[0.01][i])/3 for i in range(3)]
    print(f"  k={k:.3f}: C_ed={ex[0]*nrm:9.5f} (pred {-k*k/3:9.5f})  C_ee={ex[1]*nrm:9.5f}  C_ee/C_ed={ex[1]/ex[0]:.4f}  C_dd={ex[2]*nrm:9.5f}")
