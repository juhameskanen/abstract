"""
Stage 1 of test 2: does the spectrum of a discrete Laplacian carry the
Weyl / heat-kernel structure that Route A relies on?   (No matter yet.)

Heat trace in d dims:  K(t)=Tr e^{-tL} ~ (4 pi t)^{-d/2} [ V + (t/6)∫R + O(t^2) ]
Checks (all reference values are standard results, nothing invented):
  A. flat torus grid, d=2  : leading Weyl term only (∫R = 0).  Control: random graph.
  B. round S^2 (icosphere, cotan Laplacian): K(t) = 1/t + 1/3 + t/15 + ...  The 1/3
     is the Einstein-Hilbert-type coefficient (∫R/(24π) = chi/6).
FREE PARAMETERS / ASSUMPTIONS flagged in output: mesh resolution, t-window, operator choice.
"""
import numpy as np, scipy.linalg as sla, itertools

# ---------- A. torus grid vs random graph ----------
def torus_laplacian(n):
    a = 1.0/n                       # lattice spacing, box side 1
    N = n*n; L = np.zeros((N, N))
    idx = lambda i, j: (i % n)*n + (j % n)
    for i in range(n):
        for j in range(n):
            p = idx(i, j)
            for q in (idx(i+1, j), idx(i-1, j), idx(i, j+1), idx(i, j-1)):
                L[p, p] += 1; L[p, q] -= 1
    return L/a**2                   # 5-point Laplacian in continuum units

def weyl_count_check(ev, V, d, label):
    # N(lam) ~ V * omega_d * lam^{d/2} / (2 pi)^d ; d=2: V*lam/(4 pi)
    assert d == 2
    out = []
    for lam in (200, 500, 1000, 2000):
        n_true = np.sum(ev <= lam); n_weyl = V*lam/(4*np.pi)
        out.append((lam, n_true, n_weyl, n_true/n_weyl))
    print(f"[{label}] lam, N_true, N_Weyl, ratio")
    for r in out: print("   %6.0f %7d %9.1f %7.3f" % r)

n = 40
ev = np.linalg.eigvalsh(torus_laplacian(n))
weyl_count_check(ev, 1.0, 2, "torus 40x40")

# control: random graph with same node count, edge weights rescaled the same way (no geometry)
rng = np.random.default_rng(0)
N = n*n; A = (rng.random((N, N)) < 4/N).astype(float); A = np.triu(A, 1); A = A + A.T
Lr = (np.diag(A.sum(1)) - A)/ (1.0/n)**2
weyl_count_check(np.linalg.eigvalsh(Lr), 1.0, 2, "CONTROL random graph")

# ---------- B. icosphere heat trace ----------
def icosphere(sub):
    t = (1+5**.5)/2
    V = [(-1,t,0),(1,t,0),(-1,-t,0),(1,-t,0),(0,-1,t),(0,1,t),(0,-1,-t),(0,1,-t),(t,0,-1),(t,0,1),(-t,0,-1),(-t,0,1)]
    V = [np.array(v)/np.linalg.norm(v) for v in V]
    F = [(0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),(1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
         (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),(4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1)]
    for _ in range(sub):
        cache = {}; newF = []
        def mid(a, b):
            k = tuple(sorted((a, b)))
            if k not in cache:
                m = V[a]+V[b]; V.append(m/np.linalg.norm(m)); cache[k] = len(V)-1
            return cache[k]
        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            newF += [(a,ab,ca),(b,bc,ab),(c,ca,bc),(ab,bc,ca)]
        F = newF
    return np.array(V), np.array(F)

def cotan_system(V, F):
    N = len(V); W = np.zeros((N, N)); M = np.zeros(N)
    for tri in F:
        for k in range(3):
            i, j, o = tri[k], tri[(k+1) % 3], tri[(k+2) % 3]
            u, w = V[i]-V[o], V[j]-V[o]
            cot = (u@w)/np.linalg.norm(np.cross(u, w))
            W[i, j] += cot/2; W[j, i] += cot/2
        area = np.linalg.norm(np.cross(V[tri[1]]-V[tri[0]], V[tri[2]]-V[tri[0]]))/2
        for v in tri: M[v] += area/3
    L = np.diag(W.sum(1)) - W
    return L, M

def exact_sphere_K(t, lmax=400):
    l = np.arange(lmax); return np.sum((2*l+1)*np.exp(-t*l*(l+1)))

print("\n[B] round S^2, K(t) vs exact and vs 1/t + 1/3 + t/15")
for sub in (3, 4):
    V, F = icosphere(sub); L, M = cotan_system(V, F)
    ev = sla.eigh(L, np.diag(M), eigvals_only=True)
    print(f" icosphere sub={sub}: N={len(V)}, area={M.sum():.4f} (4pi={4*np.pi:.4f}), lowest nonzero evs {ev[1]:.3f},{ev[4]:.3f} (exact 2, 6)")
    print("    t      K_mesh     K_exact    K-1/t (mesh)  K-1/t (exact)  [asymptotic 1/3+t/15]")
    for t in (0.05, 0.1, 0.2, 0.4):
        Km = np.sum(np.exp(-t*ev)); Ke = exact_sphere_K(t)
        print("  %5.2f  %9.4f  %9.4f  %10.4f  %12.4f   %10.4f" % (t, Km, Ke, Km-1/t, Ke-1/t, 1/3+t/15))
