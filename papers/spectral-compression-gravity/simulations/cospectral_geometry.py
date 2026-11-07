"""
Many-to-one test: one spectrum, several 'geometries' (graphs).  Finite graphs stand in for geometries (A: Laplacian L=D-W).
 1. Find connected, non-isomorphic graphs with identical Laplacian spectrum (networkx atlas, <=7 nodes).
 2. Spectral cost C = Tr f(L), f=e^{-x}: identical BY CONSTRUCTION (depends only on eigenvalues)  -> TAUTOLOGICAL, flagged.
 3. Add state data: for each vertex-localized state delta_v define the basis-invariant spectral measure
        p_v(lambda) = || P_lambda delta_v ||^2   (P_lambda = projector on the eigenspace; independent of the choice of basis in degenerate eigenspaces)
    and its Shannon entropy H_v (bits) = 'spectral description length' of that state.  Compare the sorted multisets {H_v} of the two graphs.
 4. Control: relabelling vertices (a 'gauge' redescription of ONE geometry) must change nothing.
"""
import numpy as np, networkx as nx
from networkx.generators.atlas import graph_atlas_g
def lap_spec(G):
    L=nx.laplacian_matrix(G).toarray().astype(float); w,V=np.linalg.eigh(L); return L,w,V
def spec_measures(L,w,V,tol=1e-8):
    # group eigenvalues into eigenspaces
    groups=[]; cur=[0]
    for i in range(1,len(w)):
        if abs(w[i]-w[cur[0]])<tol: cur.append(i)
        else: groups.append(cur); cur=[i]
    groups.append(cur)
    P=[V[:,g]@V[:,g].T for g in groups]
    return [np.array([Pk[v,v] for Pk in P]) for v in range(len(w))], groups     # ||P delta_v||^2 = P[v,v]
def entropies(L,w,V):
    ms,_=spec_measures(L,w,V)
    H=[-(p[p>1e-12]*np.log2(p[p>1e-12])).sum() for p in ms]
    return np.sort(np.array(H))
atlas=[G for G in graph_atlas_g() if G.number_of_nodes()>=2 and nx.is_connected(G)]
buckets={}
for G in atlas:
    L,w,V=lap_spec(G); key=(G.number_of_nodes(),tuple(np.round(w,6))); buckets.setdefault(key,[]).append((G,L,w,V))
pairs=[(k,v) for k,v in buckets.items() if len(v)>1]
nconn={n:sum(1 for G in atlas if G.number_of_nodes()==n) for n in range(2,8)}
print("connected graphs by node count:",nconn)
print("Laplacian-cospectral classes (size>1) by node count:",{n:sum(1 for k,v in pairs if k[0]==n) for n in range(2,8)})
f=lambda x:np.exp(-x)
sep=0; tot=0
for (n,spec),grs in pairs:
    for i in range(len(grs)):
        for j in range(i+1,len(grs)):
            (G1,L1,w1,V1),(G2,L2,w2,V2)=grs[i],grs[j]; tot+=1
            C1,C2=f(w1).sum(),f(w2).sum()
            H1,H2=entropies(L1,w1,V1),entropies(L2,w2,V2)
            d=np.abs(H1-H2).max(); sep+= d>1e-6
            if tot<=6: print(f" n={n}: edges {G1.number_of_edges()},{G2.number_of_edges()}  |C1-C2|={abs(C1-C2):.1e}  max|H1-H2|(sorted per-vertex spectral entropies)={d:.4f}  separated={d>1e-6}")
print(f"\ncospectral non-isomorphic pairs found: {tot};  separated by per-vertex spectral-measure entropies: {sep}")
# control: relabel one graph
G=grs[0][0]; perm=np.random.default_rng(1).permutation(G.number_of_nodes())
Gp=nx.relabel_nodes(G,{i:int(perm[i]) for i in range(G.number_of_nodes())})
L,w,V=lap_spec(G); Lp,wp,Vp=lap_spec(Gp)
print("control (relabelled copy of one graph): max|dC|=%.1e, max|dH sorted|=%.1e"%(abs(f(w).sum()-f(wp).sum()),np.abs(entropies(L,w,V)-entropies(Lp,wp,Vp)).max()))

# ---- finer invariants for the pairs NOT separated by the entropy invariant ----
def proj_list(L,w,V,tol=1e-8):
    _,groups=spec_measures(L,w,V,tol); return [V[:,g]@V[:,g].T for g in groups]
def level2(L,w,V):   # multiset of full per-vertex spectral measures (vector of ||P_k delta_v||^2)
    P=proj_list(L,w,V); vs=[tuple(np.round([Pk[v,v] for Pk in P],7)) for v in range(len(w))]; return sorted(vs)
def level3(L,w,V):   # multiset over unordered vertex pairs of (|P_k[u,v]|)_k  together with the vertex vectors
    P=proj_list(L,w,V); n=len(w); out=[]
    for u in range(n):
        for v in range(u+1,n):
            a=tuple(np.round([Pk[u,u] for Pk in P],7)); b=tuple(np.round([Pk[v,v] for Pk in P],7))
            out.append((tuple(sorted((a,b))),tuple(np.round([abs(Pk[u,v]) for Pk in P],7))))
    return sorted(out)
n_unsep=0; n_l2=0; n_l3=0
for (n,spec),grs in pairs:
    for i in range(len(grs)):
        for j in range(i+1,len(grs)):
            (G1,L1,w1,V1),(G2,L2,w2,V2)=grs[i],grs[j]
            if np.abs(entropies(L1,w1,V1)-entropies(L2,w2,V2)).max()>1e-6: continue
            n_unsep+=1
            n_l2+= level2(L1,w1,V1)!=level2(L2,w2,V2)
            n_l3+= level3(L1,w1,V1)!=level3(L2,w2,V2)
print(f"\npairs not separated by entropy invariant: {n_unsep};  of these separated by the full per-vertex spectral-measure multiset: {n_l2};  by the pairwise-projector multiset: {n_l3}")
