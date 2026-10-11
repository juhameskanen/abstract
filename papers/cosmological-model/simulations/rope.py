#!/usr/bin/env python3
"""
rope.py - Minimal "rope with knots": entropy growth = space expansion;
knots (filter matches) replace w bits by one symbol => space contracts.

  python rope.py [--n 300] [--mask 0100] [--overlap] [--contract per_knot|union] [--plot]

CHOICES (mine, not derived from the framework):
  C1  entropy      : S = N*h(p), h = binary Shannon entropy in bits (full string)
  C2  length       : L = 1 + S - contraction            (clipped at 1; see warnings)
  C3  contraction  : per_knot: (w-1)*M      -- your rule; double-counts shared bits if --overlap
                     union   : |covered bits| - M   -- shared bits counted once
                     (identical without --overlap)
  C4  matching     : default greedy non-overlapping; --overlap = every sliding match
  C5  mask         : a ones, b zeros -> hump peak at p=a/(a+b), needs <0.5 for a hump
  C6  dynamics     : one random bit-flip per step, start all zeros
  Mixed units: S is in entropy-bits, contraction in symbol counts (equal only at p=1/2).
"""
import argparse, math, random

def h(p): return 0.0 if p <= 0 or p >= 1 else -(p*math.log2(p)+(1-p)*math.log2(1-p))

def borders(mask):
    w = len(mask); return [w-k for k in range(1, w) if mask[k:] == mask[:w-k]]

def find_matches(bits, mask, overlap):
    w, i, out = len(mask), 0, []
    while i <= len(bits)-w:
        if bits[i:i+w] == mask:
            out.append(i); i += 1 if overlap else w
        else: i += 1
    return out

def measure(bits, mask, overlap, contract):
    n, w = len(bits), len(mask)
    starts = find_matches(bits, mask, overlap)
    covered = [False]*n
    for s in starts:
        for j in range(s, s+w): covered[j] = True
    M, U = len(starts), sum(covered)
    S = n*h(sum(bits)/n)
    c = (w-1)*M if contract == "per_knot" else U-M
    L_raw = 1 + S - c
    return starts, covered, max(1.0, L_raw), 1+S, L_raw < 1-1e-9

def draw(bits, mask, overlap, contract):
    n, w = len(bits), len(mask)
    starts, covered, L, _, _ = measure(bits, mask, overlap, contract)
    M = len(starts)
    # chains of overlapping matches, drawn  o=o ; free bits between chains
    chains = []
    for s in starts:
        if chains and s < chains[-1][-1]+w: chains[-1].append(s)
        else: chains.append([s])
    free_runs, prev = [], 0
    for ch in chains:
        free_runs.append(ch[0]-prev); prev = ch[-1]+w
    free_runs.append(n-prev)
    D = max(0, round(L)-M) if M else round(L)
    D = max(D, 1 if M == 0 else 0)
    tot = sum(free_runs) or 1
    acc_f = acc_d = 0; out = ""
    for k, g in enumerate(free_runs):
        acc_f += g; tgt = round(D*acc_f/tot); out += "-"*(tgt-acc_d); acc_d = tgt
        if k < len(chains): out += "=".join("o"*1 for _ in chains[k])
    return out, L, M

def run(n, mask, steps, seed, overlap, contract):
    rng = random.Random(seed); bits = [0]*n; traj = []
    for t in range(steps+1):
        st, _, L, L0, bad = measure(bits, mask, overlap, contract)
        traj.append((t, len(st), L, L0, bad))
        bits[rng.randrange(n)] ^= 1
    return bits, traj

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--mask", default="0100")
    ap.add_argument("--steps", type=int, default=0); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--runs", type=int, default=20); ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--contract", choices=["per_knot", "union"], default="per_knot")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args(); mask = [int(c) for c in a.mask]; steps = a.steps or a.n
    print(f"N={a.n} mask={a.mask} overlap={a.overlap} contraction={a.contract} "
          f"possible shared-bit counts between knots={borders(mask) or 'none (overlap has no effect)'}\n")
    rng = random.Random(a.seed); bits = [0]*a.n
    ts = sorted(set(max(0, int(round(steps**(k/(a.frames-1))))-1) for k in range(a.frames)))
    print(f"{'step':>6} {'L':>7} {'knots':>5}  rope  ('=' joins knots sharing bits)")
    for tt in range(steps+1):
        if tt in ts:
            s, L, M = draw(bits, mask, a.overlap, a.contract); print(f"{tt:6d} {L:7.1f} {M:5d}  {s}")
        bits[rng.randrange(a.n)] ^= 1
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(7, 9), sharex=True); clipped = {}
        for ov, c, lab in [(False,"per_knot","no overlap"),(True,"per_knot","overlap, (w-1)/knot"),(True,"union","overlap, union")]:
            R = [run(a.n, mask, steps, a.seed+r, ov, c)[1] for r in range(a.runs)]
            T = [x[0] for x in R[0]]; av = lambda k: [sum(r[i][k] for r in R)/len(R) for i in range(len(T))]
            M, L, L0 = av(1), av(2), av(3)
            clipped[lab] = sum(x[4] for r in R for x in r)/ (len(R)*len(T))
            ax[0].plot(T, L, label=lab); ax[1].plot(T, M, label=lab)
            ax[2].plot(T, [l/l0 for l, l0 in zip(L, L0)], label=lab)
        ax[0].plot(T, L0, "k--", label="no knots"); ax[0].set_ylabel("L"); ax[0].legend()
        ax[1].set_ylabel("knots"); ax[2].set_ylabel("L / L_noknots"); ax[2].set_xlabel("step")
        plt.tight_layout(); plt.savefig("rope.png", dpi=120)
        print("\nfraction of steps where L was clipped at 1 (rule gave L<1):", clipped)
        
