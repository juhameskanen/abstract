import random
from unary import Unary as U
# ---------- binary reference: 8 bits, observer = counter 0..255 ----------
binary_frames = [format(k,'08b') for k in range(256)]

# ---------- Implementation A: one long unary string; frame = prefix ----------
universe = U.from_int(255)                 # 255 marks (n = 255)
frames, f = [U()], U()
while f < universe:                        # observer's own progress = successor
    f = f.succ(); frames.append(f)
print("A: n =", len(universe.s), "marks; frames =", len(frames))
assert all(frames[i] < frames[i+1] and frames[i+1]==frames[i].succ() for i in range(255))
labels=[x.to_binary().zfill(8) for x in frames]
print("A: frame labels (by pairing-off) == binary counter:", labels==binary_frames)
print("A: info per frame: unary", len(frames[-1].s), "marks vs binary 8 bits")

# ---------- Implementation B: 8 indistinguishable digits; frame = subset marked ----------
# Raw frames are subsets of 8 digit positions; the counting traversal:
def traversal(perm):
    # perm[j] = which physical digit plays 'bit j'; physical digits carry no label of their own
    return [frozenset(perm[j] for j in range(8) if (k>>j)&1) for k in range(256)]
base = traversal(list(range(8)))
shuf = list(range(8)); random.Random(3).shuffle(shuf)
other = traversal(shuf)
print("\nB: 256 distinct frames:", len(set(base))==256, "| relabelled traversal differs as raw sets:", base!=other)
# reader recovers digit roles from the traversal alone: flip period of each digit
def roles(tr):
    per={}
    for d in range(8):
        flips=[k for k in range(1,256) if (d in tr[k])!=(d in tr[k-1])]
        per[d]=flips[0]            # first flip time: 1,2,4,...,128
    return sorted(per.values())
print("B: digit roles recovered from flip timing (base):   ", roles(base))
print("B: digit roles recovered from flip timing (shuffled):", roles(other))
# cardinality-only view (what unary digits themselves carry):
print("B: frames distinguishable by cardinality alone:", len({len(s) for s in base}), "classes (0..8)")
