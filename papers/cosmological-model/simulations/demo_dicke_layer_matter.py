"""
Demo: psi-layer clock + entanglement entropy + the quantum MATTER cascade
(dicke_cascade.recursive_cascade), all at the actual working scale
(n_bits=184, scales=[6,12,20]).

This extends the original dicke_layer-only demo. The new piece is the
bottom block of panels: at every point along the clock trajectory k(tau),
we run the exact recursive cascade (dicke_cascade.recursive_cascade) and
record, at EACH scale in --scales, what fraction of the probability mass
still "alive" going into that scale:

    - fell back to fabric      (j=0 in that window: no local structure)
    - formed structure here    (0<j<thresh: partial local excess, but not
                                 enough to be promoted -- this is "matter"
                                 that stops at this scale)
    - got promoted             (j>=thresh: enough local excess to survive
                                 as input to the NEXT, finer scale --
                                 i.e. this is the quantum analogue of
                                 multiclock.level_state's "rise" branch)

All three curves are normalized so they sum to 1 AT EACH SCALE (i.e. as a
fraction of the probability mass entering that scale, not of the original
whole) -- this is what makes the three scales visually comparable as a
nested hierarchy, exactly like the classical fall/bump/rise family
fractions in the D-layer model. Everything here is exact closed-form
combinatorics (hypergeom.pmf), same cost profile as the rest of this
codebase -- no state-vector construction anywhere.

Caveat carried over honestly from dicke_cascade.py's own docstring: this
peels off literal DISJOINT modes level by level (N shrinks), which is NOT
proven identical to multiclock's classical level_state cascade (that
rescales an abstract pool via retarded time on a FIXED n_bits). Only level
0 is guaranteed to match the classical fractions by construction. This
demo is showing the quantum-cascade's own self-consistent matter
hierarchy, not (yet) a verified match to the classical one.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import multiclock as mc
import dicke_layer as dl
import dicke_cascade as dc

N_BITS = 184
SCALES = [6, 12, 20]
STEPS = 800

sim = mc.run_simulation(n_bits=N_BITS, scales=SCALES, steps=STEPS)
t_bf = sim.t_bf

cmap = plt.get_cmap("plasma")
colors = [cmap(0.15 + 0.7 * i / max(len(SCALES) - 1, 1)) for i in range(len(SCALES))]

# ---------------------------------------------------------------------
# Clock: k(tau), same closed form as multiclock, reused (not re-derived)
# ---------------------------------------------------------------------
k_vals = N_BITS * (0.5 * (1 - np.exp(-mc.TRUE_K_RATE * t_bf / N_BITS)))
k_int = np.maximum(1, np.round(k_vals).astype(int))

# ---------------------------------------------------------------------
# Entropy per scale: exact closed form, independent window at each w
# ---------------------------------------------------------------------
S_per_scale = {
    w: np.array([dl.entanglement_entropy(N_BITS, k, w) for k in k_int])
    for w in SCALES
}

# ---------------------------------------------------------------------
# Matter cascade: run the exact recursive cascade at every tau, record
# the per-level (fabric, structure, promoted) fractions, normalized so
# they sum to 1 at each level (i.e. conditional on reaching that level).
# ---------------------------------------------------------------------
n_levels = len(SCALES)
fabric_frac = np.zeros((n_levels, STEPS))
structure_frac = np.zeros((n_levels, STEPS))
promoted_frac = np.zeros((n_levels, STEPS))
entropy_contrib = np.zeros((n_levels, STEPS))

for step, k in enumerate(k_int):
    summaries, _ = dc.recursive_cascade(N_BITS, int(k), SCALES)
    for lvl, s in enumerate(summaries):
        total = s.fabric_prob + s.structure_prob + s.promoted_prob
        if total > 0:
            fabric_frac[lvl, step] = s.fabric_prob / total
            structure_frac[lvl, step] = s.structure_prob / total
            promoted_frac[lvl, step] = s.promoted_prob / total
        entropy_contrib[lvl, step] = s.entropy_contributed

# ---------------------------------------------------------------------
# Figure: clock, entropy, then one stacked-fraction panel per scale
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(10, 4 + 2.4 * (2 + n_levels)))
fig.patch.set_facecolor("white")
gs = GridSpec(2 + n_levels, 1, height_ratios=[1.4, 1.4] + [1.0] * n_levels, figure=fig)

# --- panel 1: the clock, with the Born-rule-typical band ---
ax_k = fig.add_subplot(gs[0])
mean_eq = N_BITS / 2
std_eq = np.sqrt(N_BITS * 0.25)
ax_k.axhspan(mean_eq - std_eq, mean_eq + std_eq, color="gray", alpha=0.15,
             label="typical band (mean $\\pm$ 1 std of Binomial(n,1/2))")
ax_k.plot(t_bf, k_vals, color="black", lw=2, label="k($\\tau$) = n$\\cdot$p($\\tau$)  (the clock)")
ax_k.axhline(mean_eq, color="gray", lw=1, ls=":")
ax_k.set_ylabel("excitation count k")
ax_k.set_title(f"Clock trajectory vs. the flat state's typical (Born-rule) band  (n={N_BITS})")
ax_k.legend(loc="lower right", fontsize=8)

# --- panel 2: entanglement entropy per scale (independent windows) ---
ax_s = fig.add_subplot(gs[1], sharex=ax_k)
for w, color in zip(SCALES, colors):
    ax_s.plot(t_bf, S_per_scale[w], color=color, lw=1.8, label=f"w={w}")
ax_s.set_ylabel("$S_{vN}$ (bits)")
ax_s.set_title("Exact entanglement entropy of the Dicke-state window (closed form)")
ax_s.legend(loc="lower right", fontsize=8)

# --- panels 3..: matter cascade, one stacked-fraction plot per scale ---
for lvl, (w, color) in enumerate(zip(SCALES, colors)):
    ax_m = fig.add_subplot(gs[2 + lvl], sharex=ax_k)
    ax_m.stackplot(
        t_bf,
        fabric_frac[lvl], structure_frac[lvl], promoted_frac[lvl],
        labels=["fabric (j=0, no structure)",
                "structure (formed here, not promoted)",
                "promoted (feeds next scale)"],
        colors=["#dddddd", color, "black"],
        alpha=0.85,
    )
    ax_m.set_ylim(0, 1)
    ax_m.set_ylabel(f"fraction\n(w={w})")
    ax_m.set_title(f"Scale {lvl}: w={w} -- matter cascade (conditional on reaching this scale)",
                   fontsize=9)
    if lvl == 0:
        ax_m.legend(loc="upper right", fontsize=7, ncol=3)

ax_m.set_xlabel("coordinate time $\\tau$")
fig.tight_layout()
fig.savefig("dicke_layer_matter_demo.png", dpi=150, facecolor="white")
print("saved dicke_layer_matter_demo.png")
