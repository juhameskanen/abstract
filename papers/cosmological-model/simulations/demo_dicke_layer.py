"""
Demo: the psi-layer's entanglement entropy, computed alongside the classical
run, at the actual working scale (n_bits=184, scales=[6,12,20]) -- not a
small-n toy. Entirely closed-form (dicke_layer.entanglement_entropy), so
this costs nothing extra even though n_bits is far too large for the
Tier-2 exact-statevector sandbox (that's capped at n<=24, used only to
verify the formulas this demo relies on -- see test_dicke_layer.py).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

fig, (ax_k, ax_s) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.patch.set_facecolor("white")

# top: k(tau) -- the clock -- with the Born-rule-typical band (mean +/- std of Binomial(n,1/2))
k_vals = mc.n_bits if False else N_BITS * (0.5 * (1 - np.exp(-mc.TRUE_K_RATE * t_bf / N_BITS)))
mean_eq = N_BITS / 2
std_eq = np.sqrt(N_BITS * 0.25)
ax_k.axhspan(mean_eq - std_eq, mean_eq + std_eq, color="gray", alpha=0.15,
             label="typical band (mean $\\pm$ 1 std of Binomial(n,1/2))")
ax_k.plot(t_bf, k_vals, color="black", lw=2, label="k($\\tau$) = n$\\cdot$p($\\tau$)  (the clock)")
ax_k.axhline(mean_eq, color="gray", lw=1, ls=":")
ax_k.set_ylabel("excitation count k")
ax_k.set_title(f"Clock trajectory vs. the flat state's typical (Born-rule) band  (n={N_BITS})")
ax_k.legend(loc="lower right", fontsize=8)

# bottom: entanglement entropy per scale, exact closed form, at the SAME k(tau) trajectory
for w, color in zip(SCALES, colors):
    S = np.array([dl.entanglement_entropy(N_BITS, max(1, int(round(k))), w) for k in k_vals])
    ax_s.plot(t_bf, S, color=color, lw=1.8, label=f"w={w}")
ax_s.set_xlabel("coordinate time $\\tau$")
ax_s.set_ylabel("$S_{vN}$ (bits)")
ax_s.set_title("Exact entanglement entropy of the Dicke-state window (closed form, no state-vector cost)")
ax_s.legend(loc="lower right", fontsize=8)

fig.tight_layout()
fig.savefig("dicke_layer_demo.png", dpi=150, facecolor="white")
print("saved dicke_layer_demo.png")
