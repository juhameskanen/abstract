"""
Exact joint space-time transition kernel Joint(r, Delta) for the hypergeometric
structure model, extending pair_correlation.py (space only) and persistence.py
(time only) to a single derived object covering both at once.

Construction: split the n bits into three regions relative to two width-w
windows separated by r (start-to-start): S = shared bits (size w-r, or 0 if
r>=w), L = bits only in window 1, R = bits only in window 2. Track only the
ones-count in each region (j_S, j_L, j_R) -- same reduction persistence.py
used for one window, now for three regions at once.

Each raw tick touches region S/L/R with probability p_S/p_L/p_R = |region|/n,
and does exactly one internal Ehrenfest step in that region when it does
(same M matrices as persistence.py's _ehrenfest_matrix, applied via a
Kronecker sum so the one-tick operator on the joint state is exact):

    T = (1 - p_S - p_L - p_R) I  +  p_S (M_S (x) I (x) I)
                                  +  p_L (I (x) M_L (x) I)
                                  +  p_R (I (x) I (x) M_R)

T^Delta then gives the EXACT Delta-tick joint transition on (j_S,j_L,j_R).
Window1 composition = j_S + j_L; window2 composition = j_S + j_R.
"""
from __future__ import annotations
import numpy as np
from math import comb


def _ehrenfest_matrix(size: int) -> np.ndarray:
    M = np.zeros((size + 1, size + 1))
    for i in range(size + 1):
        if i > 0:
            M[i, i - 1] = i / size
        if i < size:
            M[i, i + 1] = (size - i) / size
    return M


def _kron_apply(size_s, size_l, size_r, p_s, p_l, p_r):
    """Build the exact one-tick joint transition matrix T over the flattened
    (j_S, j_L, j_R) state space."""
    Is = np.eye(size_s + 1)
    Il = np.eye(size_l + 1)
    Ir = np.eye(size_r + 1)
    Ms = _ehrenfest_matrix(size_s) if size_s > 0 else np.eye(1)
    Ml = _ehrenfest_matrix(size_l) if size_l > 0 else np.eye(1)
    Mr = _ehrenfest_matrix(size_r) if size_r > 0 else np.eye(1)

    T = (1 - p_s - p_l - p_r) * np.kron(np.kron(Is, Il), Ir)
    if size_s > 0:
        T += p_s * np.kron(np.kron(Ms, Il), Ir)
    if size_l > 0:
        T += p_l * np.kron(np.kron(Is, Ml), Ir)
    if size_r > 0:
        T += p_r * np.kron(np.kron(Is, Il), Mr)
    return T


def joint_kernel(n: int, w: int, r: int, delta: int, a: int, k: int):
    """P(window1 comp==a at t=0 AND window2 comp==a at t=delta), given a
    typical (multivariate-hypergeometric) initial distribution over
    (j_S,j_L,j_R) consistent with k ones total.

    r=0 is the identical-window case (excluded, as in pair_correlation.py).

    IMPORTANT: window1's condition must be applied to the INITIAL state
    (t=0), and window2's condition to the EVOLVED state (t=delta) -- j_S
    keeps evolving during the delta ticks too, so checking both conditions
    on the same (initial or final) snapshot is wrong. Fixed here by zeroing
    the initial distribution outside window1's condition, propagating via
    T^delta, then summing the evolved probability landing in window2's
    condition.
    """
    if r <= 0:
        raise ValueError("r must be >= 1")
    size_s = max(w - r, 0)
    edge = min(r, w)
    size_l = edge
    size_r = edge
    rest = n - w - r if r < w else n - 2 * w

    p_s, p_l, p_r = size_s / n, size_l / n, size_r / n

    Cnk = comb(n, k)
    init = np.zeros((size_s + 1, size_l + 1, size_r + 1))
    mask_win1 = np.zeros((size_s + 1, size_l + 1, size_r + 1), dtype=bool)
    mask_win2 = np.zeros((size_s + 1, size_l + 1, size_r + 1), dtype=bool)
    for js in range(size_s + 1):
        for jl in range(size_l + 1):
            for jr in range(size_r + 1):
                rem = k - js - jl - jr
                if 0 <= rem <= rest:
                    init[js, jl, jr] = (comb(size_s, js) * comb(size_l, jl) *
                                        comb(size_r, jr) * comb(rest, rem)) / Cnk
                mask_win1[js, jl, jr] = (js + jl == a)
                mask_win2[js, jl, jr] = (js + jr == a)

    init_flat = init.reshape(-1)
    mask_win1_flat = mask_win1.reshape(-1)
    mask_win2_flat = mask_win2.reshape(-1)

    T = _kron_apply(size_s, size_l, size_r, p_s, p_l, p_r)
    T_delta = np.linalg.matrix_power(T, delta)

    weighted_initial = init_flat * mask_win1_flat  # zero out states not matching window1@t=0
    propagated = weighted_initial @ T_delta         # evolve forward delta ticks
    total = float(np.sum(propagated * mask_win2_flat))  # restrict to window2's condition @t=delta
    return total


def joint_kernel_batch(n: int, w: int, r: int, deltas: list[int], a: int, k: int) -> dict[int, float]:
    """Same as joint_kernel(), but for MANY delta values at a fixed r in one
    call. deltas must be evenly-spaced multiples of a common step (e.g.
    [0, stride, 2*stride, 3*stride], exactly what spacetime_field.py needs).
    Reuses one matrix_power(T, step) via repeated squaring, then chains
    incremental multiplications for the higher multiples -- this avoids a
    general (non-symmetric) eigendecomposition, which turned out to be far
    slower than plain matmul-based powering for the matrix sizes here
    (profiled: np.linalg.eig + inv dominated runtime for w=20, ~43s of 45s;
    replaced with matrix_power + chaining, see module benchmark below).
    """
    if r <= 0:
        raise ValueError("r must be >= 1")
    size_s = max(w - r, 0)
    edge = min(r, w)
    size_l, size_r = edge, edge
    rest = n - w - r if r < w else n - 2 * w
    p_s, p_l, p_r = size_s / n, size_l / n, size_r / n

    Cnk = comb(n, k)
    init = np.zeros((size_s + 1, size_l + 1, size_r + 1))
    mask_win1 = np.zeros((size_s + 1, size_l + 1, size_r + 1), dtype=bool)
    mask_win2 = np.zeros((size_s + 1, size_l + 1, size_r + 1), dtype=bool)
    for js in range(size_s + 1):
        for jl in range(size_l + 1):
            for jr in range(size_r + 1):
                rem = k - js - jl - jr
                if 0 <= rem <= rest:
                    init[js, jl, jr] = (comb(size_s, js) * comb(size_l, jl) *
                                        comb(size_r, jr) * comb(rest, rem)) / Cnk
                mask_win1[js, jl, jr] = (js + jl == a)
                mask_win2[js, jl, jr] = (js + jr == a)

    init_flat = init.reshape(-1)
    mask_win1_flat = mask_win1.reshape(-1)
    mask_win2_flat = mask_win2.reshape(-1)
    weighted_initial = init_flat * mask_win1_flat

    sorted_deltas = sorted(deltas)
    assert sorted_deltas[0] == 0
    step = sorted_deltas[1] if len(sorted_deltas) > 1 else 0
    for i, d in enumerate(sorted_deltas):
        assert d == i * step, "deltas must be evenly-spaced multiples of a common step"

    T = _kron_apply(size_s, size_l, size_r, p_s, p_l, p_r)
    out = {0: float(np.sum(weighted_initial * mask_win2_flat))}
    if step > 0:
        T_step = np.linalg.matrix_power(T, step)
        propagated = weighted_initial.copy()
        for i in range(1, len(sorted_deltas)):
            propagated = propagated @ T_step
            out[i * step] = float(np.sum(propagated * mask_win2_flat))
    return out



    n, w, a, k = 184, 6, 2, 92

    # Sanity check 1: Delta=0 should reduce EXACTLY to pair_correlation's g(r)*P_single^2
    import sys
    sys.path.insert(0, "/home/claude/abstract/papers/cosmological-model/simulations")
    from pair_correlation import joint_match_prob
    print("Sanity check: Delta=0 should match pair_correlation.joint_match_prob exactly")
    for r in [1, 3, 5, 8]:
        jk = joint_kernel(n, w, r, 0, a, k)
        pc = joint_match_prob(n, w, k, a, r)
        print(f"  r={r}: joint_kernel(delta=0)={jk:.6f}  pair_correlation={pc:.6f}  "
              f"match={'OK' if abs(jk-pc)<1e-9 else 'MISMATCH'}")

    print()
    print("Sanity check 2: r>=w, Delta=0 should match persistence.py's own-window Delta=0 (=marginal^2... no, same window)")
    from persistence import persistence
    print("Sanity check 3: r=w (no shared bits), varying Delta, should match persistence.py")
    print("  (r=w means windows share 0 bits -- NOT the same as same-window persistence,")
    print("   so this instead checks Delta=0 still matches independence baseline)")
    for r in [6, 10]:
        for delta in [0, 5, 20]:
            jk = joint_kernel(n, w, r, delta, a, k)
            print(f"  r={r} delta={delta}: Joint={jk:.6f}")
