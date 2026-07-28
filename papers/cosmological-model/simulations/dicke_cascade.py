"""
dicke_cascade.py -- specific-composition, persistence-aware multi-scale cascade.

Builds the --scales cascade dicke_layer.py flagged as NOT YET BUILT, using
three pieces already independently verified in this project rather than
inventing new machinery:

1. EXACT FACTORIZATION (dicke_cascade.py, verified by direct linear algebra,
   overlap=1.0 to machine precision): conditioning |D_n^k> on a window
   having exactly j excitations gives EXACTLY |D_w^j> (x) |D_{n-w}^{k-j}>.
   Used here to chain levels: level L's substrate is level (L-1)'s EXACT
   leftover (n - w_{L-1}, k - a_{L-1}), not the original n_bits again.

2. SPECIFIC-COMPOSITION MATCHING (dicke_layer.pattern_probability, verified
   against exact statevector diagonalization): the Born-rule probability of
   ONE SPECIFIC (a,b) pattern, a<b chosen for genuine hump behavior per the
   Entropy-and-Emergent-Structures wiki page. This REPLACES multiclock's
   level_state "j >= ceil(w/2)" occupancy-threshold promotion rule, which
   was arbitrary and never derived.

3. PERSISTENCE / SURVIVAL (derived and verified against direct simulation
   of the classical Ehrenfest process in this conversation): a structure
   only counts as "promoted" -- eligible to seed the next level -- if it
   also survives long enough to be a stable substrate, not just matched at
   a single instant. Uses the closed form

       survival(w, delta) = ((n_bits - w) / n_bits) ^ delta

   which is the probability NONE of the w bits in the window get touched
   over delta raw ticks (each tick flips one of the FULL n_bits uniformly
   at random -- this is why n_bits, not the leftover population size,
   appears here: ticks act on the whole substrate regardless of which
   "level" a bit conceptually belongs to). VERIFIED exactly against direct
   Ehrenfest simulation (see chat: closed-form vs simulated agree to
   simulation noise, delta=10..100, n=184, w=6).

FLAGGED, NOT DERIVED -- read before trusting numbers from this module:

  - Compositions (a, b) per level remain a modeling choice. No first-
    principles derivation exists for which composition is "the" physical
    one at a given width. Earlier exploration (a=w//3, then fixed a=1,2,3
    swept against a cosmological abundance ratio) showed the dependence on
    a is smooth and monotonic -- i.e. curve-fittable to nearly anything --
    so do not read agreement with any external ratio as validation unless
    a is fixed by an INDEPENDENT argument first.

  - The survival probability used is the CONSERVATIVE "literal freeze"
    notion: zero tolerance for any bit in the window being touched, even
    if the touch doesn't change the window's (a,b) classification. This is
    a lower bound on true persistence, not the tight answer. The more
    accurate "composition-preserving" survival (tolerates internal
    reshuffling that keeps the same a-vs-b class) has NOT been derived --
    it's a proper Markov chain over composition space, not a simple
    per-bit product, and is flagged as the next real piece of math needed
    here, not yet built.

  - delta (how many ticks the survival check spans) is set to the level's
    own width w -- "does the structure survive as long as it took to form"
    -- which is a natural, motivated choice (echoes the retard()/lapse
    mechanism's "duration = own size" idea) but is still a choice, not a
    forced consequence of anything derived.

VECTORIZED TIME-SERIES ADDITION (this revision)
------------------------------------------------
`run_cascade` (original, unchanged in spirit) evaluates the cascade at one
scalar global excitation count k -- useful for spot checks / tests. Driving
a plot over a t_bf grid of a few thousand points by calling it in a Python
loop would work but is wasteful and awkward to slice into per-level arrays
afterward. `run_cascade_series` below evaluates the SAME closed-form
cascade at every entry of a k-array at once:

  - n_substrate per level is a fixed integer (n_bits minus the widths of
    all earlier levels) -- it does not depend on k or time at all, so it's
    computed once per level, not per time-step.
  - survival_prob per level is likewise a constant (survival_probability()
    depends only on n_bits, width, delta -- never on k), so it is also
    computed once and simply broadcast across time via `survival_so_far`.
  - Only match_prob truly varies with time, through k_substrate(t) = k(t)
    minus the cumulative a's consumed by earlier levels -- and
    dl.pattern_probability already accepts an array of k values, so that
    call vectorizes for free.

This also fixes a piece of leftover dead code in the original `run_cascade`:
the first assignment to `cumulative_persistent` inside the loop was
immediately discarded and overwritten by the "simpler, unambiguous
recompute" line right after it. Harmless (the second line always won), but
confusing to read and easy to mistrust -- removed here, output unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

import dicke_layer as dl
from multiclock import combinatorial_entropy_bits


@dataclass
class LevelSpec:
    """One level's window width and composition. Composition is a flagged
    modeling choice -- see module docstring."""
    width: int
    a: int
    b: int

    def __post_init__(self) -> None:
        if self.a + self.b != self.width:
            raise ValueError(f"a+b must equal width: {self.a}+{self.b} != {self.width}")


@dataclass
class CascadeLevelResult:
    spec: LevelSpec
    n_substrate: int          # size of this level's EXACT leftover substrate
    k_substrate: int          # exact excitation count in that substrate
    match_prob: float         # Born-rule probability of matching (a,b) AT this level, GIVEN reaching it
    survival_prob: float      # probability the match also persists for `width` ticks
    cumulative_match_prob: float      # probability of matching every level 0..this one, in sequence
    cumulative_persistent_prob: float  # cumulative_match_prob * product of survival probs so far
    n_windows_available: int  # n_substrate // width -- how many disjoint copies could exist
    entanglement_entropy: float  # S_vN of this level's window, for reference (same n_substrate/k_substrate)


def survival_probability(n_bits: int, w: int, delta: float) -> float:
    """Closed-form, VERIFIED probability that none of a w-bit window's bits
    are touched over delta raw ticks, given each tick flips one of the FULL
    n_bits uniformly at random. See module docstring for the simulation
    check. This is the CONSERVATIVE (literal-freeze) notion -- flagged."""
    if w >= n_bits:
        return 0.0
    return ((n_bits - w) / n_bits) ** delta


def run_cascade(n_bits: int, k: int, levels: Sequence[LevelSpec]) -> list[CascadeLevelResult]:
    """Chain levels through the EXACT factorization at a single scalar k,
    multiplying in the survival factor at each level. A single
    deterministic chain (not a branching tree over all outcomes) because
    we specifically want the probability of matching THIS SPECIFIC
    composition at every level, not the full distribution over outcomes.
    """
    results: list[CascadeLevelResult] = []
    n_substrate, k_substrate = n_bits, k
    cumulative_match = 1.0
    survival_so_far = 1.0

    for spec in levels:
        if n_substrate <= 0 or k_substrate < 0 or spec.width > n_substrate:
            # substrate exhausted -- record zeros and stop contributing further
            results.append(CascadeLevelResult(
                spec=spec, n_substrate=max(n_substrate, 0), k_substrate=max(k_substrate, 0),
                match_prob=0.0, survival_prob=0.0,
                cumulative_match_prob=0.0, cumulative_persistent_prob=0.0,
                n_windows_available=0, entanglement_entropy=0.0,
            ))
            continue

        match_p = dl.pattern_probability(n_substrate, k_substrate, spec.a, spec.b)
        surv_p = survival_probability(n_bits, spec.width, spec.width)  # delta = own width, using FULL n_bits
        S_vN = dl.entanglement_entropy(n_substrate, k_substrate, spec.width) if k_substrate > 0 else 0.0

        cumulative_match *= match_p
        survival_so_far *= surv_p
        cumulative_persistent = cumulative_match * survival_so_far

        results.append(CascadeLevelResult(
            spec=spec, n_substrate=n_substrate, k_substrate=k_substrate,
            match_prob=match_p, survival_prob=surv_p,
            cumulative_match_prob=cumulative_match,
            cumulative_persistent_prob=cumulative_persistent,
            n_windows_available=n_substrate // spec.width,
            entanglement_entropy=S_vN,
        ))

        # EXACT leftover substrate for the next level (verified factorization)
        n_substrate = n_substrate - spec.width
        k_substrate = k_substrate - spec.a

    return results


# ===========================================================================
# VECTORIZED TIME-SERIES CASCADE (new)
# ===========================================================================

@dataclass
class CascadeSeriesResult:
    """Same fields as CascadeLevelResult, but every probability/entropy
    field is an array aligned with the k_array / t_bf grid passed in.
    n_substrate, survival_prob and n_windows_available stay scalars: they
    never depend on k or time, only on the (fixed) widths of this and
    earlier levels."""
    spec: LevelSpec
    n_substrate: int
    survival_prob: float
    n_windows_available: int
    match_prob: np.ndarray
    cumulative_match_prob: np.ndarray
    cumulative_persistent_prob: np.ndarray
    entanglement_entropy: np.ndarray


def run_parallel_series_retarded(
    n_bits: int, tau_raw: np.ndarray, levels: Sequence[LevelSpec],
    mode: str = "class",
) -> list[CascadeSeriesResultRetarded]:
    """Retarded-clock counterpart of run_parallel_series.

    Chaining (substrate nesting) and retardation (per-width clock lag)
    are independent axes: whether level i's substrate is level (i-1)'s
    leftover, or the same shared n_bits every level reads from, says
    nothing about whether a width-w structure ticks its own proper time
    at lapse = 1/w. It does -- a width-20 structure retards its own
    clock whether or not it happens to nest inside another level's
    leftover. This function keeps run_parallel_series's independence
    (no cumulative_a peeling, nothing multiplied across levels) while
    giving every level its own tau_local = retard(tau_raw, width), same
    as run_cascade_series_retarded.
    """
    if mode not in ("specific", "class"):
        raise ValueError(f"mode must be 'specific' or 'class', got {mode!r}")
    prob_fn = dl.pattern_probability if mode == "specific" else dl.class_probability

    tau_raw = np.asarray(tau_raw, dtype=float)
    results: list[CascadeSeriesResultRetarded] = []

    k_true_full = dl.k_of_tau(n_bits, tau_raw)

    for spec in levels:
        if spec.width > n_bits:
            zeros = np.zeros_like(tau_raw)
            results.append(CascadeSeriesResultRetarded(
                spec=spec, n_substrate=n_bits, survival_prob=0.0,
                n_windows_available=0, lapse=1.0 / spec.width, tau_local=zeros,
                match_prob=zeros, cumulative_match_prob=zeros,
                cumulative_persistent_prob=zeros, entanglement_entropy=zeros,
                pending=zeros,
            ))
            continue

        tau_local = spec.width * np.floor(tau_raw / spec.width)
        k_local_full = dl.k_of_tau(n_bits, tau_local)

        k_true = np.clip(np.round(k_true_full), 0, n_bits)
        k_eff = np.clip(np.round(k_local_full), 0, n_bits)

        match_p = prob_fn(n_bits, k_eff, spec.a, spec.b)
        surv_p = survival_probability(n_bits, spec.width, spec.width)
        persistent = match_p * surv_p  # NOT multiplied against any other level

        S_vN = np.array([
            dl.entanglement_entropy(n_bits, int(round(kk)), spec.width) if kk > 0 else 0.0
            for kk in k_eff
        ])

        entropy_true = n_bits * combinatorial_entropy_bits(n_bits, k_true)
        entropy_eff = n_bits * combinatorial_entropy_bits(n_bits, k_eff)
        pending = np.clip(entropy_true - entropy_eff, 0.0, None)

        results.append(CascadeSeriesResultRetarded(
            spec=spec, n_substrate=n_bits, survival_prob=surv_p,
            n_windows_available=n_bits // spec.width, lapse=1.0 / spec.width,
            tau_local=tau_local,
            match_prob=match_p,
            cumulative_match_prob=match_p.copy(),
            cumulative_persistent_prob=persistent.copy(),
            entanglement_entropy=S_vN,
            pending=pending,
        ))

    return results


def run_parallel_series(
    n_bits: int, k_array: np.ndarray, levels: Sequence[LevelSpec],
    mode: str = "class",
) -> list[CascadeSeriesResult]:
    """Independent-per-level counterpart to run_cascade_series.

    run_cascade_series CHAINS levels: level i's substrate is level (i-1)'s
    EXACT leftover (n_substrate -= width, k_substrate -= a), and
    cumulative_persistent_prob multiplies in every earlier level's
    match_prob and survival_prob. That's the right model for a genuine
    FORMATION HIERARCHY (e.g. "does dark matter's own leftover, after it
    forms, go on to also host visible structure") -- but it is the WRONG
    model for species that simply coexist rather than nest inside one
    another (e.g. an electron-type species and a quark-type species are
    not "the quark only exists among whatever the electron didn't use").
    Chaining forces exactly that kind of nesting, and because match
    probabilities are each < 1, chaining compounds them multiplicatively
    -- in practice this means whichever level is listed first dominates
    every later level by orders of magnitude, regardless of physical
    intent (see chat).

    This function instead evaluates every level on the SAME shared
    (n_bits, k(t)) -- dicke_layer.py's own original "species/scales share
    the SAME n and the SAME k(tau)" assumption, taken literally instead
    of overridden by a peeling recursion. Each level still gets its own
    survival_probability(n_bits, width, width) factor (that piece was
    independently derived/verified and has nothing to do with chaining),
    but nothing is multiplied ACROSS levels: level i's
    cumulative_persistent_prob is just its own match_prob * survival_prob,
    full stop.

    Use this for a set of levels meant to represent coexisting categories
    (e.g. dark matter as one entry, several visible fermion species as
    others). Use run_cascade_series for a genuine nested formation
    hierarchy instead.
    """
    if mode not in ("specific", "class"):
        raise ValueError(f"mode must be 'specific' or 'class', got {mode!r}")
    prob_fn = dl.pattern_probability if mode == "specific" else dl.class_probability

    k_array = np.asarray(k_array, dtype=float)
    results: list[CascadeSeriesResult] = []

    for spec in levels:
        if spec.width > n_bits:
            zeros = np.zeros_like(k_array)
            results.append(CascadeSeriesResult(
                spec=spec, n_substrate=n_bits, survival_prob=0.0,
                n_windows_available=0,
                match_prob=zeros, cumulative_match_prob=zeros,
                cumulative_persistent_prob=zeros, entanglement_entropy=zeros,
            ))
            continue

        k_clipped = np.clip(k_array, 0, n_bits)
        match_p = prob_fn(n_bits, k_clipped, spec.a, spec.b)
        surv_p = survival_probability(n_bits, spec.width, spec.width)
        persistent = match_p * surv_p  # NOT multiplied against any other level

        S_vN = np.array([
            dl.entanglement_entropy(n_bits, int(round(kk)), spec.width) if kk > 0 else 0.0
            for kk in k_clipped
        ])

        results.append(CascadeSeriesResult(
            spec=spec, n_substrate=n_bits, survival_prob=surv_p,
            n_windows_available=n_bits // spec.width,
            match_prob=match_p,
            cumulative_match_prob=match_p.copy(),        # no chaining: same as match_prob
            cumulative_persistent_prob=persistent.copy(),  # own survival only, not chained
            entanglement_entropy=S_vN,
        ))

    return results


@dataclass
class CascadeSeriesResultRetarded:
    """Same as CascadeSeriesResult, but computed on each level's OWN
    retarded clock (tau_local = width * floor(tau_raw / width)) instead
    of the shared raw tau_raw -- the Dicke-side counterpart of
    multiclock.ScaleResult / wavefunction_model.QuantumScaleResult.

    `pending` is the entropy-bit gap between what the raw tau_raw already
    knows about this level's substrate and what this level's own lagged
    clock has caught up to -- same ledger role as multiclock's
    pool_true - pool_eff. Wider (heavier) levels retard more slowly
    (lapse = 1/width), so they carry a larger, longer-lived pending
    backlog -- this is the GR-like "more bits -> slower proper time"
    effect, previously present in the classical/wavefunction backends
    but absent here.
    """
    spec: LevelSpec
    n_substrate: int
    survival_prob: float
    n_windows_available: int
    lapse: float
    tau_local: np.ndarray
    match_prob: np.ndarray
    cumulative_match_prob: np.ndarray
    cumulative_persistent_prob: np.ndarray
    entanglement_entropy: np.ndarray
    pending: np.ndarray


def run_cascade_series_retarded(
    n_bits: int, tau_raw: np.ndarray, levels: Sequence[LevelSpec],
    mode: str = "class",
) -> list[CascadeSeriesResultRetarded]:
    """Retarded-clock counterpart of run_cascade_series.

    The peeling of each level's fixed (a, b) target off the substrate is
    time-independent (a, b are constants, not functions of k or tau), so
    the leftover-substrate chain (n_substrate, cumulative_a) is exactly
    the same bookkeeping as the unretarded version. What changes is WHICH
    tau feeds dl.k_of_tau for each level's own match_prob/entanglement:
    level i uses its own tau_local_i = retard(tau_raw, width_i), not the
    shared tau_raw -- mirroring multiclock.retard()'s "a width-w structure
    only advances its own clock every w raw flips" rule exactly.
    """
    if mode not in ("specific", "class"):
        raise ValueError(f"mode must be 'specific' or 'class', got {mode!r}")
    prob_fn = dl.pattern_probability if mode == "specific" else dl.class_probability

    tau_raw = np.asarray(tau_raw, dtype=float)
    results: list[CascadeSeriesResultRetarded] = []

    n_substrate = n_bits
    cumulative_a = 0
    cumulative_match = np.ones_like(tau_raw)
    survival_so_far = 1.0

    k_true_full = dl.k_of_tau(n_bits, tau_raw)

    for spec in levels:
        if n_substrate <= 0 or spec.width > n_substrate:
            zeros = np.zeros_like(tau_raw)
            results.append(CascadeSeriesResultRetarded(
                spec=spec, n_substrate=max(n_substrate, 0), survival_prob=0.0,
                n_windows_available=0, lapse=1.0 / spec.width, tau_local=zeros,
                match_prob=zeros, cumulative_match_prob=zeros,
                cumulative_persistent_prob=zeros, entanglement_entropy=zeros,
                pending=zeros,
            ))
            continue

        tau_local = spec.width * np.floor(tau_raw / spec.width)
        k_local_full = dl.k_of_tau(n_bits, tau_local)

        k_substrate_true = np.clip(np.round(k_true_full - cumulative_a), 0, n_substrate)
        k_substrate_eff = np.clip(np.round(k_local_full - cumulative_a), 0, n_substrate)

        match_p = prob_fn(n_substrate, k_substrate_eff, spec.a, spec.b)
        surv_p = survival_probability(n_bits, spec.width, spec.width)
        survival_so_far *= surv_p

        cumulative_match = cumulative_match * match_p
        cumulative_persistent = cumulative_match * survival_so_far

        S_vN = np.array([
            dl.entanglement_entropy(n_substrate, int(round(kk)), spec.width) if kk > 0 else 0.0
            for kk in k_substrate_eff
        ])

        entropy_true = n_substrate * combinatorial_entropy_bits(n_substrate, k_substrate_true)
        entropy_eff = n_substrate * combinatorial_entropy_bits(n_substrate, k_substrate_eff)
        pending = np.clip(entropy_true - entropy_eff, 0.0, None)

        results.append(CascadeSeriesResultRetarded(
            spec=spec, n_substrate=n_substrate, survival_prob=surv_p,
            n_windows_available=n_substrate // spec.width, lapse=1.0 / spec.width,
            tau_local=tau_local,
            match_prob=match_p,
            cumulative_match_prob=cumulative_match.copy(),
            cumulative_persistent_prob=cumulative_persistent.copy(),
            entanglement_entropy=S_vN,
            pending=pending,
        ))

        n_substrate = n_substrate - spec.width
        cumulative_a += spec.a

    return results


def run_cascade_series(
    n_bits: int, k_array: np.ndarray, levels: Sequence[LevelSpec],
    mode: str = "specific",
) -> list[CascadeSeriesResult]:
    """Vectorized form of run_cascade: evaluate the whole cascade at every
    entry of k_array (typically one entry per simulation time-step) in
    closed form, without a Python loop over time.

    Exploits two facts that hold for every level regardless of k or t:
      - n_substrate is fixed once the earlier levels' widths are fixed.
      - survival_prob is fixed (survival_probability depends only on
        n_bits, width, delta -- never on k).
    Only match_prob genuinely varies with k, and dl.pattern_probability /
    dl.class_probability already vectorize over an array of k values, so
    that call is the only per-level array operation needed.

    Args:
        mode: "specific" (default, matches the original run_cascade) uses
            dl.pattern_probability -- the probability of one exact bit
            ordering. "class" uses dl.class_probability -- the probability
            of the whole (a,b) composition class, any ordering, matching
            the classical layer's count-based threshold. "specific"
            answers "how much distinguishing quantum information is
            here"; "class" answers "how much matter has formed". Picking
            "specific" for a matter signal silently suppresses it by
            C(width, a) per level -- see dicke_layer.class_probability's
            docstring. Default is kept as "specific" for backward
            compatibility with existing callers; pass "class" explicitly
            for the macroscopically-relevant matter signal.
    """
    if mode not in ("specific", "class"):
        raise ValueError(f"mode must be 'specific' or 'class', got {mode!r}")
    prob_fn = dl.pattern_probability if mode == "specific" else dl.class_probability

    k_array = np.asarray(k_array, dtype=float)
    results: list[CascadeSeriesResult] = []

    n_substrate = n_bits
    k_substrate = k_array.copy()
    cumulative_match = np.ones_like(k_array)
    survival_so_far = 1.0

    for spec in levels:
        if n_substrate <= 0 or spec.width > n_substrate:
            zeros = np.zeros_like(k_array)
            results.append(CascadeSeriesResult(
                spec=spec, n_substrate=max(n_substrate, 0), survival_prob=0.0,
                n_windows_available=0,
                match_prob=zeros, cumulative_match_prob=zeros,
                cumulative_persistent_prob=zeros, entanglement_entropy=zeros,
            ))
            continue

        valid = k_substrate >= 0
        k_clipped = np.clip(k_substrate, 0, n_substrate)

        match_p = prob_fn(n_substrate, k_clipped, spec.a, spec.b)
        match_p = np.where(valid, match_p, 0.0)

        surv_p = survival_probability(n_bits, spec.width, spec.width)
        survival_so_far *= surv_p

        cumulative_match = cumulative_match * match_p
        cumulative_persistent = cumulative_match * survival_so_far

        S_vN = np.array([
            dl.entanglement_entropy(n_substrate, int(round(kk)), spec.width) if ok and kk > 0 else 0.0
            for kk, ok in zip(k_clipped, valid)
        ])

        results.append(CascadeSeriesResult(
            spec=spec, n_substrate=n_substrate, survival_prob=surv_p,
            n_windows_available=n_substrate // spec.width,
            match_prob=match_p,
            cumulative_match_prob=cumulative_match.copy(),
            cumulative_persistent_prob=cumulative_persistent.copy(),
            entanglement_entropy=S_vN,
        ))

        # EXACT leftover substrate for the next level (verified factorization)
        n_substrate = n_substrate - spec.width
        k_substrate = k_substrate - spec.a

    return results
