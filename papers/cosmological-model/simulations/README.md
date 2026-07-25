# Cosmological toy model: three-fold expansion without fine-tuning

## The idea

Take a length-`n` bitstring that starts all-zero and relaxes under the
symmetric Ehrenfest process (flip one random bit per tick). Two curves fall
out of that process with no free parameters:

1. **Entropy** — the substrate's Shannon entropy rises monotonically from 0
   toward its equilibrium maximum as the string randomizes.
2. **Emergence** — the probability of any fixed "structured" bit-pattern
   (composition `a` ones, `b` zeros with `a < b`) does *not* rise
   monotonically. It grows, peaks, and then decays back down to a small
   equilibrium plateau — a hump. (Derivation and exact closed forms:
   [Entropy and Emergent Structures](https://github.com/juhameskanen/abstract/wiki/Entropy-and-Emergent-Structures).)

An internal observer is itself built out of these emergent structures, so it
can only measure spacetime/resolution using the entropy budget *left over*
after the hump-shaped structures have taken their cut:

```
resolution(tau) ~ entropy(tau) - structure(tau)
```

Entropy minus a hump is, generically, a three-phase curve: fast early growth
(structures are still rare, almost all entropy goes into resolution), a
slowdown while the hump is near its peak (structure formation eats the
budget), and a late recovery as the hump decays back toward its small
equilibrium plateau and the leftover entropy budget grows again. This
three-fold expansion profile (rapid growth → matter-loading slowdown → late
recovery) falls out of the bit-flip relaxation process itself — no inflaton,
no dark energy, no tuned cosmological constant required.

This folder contains three independent implementations of that same idea,
built on different levels of formal machinery, all producing the same
qualitative three-fold profile.

## The three models

| Mode | Script | What it computes |
| --- | --- | --- |
| `statistical` | `cosmic_d.py` (math in `multiclock.py`) | Classical bitstring: mean-field Ehrenfest relaxation `p(tau)`, exact combinatorial entropy `log2 C(n,k)`, and exact hypergeometric fall/hump/rise window probabilities at each of several nested scales. |
| `dicke` | `cosmic_psi.py` (math in `dicke_layer.py`, `dicke_cascade.py`) | Quantum Dicke-state layer: the same relaxation viewed as an equal-superposition state over Hamming-weight sectors, with exact Born-rule window/pattern probabilities, entanglement entropy, and a cascade that chains scales through their exact leftover substrate. |
| `wavefunction` | `cosmic_wavefunction.py` (math in `wavefunction_model.py`) | General complex wavefunction `psi_tau(x) = sqrt(P_tau(x)) exp(i*Phi_tau(x))` with a compact spectral phase codec, so the state is genuinely complex/entangled/position-dependent, while a Born measurement in the fabric basis reproduces the same statistical shadow as the other two backends exactly. |

All three share the same underlying relaxation law and the same multi-scale
"nested clocks" construction (a width-`w` structure only advances its own
proper time once every `w` raw bit-flips, so heavier/larger structures age
more slowly — a GR-flavored lapse), just built up with progressively more
machinery (classical counting → exact quantum combinatorics → explicit
complex state).

## Requirements

```bash
pip install numpy scipy matplotlib
```

## Usage

Run any of the three backends through the unified launcher:

```bash
python cosmic.py statistical --scales 6,12,20
python cosmic.py dicke        --scales 6,12,20
python cosmic.py wavefunction --scales 6,12,20
```

Each accepts its own set of options (`--n_bits`, `--t_bf_max`, `--steps`,
`--matter_power`, `--output`, ...); run e.g. `python cosmic_d.py --help` for
the full list of a given backend. `--scales` sets the nested structure
widths (in bits); `--matter_power` sets the exponent `q` in the order-
parameter reweighting `M_i(tau) = structure_i(tau) * eta(tau)^q` used to
turn raw hump probability into "matter."

A convenience script runs all three with one shared configuration and saves
three PNGs:

```bash
./run
```

There is also an animated variant of the statistical backend that visualizes
differential time dilation between scales:

```bash
python cosmic_d_anim.py --scales 6,12,20 --output time_dilation.gif
```

Each run prints diagnostics to stdout (conservation-ledger error, where
"now" was auto-detected as the peak of total matter, and — for the
wavefunction backend — the analytic-vs-sampled Born check and a three-stage
detector for the rapid-growth / slowdown / recovery profile) and saves a
4-panel plot showing the resolution/size curve, the entropy budget, the
hump-shaped structure probabilities, and an expansion-rate proxy.

## Validation

```bash
python -m unittest -v test_wavefunction_model.py
```

The tests cover: the discrete Ehrenfest chain reproduced exactly from first
principles, an observer/history bridge checked against brute-force
enumeration, exact Born probabilities against direct statevector
computation, invariance of the fabric-basis Born distribution under the
phase codec, Monte Carlo sampling matching the analytic one-fraction, the
hump family being an interior peak rather than a monotonic edge case, and
the end-to-end run exhibiting the matter-driven three-stage profile.


## Files

- `multiclock.py` — classical Ehrenfest relaxation, exact combinatorial
  entropy, hypergeometric fall/hump/rise window probabilities, multi-scale
  nested-clock cascade.
- `dicke_layer.py` — Dicke-sector (equal-superposition) quantum primitives:
  sector probabilities, window marginals, entanglement entropy, exact
  pattern probabilities.
- `dicke_cascade.py` — chains `dicke_layer.py` levels through their exact
  quantum leftover substrate, with a persistence/survival correction.
- `wavefunction_model.py` — general complex wavefunction with a compact
  spectral phase codec; exact Born shadow; finite-budget cosmological
  ledger; diagnostics for the three-stage profile.
- `cosmic.py` — unified `{statistical|dicke|wavefunction}` launcher.
- `cosmic_d.py` / `cosmic_psi.py` / `cosmic_wavefunction.py` — per-backend
  plotting and CLI drivers.
- `cosmic_d_anim.py` — animated time-dilation demo built on `multiclock.py`.
- `test_wavefunction_model.py` — automated validation (see above).
- `run` — example shell script running all three backends with one shared
  configuration.
- `spectrum.json` — reference real-cosmology number densities by epoch
  (flat ΛCDM, Planck 2018 parameters), for external comparison; not
  consumed by any script in this folder.
