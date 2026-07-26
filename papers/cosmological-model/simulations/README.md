# README

## Objective & Theoretical Framework

This simulation models cosmological evolution within the $D\text{-}\psi\text{-}G$ framework.
By assuming an initially zero-entropy state evolving toward thermal equilibrium in a typical relaxed system,
and applying the proven hump-shaped measure distribution for emergent substructures,
the framework yields a three-phase cosmological expansion profile as a high-probability outcome for an intrinsic observer.

## Key Assumptions & Conditioning

**Observer-Conditioned Dynamics:** Observational outcomes in this framework are conditioned on the existence of an observer.
While macroscopic observer states are not explicitly modeled, the system simulates micro-structural emergent entities 
analogous to elementary particles across scales. The peak in the probability density for emergent micro-structures corresponds
directly to the domain where the probability of observer emergence is maximal.

**Thermodynamic Arrow:** The monotonic increase of entropy is treated here as an empirical fact.
Within the broader mathematical framework, this thermodynamic arrow can be formally derived via Solomonoff induction.


## Usage

This folder implements three different cosmological toy models predicting the three-fold expansion profile.

They can be executed:

```bash
python cosmic.py statistical --scales 6,12,20
python cosmic.py dicke --scales 6,12,20
python cosmic.py wavefunction --scales 6,12,20 --t_bf_max 6
```

Or run all three with the checked-in configuration:

```bash
bash run
```

## Wavefunction 

The implicit state is

$$
\psi_\tau(x)=\sqrt{P_\tau(x)}e^{i\Phi_\tau(x)},
\qquad
P_\tau(x)=\prod_j p(\tau)^{x_j}[1-p(\tau)]^{1-x_j}.
$$

The phase residual is generated from a small DCT-like coefficient rule and
contains pair terms. Consequently, the state can be complex, entangled, and
position dependent; it is not restricted to one Dicke sector. The phase is
diagonal in the fabric basis, so

$$
|\psi_\tau(x)|^2=P_\tau(x)
$$

exactly. Born sampling therefore yields the statistical bitstring model seen
from inside.

## Cosmological ledger

The screen/Born entropy $nH_2[p(\tau)]$ is interpreted as unfolded spacetime resolution.
At each `--scale`, the exact binomial window probabilities are divided into falling, hump, and rising composition families.
The hump family supplies elementary microstructures, and *is* the matter observable directly:

$$
M_i(\tau)=\mathrm{structure}_i(\tau).
$$

There is no order-parameter weighting on top of this. An earlier version multiplied by
$\eta(\tau)^q$ (`--matter_power`, now removed): this forces $M_i\to 0$ at equilibrium for any
$q>0$ purely by construction, independent of what the hump family itself is doing there — i.e.
it silently erases the model's own equilibrium prediction (see below) rather than testing it.
Since `structure_i(\tau)` is already an exact Born-rule quantity, weighting it is not
needed and was hiding the real content of the model.

The size curve is

$$
R_Q=\frac{S_{\rm Born}-P_{\rm pending}-M_{\rm bits}}{n}.
$$

The default run detects rapid early growth, a matter-loading slowdown, and a
late positive recovery as the hump populations decline. There is no observer
bitmap or observer tick. An observer is a later higher-order assembly of the
same microstructures.

## Equilibrium remnant and the Sanov/large-deviation floor

As $\tau\to\infty$, $p(\tau)\to 1/2$ and the Born entropy saturates to $n$ bits exactly, but the
hump-family probability at each scale does **not** go to zero — a Bernoulli(1/2) window still has
some probability of landing on the target composition class, it's just exponentially small. This
is the model's own prediction of a permanent, non-fine-tuned remnant of structure at full
equilibrium (loosely: a Hawking-radiation-style tail, not a clean vacuum).

For the default composition rule (`dicke_layer.default_composition`, $a/w\to 1/3$), the
equilibrium hump probability at width $w$ is exactly a large-deviations (Sanov) rate:

$$
P_{\rm bump}(w;\,p=\tfrac12) \sim \exp\!\big(-w\,D_{\rm KL}(\tfrac13\,\|\,\tfrac12)\big),
\qquad D_{\rm KL}(\tfrac13\|\tfrac12)\approx 0.0566\ \text{nats/site}.
$$

This was checked directly against the code (not assumed): fitting $\log P_{\rm bump}(w)$ across
$w\in\{30,\dots,300\}$ gives an empirical decay rate of $0.0603$ nats/site against the predicted
$0.0566$ — the residual gap is the usual polynomial (Stirling) prefactor, not a discrepancy in
the exponent. So the "complex/large structures are exponentially suppressed" statement isn't a
qualitative gloss; it is the exact rate function of the binomial tail, derived, not fit.

Across a multi-scale hierarchy this cascades: each level's pool is the previous level's
**rise**-family probability (mass not yet consumed as fall or hump), so the total equilibrium
remnant fraction of $n$ is

$$
\rho(\text{scales}) \;=\; \sum_i \Big(\textstyle\prod_{j<i} f_{\rm rise}(w_j;\,p=\tfrac12)\Big)\, f_{\rm bump}(w_i;\,p=\tfrac12),
$$

implemented as `equilibrium_residual_fraction(scales)` in `wavefunction_model.py`. **This is the
correct target for "recovery,"** not zero: `end_suppression` should converge to $\rho$, and the
`three_stage_detected` diagnostic now checks convergence of the *excess* suppression above $\rho$,
not raw suppression. Convergence needs enough internal time — `--t_bf_max 2.0` is too short for
the default scales (`end_excess/peak_excess ≈ 15%` at that point); `--t_bf_max` of `4`–`6` converges
to machine precision against the analytic $\rho$. The shipped `run` script uses `6.0` for the
wavefunction backend for this reason.

## Robustness: `scales` is the one real free parameter

Everything else in the wavefunction backend either doesn't affect the physical (fabric-basis,
matter/size) observables at all, or shouldn't be treated as free:

- `--phase_strength`, `--spatial_modes`, `--temporal_modes`, `--pair_range`, `--phase_seed`,
  `--phase_topology` only affect the phase codec, i.e. entanglement/coherence diagnostics. They
  are provably inert for `entropy_bits`, `pending`, `matter_bits`, and `size_measure` — those
  depend only on the mean-field `p(\tau)` and the window-family split, never on the phase.
- `--matter_power` is gone (see above).
- `--n_bits`, `--t_bf_max` set the observation window and grid resolution, not the physics; results
  should be (and were checked to be) stable once `t_bf_max` is large enough to relax.

That leaves `--scales` — the window widths standing in for elementary-particle scales, per the
observer-centric reading (structure-probability peaks are where observer-probability peaks) — as
the genuine free input. It was swept over single scales, coprime triplets, and a 5-level
hierarchy (widths 2–50); the three-stage profile was detected in all cases once `t_bf_max` was
large enough. The remnant fraction $\rho$ varies meaningfully across these (from ~0.004 at
`w=50` to ~0.59 for a 5-level cascade), so `scales` is doing real, falsifiable work rather than
just toggling a fixed shape on and off.

Open thread, not yet resolved: `peak_t / n` (when matter formation peaks, in units of $n$) is
non-monotonic in $w$ at small widths and detection degrades by $w\gtrsim 100$ at the default grid
resolution (`--steps 3000`) — needs to be separated into "real physics" vs. "`default_composition`
discretization / grid-resolution artifact" before the $w\to$ epoch relationship (the part that
would actually connect a particle scale to a cosmic time) can be trusted.

## Files

- `wavefunction_model.py`: implicit complex state, phase codec, Born entropy,
  scale hierarchy, finite-budget cosmology, diagnostics.
- `cosmic_wavefunction.py`: plotting and CLI driver.
- `cosmic.py`: unified three-backend launcher.
- `test_wavefunction_model.py`: exact Born-shadow, entanglement, hump, ledger,
  and three-stage tests.

## Validation

Run:

```bash
python -m unittest -v test_wavefunction_model.py
python cosmic.py wavefunction --scales 6,12,20 --t_bf_max 6 --output general_wavefunction_cosmology.png
```

The wavefunction backend's own printout reports the analytic residual floor
(`equilibrium_residual_fraction`) alongside the numerically observed
`end_suppression`; they should agree to several decimal places once
`t_bf_max` is large enough. This is the strongest available cross-check,
since the floor is derived independently of the time-series simulation.

The phase codec is a finite MDL proxy, not a computation of uncomputable
Solomonoff complexity. It supplies one compact, symmetry-breaking complex lift
whose Born shadow is exact. A future universal-mixture layer can place an
induced measure over many such short codecs.
