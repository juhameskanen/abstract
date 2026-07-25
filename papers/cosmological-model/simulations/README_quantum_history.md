# Observer-conditioned compressed quantum-history backend

This patch adds a third simulation choice beside the existing statistical
`cosmic_d.py` and Dicke `cosmic_psi.py` drivers.

```text
python cosmic.py statistical [existing cosmic_d.py arguments]
python cosmic.py dicke       [existing cosmic_psi.py arguments]
python cosmic.py history     [new cosmic_history.py arguments]
```

## Files

Copy these files into `papers/cosmological-model/simulations/`:

- `quantum_history.py` — exact observer-conditioned history and complex
  wavefunction core.
- `cosmic_history.py` — CLI, plots, and Born-sampling validation.
- `cosmic.py` — unified three-backend launcher. It does not modify the two
  existing drivers.
- `test_quantum_history.py` — regression tests.

No dependency beyond NumPy and Matplotlib is added.

## What is implemented

### 1. Exact discrete substrate history

The substrate uses the exact one-flip-per-tick Ehrenfest rule. There is no
mean-shell rounding and no independent-bit approximation in the history
measure.

### 2. Exact observer conditioning

An observer record is a finite pattern `O` required at an internal tick. The
selected pixels divide the bitstring into three exchangeable groups:

1. pixels required to be one;
2. pixels required to be zero;
3. all remaining fabric pixels.

A forward/backward Markov bridge computes

```text
P(group counts at tick t | observer record O)
```

for every internal tick. Thus the early black boundary and all later frame
marginals belong to the same complete history measure conditioned on the
observer.

### 3. Complex quantum lift

Every conditional frame is lifted to

```text
psi_t(x | O) = sqrt(P_t(x | O)) * exp(i Phi_t(x)).
```

`Phi_t(x)` is a low-rank spatial/temporal cosine phase codec with pairwise
controlled-phase terms. It is diagonal in the pixel basis, so

```text
|psi_t(x | O)|^2 = P_t(x | O)
```

exactly. Computational-basis Born samples therefore reproduce the
observer-conditioned statistical universe while retaining a genuinely complex
coherence residual.

The spatial phase basis is an inexpensive DCT-like proxy. It can later be
replaced by the intrinsic spectrum of the derived relational graph.

### 4. Scalable representation

For the default `n=184`, the code never allocates a `2**184` statevector. It
stores the exact count-space bridge. Full statevectors and half-chain
entanglement are constructed only for validation systems with `n <= 16` by
default.

### 5. Matter-pattern compatibility

The new driver accepts the same scale/composition convention as
`cosmic_psi.py`:

```text
--scales 6,12,20
--compositions 2:4,4:8,6:14
```

Pattern probabilities are calculated directly from the Born distribution in
the non-observer bulk, not from a selected Dicke sector. Literal persistence
and parallel/cascade choices are retained as explicit toy-model options.

## Examples

Default observer-conditioned run:

```bash
python cosmic.py history \
  --n_bits 184 \
  --observer gaussian:9 \
  --phase_topology ring \
  --phase_strength 0.9 \
  --output observer_conditioned_quantum_history.png
```

Explicit pixel observer:

```bash
python cosmic.py history \
  --observer pattern:00111100 \
  --observer_tick 184
```

Identity conditioning, useful for checking the exact unconditioned Ehrenfest
chain:

```bash
python cosmic.py history --observer none
```

Small exact statevector validation:

```bash
python cosmic.py history \
  --n_bits 12 \
  --t_bf_max 2 \
  --observer gaussian:5 \
  --observer_tick 10
```

Run tests:

```bash
python -m unittest -v test_quantum_history.py
```

## What the implementation demonstrates

- A globally specified, observer-conditioned history can begin at the all-zero
  boundary and display increasing Born-visible entropy.
- A complex phase residual can be added without changing the statistical pixel
  universe seen in the computational basis.
- The fixed-sector Dicke state is not required. The amplitudes range across the
  observer-conditioned count sectors, and the phase codec breaks full
  permutation symmetry.
- At a fixed discrete tick, parity remains visible. Near equilibrium the exact
  screen entropy approaches roughly `n - 1` bits rather than `n` because only
  one Hamming-weight parity is accessible at a time.

## Deliberate limitations

This is a new backend, not the final universal quantum model.

1. **Finite MDL proxy, not Solomonoff induction.** The codec-description number
   is computable bookkeeping. It is not Kolmogorov complexity and does not yet
   sum over all programs.
2. **The codec is selected by configuration.** A pixel-only record cannot
   distinguish different phase codecs that have the same Born magnitudes. A
   future interference-sensitive observer record or universal generator
   mixture must select the phase law.
3. **Frame wavefunctions share one smooth codec, but a single constrained
   global history-state Hilbert space is not yet constructed.** The classical
   history measure is conditioned exactly; the complex lift is presently a
   consistent family of conditional frame states.
4. **Matter definitions remain inherited toy classifiers.** Persistent,
   non-overlapping microstructures should eventually replace fixed
   compositions and the literal-freeze survival rule.
5. **The observer record is supplied explicitly.** The next model-selection
   layer should compare all short observer-compatible generators rather than
   accepting one codec as input.

These limitations are exposed in the API rather than hidden, so the backend can
serve as a controlled bridge from the old statistical/Dicke models to a future
universal algorithmic quantum-history model.
