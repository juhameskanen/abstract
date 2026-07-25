# README

This folder implements three different cosmological toy models predicting the three-fold expansion profile.

## Usage

```bash
python cosmic.py statistical --scales 6,12,20
python cosmic.py dicke --scales 6,12,20
python cosmic.py wavefunction --scales 6,12,20
```

## Wavefunction 

The implicit state is

\[
\psi_\tau(x)=\sqrt{P_\tau(x)}e^{i\Phi_\tau(x)},
\qquad
P_\tau(x)=\prod_j p(\tau)^{x_j}[1-p(\tau)]^{1-x_j}.
\]

The phase residual is generated from a small DCT-like coefficient rule and
contains pair terms. Consequently, the state can be complex, entangled, and
position dependent; it is not restricted to one Dicke sector. The phase is
diagonal in the fabric basis, so

\[
|\psi_\tau(x)|^2=P_\tau(x)
\]

exactly. Born sampling therefore yields the statistical bitstring model seen
from inside.

## Cosmological ledger

1. The screen/Born entropy \(nH_2[p(\tau)]\) is interpreted as unfolded
   spacetime resolution.
2. At each `--scale`, the exact binomial window probabilities are divided into
   falling, hump, and rising composition families.
3. The hump family supplies elementary microstructures. These use the same
   finite information budget as spacetime fabric.
4. The existing order-parameter weighting is retained explicitly:

   \[
   M_i(\tau)=\mathrm{structure}_i(\tau)\,\eta(\tau)^q,
   \qquad q=\texttt{--matter\_power}.
   \]

   The default is `q=1`, matching the current statistical backend. Setting
   `--matter_power 0` is the raw-hump/plateau ablation.
5. The size curve is

   \[
   R_Q=\frac{S_{\rm Born}-P_{\rm pending}-M_{\rm bits}}{n}.
   \]

The default run detects rapid early growth, a matter-loading slowdown, and a
late positive recovery as the weighted hump populations decline. There is no observer
bitmap or observer tick. An observer is a later higher-order assembly of the
same microstructures.

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
python cosmic.py wavefunction --output general_wavefunction_cosmology.png
```

The phase codec is a finite MDL proxy, not a computation of uncomputable
Solomonoff complexity. It supplies one compact, symmetry-breaking complex lift
whose Born shadow is exact. A future universal-mixture layer can place an
induced measure over many such short codecs.
