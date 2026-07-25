# Validation performed

The supplied code was executed in the current environment.

## Automated tests

```text
Ran 7 tests
OK
```

The tests verify:

1. identity conditioning reproduces the exact discrete Ehrenfest
   Hamming-weight chain;
2. the forward/backward observer bridge agrees with brute-force enumeration of
   all small flip histories;
3. the required observer pattern is certain at the observer tick;
4. exact statevector Born probabilities equal the count-space distribution;
5. adding the complex phase codec changes phases but not pixel-basis Born
   probabilities;
6. Monte Carlo Born samples reproduce the analytic one fraction;
7. the end-to-end matter/size simulation remains finite and normalized.

## Default-size run

A run with `n=184`, a nine-pixel Gaussian observer, and the default scales
completed successfully. The scalable bridge was used; no `2**184` statevector
was allocated.

Representative diagnostics:

```text
P(observer)                         1.575140849384e-03
maximum Born entropy / n           0.99456522
maximum matter allocation / n      0.16576753
sampled vs analytic one-fraction   absolute error 6.956e-04 (512 samples)
```

The limiting `0.99456522` is `183/184`: at one exact discrete tick the
one-flip chain occupies only one parity sector, so the maximum screen entropy
is approximately `n - 1` bits rather than `n`.
