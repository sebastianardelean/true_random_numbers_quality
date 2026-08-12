# Tests to Compare TRNG vs PRNG

Beyond what's already done (moments, chi-square, Wasserstein, MFDFA), the main
gaps are **independence/correlation**, **entropy**, and **binary-level** tests.

## Dependence / correlation
- Autocorrelation (lag plot, `acf`) and **Ljung-Box** test for serial correlation
- **Runs test** (`statsmodels.stats.runstest_1samp`) for above/below-median patterns
- Mutual information / permutation entropy as nonlinear dependence measures

## Entropy
- Shannon entropy on bins/blocks, block-entropy saturation vs. n (PRNGs can be
  detected via block entropy)
- Min-entropy / compression ratio (e.g. `gzip`/`lzma` on byte stream)

## Distribution (hypothesis tests, stronger than chi-square)
- **Kolmogorov-Smirnov** (`scipy.stats.kstest` vs. `randint`) and **Anderson-Darling**
- **Epps-Pulley**, or a two-sample **KS/Mann-Whitney** comparing TRNG vs PRNG directly

## Fractal / time-series (extends MFDFA)
- **Hurst exponent** via R/S analysis or DFA order dependence
- **Lempel-Ziv complexity** and **approximate/sample entropy** (lower = more structure)

## Standard suites (the gold standard)
- **NIST STS** (15 tests: frequency, runs, serial, spectral, longest-run, etc.)
- **Dieharder** or **TestU01** (Crush/BigCrush) - test the byte streams directly

## Classical battery
- Gap, poker, coupon-collector, birthday-spacing, overlapping-sums tests

## Two-sample direct comparison (TRNG vs PRNG, not vs. uniform)
- Two-sample KS, Mann-Whitney, permutation test on the two streams

## Spectral / practical
- **Spectral test** - FFT of the numeric stream to find periodicity (stronger
  than NIST's bit-level spectral test)
- **Monte Carlo integration** - accuracy/convergence of a pi estimate as a
  practical quality proxy
- **Ergodicity** - consistency of block-wise means/std across the stream
  (time avg vs. ensemble avg)

## Cellular automata (entropy-amplification / mixing test)
- Convert each generator to a bit stream, feed it as the 1D initial condition of
  a **chaotic CA** (rule 30 is ideal; rule 90 is linear/XOR and just propagates
  correlations)
- Evolve both for T steps; measure density of live cells over time, block
  entropy, and spectrum of the evolved stream
- A truly random seed stays "featureless" under rule 30; a structured seed
  leaves artifacts that survive/amplify
- Compare the two space-time diagrams directly with a divergence metric
- Caveat: a strong PRNG will usually pass this (rule 30 is itself a good RNG);
  useful as an additional dimension, not a discriminator by itself

## ML discriminator (strongest modern approach)
- **GST / ML discriminator** - train a classifier (e.g. LSTM) to distinguish
  TRNG from PRNG streams; the CA test is essentially a hand-crafted version
  of this

## Note
For a fair apples-to-apples comparison, convert both generators to uniform
**bits/bytes** first, then run the binary suites - that's where TRNGs vs PRNGs
typically separate (e.g. NIST p-values across trials, autocorrelation at bit level).
