# HMM Forward-Step Contract

## Scope

This internal contract covers the shared forward calculation used by `pthmm` and `hmm`.
It does not change the public Rust, CLI, or Python interfaces.

## Inputs

- An observation-probability vector with one value per score state.
- A banded transition matrix with one row per score state.
- The initial distribution when no prior forward variable exists; otherwise a
  state-sized previous forward variable.

The two observation and state vectors must have equal lengths. Existing constructors and
input validation remain responsible for producing compatible dimensions.

## Required behavior

1. On the first observation, use the initial distribution as the prior.
2. On subsequent observations, compute the same mathematical transition product as the full
   banded matrix operation, while omitting rows with zero forward probability and columns
   unreachable from the non-zero rows.
3. Preserve source-row order when accumulating destination probabilities.
4. Multiply the prior by the observation probabilities and normalize by
   `max(sum(values), 1e-6)`.
5. If the previous vector is all zero, return a zero forward vector without division by zero.
6. Select the first maximum as the current state, then preserve existing position, path,
   tempo, and debug-state behavior.

Optimized and full calculations must produce equivalent probabilities within the existing
HMM golden tolerances. The benchmark must demonstrate the intended work reduction on long
score inputs without a greater than 10% median regression on existing benchmark methods.

## Compatibility

- No configuration key, method selector, output shape, or error behavior changes.
- The Python reference environment identifies both pymatchmaker 0.5.2 and the locked
  python-hiddenmarkov version.
- Current user-facing documentation identifies the active release; historical release
  comparisons and asset attribution remain historically accurate.
