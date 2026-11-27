# Data Model: pymatchmaker 0.5.2 Sync

This feature changes no persisted runtime entities or public data formats. The following
concepts define the reference record and the shared HMM calculation that the implementation
and tests must preserve.

## Pinned Reference Record

| Field | Meaning | Validation |
|---|---|---|
| pymatchmaker tag | Released source that defines the expected behavior | Must identify `v0.5.2`. |
| python-hiddenmarkov version | Locked reference dependency used by pymatchmaker's HMM methods | Must satisfy `>=0.3.0`; the exact resolved version is recorded in generated metadata. |
| lockfile digest | Identity of the complete reference environment | Generated metadata must match the checked-in `tools/golden/uv.lock`. |

The two generators (MIDI and audio) must describe the same pymatchmaker release and lock
state. The hiddenmarkov version is relevant to the forward-step change and must be
inspectable without reconstructing the environment from a hash alone.

## Banded Transition Matrix

The HMM transition matrix is represented by one row band per score state:

- **State count**: number of score states.
- **Band start**: first represented destination column in that row.
- **Band values**: transition probabilities for consecutive destination columns.
- **Implicit entries**: all columns outside the represented band have probability zero.

For each populated row, represented probability values are non-negative and the row's
normalization is established when the transition model is constructed. No dense transition
matrix is required by the optimized forward step.

## Forward Variable and Reachable Range

The forward variable is either absent (first observation) or a state-sized vector of
normalized probabilities. For a populated vector, its **reachable range** is the half-open
interval from the first non-zero state through one past the last non-zero state. Its
**reachable destination bound** is one past the highest non-zero transition column in rows
of that range.

The reachable range can be empty when all forward probabilities are zero. This is a valid
degenerate state: the transition prior remains zero, observation weighting remains zero, and
normalization uses the existing `1e-6` lower bound rather than dividing by zero.

## Forward-Step State Transition

For an initial observation, the prior is the transition model's initial distribution. For
subsequent observations, the prior is the transpose product of the transition matrix and the
previous forward variable. The optimized calculation limits row and destination work to the
reachable range; it does not change the probability equation:

```text
next = observation_probability * prior
next /= max(sum(next), 1e-6)
```

The resulting state vector feeds the existing argmax, score-position lookup, and alignment
path append in both `pthmm` and `hmm`.

## Relationships

- The pinned reference record describes the input data used to create golden expectations.
- The banded transition matrix and forward variable belong to one HMM follower instance.
- Each observation advances the forward variable, current state, score position, and
  alignment path in the existing order.
- `pthmm` and `hmm` share the same transition and forward-step implementation; `hmm`
  additionally applies its IOI observation weight and tempo update around that shared step.
