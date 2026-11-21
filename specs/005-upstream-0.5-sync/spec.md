# Feature Specification: Follow Upstream pymatchmaker 0.5.1

**Feature Context**: `005-upstream-0.5-sync`
**Created**: 2026-11-21
**Status**: Draft
**Input**: User description: "es liegt eine neue version von pymatchmaker vor. ziehe die änderungen hier nach." (A new pymatchmaker release exists; pull its changes into this port.)

The pinned reference moves from pymatchmaker 0.4.0 to 0.5.1 (the latest release). The
upstream diff `v0.4.0..v0.5.1` touches twelve files; after removing pure reformatting and
code this port deliberately does not cover, four change groups remain:

1. **New follower `dannenberg`** (`matchmaker/dp/dannenberg.py`, a `methods.yaml` entry,
   and the new `score_cevts` reference builder in `matchmaker/registry.py`): a MIDI
   event-level dynamic-programming matcher after Bloch & Dannenberg (1985) whose reference
   is the score's compound events (one pitch set per unique onset beat).
2. **Evaluation smoothness metric** (`matchmaker/utils/eval.py`): the SPARC position path
   is now resampled at 3000 Hz instead of 50 Hz, windows with fewer than two samples report
   "insufficient samples", and per-window values are produced by a sliding-spectrum
   computation documented to equal the previous per-window result.
3. **MIDI HMM follower internals** (`matchmaker/prob/hmm.py`): the Bernoulli pitch
   observation model evaluates the shared uniform profile of inserted (odd) states once,
   and `PitchIOIHMM.step` restricts the forward step to reachable states with a guarded
   normalization (`max(sum, 1e-6)`). Both are documented upstream as value-preserving.
4. **Multi-fold hypothesis enumeration** (`matchmaker/dp/oltw_arzt_multi_fold.py`): the
   first four intermediate unfold variants are now produced lazily without enumerating all
   variants; traversal order and the resulting clones are unchanged.

Out of scope, with reasons recorded in `UPSTREAM.md`:

- `matchmaker/io/audio.py` reworks the real-time `wait` pacing of file input onto a feeder
  thread. Live-timing/thread plumbing is on the existing "Not ported" list and stays there.
- Pure reformatting and docstring edits in `particle_filter_korzeniowski.py`, `skf.py`
  and `oltw_arzt.py` change no behavior and require no port change.
- Upstream dependency floors (`partitura>=1.9.0`, `python-hiddenmarkov>=0.1.4`) are
  unchanged; those pins stay as the lockfile resolves them.

## User Scenarios & Testing

### User Story 1 - Existing methods match the 0.5.1 reference (Priority: P1)

A user runs any already delivered follower, processor, or evaluation and gets results that
match pymatchmaker 0.5.1, with golden data regenerated from the new pin.

**Why this priority**: Reference parity is the project's non-negotiable foundation; every
other change builds on a correctly re-pinned reference.

**Independent Test**: Bump the pins, regenerate the golden data, and run the full golden
suite; review that every changed golden number traces to one of the upstream change groups
above.

**Acceptance Scenarios**:

1. **Given** the pinned reference at 0.5.1, **When** golden data is regenerated, **Then**
   all golden tests pass within their documented tolerances.
2. **Given** the regenerated golden data diff, **When** a number changed, **Then** it is
   attributable to the SPARC sample-rate change or another listed upstream change, and
   numbers of untouched stages are identical.
3. **Given** an alignment evaluated for smoothness, **When** a window's position path has
   fewer than two samples, **Then** the window is omitted from the mean, as in the
   reference (which classifies it as having insufficient samples).
4. **Given** the MIDI HMM followers (`pthmm`, `hmm`), **When** they run on the golden
   inputs, **Then** positions and paths still meet the reference, including the
   inserted-states and restricted-forward-step revisions.
5. **Given** `arzt_multi_fold` on the repeat-structure fixtures, **When** hypotheses are
   built, **Then** the same clones are selected as before (the upstream change is
   enumeration-only).

---

### User Story 2 - Follow a MIDI performance with `dannenberg` (Priority: P2)

A user selects the new `dannenberg` method to follow a live or recorded MIDI performance
and obtains score positions from the windowed dynamic-programming matcher, with the same
defaults as the reference.

**Why this priority**: The only new user-facing capability of the release; valuable on its
own once the reference pin is correct.

**Independent Test**: Follow the fixed MIDI golden performances with `dannenberg` and
compare positions, alignment path, and evaluation metrics with the regenerated reference.

**Acceptance Scenarios**:

1. **Given** a score and a recorded MIDI performance, **When** `dannenberg` runs with
   defaults, **Then** positions and the alignment path meet the documented reference
   comparison limits.
2. **Given** the method registry, **When** `dannenberg` is selected by name through the
   library, the command line, or the Python bindings, **Then** it accepts the reference's
   configuration keys (`c_m`, `c_e`, `epsilon`, `epsilon_fraction`, `window`, `smoothing`,
   initial tempo) with the reference defaults.
3. **Given** a performance containing wrong, extra, or missing notes, **When** `dannenberg`
   follows it, **Then** the rating penalties and the best-row updates behave as in the
   reference (wrong first notes do not lock onto an arbitrary row).
4. **Given** no new input after a matched note, **When** the current position is queried,
   **Then** it advances with the smoothed tempo but never past the next score event, as in
   the reference.
5. **Given** the existing methods, **When** `dannenberg` is added, **Then** their defaults
   and golden results remain unchanged.

---

### User Story 3 - Updated upstream record and version surfaces (Priority: P3)

A contributor reads the repository and finds the pinned version, the module map, and the
not-ported list consistent with 0.5.1.

**Why this priority**: Keeps the port auditable; purely documentary once US1/US2 are done.

**Independent Test**: Check that the version table, the module map (a row for the new
follower module), the not-ported list, and the README's pinned release all state 0.5.1
state.

**Acceptance Scenarios**:

1. **Given** `UPSTREAM.md`, **When** the sync is complete, **Then** the version table says
   pymatchmaker 0.5.1, the module map has a row for the Dannenberg follower, and the
   not-ported list covers the real-time `wait` feeder.
2. **Given** the benchmark reference runner and its lockfile, **When** the pins are bumped,
   **Then** the locked environment installs pymatchmaker 0.5.1 and the benchmark compares
   against it.

---

### Edge Cases

- A score whose first performance note is wrong: the Dannenberg matcher's initial column
  penalizes skipped score events, so matching must start at the first compound event rather
  than locking onto any row in the window.
- A score with a single compound event, or a window reaching past the last event: the
  matcher must clamp its row window and still report whether it is following.
- Notes arriving closer together than the grouping threshold (chords): the epsilon and
  epsilon-fraction tests classify them into one performance compound event; the reference
  computes but currently ignores this classification during matching — the port mirrors
  that behavior rather than "fixing" it.
- Smoothness windows at 3000 Hz on short paths: windows whose resampled path has fewer
  than two samples report insufficient samples; entirely stationary or band-undefined
  windows keep their existing reasons.
- The restricted HMM forward step when the forward variable is all zeros or on the first
  observation: the reference falls back to the unrestricted step; degenerate sums are
  guarded, never divide by zero.

## Requirements

### Functional Requirements

- **FR-001**: The pinned reference MUST move to pymatchmaker 0.5.1 in the locked golden
  environment, and the golden data MUST be regenerated from that pin.
- **FR-002**: All previously delivered followers, processors, score handling, and
  evaluation MUST meet their golden tests against the regenerated 0.5.1 data within the
  tolerances documented next to each test.
- **FR-003**: The smoothness metric MUST resample the held position path at the
  reference's new rate (3000 Hz), MUST report windows with fewer than two samples as
  insufficient, and MUST keep per-window values equal to the previous per-window
  computation at the same rate.
- **FR-004**: A new `dannenberg` method MUST follow MIDI performances with the windowed
  dynamic-programming matcher of the reference, using score compound events (one pitch set
  per unique onset beat) as its reference features and exposing the reference defaults
  (`c_m` 0.5, `c_e` 0.0, `epsilon` 0.090 s, `epsilon_fraction` 0.25, `window` 50,
  `smoothing` 0.5, initial tempo 120).
- **FR-005**: `dannenberg` MUST be selectable wherever existing MIDI methods are
  (library, command line, Python bindings) and MUST reject configuration keys the
  reference does not accept, consistent with existing method-configuration handling.
- **FR-006**: `dannenberg` MUST report its current position by interpolating from the last
  matched event with the smoothed tempo, capped at the next score event, and MUST report
  that it stopped following only past the last score event, as in the reference.
- **FR-007**: The MIDI HMM followers MUST adopt the reference's revisions (shared uniform
  profile for inserted states evaluated once; forward step restricted to reachable states
  with guarded normalization) in a way that keeps their golden results within the
  documented tolerances; where the port already realizes a revision equivalently, that
  equivalence MUST be recorded instead of changed, and any observable deviation MUST be
  traced to a specific upstream line and documented.
- **FR-008**: Multi-fold hypothesis building MUST produce the same bounded set of unfold
  variants as the reference's lazy enumeration; if the port already enumerates
  equivalently, that equivalence MUST be recorded instead of changed.
- **FR-009**: `UPSTREAM.md` (version table, module map, not-ported list) and the README's
  pinned release MUST state the 0.5.1 state, including the not-ported real-time `wait`
  feeder of file input.
- **FR-010**: The new follower MUST satisfy the same cross-cutting rules as existing
  methods: golden tests, exportable intermediate results, documented public items, and no
  panics on invalid user input.

### Key Entities

- **Score compound event**: the set of MIDI pitches sharing one unique onset beat; the
  reference feature sequence of the Dannenberg matcher, ordered by onset beat.
- **Dannenberg matcher state**: the current and previous rating columns over score
  compound events, the pitches already credited per event, the best rating and row, and
  the smoothed tempo anchor used for position interpolation.
- **Smoothness window**: an onset-anchored time window over the resampled held position
  path, carrying a value and a reason ("ok", "stationary", "undefined band", "uncovered",
  "insufficient samples").

## Success Criteria

### Measurable Outcomes

- **SC-001**: With the reference pinned at 0.5.1, the complete golden suite passes, and
  the regenerated golden-data diff contains no change that cannot be traced to a listed
  upstream change group.
- **SC-002**: On the fixed MIDI golden performances, `dannenberg` reproduces the
  reference's positions and alignment path within the documented comparison limits, and
  its evaluation metrics equal the reference's for the same runs.
- **SC-003**: Smoothness values computed for the existing golden alignments equal the
  0.5.1 reference values; windows shorter than two samples are reported as insufficient
  rather than skewing the mean.
- **SC-004**: The benchmark report after the sync shows no regression beyond the
  project's 10 % latency threshold for existing methods, and includes the new method side
  by side with the reference.
- **SC-005**: A reader of the upstream record finds every 0.4.0→0.5.1 upstream change
  either mapped to a port change or listed as not ported with a reason.

## Assumptions

- The sync targets 0.5.1 (the latest release), not the intermediate 0.5.0; 0.5.1 contains
  0.5.0's changes plus the smoothness, HMM, and file-pacing revisions.
- The real-time `wait` feeder of file input remains out of scope, like all live
  timing/thread plumbing; offline frame iteration is unaffected by the upstream rework.
- Upstream's HMM revisions are value-preserving as documented; golden regeneration is the
  arbiter, and any measured deviation is handled under FR-007.
- The reference computes but ignores the performance compound-event grouping during
  Dannenberg matching; the port mirrors the reference's actual behavior at the tag, per
  the project's parity principle.
- Dependency floors of partitura and python-hiddenmarkov are unchanged; the locked
  versions may still move within those floors when the lockfile is regenerated, which the
  golden regeneration absorbs.
- No new Git branch was requested; the spec directory name stands alone.
