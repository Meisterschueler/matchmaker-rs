# Feature Specification: MIDI Score Following (pthmm, hmm, outerhmm)

**Feature Branch**: `001-midi-score-following`

**Created**: 2026-10-28

**Status**: Draft

**Input**: User description: "Port pymatchmaker (v0.4.0) to pure Rust. First increment: online
MIDI score following with the HMM methods `pthmm` (MIDI default), `hmm` (PitchIOIHMM) and
`outerhmm` (OuterProductHMM), MIDI processors `pitch` and `chord_onset`, MusicXML and MIDI
scores. Must be easy to understand and use, performance must always be visible and
comparable to the Python reference, easy to debug. Interfaces: library, CLI, Python
bindings."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Follow a recorded MIDI performance with `pthmm` (Priority: P1)

A developer has a score (MusicXML or MIDI) and a recorded MIDI performance of it. They run the
score follower in simulation mode and receive, for each incoming observation, the current score
position in beats, and at the end the complete alignment path (performance time ↔ score beat).

**Why this priority**: This is the core value of the library and the MIDI default method of the
reference. Everything else (comparison, bindings) builds on it.

**Independent Test**: Run the follower on the bundled example `simple_mozart_k265_var1`
(score + performance MIDI) and check that positions are produced, are valid score onsets, and
that the alignment path covers the performance.

**Acceptance Scenarios**:

1. **Given** the bundled MusicXML score and its performance MIDI, **When** the developer runs
   `pthmm` in simulation mode, **Then** a score position (in beats) is reported for every
   consumed observation and the final alignment path has one (performance time, score beat)
   pair per consumed observation; like the reference, the follower stops consuming once it
   reaches the last score state.
2. **Given** the same inputs as a MIDI score instead of MusicXML, **When** the follower runs,
   **Then** it produces positions on the same beat grid as the reference does for that MIDI
   score.
3. **Given** a score with repeat signs, **When** the follower runs with repeat unfolding
   enabled (default), **Then** positions refer to the unfolded score, as in the reference.
4. **Given** the CLI, **When** the developer passes a score file and a performance file,
   **Then** positions are printed as they are computed and the alignment path can be written
   to a file.

---

### User Story 2 - Prove parity and compare performance with the Python reference (Priority: P1)

A maintainer wants evidence that the Rust results equal the Python reference and wants to see
how much faster (or slower) Rust is, on the same inputs and machine.

**Why this priority**: Parity and visible performance are non-negotiable project principles;
without them the port cannot be trusted.

**Independent Test**: Run the comparison on the bundled examples: it reports per method whether
Rust matches the reference within tolerance, plus side-by-side latency and accuracy figures.

**Acceptance Scenarios**:

1. **Given** reference outputs generated once by the reference release for fixed inputs,
   **When** the Rust test suite runs, **Then** features, positions and alignment paths match
   the reference within the documented tolerance, and any mismatch names the first differing
   observation.
2. **Given** an alignment path and ground-truth annotations, **When** evaluation runs,
   **Then** it reports the same metrics as the reference (`mean_b`, `median_b`, `std_b`,
   `accuracy_{0.05,0.1,0.3,0.5,1,2}b`, `skewness`, `kurtosis`, `smoothness`) with equal
   values for equal inputs.
3. **Given** the benchmark command, **When** it runs Rust and Python on the same inputs,
   **Then** it produces a report with, per method and piece: processing latency per
   observation (feature and inference separately), real-time factor and accuracy metrics,
   for both implementations side by side.
4. **Given** the full benchmark datasets ((n)ASAP, Batik, Vienna4x22), **When** CI runs,
   **Then** the full benchmark report is produced and reachable from the README.

---

### User Story 3 - Follow with tempo-aware `hmm` (PitchIOIHMM) (Priority: P2)

A developer uses the `hmm` method, which additionally uses inter-onset intervals and a tempo
model, to follow a performance.

**Why this priority**: Second MIDI method of the reference; adds the tempo model that later
methods reuse.

**Independent Test**: Run `hmm` on the bundled example and compare to the reference outputs.

**Acceptance Scenarios**:

1. **Given** the bundled example, **When** `hmm` runs, **Then** positions and alignment path
   match the reference within tolerance.
2. **Given** a run of `hmm`, **When** debugging is enabled, **Then** the tempo estimate
   (beat period) per observation is available for inspection.

---

### User Story 4 - Follow note by note with `outerhmm` (Priority: P2)

A developer uses the `outerhmm` method (outer-product HMM after Nakamura 2014), which
processes one performed note at a time and models pitch errors, repeats and skips.

**Why this priority**: Third MIDI method of the reference; event-based instead of
polling-based, so it also covers the message-by-message stream mode.

**Independent Test**: Run `outerhmm` on the bundled example and compare to the reference
outputs.

**Acceptance Scenarios**:

1. **Given** the bundled example, **When** `outerhmm` runs, **Then** positions and alignment
   path equal the reference.
2. **Given** a performance with a wrong note (e.g. a semitone off), **When** `outerhmm`
   runs, **Then** it behaves as the reference does for that input.

---

### User Story 5 - Use the Rust follower from Python (Priority: P3)

A Python user replaces `from matchmaker import Matchmaker` with the Rust-backed package and
runs the same code for the supported methods.

**Why this priority**: Lowers the barrier for existing pymatchmaker users and enables direct
A/B comparisons in Python, but the Rust library and CLI come first.

**Independent Test**: Run the README quickstart of pymatchmaker for `input_type="midi"` with
`method="pthmm"`, `method="hmm"` and `method="outerhmm"` against the Rust-backed package;
results equal the Rust library's results.

**Acceptance Scenarios**:

1. **Given** the Rust-backed Python package, **When** a user runs
   `Matchmaker(score_file=..., performance_file=..., input_type="midi", method=...)` and
   iterates `run()`, **Then** they receive beat positions and can read the alignment path
   in the same shape as the reference.
2. **Given** an unsupported method or input type, **When** the user requests it, **Then** a
   clear error lists the supported options.

---

### User Story 6 - Debug a disagreement (Priority: P3)

A maintainer sees that Rust and Python disagree on a piece and wants to find the first stage
where they differ.

**Why this priority**: Makes the parity principle practical; reduces debugging time.

**Independent Test**: Export intermediate data from Rust for one piece and load it in Python
next to the reference's data.

**Acceptance Scenarios**:

1. **Given** a run, **When** the developer enables export, **Then** score features, every
   observation (features + performance time), the state probabilities or position per step,
   and the alignment path are written in a format Python reads directly.
2. **Given** verbose logging, **When** the follower runs, **Then** per-stage timing and the
   current position are logged per observation.

### Edge Cases

- Performance contains notes not in the score (wrong notes, insertions): follower keeps going;
  behaviour matches the reference (`has_insertions`).
- Several note-ons arrive at the same time (chord) or within the chord threshold of 35 ms:
  grouped exactly as the reference processor does.
- Note-on with velocity 0: treated as note-off.
- Performance ends before or continues after the end of the score: no crash; positions stay
  within the score.
- Score without tempo marking: default tempo of 120 BPM, as in the reference.
- Score with grace notes, ties, multiple parts/voices: note list and beat positions equal the
  reference's score representation.
- Empty performance or performance without note-ons: empty alignment path, no error.
- Unreadable, unsupported or corrupt score/performance file: typed error naming the file and
  the problem; no crash.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST load scores from MusicXML (`.musicxml`, `.xml`, compressed `.mxl`)
  and MIDI (`.mid`) files.
- **FR-002**: System MUST derive the score representation used by the reference: one entry
  per note with onset in beats (`onset_beat`), pitch and duration, after unfolding repeats
  (on by default, switchable) and merging all parts.
- **FR-003**: System MUST determine the score tempo as the reference does, defaulting to
  120 BPM, and allow the user to override it.
- **FR-004**: System MUST read a performance MIDI file and replay it as a timed stream
  (simulation mode), either message by message or grouped into polling windows of a
  configurable length, with the same timing semantics as the reference.
- **FR-005**: System MUST accept MIDI messages pushed incrementally by the caller (with
  timestamps), so that any live source can feed the follower.
- **FR-006**: System MUST provide the `pitch` processor (binary pitch vector or pitch list,
  piano range on/off) and the `chord_onset` processor (35 ms chord merging) with outputs
  equal to the reference.
- **FR-007**: System MUST provide the `pthmm` score follower (pitch HMM with optional
  insertion states) with reference defaults.
- **FR-008**: System MUST provide the `hmm` score follower (pitch + inter-onset-interval HMM
  with Kalman tempo model) with reference defaults.
- **FR-009**: System MUST provide the `outerhmm` score follower (outer-product HMM with
  pitch-error, repeat and skip modelling; event-based, one note per observation) with
  reference defaults.
- **FR-010**: For every observation, the follower MUST report the current score position in
  beats; at the end it MUST return the alignment path as (performance time, score beat)
  pairs.
- **FR-011**: Methods and processors MUST be selectable by the reference names (`pthmm`,
  `hmm`, `outerhmm`, `pitch`, `chord_onset`), with the reference defaults per method;
  method-specific parameters MUST be overridable.
- **FR-012**: System MUST evaluate an alignment path against ground truth (annotation text
  files and `.match` files) with the reference metrics and mapping function.
- **FR-013**: System MUST provide a CLI to run a follower (score, performance, method,
  processor, parameters), print positions, write the alignment path, and run evaluation.
- **FR-014**: System MUST provide a Python package exposing the `Matchmaker` entry point
  for the supported methods with the reference's call signature and result shapes.
- **FR-015**: System MUST include golden tests against the outputs of the current reference
  release (v0.4.0 at time of writing) for each processor and method, generated by a
  checked-in script that can be re-run for a newer release.
- **FR-016**: System MUST provide a benchmark that runs Rust and the Python reference on the
  same inputs and reports latency per observation (feature and inference), real-time factor
  and accuracy metrics side by side, for the bundled examples and the full datasets.
- **FR-017**: System MUST measure the time spent in each stage per observation in a way that
  can be switched on without code changes.
- **FR-018**: System MUST export intermediate results (score features, observations,
  per-step position/state probabilities, tempo estimate, alignment path) in a
  Python-readable format on request.
- **FR-019**: System MUST report invalid input as typed errors naming the file and problem.
- **FR-020**: CI MUST run formatting, lint, tests including golden tests, and the full
  benchmark, and the README MUST show build status, version and a link to the latest
  benchmark report.

### Key Entities

- **Score**: list of notes (id, pitch, onset in beats, duration in beats), after unfolding;
  plus tempo and the unique onset grid used as follower states.
- **Performance stream**: time-ordered MIDI messages (type, pitch, velocity, time in seconds),
  optionally grouped into polling windows.
- **Observation**: processor output for one frame: features (pitch vector or pitch list) and
  the performance time it refers to.
- **Score follower**: method (`pthmm`/`hmm`/`outerhmm`), its parameters, and its current
  state (position, probabilities, tempo estimate).
- **Alignment path**: ordered pairs (performance time in seconds, score position in beats).
- **Ground truth**: annotated pairs (performance time, score beat) from annotation or
  `.match` files.
- **Evaluation result**: the reference metric set for one alignment.
- **Benchmark report**: per implementation, method and piece: latencies, real-time factor,
  evaluation result.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On all bundled examples and all benchmark dataset pieces, 100% of positions
  reported by `pthmm` and `outerhmm` equal the reference positions (difference < 1e-6
  beats).
- **SC-002**: On the same inputs, `hmm` positions equal the reference in ≥ 99.9% of
  observations; every evaluation metric differs by at most 0.01 from the reference.
- **SC-003**: Processor outputs (`pitch`, `chord_onset`) equal the reference for 100% of
  frames of the bundled examples.
- **SC-004**: Mean inference latency per observation in Rust is at least 5× lower than the
  Python reference on the same machine, as shown in the benchmark report.
- **SC-005**: A developer new to the project gets a first alignment of the bundled example
  from a fresh checkout with at most 3 documented commands.
- **SC-006**: Every CI run on `master` publishes the full benchmark report; the README links
  to it.
- **SC-007**: For any parity failure, the test output names the method, the input and the
  index of the first differing observation.

## Assumptions

- The reference is the current pymatchmaker release (tag `v0.4.0`); a newer release may be
  adopted later by regenerating the golden data, never an unreleased state. Its bundled
  assets (`simple_mozart_k265_var1`, `mozart_k265_var1`, `Bach-fugue_bwv_858`) are used as
  fixed test inputs.
- Python reference outputs are generated by maintainers with pymatchmaker installed and
  checked into the repository; normal Rust builds and tests do not need Python. The
  benchmark comparison and CI benchmark job do need the reference installed.
- The benchmark datasets ((n)ASAP, Batik, Vienna4x22) are downloadable in CI and their
  licenses allow use for evaluation; data is not redistributed in this repository.
- MEI and other score formats supported by partitura, live MIDI device input, audio input,
  and all other methods (`arzt`, `dixon`, `pfkorz`, parangonar methods; audio `outerhmm`)
  are out of scope for this feature.
- The `chord_onset` processor is included now (selected for the first increment) although its
  consumers (MIDI `arzt`/`dixon`) come in a later feature.
- Results are deterministic for fixed inputs; no random seeds are needed for `pthmm`, `hmm`
  or `outerhmm`.
