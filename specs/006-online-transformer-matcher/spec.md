# Feature Specification: Online Transformer Matcher (OTM)

**Feature Branch**: `006-online-transformer-matcher`

**Created**: 2026-11-24

**Status**: Draft

**Input**: User description: "ich möchte den OnlineTransformerMatcher migrieren. die anderen drei methoden sollen später kommen."

This feature ports pymatchmaker's MIDI method `OTM`, backed by Parangonar's
`OnlineTransformerMatcher`, into the existing score-following product. The scope is limited
to this method; `OPTM`, `SLT_OLTW`, and `SL_OLTW` are deferred. The offline
`DualDTWNoteMatcher` is not a score-following method and is out of scope.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Follow a MIDI performance with OTM (Priority: P1)

A developer has a score and a recorded or incrementally supplied MIDI performance. They
select `OTM` and receive the current score position in beats as each performance note is
consumed, followed by the completed alignment path.

**Why this priority**: This is the requested capability and the first of the four Parangonar
methods to be migrated.

**Independent Test**: Run `OTM` on a fixed score/performance pair and verify that the
library, CLI, and Python package each produce score positions and an alignment path.

**Acceptance Scenarios**:

1. **Given** a supported MusicXML or MIDI score and a recorded MIDI performance, **When** a
   user selects `OTM`, **Then** the follower reports positions in score beats and returns
   the final alignment path.
2. **Given** a stream of MIDI events, **When** the user selects `OTM`, **Then** the follower
   processes each note event in order without merging notes into polling-window chords.
3. **Given** the CLI or Python package, **When** the user selects `OTM` for MIDI input,
   **Then** it exposes the same positions and path semantics as the library.
4. **Given** the method's supported configuration, **When** a user overrides a parameter,
   **Then** that value affects the run; an unsupported parameter is reported as an error.
5. **Given** a user requests `OPTM`, `SLT_OLTW`, or `SL_OLTW`, **When** the method is
   validated, **Then** it is rejected as unsupported and the available methods are listed.

---

### User Story 2 - Verify OTM against its reference (Priority: P1)

A maintainer wants to confirm that the migrated follower follows the same score positions
and alignment path as the pinned pymatchmaker reference on repeatable inputs.

**Why this priority**: Reference parity is a non-negotiable project principle and is
especially important for a model-based matcher.

**Independent Test**: Generate OTM reference results from the pinned release and compare the
Rust outputs for the same scores, performances, model, and configuration.

**Acceptance Scenarios**:

1. **Given** the pinned reference and fixed golden inputs, **When** the OTM golden tests run,
   **Then** positions and alignment paths meet the documented comparison tolerances.
2. **Given** a parity failure, **When** the comparison reports it, **Then** it identifies the
   input, method, output, and first differing event.
3. **Given** repeated runs with the same inputs and configuration, **When** OTM is executed
   five times, **Then** the produced positions and alignment paths are identical.
4. **Given** the benchmark inputs, **When** Rust and the reference are benchmarked, **Then**
   the report includes per-event inference latency and real-time factor for both.

---

### User Story 3 - Diagnose and handle OTM run failures (Priority: P2)

A developer needs to inspect a run or understand why OTM cannot start, rather than receiving
an incomplete or success-shaped result.

**Why this priority**: Model-based inference adds assets and failure modes that must be
visible to users and maintainers.

**Independent Test**: Export an OTM run for comparison with the reference, then exercise
invalid inputs and unavailable or invalid model assets and verify explicit errors.

**Acceptance Scenarios**:

1. **Given** a completed OTM run with export enabled, **When** the export is read by the
   existing comparison tools, **Then** it includes the score reference, processed
   observations, per-event model scores and tempo diagnostics, positions, and alignment
   path in the established formats.
2. **Given** audio input, invalid configuration, or unavailable/corrupt model assets,
   **When** OTM is requested, **Then** the user receives an actionable error and no
   fabricated alignment result.
3. **Given** an empty performance, **When** OTM runs, **Then** it returns an empty path
   without panicking.

## Edge Cases

- Several notes arrive with the same timestamp: process them individually and in stream
  order, as the reference does.
- The performance includes notes absent from the score, or omits score notes: preserve the
  reference matcher's alignment and position behavior.
- Grace notes are present in the score: include them in the score representation used by OTM,
  as the reference does.
- The performance ends before the score or continues after the final score event: preserve
  the reference's stopping and final-position behavior.
- A score or performance is empty, unreadable, or malformed: return an explicit error or an
  empty path according to the reference behavior; do not panic.
- Model files are missing, unreadable, incompatible, or corrupt: report the failure
  explicitly; do not silently substitute another method or model.
- A performance pitch cannot be represented by the reference OTM token vocabulary: report an
  explicit input error rather than indexing outside the model vocabulary.
- The user requests `OTM` with audio input: reject it and list the supported input/method
  choices.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST expose the MIDI method under the reference selector name `OTM` in
  the Rust library, CLI, and Python package. It MUST NOT change the default MIDI method.
- **FR-002**: OTM MUST use the same score-note representation as the pinned reference,
  including grace notes, and support the score formats already supported by the existing
  MIDI pipeline.
- **FR-003**: OTM MUST consume MIDI note observations one at a time in event order. It MUST
  NOT merge simultaneous or near-simultaneous note events into a chord observation.
- **FR-004**: OTM MUST reproduce the `OnlineTransformerMatcher` behavior, including its
  tempo-model post-processing, rather than the distinct `OnlinePureTransformerMatcher`
  behavior.
- **FR-005**: OTM MUST preserve the reference defaults: processor `parangonar`, no MIDI
  polling window, `token_number=91`, `dim_model=64`, `dim_class=2`, `num_heads=8`,
  `num_decoder_layers=6`, `dropout_p=0.1`, `init_beat_period=0.5`, and `lookback=1`.
- **FR-006**: Users MUST be able to override the configuration parameters accepted by the
  reference OTM matcher. Unknown parameters and values incompatible with the supplied model
  MUST produce an explicit error.
- **FR-007**: For each consumed performance note, OTM MUST report a score position in beats
  and MUST return the final alignment path with the same meaning and shape as the existing
  score-following API and the pinned reference.
- **FR-008**: Users MUST be able to run OTM from the standard matchmaker-rs distribution
  without installing the Python reference or its runtime. Missing or invalid model assets
  MUST be reported explicitly.
- **FR-009**: OTM MUST have golden tests generated from the repository's pinned pymatchmaker
  release and its locked dependencies. Tests MUST cover score reference construction,
  processed note observations, positions, and alignment path.
- **FR-010**: OTM run exports MUST include the score reference, processed observations,
  per-event model scores and tempo diagnostics, positions, and final alignment path in
  formats accepted by the existing comparison tools.
- **FR-011**: The benchmark MUST compare Rust and the pinned reference on identical inputs
  and report per-event inference latency and real-time factor using the existing benchmark
  conventions.
- **FR-012**: This feature MUST NOT add `OPTM`, `SLT_OLTW`, or `SL_OLTW`; those methods are
  deferred. It MUST NOT add offline note alignment such as `DualDTWNoteMatcher`.
- **FR-013**: The README and `UPSTREAM.md` MUST identify OTM as ported and the other
  Parangonar methods as not yet ported.
- **FR-014**: Any distributed model asset MUST include its source, checksum, and required
  attribution and license notices. Its redistribution terms MUST be verified before it is
  included in a release.

### Key Entities

- **Score note reference**: The ordered score-note data OTM uses, including note identity,
  pitch, onset, duration, and grace-note status.
- **Performance note observation**: One performed MIDI note and its performance timestamp,
  delivered in stream order.
- **OTM alignment state**: The transformer's local alignment decision together with the
  tempo-model state used to produce the current score position.
- **Alignment path**: The sequence of performance-time and score-beat pairs returned by the
  follower.
- **Model assets**: The trained parameters required to make OTM predictions.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On every checked-in OTM golden fixture, 100% of returned score positions and
  alignment-path entries match the pinned reference within the documented tolerances; beat
  positions differ by less than `1e-6` where exact equality is not possible.
- **SC-002**: For every golden fixture, the Rust and reference runs consume the same number
  and order of note observations, and their score-reference and processor outputs match the
  documented comparison tolerances.
- **SC-003**: The Rust library, CLI, and Python package produce equivalent OTM positions and
  alignment paths on the same fixtures, with no more than `1e-6` beat difference.
- **SC-004**: Five repeated runs for each fixed OTM fixture produce identical Rust positions
  and alignment paths.
- **SC-005**: Over at least five runs on identical bundled inputs, the benchmark reports
  median per-event inference latency and real-time factor for Rust and the reference; Rust's
  median real-time factor is below `1.0`.
- **SC-006**: All existing non-OTM golden tests continue to pass without changes to their
  expected outputs.

## Assumptions

- The reference is pymatchmaker `v0.5.1`, already pinned by the repository. Its
  `methods.yaml` registers `OTM` and delegates it to Parangonar's
  `OnlineTransformerMatcher`.
- The existing MIDI score loading, repeat handling, streaming, CLI, Python bindings, export,
  and benchmark surfaces are reused; this feature adds no live-device I/O.
- The migration uses the standard matchmaker-rs distribution, and inference is expected to
  work on CPU.
- The pretrained OTM model assets can be obtained and distributed under terms compatible
  with this project. Their provenance, license, and packaging must be verified before
  distribution.
- `OPTM`, `SLT_OLTW`, and `SL_OLTW` will be considered in later features and are not implied
  by OTM support.
