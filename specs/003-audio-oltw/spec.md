# Feature Specification: Audio Online Time Warping (arzt, dixon)

**Feature Branch**: `003-audio-oltw`

**Created**: 2026-11-01

**Status**: Draft

**Input**: User description: "Add the audio matchers after MIDI feature 002, on a dedicated branch, following the same feature workflow."

Builds on [001-midi-score-following](../001-midi-score-following/spec.md) and [002-midi-oltw](../002-midi-oltw/spec.md). This feature adds the audio variants of `arzt` and `dixon` to the existing symbolic-score following workflow.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Follow recorded audio with `arzt` (Priority: P1)

A musician or developer provides a MusicXML or MIDI score and a recording of its performance. They select audio input and `arzt`, then receive score positions over the recording and an alignment path when processing finishes.

**Why this priority**: It delivers the first usable audio OLTW matcher and reuses the event-level method already available for MIDI.

**Independent Test**: Run the bundled reference audio fixture against its symbolic score and compare frame features, score positions, and alignment path with the pinned Python reference.

**Acceptance Scenarios**:

1. **Given** a supported score and recorded performance, **When** audio `arzt` processes the recording, **Then** it yields score positions and a final alignment path matching the reference within documented tolerances.
2. **Given** a supported audio recording with no musical activity, **When** audio `arzt` processes it, **Then** it follows the reference behavior without crashing.
3. **Given** the CLI or Python facade, **When** a user selects `arzt` with audio input, **Then** the method uses its reference-defined defaults and returns the same alignment as the library.

### User Story 2 - Follow recorded audio with `dixon` (Priority: P1)

A musician or developer follows the same workflow using audio `dixon` and receives score positions and a final alignment path.

**Why this priority**: `dixon` is the second audio OLTW method named in the project roadmap and provides an independent alignment strategy.

**Independent Test**: Run audio `dixon` on the same bundled fixture and compare frame features, score positions, and alignment path with the pinned Python reference.

**Acceptance Scenarios**:

1. **Given** a supported score and recorded performance, **When** audio `dixon` processes the recording, **Then** it yields score positions and a final alignment path matching the reference within documented tolerances.
2. **Given** a short recording or a score shorter than the reference search window, **When** audio `dixon` processes it, **Then** it follows the reference behavior without an out-of-range failure.
3. **Given** the CLI or Python facade, **When** a user selects `dixon` with audio input, **Then** the method uses its reference-defined defaults and returns the same alignment as the library.

### User Story 3 - Choose audio features and inspect a run (Priority: P2)

A user chooses a supported audio processor, compares the run with reference outputs, and can inspect intermediate features and alignment data when results differ.

**Why this priority**: Feature choice and inspectability make the audio methods usable beyond a single fixture and preserve the project's parity and debugging workflow.

**Independent Test**: Run each supported processor on a fixture, compare its frame features with golden reference data, and verify that the debug export contains the corresponding audio and alignment stages.

**Acceptance Scenarios**:

1. **Given** either supported audio processor, **When** it processes a recording, **Then** its frame features and frame times match the pinned reference within documented tolerances.
2. **Given** a completed CLI or library run with export enabled, **When** the user inspects the export, **Then** audio features, per-frame positions, and the alignment path are available in Python-readable files.
3. **Given** an unsupported method, processor, or parameter, **When** a user requests it, **Then** the error identifies the invalid choice and lists supported alternatives.

### Edge Cases

- Empty, truncated, or corrupt audio recordings produce a typed error or empty result according to the reference behavior; they never panic.
- Silence or frames with no usable feature content follow the reference behavior and do not produce invalid score positions.
- Recordings shorter than one analysis frame, or with fewer frames than the score search window, are handled without an out-of-bounds failure.
- A recording that continues after the score ends keeps positions within the score and follows the reference stop rule.
- Unsupported or corrupt audio formats are rejected with an actionable error; supported multichannel inputs are downmixed to mono and source sample rates are resampled to the target rate.
- A score with one onset and an empty recording follow the existing reference run-loop behavior.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST accept a supported recorded-audio performance together with a MusicXML or MIDI score for audio score following.
- **FR-002**: System MUST expose audio `arzt` and `dixon` as selectable methods and use the defaults defined by the pinned reference.
- **FR-003**: System MUST provide the reference-supported audio processors `lse` and `chroma`, including their reference-defined feature values, frame times, and method compatibility.
- **FR-004**: Each audio method MUST report score positions for consumed audio frames and produce an alignment path when processing completes, following the reference stop and path-recording rules.
- **FR-005**: Method, processor, and supported parameter choices MUST be available through the Rust library, CLI, and Python facade; invalid choices MUST produce typed or user-facing errors that list valid alternatives.
- **FR-006**: The checked-in golden generator MUST produce reference audio features, frame times, positions, and alignment paths using the pinned released Python implementation and locked environment.
- **FR-007**: Golden tests MUST cover each delivered audio method and processor combination on fixed recordings and report the first differing frame or stage.
- **FR-008**: Users MUST be able to export audio features, per-frame alignment diagnostics, and the final path in a format readable by the existing Python comparison tools.
- **FR-009**: Audio framing, feature extraction, and inference MUST be observable in logs and measurable independently in the benchmark report.
- **FR-010**: The benchmark MUST compare Rust and the pinned reference on identical audio inputs and report feature latency, inference latency, real-time factor, accuracy, and alignment parity.
- **FR-011**: README and upstream module documentation MUST describe the delivered audio methods, processors, and remaining unsupported audio functionality.

### Key Entities

- **Audio frame**: A time-bounded portion of the recorded performance with a timestamp and extracted feature values.
- **Audio feature sequence**: The ordered feature vectors and timestamps produced by a supported processor.
- **Alignment path**: The mapping between performance time and score position produced by a follower.
- **Audio run diagnostics**: Exportable intermediate values needed to compare stages with the reference.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On every bundled audio fixture, both methods produce frame positions within the documented reference tolerance, with any known deviation tied to the first differing frame and explained by evaluation metrics.
- **SC-002**: For every supported processor and fixture, 100% of feature frames and timestamps match the reference within documented tolerances.
- **SC-003**: The benchmark demonstrates at least a 5x lower median inference latency per frame than the Python reference on the same machine, or records and explains the unmet target before release.
- **SC-004**: Each method can be selected and run through the library, CLI, and Python facade using the same inputs and produces equivalent paths.
- **SC-005**: Empty, silent, short, unsupported, and corrupt input cases produce the documented result or a typed error without panicking.

## Assumptions

- The first delivery processes prerecorded audio files; live audio devices and capture are out of scope.
- The symbolic score remains a MusicXML or MIDI file and uses the existing score-loading behavior.
- The initial audio scope is limited to frame-level `arzt` and `dixon` with the `lse` and `chroma` processors named in the project roadmap.
- Channel handling, sample-rate behavior, frame settings, and method defaults follow the pinned pymatchmaker release and are recorded during planning from its source and golden outputs.
- The Rust decoder's supported containers are documented explicitly; formats that require a non-Rust codec backend are rejected rather than silently delegated to a system library.
- Audio `outerhmm`, `skf`, `pfkorz`, `arzt_tempo`, `arzt_multi_fold`, live MIDI devices, and unrelated methods remain out of scope.
- Golden data and benchmark inputs must be redistributable or handled under their respective licenses; only derived metrics are published by CI.
