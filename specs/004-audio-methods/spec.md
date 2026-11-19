# Feature Specification: Remaining Audio Methods and Features

**Feature Context**: `004-audio-methods` (existing active feature; no new Git branch requested)
**Created**: 2026-11-03
**Status**: Ready for planning
**Input**: Deliver the remaining audio methods and features incrementally, following Spec Kit, with a separate commit for every new algorithm. The agreed roadmap also separates processors and audio ground-truth evaluation.

## User Scenarios & Testing

### User Story 1 - Tempo-aware Arzt (Priority: P1)

A user follows a recorded performance with `arzt_tempo` to track tempo changes.

**Why this priority**: Extends an already delivered follower before introducing new model families.
**Independent Test**: Follow fixed steady, accelerating, and slowing performances and compare positions and tempo state with the reference.

**Acceptance Scenarios**:
1. **Given** a score and tempo-varying audio, **When** `arzt_tempo` runs, **Then** positions, alignment path, and tempo diagnostics meet the documented reference comparison limits.
2. **Given** existing audio `arzt` and MIDI methods, **When** this method is added, **Then** their defaults and golden results remain unchanged.

### User Story 2 - Multiple score unfolding hypotheses (Priority: P1)

A user follows audio with `arzt_multi_fold` when the performed repeat path differs from maximal score unfolding.

**Why this priority**: Reuses Arzt while addressing repeat handling independently.
**Independent Test**: Compare no-repeat, taken-repeat, and skipped-repeat fixtures with the reference, including selected hypotheses.

**Acceptance Scenarios**:
1. **Given** an annotated repeat score, **When** alternate repeat paths are performed, **Then** hypothesis selection, switching, and returned positions meet the reference contract.
2. **Given** costs near the switching threshold, **When** candidates compete, **Then** selection-window and switch-margin behavior match the reference.

### User Story 3 - Evaluate audio alignment (Priority: P1)

A user supplies audio ground truth and obtains the same alignment-quality metrics as the reference evaluator.

**Why this priority**: Makes later template-based and stochastic comparisons measurable.
**Independent Test**: Evaluate a fixed path against known ground truth and compare every reported metric with the reference.

**Acceptance Scenarios**:
1. **Given** audio ground truth and a completed alignment, **When** evaluation runs, **Then** the result uses the reference mapping and metric definitions.
2. **Given** malformed, missing, or incompatible ground truth, **When** evaluation is requested, **Then** a clear error is returned without overwriting existing output.

### User Story 4 - Raw spectrum features (Priority: P2)

A user selects `raw_spectrum` and can inspect its output independently of SKF.

**Why this priority**: Supplies SKF's default observation representation.
**Independent Test**: Compare spectrum dimensions, magnitudes, timestamps, and reset behavior with fixed reference frames.

**Acceptance Scenarios**:
1. **Given** fixed audio, **When** `raw_spectrum` runs, **Then** observations match the reference with the selected transform size.
2. **Given** a compatible existing frame follower, **When** this processor is selected, **Then** score and performance use matching feature dimensions.

### User Story 5 - Switching Kalman filter (Priority: P2)

A user follows recorded audio with `skf` and inspects chord-state and tempo estimates.

**Why this priority**: Delivers a new model family after its processor and evaluation prerequisite.
**Independent Test**: Replay fixed raw-spectrum observations and compare model state, positions, and path; separately compare full-pipeline quality metrics.

**Acceptance Scenarios**:
1. **Given** a score and raw-spectrum observations, **When** SKF runs, **Then** state transitions and estimates meet the reference comparison limits.
2. **Given** synthesized or fallback templates, **When** a run is inspected, **Then** the template mode and provenance are identifiable and the compared reference uses the same mode.

### User Story 6 - Constant-Q features (Priority: P2)

A user selects `cqt` for an existing compatible frame follower.

**Why this priority**: Establishes constant-Q behavior before spectral flux and Audio-OuterHMM.
**Independent Test**: Compare frequency bins, frame trimming, normalization, timestamps, and reset behavior with the reference.

**Acceptance Scenarios**:
1. **Given** fixed pitched and silent audio, **When** `cqt` runs, **Then** dimensions and values meet the reference feature limits.
2. **Given** a processor override, **When** following begins, **Then** score and performance use the same processor settings.

### User Story 7 - Constant-Q spectral flux (Priority: P2)

A user selects `cqt_spectral_flux` and can inspect onset-sensitive observations.

**Why this priority**: Supplies the default observations for Audio-OuterHMM.
**Independent Test**: Compare consecutive frames, silence, positive spectral changes, and lifecycle behavior with the reference.

**Acceptance Scenarios**:
1. **Given** audio with repeated and changing pitches, **When** flux features are computed, **Then** values and cross-frame state match the reference limits.
2. **Given** a fresh independent run, **When** the same audio is processed, **Then** another run's processor state does not affect its observations; within a run, the released lifecycle behavior is preserved.

### User Story 8 - Audio outer-product HMM (Priority: P2)

A user follows recorded audio with `outerhmm`, distinct from MIDI `outerhmm`.

**Why this priority**: Depends on constant-Q flux and verified pitch templates.
**Independent Test**: Replay fixed flux observations with fixed templates; compare posterior diagnostics, positions, and alignment path.

**Acceptance Scenarios**:
1. **Given** fixed score events, observations, and pitch templates, **When** Audio-OuterHMM runs, **Then** model diagnostics and alignment meet the reference limits.
2. **Given** MIDI `outerhmm`, **When** audio support is added, **Then** MIDI defaults and behavior remain unchanged.

### User Story 9 - Mel spectrum features (Priority: P3)

A user selects `mel` as another observation representation for existing compatible frame followers.

**Why this priority**: Adds feature choice after the primary missing followers.
**Independent Test**: Compare mel bins, power scaling, normalization, timestamps, and silence with the reference.

**Acceptance Scenarios**:
1. **Given** fixed audio and a compatible follower, **When** `mel` is selected, **Then** exported observations and the complete run meet their reference limits.

### User Story 10 - MFCC features (Priority: P3)

A user selects `mfcc` for existing compatible frame followers.

**Why this priority**: Reuses validated spectral work after mel delivery.
**Independent Test**: Compare coefficient count, scaling, transform normalization, and silence with the reference.

**Acceptance Scenarios**:
1. **Given** fixed audio and a compatible follower, **When** `mfcc` is selected, **Then** coefficients and the complete run meet their reference limits.

### User Story 11 - Korzeniowski audio observations (Priority: P3)

A user extracts and inspects `korzeniowski` audio observations independently of the particle filter.

**Why this priority**: Establishes the particle filter's observation contract first.
**Independent Test**: Compare each observation component, window alignment, silence, and state reset with the reference.

**Acceptance Scenarios**:
1. **Given** fixed pitched and silent audio, **When** the processor runs, **Then** its structured observations meet the reference limits.
2. **Given** an incompatible follower, **When** this processor is selected, **Then** construction rejects the combination rather than misinterpreting observations.

### User Story 12 - Korzeniowski particle filter (Priority: P3)

A user follows recorded audio with `pfkorz`, selects a seed, and inspects particle and tempo estimates.

**Why this priority**: Largest remaining method, with dedicated score features and stochastic verification.
**Independent Test**: Repeat a fixed-seed run exactly within Rust; compare accuracy distributions with the reference across a fixed seed set.

**Acceptance Scenarios**:
1. **Given** a fixed score, audio, configuration, and seed, **When** a run is repeated, **Then** its positions and path are identical within the same runtime environment.
2. **Given** a declared seed set, **When** Rust and reference runs are evaluated, **Then** aggregate accuracy meets the predeclared metric limits without requiring identical random streams.

### Edge Cases

- Empty, silent, shorter-than-window, corrupt, multichannel, and final-partial-hop performances.
- Invalid rates, hops, transform sizes, dimensions, particle counts, seeds, and non-finite numeric parameters.
- One-onset scores, score overrun, tied notes, grace notes, repeats, and ambiguous hypothesis costs.
- Missing, corrupt, dimensionally incompatible, or unlicensed spectral templates and SoundFonts.
- Repeated runs after reset; unsupported method/processor pairs; invalid ground truth and output paths.
- Near-zero observation likelihoods, covariance instability, particle weight collapse, and long-stream state growth.

## Requirements

### Functional Requirements

- **FR-001**: Deliver `arzt_tempo` with reference tempo adaptation and diagnostic state (US1).
- **FR-002**: Deliver `arzt_multi_fold` with reference repeat hypotheses and switching rules (US2).
- **FR-003**: Deliver audio ground-truth evaluation using the reference alignment mapping and metrics (US3).
- **FR-004**: Deliver `raw_spectrum` with reference feature shape, values, and framing (US4).
- **FR-005**: Deliver `skf` with reference state evolution and identifiable spectral-template mode (US5).
- **FR-006**: Deliver `cqt` with reference binning, normalization, and frame selection (US6).
- **FR-007**: Deliver `cqt_spectral_flux` with reference cross-frame state and lifecycle behavior (US7).
- **FR-008**: Deliver audio `outerhmm` with reference score-event and pitch-template models (US8).
- **FR-009**: Deliver `mel` with reference power scaling and normalization (US9).
- **FR-010**: Deliver `mfcc` with reference coefficient scaling and normalization (US10).
- **FR-011**: Deliver structured `korzeniowski` audio observations and reject incompatible consumers (US11).
- **FR-012**: Deliver audio `pfkorz`, its score model, selectable seed, and reproducible runs (US12).
- **FR-013**: Each new method and processor MUST be usable through the existing library, command-line, and Python interfaces with the reference defaults and supported overrides. Standalone processor extraction MUST be possible even before its consumer method is delivered.
- **FR-014**: Existing MIDI and audio methods, processor defaults, exports, and errors MUST remain backward compatible.
- **FR-015**: Every delivered method and processor MUST have fixed-input, locked-release reference comparisons with limits documented before implementation; every failed comparison MUST identify the first differing stage.
- **FR-016**: Synthesis differences and stochastic methods MUST be assessed using the exact reference evaluator and explicitly documented metric limits, not silently relaxed feature tolerances.
- **FR-017**: Each story MUST be delivered in its own commit after tests, all repository quality gates, documentation, and benchmark evidence are complete. A commit MUST NOT introduce two new algorithms or processors. No subsequent story implementation starts before the current commit is complete. Commits require explicit user authorization when implementation begins.
- **FR-018**: All new stages MUST expose per-frame diagnostic timing and intermediate observations, model state, reference data, positions, and paths in inspectable formats.
- **FR-019**: Invalid external input MUST produce descriptive typed errors without panics, silent fallback changes, or destructive output handling.
- **FR-020**: Template and fixture provenance, exact hashes, redistribution permissions, and synthesis mode MUST be recorded before use; unverified assets block their consuming story.
- **FR-021**: The default runtime MUST remain portable and avoid native-library linkage; platform and dependency-license gates MUST pass for every increment.
- **FR-022**: Benchmarks MUST compare identical inputs with the reference, measure feature/inference latency and real-time factor plus accuracy, use at least five repetitions, and flag slowdown greater than 10% against the previous published baseline.

### Key Entities

- **Audio observation**: A timestamped feature vector or method-specific structured observation with dimensions and configuration.
- **Reference model**: Synthesized score frames, score events, repeat hypotheses, or harmonic templates consumed by one method.
- **Follower state**: Current position, tempo estimate, probabilities/costs, hypothesis selection, or weighted particles.
- **Evaluation record**: Ground-truth provenance, mapping, accuracy/error metrics, and run identity.
- **Delivery increment**: One user story, its validation evidence, and its independent commit.

## Success Criteria

### Measurable Outcomes

- **SC-001**: All five missing audio methods and all six missing audio processors are independently usable; audio ground-truth evaluation is delivered as a twelfth increment.
- **SC-002**: All deterministic replay and feature comparisons pass their predeclared limits; every synthesis or stochastic deviation has an explicit metric comparison and provenance record.
- **SC-003**: Every supported method works through all three interfaces; all previously supported configurations retain their regression results.
- **SC-004**: The twelve delivery commits each contain exactly one story and have passing quality gates and corresponding validation evidence.
- **SC-005**: Fixed-seed particle-filter repeats are identical within Rust; reference accuracy comparisons cover at least five declared seeds.
- **SC-006**: Every new method's benchmark reports feature latency, inference latency, real-time factor, and accuracy over at least five runs; every slowdown above 10% is flagged and justified before delivery.

## Assumptions

- The reference remains the released pymatchmaker 0.4.0 and its existing locked environment.
- Existing recorded-audio decoding, rendering, score loading, evaluation primitives, and test infrastructure are reused.
- Stories are executed in the numbered roadmap order, even where technical work could otherwise proceed independently.
- Live audio/MIDI devices, Parangonar, additional MIDI processors/methods, and reference-version upgrades are outside this feature.
- This request authorizes planning artifacts, not source implementation, Git commits, or a new branch.
