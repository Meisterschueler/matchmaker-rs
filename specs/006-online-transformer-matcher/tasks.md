---
description: "Implementation tasks for Online Transformer Matcher (OTM)"
---

# Tasks: Online Transformer Matcher (OTM)

**Input**: Design documents from `specs/006-online-transformer-matcher/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`,
`contracts/otm-api.md`, and `quickstart.md`

**Tests**: Included because FR-009 and SC-001–SC-006 explicitly require golden parity,
repeatability, API, error, export, and benchmark validation. In each story, the test tasks
come before the implementation tasks and must fail for the intended missing behavior.

**Organization**: Tasks are grouped by the three user stories in `spec.md`. The OTM
checkpoint licensing decision is a blocking setup gate.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because files and prerequisites do not overlap.
- **[Story]**: Maps a task to US1, US2, or US3 in `spec.md`.
- Every task names the file or output path to change or validate.

## Phase 1: Setup

**Purpose**: Clear the model-asset gate before work that packages or distributes weights.

**Execution status (2026-11-24)**: T001 is cleared by the project owner's explicit
Apache-2.0 determination for source and weights; see `research.md`. Implementation
is complete and verified. All 27 tasks are checked below. No upstream request was posted.

- [X] T001 Record the project owner's explicit Apache-2.0 determination and authorization for Parangonar source and weights in `specs/006-online-transformer-matcher/research.md` and attribution in `NOTICE`; preserve source/license metadata in `assets/otm/model.json` during conversion.

---

## Phase 2: Foundational

**Purpose**: Make reference results reproducible and define the compact inference artifact
needed by all user stories.

**⚠️ CRITICAL**: Complete T001–T003 before starting user-story implementation. T003 may
produce a distributable artifact only after T001 confirms permission.

- [X] T002 [P] Force CPU execution before importing the Python matcher in `tools/golden/generate.py` and `tools/bench/run_reference.py` so reference fixtures and benchmark timings use the same device.
- [X] T003 [P] Implement the offline checkpoint converter and versioned manifest in `tools/golden/export_otm_weights.py` and `assets/otm/model.json`; retain the checkpoint's 26 positional rows, omit unused positional rows and optimizer metadata, verify its source SHA-256, and write distributable weights only after T001 is cleared.

**Checkpoint**: Asset rights are confirmed, reference generation is CPU-pinned, and the
versioned artifact format is reproducible.

---

## Phase 3: User Story 1 — Follow a MIDI performance with OTM (Priority: P1) 🎯 MVP

**Goal**: Expose the event-based `OTM` follower in the Rust library, CLI, and Python package,
preserving score positions and path semantics.

**Independent Test**: Run the same score/performance pair through the library, CLI, and
Python API. Each consumes one note event at a time and returns beat positions plus a
`(2, T)` time/beat alignment path; configuration overrides take effect and unsupported
methods remain rejected.

### Tests for User Story 1

> Write these tests first and verify they fail for the intended missing behavior.

- [X] T004 [P] [US1] Add score-reference cases in `tests/otm_score.rs` asserting that grace status is explicit ("never infer it from duration"), onsets are ascending unique `f32` values, and if IDs are invalid "replace all with stable `s{row}` IDs".
- [X] T005 [P] [US1] Add processor cases in `tests/otm_processor.rs` asserting one positive-velocity note-on per observation, no observation for note-off/non-note messages, and rejection of "a frame with multiple positive note-ons" when `polling_period = None`.
- [X] T006 [P] [US1] Add tokenizer/model cases in `tests/otm_model.rs` asserting "26 tokens × 7 integer attributes", up to 8 performance tokens, one separator of 89, up to 16 score tokens, padding of 90 to length 26, score pitches 21–108, and explicit failure for out-of-vocabulary performance tokens.
- [X] T007 [P] [US1] Add API tests in `tests/otm_api.rs` and `crates/matchmaker-py/tests/test_otm.py` for the exact `"OTM"` selector, MIDI-only availability, unchanged default method, reference defaults, positions, and public `(2, T)` path shape.

### Implementation for User Story 1

- [X] T008 [US1] Add explicit `is_grace` state to score notes and propagate `GraceNote` identity through `src/score/mod.rs`, `src/score/part.rs`, and `src/score/musicxml.rs`; update all `ScoreNote` constructors in `src/prob/skf.rs`, `tests/golden_audio_skf.rs`, and `tests/golden_audio_outerhmm.rs` with the correct non-grace default. Preserve valid unique IDs, replace all IDs if any are empty or duplicated, and never infer grace status from duration.
- [X] T009 [US1] Build the OTM score reference in `src/score/mod.rs` with ascending unique onset positions, full grace-aware note rows, non-grace pitch sets per onset, and the reference's pitch-set iteration order; preserve IDs unless any are empty or duplicated, in which case regenerate all IDs as `s{row}`.
- [X] T010 [P] [US1] Implement and register the event-based `parangonar` processor in `src/features/parangonar.rs` and `src/features/mod.rs`; emit one positive-velocity note-on per observation, ignore non-note messages, reject multiple note-ons in a frame, and require `polling_period = None`. Document its upstream processor path, public items, and a runnable module example.
- [X] T011 [P] [US1] Implement tokenization, fixed transformer inference, and validated artifact loading in `src/dp/otm_model.rs` and `src/dp/mod.rs`; preserve the 26×7 token layout, 89 separator, 90 padding, 91-entry vocabulary, 64 model width, 2 classes, 8 heads, 6 layers, and artifact positional rows without runtime pickle loading. Document the upstream model paths, public items, and a runnable module example.
- [X] T012 [US1] Port the `OnlineTransformerMatcher.__call__` state machine and tempo model into `src/dp/otm.rs`; preserve the 0.5-second initial beat period, positive `lookback`, last-100 onset history, `[0.1, 10.0]` beat-period clamp, greedy/model/jump/stuck decision order, and clamped current score index. Document the upstream matcher path, public items, and a runnable module example.
- [X] T013 [US1] Add `Method::Otm`, OTM configuration/defaults, MIDI-only validation, and follower construction through the existing `src/matchmaker.rs` public module (no additional root re-export needed in `src/lib.rs`); keep the default MIDI method unchanged, use processor `parangonar` with `polling_period = None`, and reject unsupported parameters/model shapes.
- [X] T014 [US1] Wire `--method OTM` and OTM-specific configuration/help text into `crates/matchmaker-cli/src/main.rs`; preserve the current default method and return an actionable error for audio input or unsupported Parangonar selectors.
- [X] T015 [US1] Expose `OTM` and reference-compatible `DEFAULT_KWARGS` through `crates/matchmaker-py/src/lib.rs`; ensure Python calls use the same library positions/path and keep tracker-constructor defaults separate from the YAML-level processor/polling defaults.

**Checkpoint**: Library, CLI, and Python can follow a fixed MIDI pair with OTM; event order,
config behavior, positions, and public path semantics are independently testable.

---

## Phase 4: User Story 2 — Verify OTM against its reference (Priority: P1)

**Goal**: Prove stage-level and end-to-end parity with the locked pymatchmaker reference and
measure the online implementation on identical inputs.

**Independent Test**: Run `cargo test --test golden_otm` and the paired benchmark from
`quickstart.md`. Score/processor/model/tempo outputs match documented tolerances, repeated
Rust paths are identical, failures report the first differing event, and median Rust RTF is
below `1.0`.

### Tests for User Story 2

> Write these tests first; the reference generator and fixtures are added afterward.

- [X] T016 [P] [US2] Add stage-by-stage golden comparisons in `tests/golden_otm.rs` following `tests/golden_dannenberg.rs`; check score positions, observations, logits, tempo state, indices, positions, internal path, public path, `1e-6` beat tolerance, and first-difference context.
- [X] T017 [P] [US2] Add transformer-layer parity tests inside `src/dp/otm_model.rs` using small `(26, 7)` input fixtures and expected encoder outputs under `tests/golden/otm_model_layers/`; keep internal tensors private instead of exposing an API solely for `tests/otm_model_layers.rs`.

### Implementation for User Story 2

- [X] T018 [US2] Extend `tools/golden/generate.py` to generate OTM fixtures under `tests/golden/<piece>/OTM/` and the offline exporter to generate layer fixtures under `tests/golden/otm_model_layers/`; record the reference tag, lock hash, checkpoint hash, score/grace data, event observations, token windows, logits, tempo state, positions, and both path representations.
- [X] T019 [P] [US2] Add OTM to Rust and Python benchmark method handling in `crates/matchmaker-cli/src/bench.rs` and `tools/bench/run_reference.py`, reusing the three comparable bundled MIDI inputs already in `tools/bench/bundled.toml`; report per-event inference latency and RTF over at least five CPU runs.
- [X] T020 [US2] Run the OTM golden suite and paired benchmark from `specs/006-online-transformer-matcher/quickstart.md`; verify five-run path determinism, unchanged non-OTM goldens, and median Rust RTF `< 1.0`, and save the comparison report under `target/bench/otm-report/`.

**Checkpoint**: Every checked-in OTM fixture passes stage and end-to-end parity, and the
benchmark demonstrates the specified real-time target.

---

## Phase 5: User Story 3 — Diagnose and handle OTM run failures (Priority: P2)

**Goal**: Make invalid inputs, model failures, and intermediate OTM decisions visible without
panics, fabricated positions, or success-shaped fallbacks.

**Independent Test**: Export a valid OTM run and load the diagnostics with the existing
Python comparison tools; separately exercise the error cases and confirm explicit typed
errors. An empty performance returns an empty `(2, 0)` path.

### Tests for User Story 3

> Write these tests first and verify each intended failure/output-shape assertion fails.

- [X] T021 [P] [US3] Add failure and edge-case tests in `tests/otm_errors.rs`, `tests/otm_model.rs`, CLI/Python tests and the existing empty-performance test for audio input, invalid/unknown config, incompatible model dimensions, out-of-vocabulary pitch, missing/corrupt/wrong-checksum assets, empty score, and empty performance (empty path, no panic).
- [X] T022 [P] [US3] Add diagnostic export tests in `tests/otm_export.rs` asserting Python-readable `.npy`/JSON outputs, event-aligned masks, `(T, 26, 7)` token windows, `(T, 26, 2)` logits, `(T, 2)` tempo state, and public `(2, T)` alignment path.

### Implementation for User Story 3

- [X] T023 [US3] Reuse `src/error.rs`'s typed `InvalidParameter` in `src/dp/otm_model.rs`, `src/matchmaker.rs`, and `src/features/parangonar.rs`; name failing inputs/keys and reject missing/corrupt/incompatible assets and invalid pitches through checked calls without panics or fabricated positions. Document legacy infallible-trait rejection semantics in the contract.
- [X] T024 [US3] Add per-event OTM debug state and `DEBUG`-level tracing in `src/base.rs` and `src/dp/otm.rs` for tokens, logits when evaluated, current index, and tempo state; mark skipped inference explicitly rather than treating zero-filled values as valid.
- [X] T025 [US3] Extend `src/export.rs` to write the OTM score reference, processor observations, token/logit arrays and masks, tempo diagnostics, current indices, and public alignment path in the filenames and shapes specified by `specs/006-online-transformer-matcher/contracts/otm-api.md`.

**Checkpoint**: Exports are comparable in Python, and all invalid, missing-asset, and empty
input cases have explicit, non-panicking outcomes.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Document the delivered method and run the complete repository gates.

- [X] T026 [P] Update `README.md` and `UPSTREAM.md` to mark `OTM` ported, list the Parangonar processor/defaults, and identify `OPTM`, `SLT_OLTW`, and `SL_OLTW` as not yet ported; include model attribution only after T001 clears.
- [X] T027 Run every applicable command in `specs/006-online-transformer-matcher/quickstart.md`, including workspace tests, Python bindings (3.10, 3.12 and 3.13), `wasm32-unknown-unknown`, MSRV 1.88, `cargo deny check`, formatting, Clippy, benchmark threshold, and the model-license release gate; resolve failures before declaring the feature complete.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: T001 is a hard gate. If checkpoint rights remain unclear, stop before packaging or implementing the standard-distribution model and revise the spec/distribution requirement.
- **Foundational (Phase 2)**: T002 and T003 follow the gate and must complete before US1; T002 and T003 can run in parallel because they edit separate files.
- **US1 (Phase 3)**: Depends on Setup and Foundational completion. US1 test tasks T004–T007 come before implementation; T008–T009 are sequential because they both change `src/score/mod.rs`. T010 and T011 can proceed alongside score work after their own tests and T003; T012 waits for the score, processor, and model components; T013 precedes CLI/Python wiring.
- **US2 (Phase 4)**: Depends on US1. T016 and T017 can be written in parallel; T018 and T019 can be implemented in parallel after the Rust method is available; T020 validates both.
- **US3 (Phase 5)**: Depends on US1, with export parity validation also relying on US2 fixtures. T021 and T022 can be written in parallel; T023 and T024 can be implemented in parallel after their tests; T025 depends on the debug state from T024.
- **Polish (Phase 6)**: T026 can run alongside late US3 work; T027 depends on all implementation and documentation tasks.

### User Story Dependencies

- **US1 (P1)**: Can start after T001–T003; no dependency on another user story.
- **US2 (P1)**: Depends on US1's method and public outputs so the locked reference can be compared with the Rust implementation.
- **US3 (P2)**: Depends on US1's model/config/follower surfaces; its full export-comparison acceptance depends on US2 fixtures.

### Parallel Opportunities

- Run T002 and T003 in parallel after T001.
- Within US1, run tests T004–T007 in parallel; later run T010 and T011 in parallel with the score work, and T014/T015 in parallel after T013.
- Within US2, run T016/T017 in parallel, then T018/T019 in parallel.
- Within US3, run T021/T022 in parallel, then T023/T024 in parallel.

### Parallel Example: User Story 1

```text
Task: "T004 — score/grace reference tests in tests/otm_score.rs"
Task: "T005 — event processor tests in tests/otm_processor.rs"
Task: "T006 — token/model tests in tests/otm_model.rs"
Task: "T007 — library/Python API tests in tests/otm_api.rs and crates/matchmaker-py/tests/test_otm.py"

After those fail as expected:
Task: "T010 — Parangonar processor in src/features/parangonar.rs"
Task: "T011 — OTM model in src/dp/otm_model.rs"
```

### Parallel Example: User Story 2

```text
Task: "T016 — stage golden comparisons in tests/golden_otm.rs"
Task: "T017 — selected layer-output tests in tests/otm_model_layers.rs"

After US1:
Task: "T018 — OTM reference generation in tools/golden/generate.py"
Task: "T019 — paired benchmark support in crates/matchmaker-cli/src/bench.rs and tools/bench/"
```

### Parallel Example: User Story 3

```text
Task: "T021 — OTM failure/edge tests in tests/otm_errors.rs"
Task: "T022 — OTM export tests in tests/otm_export.rs"

After those fail as expected:
Task: "T023 — typed validation and empty-input handling in src/error.rs and src/dp/otm_model.rs"
Task: "T024 — OTM step diagnostics and tracing in src/base.rs and src/dp/otm.rs"
```

---

## Implementation Strategy

### MVP First

1. Complete T001–T003; do not bypass the model-rights gate.
2. Complete US1 to expose OTM through the library, CLI, and Python.
3. Complete the US2 golden-parity subset before treating the MVP as mergeable; Constitution
   Principle I requires reference goldens for every method and processor.
4. Validate the RTF target before calling the MVP complete.
5. Add US3 diagnostics and full error/export coverage, then finish the repository gates.

### Incremental Delivery

1. Setup and foundational tasks establish the authorized model artifact and CPU reference.
2. US1 delivers a usable OTM follower across all public interfaces.
3. US2 verifies the score/processor/model/tempo/path behavior against pinned reference data
   and establishes the benchmark result.
4. US3 adds the full diagnostic export and explicit failure behavior.
5. Polish updates the roadmap and validates the feature on repository quality gates.

**Suggested MVP scope**: US1 plus the minimum US2 golden coverage required by the
non-negotiable reference-parity principle. US3's full diagnostics can follow, but missing or
invalid assets and invalid user input must never panic or produce fallback results.
