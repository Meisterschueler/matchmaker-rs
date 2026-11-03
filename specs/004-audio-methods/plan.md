# Implementation Plan: Remaining Audio Methods and Features

**Branch**: `004-audio-methods` (already exists) | **Date**: 2026-11-03 | **Spec**: [spec.md](spec.md)
**Input**: `specs/004-audio-methods/spec.md`
**Status**: Design complete; implementation gates unexecuted. Planning does not authorize source changes or commits.

## Summary

Extend recorded audio with five methods, six processors, and ground-truth evaluation. Deliver exactly one story per commit, in the agreed order. Each story owns reference fixtures, implementation, library/CLI/Python integration, exports, documentation, benchmarks, and quality gates. Register a method only when its consumer path is complete.

Reuse AudioStream, score loading/rendering, ArztFrame, OnlineAlignment, the evaluator, golden generator, exporter, and benchmark runners. Preserve vector Observation and AudioProcessor contracts; add a separate structured Korzeniowski observation in US11. No generic registry or inference framework.

## Technical Context

**Language/Version**: Rust edition 2024, MSRV 1.88; existing Python 3.10-3.13 bindings and locked reference scripts.
**Primary Dependencies**: Existing realfft 3.5.1, rubato 5.0.1, rustysynth 1.3.8, symphonia 0.6.1, serde, tracing; PyO3/numpy in the separate binding crate.
**Storage**: Small checked-in fixtures, golden arrays, provenance metadata; temporary SoundFonts under target; no database.
**Testing**: Rust unit/integration/doc tests, CLI tests, Python pytest, reference replay and full-pipeline metric comparisons.
**Target Platform**: Linux, Windows, macOS; default core build on wasm32-unknown-unknown.
**Project Type**: Existing library workspace with CLI and separate PyO3 bindings.
**Performance Goals**: Feature/inference latency and real-time factor; five-run median, flag and justify >10% slowdown against latest published master baseline. No unsupported speedup promise.
**Constraints**: No native default-runtime dependencies or unsafe additions; locked reference 0.4.0, verified asset rights, seeded Rust particle runs.
**Scale/Scope**: Twelve serial increments, recorded audio only. Long-input tests distinguish bounded model windows/beam/particles from intentionally retained alignment paths.

No dependency is added in planning. US6 evaluates maintained pure-Rust CQT alternatives against librosa; absent a parity-compatible candidate, port the specific kernel using existing FFT/resampling after documenting the libraries-first rationale. US12 evaluates a maintained seeded Rust RNG with WASM-compatible explicit seeding. See [research.md](research.md).

## Constitution Check

*Gate: evaluate before Phase 0 research and again after Phase 1 design.*

| Principle | Pre-research design check | Post-design check / implementation gate |
|---|---|---|
| I Reference parity | PASS: released 0.4.0, existing lockfile | PASS in design: goldens for every story; exact evaluator for synthesis/stochastic comparisons. Numeric/asset evidence remains a delivery gate. |
| II Structural parity | PASS: Stream -> Processor -> OnlineAlignment | PASS: one module per method/processor, documented public items and runnable examples; no generic framework. |
| III Performance | PASS: existing harness and DEBUG spans | PASS: per-story five-run benchmarks, accuracy records, PR delta and >10% flag. |
| IV Debuggability | PASS: typed errors and NumPy-readable export | PASS: state exports, first-difference diagnostics, structured observations, seeded PF. |
| V Pure Rust, libraries first | PASS: reuse installed pure-Rust dependencies | PASS in design: CQT/RNG evaluation before new code; no FFI escape hatch; MSRV/WASM/license gates. |

No constitution exceptions requested. Design checks are not runtime success claims. Unverified template rights or a failed pure-Rust parity probe block the story without weakening Principle I or V.

## Project Structure

### Documentation (this feature)

```text
specs/004-audio-methods/
  spec.md
  plan.md
  research.md
  data-model.md
  quickstart.md
  contracts/audio-api.md
  checklists/requirements.md
  tasks.md
```

### Source Code (repository root)

```text
src/
  matchmaker.rs, base.rs, export.rs, error.rs
  dp/arzt_frame.rs, arzt_tempo.rs, arzt_multi_fold.rs
  features/raw_spectrum.rs, cqt.rs, cqt_spectral_flux.rs, mel.rs, mfcc.rs, korzeniowski.rs
  prob/skf.rs, outerhmm_audio.rs, pfkorz.rs
  score/korzeniowski.rs
  utils/tempo_models.rs, eval.rs
crates/matchmaker-cli/src/main.rs
crates/matchmaker-cli/tests/follow_audio.rs, bench_audio.rs
crates/matchmaker-py/src/lib.rs
crates/matchmaker-py/tests/test_audio.py
tests/golden_audio_*.rs, audio_config_export.rs, edge_cases.rs
tests/data/audio/, tests/golden/audio/
tools/golden/generate_audio.py, compare.py
tools/bench/audio.toml, run_reference.py, report.py
```

**Structure Decision**: Extend owning modules and existing facade/test files. New files appear only in their story. ArztTempoModel stays in utils/tempo_models.rs with a dedicated arzt_tempo method module. Vector processors stay on AudioProcessor; structured observations and score templates live in Korzeniowski modules. Add constructor/enum variants only when the full story works.

## Phase 0: Research Decisions

Use [research.md](research.md) for verified reference semantics, asset rules, dependencies, and probe gates. No scope clarification is required. Remaining CQT feasibility, asset permission, and numeric evidence are assigned to consuming stories, not presented as solved.

## Phase 1: Design and Contracts

[data-model.md](data-model.md) defines entities and state invariants. [contracts/audio-api.md](contracts/audio-api.md) defines defaults, compatibility, exports, errors, evaluation, and limits. [quickstart.md](quickstart.md) defines executable per-story validation and commit gates.

## Delivery Order and Commit Boundaries

| Increment | Story | Single implementation commit | Prerequisite |
|---|---|---|---|
| C01 | US1 | feat(audio): add arzt_tempo | Existing Arzt/audio baseline |
| C02 | US2 | feat(audio): add arzt_multi_fold | C01; existing score repeats |
| C03 | US3 | feat(cli): evaluate audio ground truth | C02; existing evaluator |
| C04 | US4 | feat(audio): add raw_spectrum processor | C03 |
| C05 | US5 | feat(audio): add skf follower | C04; verified template mode |
| C06 | US6 | feat(audio): add cqt processor | C05; CQT parity probe |
| C07 | US7 | feat(audio): add cqt_spectral_flux processor | C06 |
| C08 | US8 | feat(audio): add outerhmm follower | C07; template provenance/hash gate |
| C09 | US9 | feat(audio): add mel processor | C08 |
| C10 | US10 | feat(audio): add mfcc processor | C09 |
| C11 | US11 | feat(audio): add korzeniowski processor | C10 |
| C12 | US12 | feat(audio): add pfkorz follower | C11; score model and RNG gate |

Each row owns fixtures, tests, exports, configuration, all interfaces, docs, benchmarks, and gate evidence. Specs may be prepared together; implementation remains serial. No infrastructure precommit exposes an unfinished algorithm. Planning artifacts may have a documentation-only commit, but none is made now. Obtain explicit authorization for this commit sequence when implementation is requested; lack of authorization pauses at the boundary, never combines stories.

MVP is C01 only. Work test-first in tiny internal edits with immediate focused checks; the final coherent story is one commit. No test-only or facade-only algorithm commit. A blocked step pauses the sequence; reordering/removing a story requires user approval.

## Required Quality Gates

Implementation clarification (2026-11-03): the user approved the verified local baseline for T002, then explicitly authorized the separate C01 commit and featurebranch push after all local gates passed. The commit/publication precede remote CI; multi-platform CI remains required before delivery/merge and before C02 begins. Unverified remote checks are not treated as passed; no merge is authorized.

Each story runs narrow replay/feature and interface tests first, then all Rust/CLI tests (all features), doc tests, Python binding build/tests across supported versions in CI, fmt, clippy with warnings denied, MSRV 1.88, WASM core build, and cargo-deny. Algorithm-changing PRs include goldens and a benchmark delta; CI stays green on Linux, Windows, and macOS. Reference scripts use the lockfile and verified SoundFont provisioning.

Record results in research.md and benchmark reports. Missing platform/network tooling blocks a completed gate claim and must be disclosed. Do not fix unrelated baseline failures in an algorithm port.

## Complexity Tracking

No constitution violations or architecture layers requested. If no CQT crate meets parity, document the libraries-first evaluation before porting the specific reference kernel; this does not waive parity or Pure-Rust gates.
