---
description: "Serial delivery tasks for remaining audio methods and features"
---

# Tasks: Remaining Audio Methods and Features

**Input**: Design documents from specs/004-audio-methods/.
**Prerequisites**: [spec.md](spec.md), [plan.md](plan.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/audio-api.md](contracts/audio-api.md), [quickstart.md](quickstart.md).
**Tests**: Required by FR-015/FR-016 and Constitution I/IV. Write failing tests before porting each story; a focused filter must execute tests, not pass with zero selected.
**Organization**: Twelve serial stories, each seven tasks and one coherent separately authorized commit. Tasks remain unchecked because this artifact is planning, not implementation evidence.

## Format and common delivery contract

- Format: `- [ ] TNNN [USn] Description with exact file paths`. Setup/foundation/polish have no story label.
- No [P] tasks: the user selected serial work. Independent checks within the active story may run concurrently, but no later story implementation starts early.
- Every story implements its matching FR-001 through FR-012 and all shared FR-013 through FR-022. Explicitly apply model constraints from data-model.md and the API/numeric contract, not just the headline algorithm.
- Every final story task runs the full quickstart gate: narrow tests; all Rust/CLI/doc/Python regressions; fmt/clippy; MSRV/WASM; cargo-deny; Linux/Windows/macOS CI; five-run benchmark and accuracy/delta evidence. Never defer this to final polish.
- Write public docs and a runnable example in each new module. Include typed invalid-input errors, DEBUG timing spans, first-difference exports and reset/short/silent/degenerate-state tests in the same story.
- A final task is complete only after all gates pass, the story's separate authorized commit exists, and its hash is recorded in research.md. Failed/unrun gates or missing commit authorization pause the sequence.

## Phase 1: Setup

- [X] T001 Confirm implementation and C01-C12 commit authorization, review the active branch/worktree without reverting user edits, and record permitted scope in specs/004-audio-methods/research.md (FR-017, SC-004).

## Phase 2: Foundational verification

The shared audio pipeline already exists; do not scaffold a second pipeline or pre-register future methods.

- [X] T002 Verify the locked reference and pinned SoundFont using tools/golden/uv.lock and tests/data/audio/README.md; run existing tests/golden_audio_arzt.rs, tests/golden_audio_dixon.rs, tests/golden_audio_processors.rs and the local quickstart baseline gates; record results in specs/004-audio-methods/research.md before US1; per user clarification, multi-platform CI remains required before each delivery, not before local implementation (FR-014, FR-020, FR-021).

## Phase 3: US1 - arzt_tempo (P1, MVP, C01)

**Goal**: Tempo-aware Arzt through all interfaces with unchanged plain Arzt/MIDI behavior.
**Independent test**: Fixed steady/accelerating/slowing replay, insertion/deletion history and reset, plus facade end-to-end tests.

- [X] T003 [US1] Capture ArztTempoModel, before_step/history and tempo-dependent advance from released reference; extend tools/golden/generate_audio.py with fixed tempo-change cases and outputs under tests/golden/audio/; record config/limits in specs/004-audio-methods/research.md before porting (FR-001, FR-015).
- [X] T004 [US1] Add failing replay/tempo-state/short/reset tests in tests/golden_audio_arzt_tempo.rs and CLI/Python default, override, error and path tests in crates/matchmaker-cli/tests/follow_audio.rs and crates/matchmaker-py/tests/test_audio.py; retain existing Arzt/MIDI regression coverage.
- [X] T005 [US1] Port ArztTempoModel in src/utils/tempo_models.rs and dedicated method in src/dp/arzt_tempo.rs; minimally integrate tempo-controlled recurrence/history/reset in src/dp/arzt_frame.rs and exports in src/dp/mod.rs without changing plain Arzt defaults.
- [X] T006 [US1] Wire AudioMethod parsing/defaults/config and constructor in src/matchmaker.rs; allow audio-only method strings before MIDI enum dispatch in crates/matchmaker-cli/src/main.rs and mirror constructor/kwargs/iterator behavior in crates/matchmaker-py/src/lib.rs; add typed validation in src/error.rs as needed.
- [X] T007 [US1] Export tempo/cost history and reference metadata in src/export.rs, compare first differences in tools/golden/compare.py, and verify legacy keys plus new diagnostics in tests/audio_config_export.rs and edge cases in tests/edge_cases.rs (FR-018, FR-019).
- [X] T008 [US1] Add current-method benchmark entries/report support in tools/bench/audio.toml, tools/bench/run_reference.py and tools/bench/report.py; add smoke coverage in crates/matchmaker-cli/tests/bench_audio.rs and update public examples, README.md and UPSTREAM.md (FR-022).
- [X] T009 [US1] Run every per-story gate from specs/004-audio-methods/quickstart.md, record hashes/maxima/report delta/platform results in specs/004-audio-methods/research.md, and create only C01 `feat(audio): add arzt_tempo`; stop before US2 until its hash is recorded (FR-017, SC-002 through SC-004, SC-006).

## Phase 4: US2 - arzt_multi_fold (P1, C02)

**Goal**: Reference-compatible repeat hypotheses and switching.
**Independent test**: Taken/skipped/no-repeat fixtures, candidate costs near switch margin, exported selected hypothesis.

- [X] T010 [US2] Capture candidate unfolding/beat domain/selection semantics and generate licensed repeat fixtures/goldens using tools/golden/generate_audio.py, tests/data/audio/ and tests/golden/audio/; record selection_window=50 and switch_margin=0.05 evidence in specs/004-audio-methods/research.md (FR-002).
- [X] T011 [US2] Write failing candidate, switching/tie, short-score and path tests in tests/golden_audio_arzt_multi_fold.rs plus CLI/Python default and repeat behavior tests in crates/matchmaker-cli/tests/follow_audio.rs and crates/matchmaker-py/tests/test_audio.py.
- [X] T012 [US2] Port candidate follower/search owner in src/dp/arzt_multi_fold.rs and src/dp/mod.rs; reuse src/score/unfold.rs, src/score/audio.rs and src/dp/arzt_frame.rs; enforce valid repeat paths and selection history bounded by selection_window, without precomputing a different single unfolding.
- [X] T013 [US2] Add current-method configuration, compatibility validation and construction in src/matchmaker.rs, crates/matchmaker-cli/src/main.rs and crates/matchmaker-py/src/lib.rs; preserve MIDI/Audio Arzt/Dixon defaults.
- [X] T014 [US2] Export hypothesis identities, selection history, beat mappings and costs in src/export.rs; extend tools/golden/compare.py, tests/audio_config_export.rs and tests/edge_cases.rs to diagnose the first differing hypothesis/frame.
- [X] T015 [US2] Extend tools/bench/audio.toml, tools/bench/run_reference.py, tools/bench/report.py and crates/matchmaker-cli/tests/bench_audio.rs for repeat runs; document public module/examples, README.md and UPSTREAM.md.
- [ ] T016 [US2] Run all quickstart gates, record per-stage/metric maxima and benchmark/platform evidence in specs/004-audio-methods/research.md, and create only C02 `feat(audio): add arzt_multi_fold`; stop before US3 until committed.

## Phase 5: US3 - Audio ground truth (P1, C03)

**Goal**: Audio quality evaluation using the existing exact mapping and metrics.
**Independent test**: Known path/TSV metric equality; invalid input never truncates existing output.

- [ ] T017 [US3] Capture reference utils/eval.py mapping/threshold definitions and create small direct perf_sec/score_beat TSV plus metric goldens in tests/data/audio/ and tests/golden/audio/ using tools/golden/generate_audio.py; record provenance in specs/004-audio-methods/research.md (FR-003).
- [ ] T018 [US3] Write failing audio ground_truth CLI tests in crates/matchmaker-cli/tests/follow_audio.rs and mapping/metric/malformed input/output preservation tests in tests/golden_eval.rs; retain existing MIDI match-format tests.
- [ ] T019 [US3] Reuse GroundTruth and evaluate_alignment in src/utils/eval.rs, adding only required audio TSV validation; reject note-ID matches without an explicit MIDI performance and never route an audio file through src/performance.rs MIDI loading.
- [ ] T020 [US3] Replace the audio --ground-truth rejection in crates/matchmaker-cli/src/main.rs, validate before output writes, and preserve follow/eval metric JSON and existing library evaluator behavior; update only necessary typed errors in src/error.rs.
- [ ] T021 [US3] Add ground-truth identity and evaluator metrics to audio export/report records via src/export.rs and tools/golden/compare.py; verify non-destructive errors and legacy export schemas in tests/audio_config_export.rs and crates/matchmaker-cli/tests/follow_audio.rs.
- [ ] T022 [US3] Integrate audio evaluation records in tools/bench/audio.toml, tools/bench/run_reference.py, tools/bench/report.py and crates/matchmaker-cli/tests/bench_audio.rs; update evaluator examples, README.md and UPSTREAM.md without adding a second evaluation API.
- [ ] T023 [US3] Run all quickstart gates and exact-mapping metric comparisons, record evidence in specs/004-audio-methods/research.md, and create only C03 `feat(cli): evaluate audio ground truth`; stop before US4 until committed.

## Phase 6: US4 - raw_spectrum (P2, C04)

**Goal**: Raw magnitude observations, compatible frame-follower overrides and standalone extraction.
**Independent test**: Reference magnitudes/timestamps/reset, 257 default bins, library/CLI/Python extraction.

- [ ] T024 [US4] Generate raw_spectrum default/override/short/silent reference features with tools/golden/generate_audio.py under tests/golden/audio/ and record reference center=False/n_fft behavior in specs/004-audio-methods/research.md (FR-004).
- [ ] T025 [US4] Add failing raw_spectrum and compatibility cases in tests/golden_audio_processors.rs, tests/audio_config_export.rs, crates/matchmaker-cli/tests/follow_audio.rs and crates/matchmaker-py/tests/test_audio.py, including standalone extraction through all interfaces.
- [ ] T026 [US4] Implement src/features/raw_spectrum.rs and register it in src/features/mod.rs using realfft and the existing AudioProcessor; enforce "Feature dimensions must match the selected processor and reference model" and preserve short-frame/timestamp behavior.
- [ ] T027 [US4] Wire processor/config in src/matchmaker.rs; add minimal `features PERFORMANCE --processor KEY --export DIRECTORY` in crates/matchmaker-cli/src/main.rs and extract_audio_features in crates/matchmaker-py/src/lib.rs, reusing src/io/audio.rs and the existing exporter, not a second pipeline.
- [ ] T028 [US4] Extend src/export.rs and tools/golden/compare.py for standalone metadata/vector features; test legacy layout, invalid dimensions/rates/n_fft, reset and output preservation in tests/audio_config_export.rs and tests/edge_cases.rs.
- [ ] T029 [US4] Benchmark raw_spectrum on compatible Arzt/Dixon in tools/bench/audio.toml, tools/bench/run_reference.py, tools/bench/report.py and crates/matchmaker-cli/tests/bench_audio.rs; document module/extraction examples, README.md and UPSTREAM.md.
- [ ] T030 [US4] Run all quickstart gates and reference comparisons, record evidence in specs/004-audio-methods/research.md, and create only C04 `feat(audio): add raw_spectrum processor`; stop before SKF until committed.

## Phase 7: US5 - skf (P2, C05)

**Goal**: Switching Kalman filtering with explicit reference template modes.
**Independent test**: Fixed-template state replay plus separate synthesized/fallback pipeline metrics.

- [ ] T031 [US5] Audit/pin template synthesis asset/mode and Gaussian fallback conditions in specs/004-audio-methods/research.md and tests/data/audio/README.md; generate fixed templates/state/goldens using tools/golden/generate_audio.py under tests/golden/audio/; enforce "Asset hashes, provenance, and template mode must accompany every template comparison" (FR-005, FR-020).
- [ ] T032 [US5] Write failing fixed-template chord/age/tempo/beam/covariance/degenerate likelihood replay in tests/golden_audio_skf.rs and default/config/error/path tests in crates/matchmaker-cli/tests/follow_audio.rs and crates/matchmaker-py/tests/test_audio.py.
- [ ] T033 [US5] Port score templates, switching/merge/tempo inference and reference safeguards in src/prob/skf.rs and register in src/prob/mod.rs; reuse src/score/audio.rs/RustySynth for runtime synthesis; keep Gaussian mode and asset errors distinguishable.
- [ ] T034 [US5] Add SKF defaults (8000 Hz, hop=128, n_fft=512), raw_spectrum-only compatibility and construction in src/matchmaker.rs, crates/matchmaker-cli/src/main.rs and crates/matchmaker-py/src/lib.rs, with typed invalid-input failures in src/error.rs.
- [ ] T035 [US5] Export beam weights/tempo moments/templates/mode in src/export.rs and compare in tools/golden/compare.py; cover short/one-onset/unstable-state and legacy exports in tests/audio_config_export.rs and tests/edge_cases.rs.
- [ ] T036 [US5] Extend tools/bench/audio.toml, tools/bench/run_reference.py, tools/bench/report.py and crates/matchmaker-cli/tests/bench_audio.rs for SKF feature/inference/full-pipeline metrics; document public examples, README.md and UPSTREAM.md.
- [ ] T037 [US5] Run all quickstart gates, fixed-template limits and renderer metric limits; record asset/mode/hash/report/platform evidence in specs/004-audio-methods/research.md and create only C05 `feat(audio): add skf follower`; pause on unverified template mode.

## Phase 8: US6 - cqt (P2, C06)

**Goal**: Pure-Rust reference-compatible constant-Q features.
**Independent test**: 88 bins, norm, complex magnitude, edge trim and rate/hop overrides.

- [ ] T038 [US6] Evaluate maintained pure-Rust CQT alternatives and record version/license/MSRV/WASM/parity evidence in specs/004-audio-methods/research.md; generate small reference probes with tools/golden/generate_audio.py and tests/golden/audio/ before choosing a targeted kernel port or Cargo.toml dependency (FR-006, FR-021).
- [ ] T039 [US6] Add failing cqt probes/default/rate/hop/trim/norm tests in tests/golden_audio_processors.rs and compatible follower/extraction tests in tests/audio_config_export.rs, crates/matchmaker-cli/tests/follow_audio.rs and crates/matchmaker-py/tests/test_audio.py; failed probe blocks registration.
- [ ] T040 [US6] Implement validated CQT in src/features/cqt.rs and src/features/mod.rs using the chosen parity-compatible approach; reuse existing FFT/resampling, preserve complex magnitude and [1:-1] trimming, and update Cargo.toml/deny.toml only if the audited candidate is selected.
- [ ] T041 [US6] Enable only cqt current-story config, compatible frame-following and standalone extraction in src/matchmaker.rs, crates/matchmaker-cli/src/main.rs and crates/matchmaker-py/src/lib.rs; reject invalid rates/hops or unsupported pairs.
- [ ] T042 [US6] Export CQT rows/config/timestamps in src/export.rs, diagnose first-difference bins/frames in tools/golden/compare.py, and cover short/empty/reset and dimension errors in tests/audio_config_export.rs and tests/edge_cases.rs.
- [ ] T043 [US6] Extend tools/bench/audio.toml, tools/bench/run_reference.py, tools/bench/report.py and crates/matchmaker-cli/tests/bench_audio.rs for compatible CQT runs; document numeric/library decision, module examples, README.md and UPSTREAM.md.
- [ ] T044 [US6] Run all quickstart gates including CQT numeric/MSRV/WASM/license proof, record evidence in specs/004-audio-methods/research.md, and create only C06 `feat(audio): add cqt processor`; no approximation, FFI, or tolerance weakening to bypass a failure.

## Phase 9: US7 - cqt_spectral_flux (P2, C07)

**Goal**: CQT(+flux) observations with released cross-call matrix state.
**Independent test**: 89/88 bins, first-call zero flux, positive changes, trimmed rows and reset.

- [ ] T045 [US7] Capture the released full-matrix previous-call flux behavior, optional flux dimensions, inherited no-op reset and score-to-performance shape interaction using tools/golden/generate_audio.py with outputs under tests/golden/audio/; record any reference pipeline blocker in specs/004-audio-methods/research.md (FR-007).
- [ ] T046 [US7] Add failing cqt_spectral_flux matrix-history/first-call/disabled-flux/reset tests in tests/golden_audio_processors.rs and facade/extraction/compatibility cases in tests/audio_config_export.rs, crates/matchmaker-cli/tests/follow_audio.rs and crates/matchmaker-py/tests/test_audio.py.
- [ ] T047 [US7] Implement src/features/cqt_spectral_flux.rs and src/features/mod.rs reusing src/features/cqt.rs only where transform semantics match; retain reference matrix subtraction, inherited no-op reset and edge trimming instead of replacing them with adjacent-row flux or a clearing reset; use fresh instances for independent runs.
- [ ] T048 [US7] Wire current processor/config/validated frame-consumer pairs and extraction in src/matchmaker.rs, crates/matchmaker-cli/src/main.rs and crates/matchmaker-py/src/lib.rs; preserve optional flux width and typed shape failures.
- [ ] T049 [US7] Export flux/CQT fields/history metadata in src/export.rs and compare via tools/golden/compare.py; cover reset, short unequal shapes and legacy exports in tests/audio_config_export.rs and tests/edge_cases.rs.
- [ ] T050 [US7] Extend tools/bench/audio.toml, tools/bench/run_reference.py, tools/bench/report.py and crates/matchmaker-cli/tests/bench_audio.rs for flux extraction/compatible frame runs; document examples, README.md and UPSTREAM.md.
- [ ] T051 [US7] Run all quickstart gates, record feature/metric/report/platform evidence in specs/004-audio-methods/research.md, and create only C07 `feat(audio): add cqt_spectral_flux processor`; stop before Audio-OuterHMM until committed.

## Phase 10: US8 - Audio outerhmm (P2, C08)

**Goal**: Audio-specific two-layer outer-product HMM without changing MIDI emissions/defaults.
**Independent test**: Identical-template weighted-chord/posterior replay and separate full-pipeline metrics.

- [ ] T052 [US8] Audit bundled pitch-template archive origin/license/hash/support mask and synthesized-overlay rule before copying assets; capture registry sample-rate override quirks in specs/004-audio-methods/research.md and tests/data/audio/README.md; extend tools/golden/generate_audio.py and tests/golden/audio/ only with permitted assets (FR-008, FR-020).
- [ ] T053 [US8] Write failing fixed-template weighted/sustained-note chord, sound/pause/repeat/likelihood replay in tests/golden_audio_outerhmm.rs and facade default/override/error plus MIDI outerhmm regression tests in crates/matchmaker-cli/tests/follow_audio.rs and crates/matchmaker-py/tests/test_audio.py.
- [ ] T054 [US8] Port pitch/chord-template construction, learned-row overlay, sound/pause states, transitions and safeguards in src/prob/outerhmm_audio.rs and src/prob/mod.rs; reuse helpers from src/prob/outerhmm.rs only when behavior truly matches and leave MIDI model behavior intact.
- [ ] T055 [US8] Add audio outerhmm defaults (16000/640, cqt_spectral_flux, s_j=0.0), validated layouts and verified override semantics in src/matchmaker.rs, crates/matchmaker-cli/src/main.rs and crates/matchmaker-py/src/lib.rs; enforce descriptive asset/config errors via src/error.rs.
- [ ] T056 [US8] Export pitch/chord templates, support mask, posterior/sound-pause state and provenance in src/export.rs; extend tools/golden/compare.py, tests/audio_config_export.rs and tests/edge_cases.rs for first-difference and one-onset/overrun tests.
- [ ] T057 [US8] Extend tools/bench/audio.toml, tools/bench/run_reference.py, tools/bench/report.py and crates/matchmaker-cli/tests/bench_audio.rs for audio outerhmm replay/full-pipeline quality; update examples, README.md and UPSTREAM.md.
- [ ] T058 [US8] Run all quickstart gates and fixed-template/renderer comparisons, record provenance/hash/report/platform evidence in specs/004-audio-methods/research.md, and create only C08 `feat(audio): add outerhmm follower`; unverified learned-template rights block delivery.

## Phase 11: US9 - mel (P3, C09)

**Goal**: Reference mel-power observations for compatible frame followers and extraction.
**Independent test**: 128 bins, bank/norm/scaling/default hops, silence and reset.

- [ ] T059 [US9] Generate default/override/mel-bank/power reference outputs using tools/golden/generate_audio.py under tests/golden/audio/ and record normalization evidence in specs/004-audio-methods/research.md (FR-009).
- [ ] T060 [US9] Add failing mel/silence/reset/override goldens in tests/golden_audio_processors.rs and compatible follower/extraction/error tests in tests/audio_config_export.rs, crates/matchmaker-cli/tests/follow_audio.rs and crates/matchmaker-py/tests/test_audio.py.
- [ ] T061 [US9] Implement src/features/mel.rs and src/features/mod.rs with installed realfft, exact librosa power/filter-bank normalization and n_fft=2*hop; reuse a spectral helper only if it removes actual duplication.
- [ ] T062 [US9] Enable mel configuration, compatible consumers and extraction in src/matchmaker.rs, crates/matchmaker-cli/src/main.rs and crates/matchmaker-py/src/lib.rs; retain old processor/default behavior and typed invalid dimensions/norm errors.
- [ ] T063 [US9] Extend src/export.rs and tools/golden/compare.py for mel metadata/first-bin differences; verify empty/short/reset and legacy schemas in tests/audio_config_export.rs and tests/edge_cases.rs.
- [ ] T064 [US9] Extend tools/bench/audio.toml, tools/bench/run_reference.py, tools/bench/report.py and crates/matchmaker-cli/tests/bench_audio.rs for compatible mel runs; update examples, README.md and UPSTREAM.md.
- [ ] T065 [US9] Run all quickstart gates, record evidence in specs/004-audio-methods/research.md, and create only C09 `feat(audio): add mel processor`; stop before MFCC until committed.

## Phase 12: US10 - mfcc (P3, C10)

**Goal**: Reference MFCC observations without silently correcting the released norm default.
**Independent test**: 13 coefficients, DCT/log-power scaling, default/override behavior and compatible runs.

- [ ] T066 [US10] Verify locked librosa accepts the released norm="backward" default and generate coefficient/log-power goldens with tools/golden/generate_audio.py under tests/golden/audio/; document any reference blocker and DCT library evaluation in specs/004-audio-methods/research.md before implementing (FR-010).
- [ ] T067 [US10] Add failing mfcc coefficient/scaling/norm/silent/reset tests in tests/golden_audio_processors.rs and compatible follower/extraction/error tests in tests/audio_config_export.rs, crates/matchmaker-cli/tests/follow_audio.rs and crates/matchmaker-py/tests/test_audio.py.
- [ ] T068 [US10] Implement src/features/mfcc.rs and src/features/mod.rs, reuse validated src/features/mel.rs internals where correct, and match log-power/DCT precision and scaling; change Cargo.toml/deny.toml only for an audited complexity-reducing DCT dependency.
- [ ] T069 [US10] Enable current MFCC config, compatible consumers and extraction in src/matchmaker.rs, crates/matchmaker-cli/src/main.rs and crates/matchmaker-py/src/lib.rs; preserve reference default/errors instead of replacing backward norm with ortho.
- [ ] T070 [US10] Extend src/export.rs, tools/golden/compare.py, tests/audio_config_export.rs and tests/edge_cases.rs for coefficient metadata, first-difference diagnostics, reset and zero-energy behavior.
- [ ] T071 [US10] Extend tools/bench/audio.toml, tools/bench/run_reference.py, tools/bench/report.py and crates/matchmaker-cli/tests/bench_audio.rs for compatible MFCC runs; update examples, README.md and UPSTREAM.md.
- [ ] T072 [US10] Run all quickstart gates, record norm/numeric/report/platform evidence in specs/004-audio-methods/research.md, and create only C10 `feat(audio): add mfcc processor`; pause if the pinned reference cannot run the requested default.

## Phase 13: US11 - korzeniowski audio processor (P3, C11)

**Goal**: Dedicated structured audio observations, extractable before PF exists.
**Independent test**: Spectrum/onset/loudness components, 512 direct versus 1470 facade hop, reset and silence dB floor.

- [ ] T073 [US11] Capture reference structured fields, periodic Hann/crop/padding, unnormalized causal flux and RMS/dB behavior with tools/golden/generate_audio.py under tests/golden/audio/; record direct/facade hop distinction in specs/004-audio-methods/research.md (FR-011).
- [ ] T074 [US11] Add failing korzeniowski component/default-hop/reset/silence tests in tests/golden_audio_processors.rs and structured extraction/incompatible-pair rejection tests in tests/audio_config_export.rs, crates/matchmaker-cli/tests/follow_audio.rs and crates/matchmaker-py/tests/test_audio.py.
- [ ] T075 [US11] Add KorzeniowskiObservation and processor in src/features/korzeniowski.rs and src/features/mod.rs without replacing vector Observation/AudioProcessor; preserve optional note fields absent for audio and nonnegative unclamped onset.
- [ ] T076 [US11] Extend extraction/config in src/matchmaker.rs, crates/matchmaker-cli/src/main.rs and crates/matchmaker-py/src/lib.rs for structured fields; enforce "Structured Korzeniowski observations must not be fed to vector-only followers" and keep PF unregistered until US12.
- [ ] T077 [US11] Export spectrum/onset/loudness separately with explicit absent note metadata in src/export.rs and tools/golden/compare.py; cover state reset/short input/invalid window and old vector exports in tests/audio_config_export.rs and tests/edge_cases.rs.
- [ ] T078 [US11] Benchmark standalone structured extraction using tools/bench/audio.toml, tools/bench/run_reference.py, tools/bench/report.py and crates/matchmaker-cli/tests/bench_audio.rs; do not force it through a vector follower; document module/field examples, README.md and UPSTREAM.md.
- [ ] T079 [US11] Run all quickstart gates, record component/timing/platform evidence in specs/004-audio-methods/research.md, and create only C11 `feat(audio): add korzeniowski processor`; stop before PF until committed.

## Phase 14: US12 - pfkorz (P3, C12)

**Goal**: Seeded audio particle filter with analytical score model and inspectable state.
**Independent test**: Fixed-draw deterministic substeps, exact Rust seed-repeat, pooled and per-seed reference metrics.

- [ ] T080 [US12] Evaluate maintained seeded RNG license/MSRV/WASM behavior in specs/004-audio-methods/research.md; generate analytical score model, fixed draws and reference runs for seeds 1984-1988 using tools/golden/generate_audio.py under tests/golden/audio/, without altering the pinned reference package (FR-012, SC-005).
- [ ] T081 [US12] Write failing fixed-draw/weights/resampling/rest/tempo/seed-repeat/score-model tests in tests/golden_audio_pfkorz.rs and facade seed/count/error/path tests in crates/matchmaker-cli/tests/follow_audio.rs and crates/matchmaker-py/tests/test_audio.py; assert all five seed records exist.
- [ ] T082 [US12] Implement analytical KorzeniowskiScoreModel in src/score/korzeniowski.rs and src/score/mod.rs, and seeded PF in src/prob/pfkorz.rs and src/prob/mod.rs; enforce consistent grid lengths/template width/nearest-onset indices, handle weight collapse as the reference, and audit any Cargo.toml/deny.toml RNG change.
- [ ] T083 [US12] Add structured observation consumer, seed=1984, num_particles=500 facade defaults and rate/hop/window validation in src/matchmaker.rs, crates/matchmaker-cli/src/main.rs and crates/matchmaker-py/src/lib.rs; retain direct processor defaults and iterator/path compatibility.
- [ ] T084 [US12] Export score-grid/rest/templates, seed, particles/weights/tempo and resampling diagnostics in src/export.rs; extend tools/golden/compare.py, tests/audio_config_export.rs and tests/edge_cases.rs for exact repeat, first-difference and bounded particle-state behavior.
- [ ] T085 [US12] Extend tools/bench/audio.toml, tools/bench/run_reference.py, tools/bench/report.py and crates/matchmaker-cli/tests/bench_audio.rs for per-seed and pooled metrics using exact evaluator; add public module/score-model/seed examples, README.md and UPSTREAM.md (FR-016, FR-022).
- [ ] T086 [US12] Run all quickstart gates, exact Rust reproducibility and predeclared five-seed accuracy limits; record RNG/config/hash/report/platform evidence in specs/004-audio-methods/research.md and create only C12 `feat(audio): add pfkorz follower`; no NumPy-stream identity claim.

## Phase 15: Final audit (no deferred implementation)

- [ ] T087 Verify all C01-C12 hashes/evidence in specs/004-audio-methods/research.md, all 87 task states in specs/004-audio-methods/tasks.md, and the completed method/processor tables in README.md and UPSTREAM.md; confirm no commit mixes algorithms/processors and no per-story gate was deferred (SC-001 through SC-006).

## Dependencies and execution order

T001 -> T002 -> US1 -> US2 -> US3 -> US4 -> US5 -> US6 -> US7 -> US8 -> US9 -> US10 -> US11 -> US12 -> T087.

This strict order is a delivery constraint, not a claim that all algorithms technically depend on their predecessor. Technical prerequisites: SKF -> raw_spectrum + template mode; CQT flux -> CQT; Audio-OuterHMM -> CQT flux + permitted templates; MFCC -> validated mel/spectral computation; PF -> structured processor + analytical score model + seeded RNG + evaluator.

Within every story: reference/asset gate -> failing tests -> core -> facade -> export/edge cases -> docs/benchmark -> full gate and commit. All current-story tests must remain runnable without any later story registered. Library/vector compatibility is checked against already delivered followers.

## Parallel opportunities

No cross-story implementation parallelism or agent fan-out is planned. Once the active story's code exists, independent read-only Rust/CLI/Python checks can run together where environments do not conflict. This does not bypass focused post-edit validation or commit ordering.

Examples by story: US1 tempo replay + Python defaults; US2 repeat replay + CLI invalid margin; US3 evaluator primitives + TSV CLI; US4 spectrum goldens + extraction interfaces; US5 fixed-template replay + default kwargs; US6 CQT goldens + rate validation; US7 flux reset + optional width; US8 posterior replay + MIDI regressions; US9 mel goldens + extraction; US10 DCT goldens + norm validation; US11 structured goldens + pair rejection; US12 seed-repeat + facade counts. These are check opportunities only, never early next-story work.

## Implementation strategy

MVP is T001-T009 and C01 only. First port the exact controlling recurrence/state with a cheap replay check, then complete the same story's interfaces and gates. At each commit boundary keep the worktree scoped and record evidence. Pause on an unverified asset, failed reference default, missing pure-Rust parity, unavailable required gate, or unauthorized commit; do not reorder or add substitutes without user approval.

All shared requirements are explicitly enforced by the common delivery contract and every story's gate: FR-013 interfaces/extraction, FR-014 regressions, FR-015/016 golden and metric comparisons, FR-017 commits, FR-018 diagnostics, FR-019 validation, FR-020 provenance, FR-021 portability/licenses, FR-022 benchmark evidence. No shared algorithm work is deferred to T087.
