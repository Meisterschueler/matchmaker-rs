# Implementation Plan: Audio Online Time Warping (arzt, dixon)

**Branch**: `003-audio-oltw` | **Date**: 2026-11-01 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/003-audio-oltw/spec.md`

## Summary

Add prerecorded audio score following for frame-level `arzt` and `dixon`, with `chroma` and `lse` processors, a mono 44.1-kHz audio pipeline, synthesized score reference features, library/CLI/Python interfaces, stage exports, golden parity tests, and an audio benchmark. Preserve all existing MIDI APIs and defaults. Use a pure-Rust decoder, resampler, and synthesizer; score-synthesis differences from FluidSynth are evaluated using reference alignment metrics as required by Constitution I.

## Technical Context

**Language/Version**: Rust stable, edition 2024, MSRV 1.88; Python reference generator uses locked Python 3.12 environment.

**Primary Dependencies**: Symphonia 0.6.1 (selected WAV/FLAC/MP3/Ogg decoding features), Rubato 5.0.1 (fixed-ratio offline resampling), RustySynth 1.3.8 (SF2 synthesis), existing realfft 3.5.1 (STFT). Add MPL-2.0 to `deny.toml` with a compatibility rationale for Symphonia.

**Storage**: Input score/audio files, the configured SoundFont, generated test WAV fixtures, and existing `.npy`/JSON golden artifacts. No database.

**Testing**: Rust unit/integration/golden/doctests; Python binding tests on supported Python versions; reference generator in the locked golden environment; `cargo deny`, clippy, MSRV and WASM gates; side-by-side audio benchmark.

**Target Platform**: Windows, Linux, and macOS; default core build must continue to compile for `wasm32-unknown-unknown`. Audio file processing is offline and does not require platform audio devices.

**Project Type**: Rust library with CLI and PyO3 bindings.

**Performance Goals**: Median follower inference latency at least 5x lower than the Python reference per frame; publish feature/inference latency and real-time factor using at least five Rust runs. Flag regressions above the existing 10% benchmark threshold.

**Constraints**: Pure Rust in the default build; supported containers are WAV, FLAC, MP3, and Ogg/Vorbis. Normalize input to mono and the configured target sample rate (44.1 kHz by default), padding the final partial hop. Require a configured, pinned SF2 SoundFont for runtime score synthesis; do not bundle or download an asset until its source, checksum, and license are verified. No live capture, M4A/MP4, AAC, audio `outerhmm`, or other non-roadmap methods in this feature.

**Scale/Scope**: Two audio followers, two audio processors, all four method/processor combinations in feature tests, one deterministic bundled fixture, Python/CLI/library paths, and an audio benchmark dataset subset after provenance is verified.

## Constitution Check

*Gate before research: all five principles have a compliant design direction. Re-check after Phase 1 below.*

| Principle | Gate | Design evidence |
|---|---|---|
| I. Reference Parity | PASS | Use only pinned pymatchmaker v0.4.0; golden files per processor/method; isolate score-render differences and justify them through alignment metrics. |
| II. Readability & Structural Parity | PASS | Dedicated audio I/O, processor, score-rendering, and frame-follower modules map to upstream paths; public docs and runnable examples are required. |
| III. Visible Performance | PASS | Add per-frame tracing spans and run side-by-side feature/inference benchmarks with five or more repetitions. |
| IV. Debuggability | PASS | Export decoded/frame features, score reference, beat mapping, per-frame positions, and final path; typed file/config errors. |
| V. Pure Rust, Libraries First | PASS WITH GATES | Symphonia/Rubato/RustySynth/realfft avoid C/C++; add a justified MPL-2.0 allow-list entry; verify `cargo deny` and WASM build. SF2 asset license/checksum must be verified before bundling or downloading. |

No Constitution violation is planned. The asset and target gates are mandatory implementation prerequisites, not assumed successes.

## Project Structure

### Documentation (this feature)

```text
specs/003-audio-oltw/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── audio-api.md
└── tasks.md                 # generated in Phase 2, not by this plan
```

### Source Code (repository root)

```text
src/
├── io/audio.rs              # decoding, mono conversion, resampling, hop framing
├── features/
│   ├── chroma.rs            # ChromagramProcessor
│   └── lse.rs               # LogSpectralEnergyProcessor
├── score/audio.rs           # score-to-audio rendering and frame-to-beat mapping
├── dp/
│   ├── arzt_frame.rs        # frame-level Arzt OLTW
│   └── dixon_frame.rs       # frame-level Dixon OLTW
├── matchmaker.rs            # input-type routing, defaults, config validation
├── export.rs                # audio stages in Python-readable output
└── error.rs                 # typed audio decode/render/config errors

crates/matchmaker-cli/src/   # audio input selection, SoundFont path and run output
crates/matchmaker-py/src/    # input_type="audio" facade and audio defaults
tests/
├── golden_audio_processors.rs
├── golden_audio_arzt.rs
├── golden_audio_dixon.rs
└── data/audio/              # deterministic, redistributable WAV fixture(s)
tools/golden/
└── generate_audio.py        # audio fixture/reference golden generation
tools/bench/                 # audio manifests, reference runner, report support
```

**Structure Decision**: Extend the existing single Rust workspace and its library/CLI/Python crates. Keep event-level MIDI followers in their existing modules; put frame-level audio variants in separate `*_frame.rs` modules. Keep `Matchmaker::new` MIDI behavior stable and add explicit audio construction/configuration in the Rust facade.

## Phase 0: Research

Completed in [research.md](research.md). Decisions cover the exact method defaults and units, stream timing/cache behavior, processor equations, reference score synthesis and beat mapping, dependency/license/target tradeoffs, and a redistributable golden-fixture strategy. No unresolved `[NEEDS CLARIFICATION]` items remain.

## Phase 1: Design & Contracts

Completed in [data-model.md](data-model.md), [contracts/audio-api.md](contracts/audio-api.md), and [quickstart.md](quickstart.md). The contract fixes supported containers and defaults, records the configured SoundFont prerequisite, and preserves MIDI behavior.

## Constitution Check (Post-Design)

| Principle | Gate | Post-design evidence |
|---|---|---|
| I. Reference Parity | PASS WITH DOCUMENTED DEVIATION | Processor and frame follower goldens use the pinned Python reference. RustySynth SF2 output differs from FluidSynth SF3; compare end-to-end alignment/evaluation metrics and retain source-feature replay tests to isolate the renderer. |
| II. Readability & Structural Parity | PASS | Module paths and names are specified against `matchmaker.io.audio`, `matchmaker.features.audio`, `matchmaker.dp.oltw_arzt`, `matchmaker.dp.oltw_dixon`, and the `score_audio` reference builder. |
| III. Visible Performance | PASS | Stream, processor, rendering, and inference spans are included; benchmark uses the same fixed input files, five runs, and no active subscriber. |
| IV. Debuggability | PASS | Data model and contract require stage-level export and typed error handling. |
| V. Pure Rust, Libraries First | PASS WITH GATES | One decoder, one resampler, one synthesizer, and existing FFT are selected. Add MPL-2.0 to license allowlist only with a documented compatibility rationale; run deny and WASM gates. SoundFont provenance is verified before fixture distribution. |

## Implementation Sequence

1. Verify and pin a redistributable SF2 asset and decoder feature/target matrix; add dependency/license entries and typed audio errors.
2. Implement file decode, mono conversion, 44.1-kHz resampling, hop framing, `chroma`, and `lse`; make processor goldens pass before adding followers.
3. Implement RustySynth score rendering, reference features, and per-reference-frame beat mapping; record renderer parity via evaluation metrics.
4. Port frame-level `arzt` and `dixon` with separate audio defaults; prove replayed-observation parity before end-to-end wiring.
5. Add explicit audio routing to the Rust, CLI, and Python APIs without changing MIDI construction/defaults; add export and focused edge cases.
6. Add audio benchmark fixtures/reporting, update README/UPSTREAM/CI and quickstart, then run all Constitution gates.

Tasks and dependency ordering are generated separately by `/speckit-tasks`.
