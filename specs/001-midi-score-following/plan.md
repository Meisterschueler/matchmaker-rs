# Implementation Plan: MIDI Score Following (pthmm, hmm, outerhmm)

**Branch**: `001-midi-score-following` | **Date**: 2026-10-28 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-midi-score-following/spec.md`

## Summary

Port pymatchmaker's MIDI score following to Rust: score loading (partitura subset: MusicXML,
MIDI scores, repeat unfolding, part merging, beat map), performance MIDI streaming (event and
windowed modes), the `pitch` and `chord_onset` processors, the followers `pthmm` (`PitchHMM`),
`hmm` (`PitchIOIHMM` + Kalman tempo model) and `outerhmm` (`OuterProductHMM`), and the reference
evaluation. Delivered as a wasm-compatible core library, a CLI and PyO3 bindings, verified by
golden tests generated from the locked reference, and compared against Python by a benchmark
harness published from CI. Research established the exact reference wiring
([research.md R1](research.md#r1-what-the-reference-actually-runs-on-the-midi-path)) and that
the HMM transition rows have a bounded non-zero band, which allows an exact `O(N·752)`
forward step instead of the reference's dense `O(N²)` one (R2).

## Technical Context

**Language/Version**: Rust stable, edition 2024, MSRV 1.88 (`rust-version`; required by zip 9); Python 3.10–3.13 for bindings and tooling

**Primary Dependencies**: core: `midly`, `roxmltree`, `zip` (deflate), `realfft`, `thiserror`, `tracing`, `serde`/`serde_json`, own `npy` module; CLI: `clap`, `anyhow`, `tracing-subscriber`; bindings: `pyo3`, `numpy` (maturin). Justification per crate: [research.md R12](research.md#r12-dependencies-v)

**Storage**: Files only: inputs (MusicXML/`.mxl`/MIDI/`.match`/TSV), `.npy` + JSON outputs; golden data in `tests/golden/`; datasets in a CI cache

**Testing**: `cargo test` (unit, golden, CLI integration); `pytest` smoke tests for the bindings; golden generator under `tools/golden` with `uv.lock`

**Target Platform**: Linux, Windows, macOS; core also `wasm32-unknown-unknown`; benchmark reference platform CI `ubuntu-latest`

**Project Type**: Library + CLI + Python extension (Cargo workspace)

**Performance Goals**: SC-004 — mean inference latency per observation ≥ 5× lower than the reference on the same machine; real-time factor ≪ 1 on all dataset pieces

**Constraints**: Parity of positions (SC-001/SC-002); core reads no clock and links no C (wasm); no panics on user input; per-frame spans at `DEBUG`/`TRACE`; benchmarks without subscriber

**Scale/Scope**: Scores up to a few thousand unique onsets (≈ 2·n states with insertions), performances up to ~10⁴ notes; 3 methods, 2 processors; full (n)ASAP, Batik, Vienna4x22 in CI

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Gate | Pre-research | Post-design |
|---|---|---|---|
| I Reference Parity | Released tag only; locked golden env; golden tests per processor/method; documented tolerances; eval follows code | ✅ planned | ✅ `tools/golden` + `uv.lock`, tolerances in [golden-data.md](contracts/golden-data.md), eval port R8; deviations documented (R5 tie order in non-default event mode; R8 `.match` fallback via converter) |
| II Readability & Structural Parity | Stream → Processor → OnlineAlignment; module per method/processor key; Python path in module docs; `missing_docs`; doc examples; no `unsafe` | ✅ planned | ✅ module map in [library-api.md](contracts/library-api.md): `prob::{pthmm, hmm, outerhmm}`, `features::{pitch, chord_onset}`, `base`, `io::stream` |
| III Visible Performance | `tracing` spans per stage; Rust-vs-Python harness; CI full benchmark + published report; regression flag > 10 %, median of ≥ 5 runs | ✅ planned | ✅ R9, [benchmark-report.md](contracts/benchmark-report.md); timing in harness only |
| IV Debuggability | `.npy`/CSV export via library + CLI flag; `RUST_LOG`; typed errors; no panics | ✅ planned | ✅ `export` module = golden layout, `--export`, `Error` enum in [data-model.md](data-model.md) |
| V Pure Rust, Libraries First | No C in default build; wasm32; crates evaluated; licenses checked | ✅ planned | ✅ R11/R12; PyO3 only in a non-default crate; **action**: add `Unlicense` (midly) to the `deny.toml` allow list |
| Scope & Workflow | edition 2024 + MSRV; lib/CLI/bindings; partitura-equal score features; CI gates | ✅ | ✅ CI adds workspace tests (excluding bindings), a bindings job and a benchmark job |

No violations; Complexity Tracking lists the justified design deviations only.

## Project Structure

### Documentation (this feature)

```text
specs/001-midi-score-following/
├── plan.md              # This file
├── research.md          # Phase 0
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1
├── contracts/           # Phase 1: library-api, cli, python-api, golden-data, benchmark-report
└── tasks.md             # Phase 2 (/speckit-tasks)
```

### Source Code (repository root)

```text
Cargo.toml                      # workspace + core package `matchmaker-rs` (lib `matchmaker_rs`)
src/
├── lib.rs                      # re-exports, crate docs
├── error.rs                    # Error (thiserror)
├── matchmaker.rs               # Matchmaker facade, Method/ProcessorKind/MethodConfig, per-method defaults
├── base.rs                     # OnlineAlignment trait, AlignmentPath, run loop
├── io/
│   ├── mod.rs
│   └── stream.rs               # MidiMessage, Frame, StreamMode, framing, push mode
├── features/
│   ├── mod.rs                  # Processor trait, Observation, Features
│   ├── pitch.rs
│   └── chord_onset.rs
├── prob/
│   ├── mod.rs
│   ├── hidden_markov.rs        # forward_algorithm_step, BandedTransitionMatrix
│   ├── hmm_models.rs           # gumbel_l pdf, stable_transition_matrix, init_dist, pitch profiles, Bernoulli, ACC IOI
│   ├── pthmm.rs
│   ├── hmm.rs                  # PitchIOIHMM
│   └── outerhmm.rs
├── utils/
│   ├── mod.rs
│   ├── tempo_models.rs         # KalmanTempoModel
│   └── eval.rs                 # transfer_positions, metrics, SPARC, GroundTruth (TSV, .match)
├── score/
│   ├── mod.rs                  # Score, ScoreNote, unique_onsets, chords
│   ├── musicxml.rs             # partitura.io.importmusicxml subset (on `roxmltree`)
│   ├── midi.rs                 # partitura.io.importmidi.load_score_midi
│   ├── timeline.rs             # beat_map / quarter_map (_time_interpolator)
│   ├── unfold.rs               # get_paths / unfold_part_maximal
│   ├── merge.rs                # merge_parts
│   └── tempo.rs                # get_tempo_from_score, to_quarter_tempo, text markings
├── performance.rs              # load_performance_midi, midi_messages_from_performance
└── export.rs                   # golden-layout writer (.npy + meta.json)

crates/
├── matchmaker-cli/             # bin `matchmaker`: follow, evaluate, bench
│   └── src/main.rs
└── matchmaker-py/              # PyO3 module `matchmaker_rs` (maturin, abi3-py310), not a default member
    ├── pyproject.toml
    ├── src/lib.rs
    └── tests/test_drop_in.py

tests/
├── data/                       # bundled reference assets (Apache-2.0, see NOTICE)
├── golden/                     # generated by tools/golden (layout: contracts/golden-data.md)
├── common/mod.rs               # golden loader + first-difference assertions (SC-007)
├── golden_score.rs
├── golden_stream_processors.rs
├── golden_pthmm.rs
├── golden_hmm.rs
├── golden_outerhmm.rs
├── golden_eval.rs
└── edge_cases.rs               # empty performance, corrupt files, wrong notes, out-of-score notes

tools/
├── golden/                     # pyproject.toml, uv.lock, generate.py, compare.py, match_to_tsv.py
├── bench/                      # run_reference.py, report.py, piece manifests (bundled.toml, …)
└── datasets/                   # datasets.toml (URL, commit, sha256), fetch.py

.github/workflows/
├── ci.yml                      # + workspace tests (excl. bindings), bindings job (maturin + pytest)
└── benchmark.yml               # datasets fetch (cached), Rust + Python runs, report → GitHub Pages
```

**Structure Decision**: Cargo workspace with the core library as the root package (keeps the
existing package name and the wasm build), a CLI crate and a non-default PyO3 crate (R11).
Module names follow the Python method/processor keys (II); partitura ports live under `score/`
and `performance.rs`, each naming its partitura source.

## Implementation Order (input for /speckit-tasks)

1. Workspace setup, `Error`, `deny.toml` (`Unlicense`), CI updates; `tools/golden` env + generator; copy assets.
2. Score loading (`score/*`) with golden tests — every result depends on `onset_beat`.
3. Performance loading, stream framing, `pitch`/`chord_onset` with golden tests.
4. `prob::hidden_markov` + `hmm_models` + `pthmm` (US1), `Matchmaker` facade, CLI `follow`.
5. Evaluation + CLI `evaluate`; export + `compare.py` (US2, US6).
6. Benchmark harness (CLI `bench`, `run_reference.py`, `report.py`, `benchmark.yml`) (US2).
7. `hmm` + Kalman (US3); `outerhmm` (US4).
8. PyO3 bindings (US5); README (build status, version, report link, quickstart).

## Risks

| Risk | Mitigation |
|---|---|
| A score feature partitura handles is missed by the port | Fixture coverage check in `generate.py` (T008); dataset-wide score golden check in the benchmark job |
| Argmax flips from summation order | First-difference reports; NumPy-compatible pairwise sum where needed (R4) |
| Reference install needs a C compiler (Cython extension) | Lockfile pins wheels where available; CI installs build tools for the reference job only |
| CI time of the full Python benchmark | Cache deterministic reference accuracy by key; shard per dataset (R9, R13) |
| Unfolding edge cases (segno/coda, nested voltas) | Port `get_paths` 1:1; dataset-wide score golden check in the benchmark job |

## Complexity Tracking

> No constitution violations. Design deviations from the reference's *structure* (not its
> results), recorded for reviewers:

| Deviation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Banded transition storage instead of dense `N×N` (R2) | Memory/latency for large scores (SC-004); identical results because skipped entries are exact zeros | Dense matrix: up to hundreds of MB per piece, memory-bound |
| Stable message sort instead of NumPy's unstable `argsort` (R5) | Unstable order is CPU-dependent and not reproducible | Emulating NumPy quicksort: cannot match across CPUs; no effect in default modes |
| `.match` ground truth: no match-score rebuild; the fallback join takes the snote's `OnsetInBeats`, other versions are converted by a reference script (R8) | Rebuilding a score from match `snote` lines is a large partitura port | Full port now: delays P1 stories; parity verified on Batik, Vienna 4x22 and (n)ASAP (T049) |
| Timing in the harness instead of the library (R9) | `Instant` panics on wasm; the reference has no MIDI latency stats | In-library timing: breaks V and distorts measurements (III) |
