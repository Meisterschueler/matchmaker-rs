# Implementation Plan: MIDI Online Time Warping (arzt, dixon)

**Branch**: `002-midi-oltw` | **Date**: 2026-10-31 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/002-midi-oltw/spec.md`

## Summary

Add the event-level online time warping followers `arzt` (`OnlineTimeWarpingArztEvent`) and
`dixon` (`OnlineTimeWarpingDixonEvent`) to the existing MIDI pipeline. Both consume the
`chord_onset` observations of 001 and align them against the score's onset pianoroll (one binary
pitch vector per unique onset). The work is a new `dp` module pair, one score-side function, new
`MethodConfig` keys, and the same golden/benchmark/CLI/Python/export wiring the 001 methods have.
No new dependency, no new processor. The reference's pass-through rules make several `methods.yaml`
defaults dead; the effective values are fixed in
[research.md R1](research.md#r1-what-the-reference-actually-runs).

## Technical Context

**Language/Version**: unchanged (Rust stable, edition 2024, MSRV from `Cargo.toml`; Python 3.10–3.13 for bindings/tooling)

**Primary Dependencies**: none new (`tracing`, `thiserror` as before). Distance functions are written by hand ([R4](research.md#r4-distance-functions))

**Storage**: Files only; new golden data under `tests/golden/<piece>/{arzt,dixon}/` in the layout of [001 golden-data.md](../001-midi-score-following/contracts/golden-data.md)

**Testing**: `cargo test` (golden per method, `method_config`, `export`, `edge_cases` extended); `pytest` for bindings; generator `tools/golden/generate.py` extended

**Target Platform**: unchanged; core stays wasm-compatible (no clock, no C)

**Project Type**: Library + CLI + Python extension (existing Cargo workspace)

**Performance Goals**: SC-003 — mean inference latency per observation ≥ 5× lower than the reference

**Constraints**: positions equal the reference (SC-001); no panics on user input; per-step spans at `DEBUG`/`TRACE`

**Scale/Scope**: scores up to a few thousand onsets, performances up to ~10⁴ chords; 2 methods, 0 new processors

## Constitution Check

| Principle | Gate | Result |
|---|---|---|
| I Reference Parity | released tag; golden tests per method; tolerances documented | ✅ `generate.py`/`compare.py` get `arzt`, `dixon`; tolerances in [R2](research.md#r2-arzt-event-algorithm-and-dtypes), [R3](research.md#r3-dixon-event-algorithm-and-dtypes) |
| II Structural Parity | module per method key, Python path in module doc, `missing_docs`, doc example | ✅ `dp::oltw_arzt`, `dp::oltw_dixon` mirror `matchmaker/dp/` |
| III Visible Performance | `tracing` spans, benchmark covers methods, regression flag | ✅ `inference` span per call; `METHODS` in `tools/bench/run_reference.py` and the CLI bench extended |
| IV Debuggability | export, typed errors, no panics | ✅ `export` writes the method dir (no `probabilities.npy`: the reference has no state distribution here); typed `InvalidParameter` |
| V Pure Rust | no C, no new dependency | ✅ |
| Scope & Workflow | bindings drop-in for delivered methods | ✅ [contracts/interfaces.md](contracts/interfaces.md) |

No violations. Complexity Tracking: only the documented deviations in research (R4 metric subset, R5 parameter handling).

## Project Structure

### Documentation (this feature)

```text
specs/002-midi-oltw/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/interfaces.md   # deltas to the 001 library/CLI/Python/golden contracts
└── tasks.md                  # Phase 2 (/speckit-tasks)
```

### Source Code (changes to the 001 layout)

```text
src/
├── dp/                        # NEW, mirrors matchmaker/dp/
│   ├── mod.rs                 # Direction (direction.py), distance functions
│   ├── oltw_arzt.rs           # ArztEvent
│   └── oltw_dixon.rs          # DixonEvent
├── score/mod.rs               # + Score::onset_pianoroll(piano_range)
├── matchmaker.rs              # Method::{Arzt, Dixon}, MethodConfig keys, defaults, follower wiring
├── export.rs                  # match arms for the two methods
└── lib.rs                     # pub mod dp

crates/
├── matchmaker-cli/src/{main,bench}.rs   # method lists, --set keys
└── matchmaker-py/src/lib.rs             # default_kwargs, pass-through exclusions, polling default
tests/
├── golden_arzt.rs             # NEW: per piece + midiscore + wrong-note variants
├── golden_dixon.rs            # NEW: same
├── {method_config,export,edge_cases,golden_eval}.rs   # method lists extended
└── golden/<piece>/{arzt,dixon}/                       # generated
tools/
├── golden/{generate,compare}.py   # METHODS, tolerances
└── bench/run_reference.py         # METHODS
```

**Structure Decision**: One module per method key under a new `dp` module, as upstream
(`matchmaker/dp/oltw_arzt.py`, `oltw_dixon.py`). The Cython `dtw_loop.pyx` is **not** ported: the
event classes use their own pure-Python loop (it is for the frame-level audio variant).

## Complexity Tracking

| Deviation | Why | Rejected alternative |
|---|---|---|
| Only `cosine` and `euclidean` as `distance_func` | the reference accepts every SciPy metric; no bundled config uses another | porting all of `cdist` (large, untestable without use) |
| `window_size`/`start_window_size` rejected for `arzt` in the core | dead in the reference (pass-through exclusion); silent ignoring hides mistakes | accepting and ignoring them in the core (surprising API) |
