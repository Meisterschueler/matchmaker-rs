# Implementation Plan: Online Transformer Matcher (OTM)

**Branch**: `006-online-transformer-matcher` | **Date**: 2026-11-24 | **Spec**: [spec.md](spec.md)

**Input**: `specs/006-online-transformer-matcher/spec.md`

**Status**: Design complete; implementation gates unexecuted. Planning does not authorize
source changes or commits.

## Summary

Port the `OTM` method from the pinned pymatchmaker `v0.5.1` release and its locked
Parangonar `3.4.1` dependency. The deliverable is the event-based `OnlineTransformerMatcher`
behavior, including score-note preprocessing, tokenization, fixed transformer inference,
tempo-aware matching heuristics, and the beat-position interface. The design reuses the
existing MIDI, library, CLI, Python, export, golden, and benchmark surfaces. It adds no
runtime Python/PyTorch dependency and does not add `OPTM`, `SLT_OLTW`, `SL_OLTW`, or offline
note alignment.

The model has a fixed 26-position input and 64-wide, six-layer transformer. Implement its
inference operations in Rust without an ML framework, and convert the released checkpoint
offline into a compact, versioned asset. Before distributing that asset, verify its
redistribution terms and attribution; the upstream package is Apache-2.0, but the checkpoint
has no separate license or disclosed training-data provenance.

## Technical Context

**Language/Version**: Rust edition 2024, MSRV 1.88; Python 3.10–3.13 for the binding and
locked reference tooling.
**Primary Dependencies**: No new runtime crate. Reuse `libm`, `serde`, `serde_json`,
`tracing`, and existing MIDI/score infrastructure. Reference generation uses the existing
locked `pymatchmaker==0.5.1`, `parangonar==3.4.1`, and PyTorch environment.
**Storage**: Compact model weights and metadata under `assets/otm/`; reference and Rust
goldens under `tests/golden/`; no database.
**Testing**: Rust unit, doc, golden, CLI, Python-binding, WASM and license checks; locked
reference generation through `tools/golden/`; Rust/Python benchmark comparison.
**Target Platform**: Linux, Windows, macOS; core remains buildable for
`wasm32-unknown-unknown`. OTM inference is CPU-only.
**Project Type**: Existing Rust library, CLI binary, and PyO3 package.
**Performance Goals**: Median real-time factor below `1.0` on the bundled OTM benchmark;
report per-event inference latency and real-time factor for both implementations over at
least five runs.
**Constraints**: Released reference only; event-based MIDI (one note-on per observation);
no unsafe code, no C/C++ runtime, no Python/PyTorch at OTM runtime, no panic on invalid
input; checkpoint redistribution and attribution must be verified before release.
**Scale/Scope**: One MIDI method, one compact pretrained model artifact, one note event per
call, 26 token positions, model width 64, 8 attention heads, and 6 transformer layers.

No technical clarification markers remain. The model-asset licensing/provenance item is a
release gate recorded in [research.md R5](research.md#r5-model-asset-licensing-and-provenance),
not permission to redistribute an uncleared artifact.

## Constitution Check

*Gate: evaluated before Phase 0 research and re-evaluated after Phase 1 design.*

| Principle | Pre-research check | Post-design check / implementation gate |
|---|---|---|
| I Reference parity | PASS: released pymatchmaker `v0.5.1` is already pinned; use the checked-in lockfile and generate OTM goldens from it. | PASS: score reference, tokens, model decisions, tempo state, positions and alignment path all have golden comparisons. |
| II Structural parity | PASS: preserve one OTM method module and the named upstream paths. | PASS: `dp::otm`, `dp::otm_model`, and `features::parangonar` map to the reference matcher, model and processor; grace-note state remains in the score model. |
| III Visible performance | PASS: add OTM to existing inference tracing and benchmark surfaces. | PASS: record per-event inference latency and real-time factor; retain five-run median reporting. The `<1.0` real-time-factor target is a delivery gate. |
| IV Debuggability | PASS: use typed errors and the existing export/comparison format. | PASS: export token inputs, logits when evaluated, tempo diagnostics, positions and alignment path; missing/invalid model assets fail explicitly. |
| V Pure Rust, libraries first | PASS with research required: assess maintained inference crates, license, C/C++ linkage and WASM compatibility before choosing. | PASS with a justified narrow deviation: fixed OTM operations use Rust code rather than a heavyweight inference framework; no new runtime dependency or native link is added. Converted model redistribution remains blocked until its terms are confirmed. |

No principle exception is requested. The custom fixed-graph inference decision is justified
under Complexity Tracking below.

## Project Structure

### Documentation (this feature)

```text
specs/006-online-transformer-matcher/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/otm-api.md
├── checklists/requirements.md
└── tasks.md                 # /speckit-tasks output; not created here
```

### Source Code (repository root)

```text
assets/otm/
├── model.bin                       # converted inference-only weights; gated by rights review
└── model.json                      # format/version, source, checksum and tensor layout

src/
├── dp/
│   ├── mod.rs                      # export OTM module
│   ├── otm.rs                      # OnlineTransformerMatcher state and online heuristics
│   └── otm_model.rs                # tokenization and fixed transformer inference
├── features/
│   ├── mod.rs                      # processor registry and Observation wiring
│   └── parangonar.rs               # one event-based MIDI note per observation
├── score/
│   ├── part.rs                     # retain explicit grace-note identity
│   ├── musicxml.rs                 # propagate GraceNote kind
│   └── mod.rs                      # OTM note-array-with-grace reference builder
├── base.rs                         # OTM step diagnostics
├── matchmaker.rs                   # Method, defaults, config, follower wiring
└── export.rs                       # OTM tokens/logits/tempo diagnostics

crates/
├── matchmaker-cli/src/             # expose OTM and benchmark method
└── matchmaker-py/src/              # expose OTM, defaults and parameter validation

tools/
├── golden/
│   ├── generate.py                 # OTM reference outputs
│   └── export_otm_weights.py       # one-time, reproducible checkpoint conversion
└── bench/
    ├── run_reference.py            # add OTM
    └── bundled.toml                # use existing bundled pieces

tests/
├── golden_otm.rs                   # stage-by-stage parity tests
├── method_config.rs                # OTM defaults and invalid-config cases
├── export.rs                       # diagnostics and model asset errors
└── golden/<piece>/OTM/             # generated OTM reference outputs
```

**Structure Decision**: Keep method state and its fixed model under the existing `dp`
module, with a separate processor module matching the method's `parangonar` processor key.
Add `is_grace` to the score model and a dedicated reference builder rather than trying to
derive grace status from note duration. Extend the existing method/config and export paths;
do not add a crate or general model registry.

## Phase 0: Research Decisions

[research.md](research.md) records the release and locked dependencies (R1), adapter behavior
and OTM online contract (R2), fixed model and compact weight conversion (R3), inference
runtime options (R4), asset redistribution gate (R5), score-model delta (R6), and parity,
export and benchmark plan (R7).

## Phase 1: Design and Contracts

- [data-model.md](data-model.md) defines the grace-aware score reference, event observation,
  token window, model, tempo state, and alignment path.
- [contracts/otm-api.md](contracts/otm-api.md) specifies the `OTM` selector, MIDI-only
  event input, defaults/configuration, result shape, errors, and diagnostic export.
- [quickstart.md](quickstart.md) provides runnable parity, CLI, Python, benchmark, and
  platform-validation scenarios.

## Delivery Order

1. **Asset and reference preflight**: confirm checkpoint distribution terms; record source,
   SHA-256 and model format; add deterministic conversion and reference fixtures. Do not
   bundle the weights until rights are cleared.
2. **Score and model foundation**: propagate `is_grace`, implement OTM reference building,
   the one-note processor, tokenizer, and transformer operations; validate each stage against
   Python outputs.
3. **Online follower and surfaces**: port the OTM matching/tempo state, add method/config
   handling, and connect library, CLI, Python, tracing and export paths.
4. **Parity, performance and docs**: run fixed golden fixtures, test repeatability and
   invalid assets/config, add benchmark coverage, update README and `UPSTREAM.md`, then pass
   repository quality gates.

## Required Quality Gates

- Generate OTM goldens from the locked reference with CPU inference and record the pinned
  Parangonar version and checkpoint hash in metadata. The reference generator and benchmark
  must disable CUDA before importing the Python matcher so results and timings use CPU.
- Compare score-note reference, processor observations, token windows, selected encoder
  layer outputs, final logits, tempo diagnostics, yielded positions and alignment path.
  Keep layer-output fixtures small and test-only; failures name the method, piece, stage
  and first differing event.
- Run existing and OTM-specific Rust, CLI and Python tests; run the default core build for
  `wasm32-unknown-unknown`, `cargo deny check`, formatting and Clippy.
- Benchmark Rust and Python on identical inputs over at least five runs; report median
  per-event inference latency and RTF. OTM's bundled benchmark RTF must be below `1.0`;
  existing-method outputs and performance must not regress.
- Verify model checksum, source attribution and redistribution terms before any release
  includes the converted weights. If terms remain unclear, stop before distribution and
  obtain maintainer/legal guidance rather than silently bundling the artifact.

## Complexity Tracking

| Deviation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| Implement the fixed transformer forward pass rather than add a general inference crate (Principle V: libraries first) | OTM uses one small fixed graph; custom operations avoid a large runtime dependency and preserve the default no-C/C++ and WASM build constraints. | Candle/Burn add substantial model/runtime dependencies; tract adds a larger ONNX runtime and an MSRV above 1.88; ONNX Runtime links C/C++. This is narrowly scoped and must pass layer-level parity and the real-time benchmark gate. |
