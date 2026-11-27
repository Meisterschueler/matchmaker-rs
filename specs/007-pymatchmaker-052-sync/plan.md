# Implementation Plan: Synchronize pymatchmaker 0.5.2

**Branch**: `007-pymatchmaker-052-sync` (planned; not checked out) | **Date**: 2026-11-27 | **Spec**: [spec.md](spec.md)

**Input**: `specs/007-pymatchmaker-052-sync/spec.md`

## Summary

Pin the golden/reference environment to the released pymatchmaker 0.5.2 and its required
`python-hiddenmarkov>=0.3.0`. Port the dependency's reachable-state forward-step optimization
into the existing banded Rust HMM implementation so both `pthmm` and `hmm` use it, while
preserving their normalized forward probabilities, positions, paths, and public
configuration. Regenerate and review reference data, then reconcile every current
pymatchmaker version or behavior reference and revise the release-sync procedure so future
upstream changes are assessed even when their files are not yet in the module map.

The official `v0.5.1...v0.5.2` diff contains four files and one functional change: adoption
of `python-hiddenmarkov` 0.3.0's optimized forward step. The dependency's unrelated Viterbi,
packaging, and documentation changes are not part of pymatchmaker's score-following behavior
and will not be ported.

## Technical Context

**Language/Version**: Rust stable, edition 2024, MSRV 1.88; Python 3.12 for the locked golden environment and Python 3.10–3.13 for the bindings.

**Primary Dependencies**: Existing Rust HMM implementation and banded transition representation; `uv`-locked pymatchmaker 0.5.2 and python-hiddenmarkov 0.3.0+ for reference generation. No new Rust dependency.

**Storage**: `tools/golden/uv.lock` and generated golden arrays/metadata; no database or persistent runtime state.

**Testing**: `cargo test --workspace --exclude matchmaker-py`; targeted HMM/golden tests; locked reference regeneration; existing Rust/Python benchmark harness; formatting, Clippy, MSRV, WASM, and dependency-license checks.

**Target Platform**: Linux, Windows, macOS; core Rust build remains compatible with `wasm32-unknown-unknown`.

**Project Type**: Rust library, CLI, and Python bindings, with Python reference-generation and benchmark tools.

**Performance Goals**: Demonstrate the optimized forward step on long HMM fixtures and avoid a greater than 10% median latency regression on existing benchmarked methods; use at least five runs for benchmark comparisons.

**Constraints**: Use only released pymatchmaker 0.5.2; lock the Python reference environment; preserve output parity and existing interfaces; do not add unsafe code, new Rust dependencies, or native dependencies to the default build. Keep historical release records and asset attribution accurate.

**Scale/Scope**: One upstream release sync. The expected algorithm change is localized to the shared HMM forward step; the reference pin, golden provenance, current release surfaces, and contributor sync guidance are updated consistently.

## Constitution Check

*Gate: evaluate before Phase 0 research and again after Phase 1 design.*

| Principle | Pre-research check | Post-design check |
|---|---|---|
| I. Reference parity | PASS: target is the released `v0.5.2`; the Python environment remains locked and goldens will be regenerated. | PASS: the pinned version and dependency are verified; golden comparisons cover both HMM methods and investigate every changed value. |
| II. Readability and structural parity | PASS: reuse the shared `hidden_markov` implementation and existing method modules rather than introducing a parallel HMM path. | PASS: document the upstream forward-step source and keep the optimized operation recognizable as a banded forward step. No public API change is required. |
| III. Visible performance | PASS: an algorithm optimization is benchmarked under the existing five-run median and regression policy. | PASS: compare existing and optimized HMM timings, report the benchmark delta, and require no greater than 10% regression on existing methods. |
| IV. Debuggability | PASS: preserve the existing forward-variable and state diagnostics; invalid-input behavior is unchanged. | PASS: existing exports and typed error handling remain in place; focused tests inspect probability vectors and state/path parity. |
| V. Pure Rust, libraries first | PASS: use the existing banded matrix representation; do not add a Rust dependency or native code. | PASS: default build and WASM/MSRV/license gates remain unchanged. The Python-only reference dependency is updated in its locked environment. |

No constitution violations or exceptions are requested.

## Project Structure

### Documentation (this feature)

```text
specs/007-pymatchmaker-052-sync/
├── spec.md
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── forward-step.md
├── checklists/
│   └── requirements.md
└── tasks.md                 # Created by /speckit-tasks, not by this plan
```

### Source Code (repository root)

```text
tools/golden/pyproject.toml, uv.lock
tools/golden/generate.py, generate_audio.py
src/prob/hidden_markov.rs
src/prob/pthmm.rs, src/prob/hmm.rs
tests/ and tests/golden/
tools/bench/
README.md, UPSTREAM.md
```

**Structure Decision**: Keep the existing single-workspace structure. Implement the shared
optimization in `src/prob/hidden_markov.rs`, keep `pthmm` and `hmm` as thin callers, and
reuse existing golden, benchmark, and documentation locations. Do not add a crate or public
method.

## Phase 0: Research

[research.md](research.md) records the release diff, the `python-hiddenmarkov` optimized
forward-step contract, its equivalence and performance implications for the banded Rust
implementation, and the repository surfaces that are current versus historical. The
comparison found no unresolved design choices.

## Phase 1: Design and Contracts

[data-model.md](data-model.md) defines the pinned-reference record and the HMM forward-step
state and invariants. [contracts/forward-step.md](contracts/forward-step.md) specifies the
internal probability contract and the unchanged external method behavior.
[quickstart.md](quickstart.md) gives the locked regeneration, parity, audit, benchmark, and
quality-gate commands.

## Delivery Sequence

| Increment | Content | Completion evidence |
|---|---|---|
| 1. Re-pin and record provenance | Update the golden environment to pymatchmaker 0.5.2 and python-hiddenmarkov 0.3.0+, update both generator tags and reference metadata, and record the full upstream change classification. | Locked environment reports the target versions; generated metadata identifies them. |
| 2. Optimize shared HMM forward step | Restrict transition work to rows with non-zero forward probability and the columns reachable from those rows; retain initial-step and zero-normalization behavior. | Focused equivalence/edge tests pass; `pthmm` and `hmm` golden results stay within existing tolerances. |
| 3. Reconcile current references and validate | Update active version surfaces and sync guidance; audit all pymatchmaker mentions; regenerate goldens and run test, benchmark, and platform gates. | No contradictory current reference; every upstream change is mapped; parity and benchmark evidence is recorded. |

## Complexity Tracking

No constitution violations. No extra abstraction, Rust dependency, public option, or new
project layer is needed.
