# Implementation Plan: Follow Upstream pymatchmaker 0.5.1

**Branch**: `005-upstream-0.5-sync` (no new branch requested) | **Date**: 2026-11-21 | **Spec**: [spec.md](spec.md)
**Input**: `specs/005-upstream-0.5-sync/spec.md`
**Status**: Design complete; implementation gates unexecuted. Planning does not authorize source changes or commits.

## Summary

Move the pinned reference from pymatchmaker 0.4.0 to 0.5.1 following the procedure in
`UPSTREAM.md`. Four upstream change groups reach the port: (1) a new MIDI follower
`dannenberg` (new module `src/dp/dannenberg.rs`, a `score_cevts` reference builder on
`Score`, a sixth `Method` variant with defaults, CLI/Python exposure, goldens, benchmark);
(2) the SPARC smoothness metric resamples at 3000 Hz and gains the reference's
sliding-spectrum per-window computation (`src/utils/eval.rs`, regenerated evaluation
goldens); (3) the HMM revisions (inserted-states profile evaluated once, reachable-state
forward restriction) are already realized by the port's banded transition storage and
guarded normalization — record the equivalence in doc comments and let golden regeneration
arbitrate; (4) the multi-fold lazy unfold enumeration matches the port's existing
`iter_unfolded_parts(part, 4)` — record equivalence only. The upstream `wait`-mode feeder
rework of file input stays on the not-ported list. All pins, `REFERENCE_TAG` constants,
version tables, and goldens move together; every changed golden number must trace to an
upstream line.

## Technical Context

**Language/Version**: Rust edition 2024, MSRV 1.88; Python 3.10–3.13 bindings; locked reference scripts under `tools/golden/`.
**Primary Dependencies**: Existing realfft, serde, tracing, midly, PyO3/numpy. No new Rust dependency: the Dannenberg matcher needs only `Vec`, pitch-set bitmasks, and existing stream/processor plumbing.
**Storage**: Checked-in golden arrays and JSON metadata under `tests/golden/`; no database.
**Testing**: Rust unit/integration/doc tests, golden replay tests, CLI tests, Python pytest; reference regeneration via `tools/golden/generate.py` and `generate_audio.py`.
**Target Platform**: Linux, Windows, macOS; default core build on wasm32-unknown-unknown.
**Performance Goals**: No >10% latency regression against the latest master benchmark; SPARC at 3000 Hz must not blow up evaluation time (hence the upstream sliding-spectrum port).
**Constraints**: Locked reference 0.5.1 (git tag in `tools/golden/pyproject.toml` + `uv.lock`); no unsafe; no native default-build dependencies.
**Scale/Scope**: One version sync: ~6 Rust source files touched, one new module, one new golden method directory, regenerated evaluation goldens, documentation updates.

No NEEDS CLARIFICATION remain; [research.md](research.md) records the verified upstream
diff `v0.4.0..v0.5.1` and all decisions.

## Constitution Check

*Gate: evaluate before Phase 0 research and again after Phase 1 design.*

| Principle | Pre-research design check | Post-design check / implementation gate |
|---|---|---|
| I Reference parity | PASS: target is the released tag v0.5.1; lockfile stays checked in; golden regeneration is part of the plan | PASS in design: every change group maps to goldens (new `dannenberg` goldens, regenerated evaluation values); equivalence claims (HMM, multi-fold) are verified by regenerated goldens, not asserted. Numeric evidence remains a delivery gate. |
| II Structural parity | PASS: one new module per new method; `Port of`/`Upstream:` headers; `windowed_spectral_arc_lengths` ported with the reference's structure | PASS: Dannenberg keeps the Python step structure (three predecessors, rating column); pitch sets use a documented `u128` bitmask without changing the algorithm's shape. |
| III Performance | PASS: existing harness; `dannenberg` added to the MIDI benchmark | PASS: the 3000 Hz SPARC adopts the reference's sliding computation instead of naive 2^21-point FFTs per window; five-run medians and >10% flag as usual. |
| IV Debuggability | PASS: typed errors, seeded where stochastic (Dannenberg is deterministic) | PASS: `debug_state` exports the DP rating column and best row; existing export plumbing reused. |
| V Pure Rust, libraries first | PASS: no new dependency | PASS: nothing links C; WASM/MSRV/deny gates unchanged. |

No constitution exceptions requested.

## Project Structure

### Documentation (this feature)

```text
specs/005-upstream-0.5-sync/
  spec.md
  plan.md
  research.md
  data-model.md
  quickstart.md
  contracts/sync-api.md
  checklists/requirements.md
  tasks.md            # /speckit-tasks output, not created here
```

### Source Code (repository root)

```text
tools/golden/pyproject.toml, uv.lock          # pin v0.4.0 -> v0.5.1
tools/golden/generate.py, generate_audio.py   # REFERENCE_TAG, METHODS += dannenberg, pitch-list handling
tools/bench/*.toml, run_reference.py          # dannenberg benchmark row
src/dp/dannenberg.rs                          # NEW: Port of matchmaker.dp.dannenberg.DPDannenberg
src/dp/mod.rs                                 # module export
src/score/mod.rs                              # score_cevts reference builder (compound events)
src/matchmaker.rs                             # Method::Dannenberg, config keys, construction
src/utils/eval.rs                             # SPARC_FS 3000, windowed_spectral_arc_lengths port
src/prob/hidden_markov.rs, hmm.rs, pthmm.rs   # doc-comment equivalence notes only
src/score/unfold.rs                           # doc-comment equivalence note only
crates/matchmaker-py/src/lib.rs               # default_kwargs, available methods
crates/matchmaker-py/tests/                   # dannenberg binding test
crates/matchmaker-cli/tests/                  # method list / follow test extension
tests/golden_dannenberg.rs                    # NEW golden replay test
tests/golden_eval.rs, method_config.rs        # regenerated values, new config keys
tests/golden/<piece>/dannenberg/              # NEW golden data
tests/golden/**/evaluation.json               # regenerated (smoothness values change)
UPSTREAM.md, README.md, tests/data/audio/README.md  # version table, module map, not-ported list
```

**Structure Decision**: Extend owning modules and existing facade/test files, as in
previous features. The compound-event builder lives next to `onset_pianoroll` on `Score`
(both port registry/feature builders over the note array). No new crate, no registry
framework.

## Phase 0: Research Decisions

See [research.md](research.md): verified upstream diff, decision per change group
(R1 pin/regeneration, R2 SPARC, R3 HMM equivalence, R4 Dannenberg design, R5 multi-fold
equivalence, R6 out-of-scope `wait` feeder, R7 docs/benchmark), and the identified numeric
risks (ulp-level shifts from the reference's own restructured forward step; lockfile
moves of transitive dependencies).

## Phase 1: Design and Contracts

[data-model.md](data-model.md) defines the compound-event reference, the Dannenberg
matcher state and invariants, and the smoothness-window semantics.
[contracts/sync-api.md](contracts/sync-api.md) defines the `dannenberg` method surface
(keys, defaults, errors), the SPARC contract at 3000 Hz, the golden-data layout, and the
equivalence-record contract. [quickstart.md](quickstart.md) is the executable validation
guide (regeneration, tests, gates).

## Delivery Order and Commit Boundaries

| Increment | Content | Single implementation commit | Prerequisite |
|---|---|---|---|
| C01 | Pin bump, `REFERENCE_TAG`s, regenerated goldens for existing methods, SPARC 3000 Hz + sliding computation, HMM/multi-fold equivalence notes, doc updates | chore(upstream): follow pymatchmaker 0.5.1 | — |
| C02 | `dannenberg` follower: module, `score_cevts`, Method variant, CLI/Python surface, goldens, benchmark row | feat(midi): add dannenberg follower | C01 |

C01 must land first: the regenerated goldens define the 0.5.1 baseline that C02's new
goldens join. Each commit owns its tests, docs, and gate evidence. Obtain explicit
authorization for commits when implementation is requested.

## Required Quality Gates

Per commit: narrow replay/feature tests first, then `cargo test --workspace --exclude
matchmaker-py` (goldens compare stage by stage), doc tests, Python binding build/tests,
`cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, MSRV 1.88
check, WASM core build, `cargo deny check`. Golden regeneration requires the locked
environment (`cd tools/golden; uv lock; uv sync --locked`) and, for audio goldens, the
pinned SoundFont provisioning. Algorithm-changing commits include goldens and a benchmark
delta. Every changed golden number is reviewed against the upstream diff before commit.

## Complexity Tracking

No constitution violations. Two deliberate non-ports are documentation, not exceptions:
the `wait`-mode feeder thread (existing not-ported scope, principle V keeps threads out of
the core) and upstream's dead compound-event grouping branch, which is ported as the
reference wrote it (computed, unused) to keep side-by-side readability.
