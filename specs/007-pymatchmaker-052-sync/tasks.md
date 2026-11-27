# Tasks: Synchronize pymatchmaker 0.5.2

**Input**: Design documents from `specs/007-pymatchmaker-052-sync/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/forward-step.md](contracts/forward-step.md),
[quickstart.md](quickstart.md)

**Tests**: Focused HMM tests, regenerated golden tests, and benchmark validation are
included because parity and performance verification are explicit requirements.

**Organization**: Tasks are grouped by user story. No project scaffolding or new shared
infrastructure is needed.

## Format

- `[P]` marks tasks that can run in parallel because they edit different files and have no
  dependency on unfinished work.
- `[US1]`, `[US2]`, and `[US3]` map to the user stories in `spec.md`.
- Every task names its target file or files.

## Phase 1: Setup

**Purpose**: Project initialization and basic structure.

No setup tasks are needed. The repository, locked reference environment, HMM modules,
golden tests, and benchmark harness already exist.

## Phase 2: Foundational

**Purpose**: Blocking prerequisites shared by all user stories.

No separate foundational work is needed. The user-story tasks use the existing shared
implementation and validation infrastructure.

---

## Phase 3: User Story 1 - Use the 0.5.2 reference (Priority: P1) 🎯 MVP

**Goal**: Pin the released reference and its updated dependency, port the optimized HMM
forward step, and preserve parity for supported methods.

**Independent Test**: The locked environment reports pymatchmaker 0.5.2 and
python-hiddenmarkov `>=0.3.0`; optimized-versus-full forward-step cases pass; regenerated
`pthmm` and `hmm` goldens pass within their existing tolerances.

### Tests for User Story 1

- [X] T001 [P] [US1] Add optimized-versus-full forward-step tests in `src/prob/hidden_markov.rs` for the first observation, concentrated and non-contiguous non-zero forward states, unreachable columns, and an all-zero forward vector; use the existing full banded product as the oracle and confirm the new optimized-path tests fail until that path is implemented.

### Implementation for User Story 1

- [X] T002 [P] [US1] Pin pymatchmaker to the released `v0.5.2` tag in `tools/golden/pyproject.toml`, run `uv lock` in `tools/golden/`, and verify `tools/golden/uv.lock` resolves python-hiddenmarkov to a version satisfying `>=0.3.0`.
- [X] T003 [P] [US1] Update `REFERENCE_TAG`, release-version checks, and reference metadata in `tools/golden/generate.py` and `tools/golden/generate_audio.py` to identify pymatchmaker `v0.5.2`, record the resolved python-hiddenmarkov version, and retain the current `tools/golden/uv.lock` digest.
- [X] T004 [US1] Implement the reachable-row and reachable-column forward calculation in `src/prob/hidden_markov.rs`; preserve initial-distribution behavior, source-row accumulation order, observation weighting, first-maximum selection, and normalization by `max(sum(values), 1e-6)`, including the all-zero case.
- [X] T005 [P] [US1] Update the upstream mapping comments in `src/prob/hmm.rs` and `src/prob/pthmm.rs` to describe their shared use of the optimized hiddenmarkov forward step without changing the public configuration or method behavior.
- [X] T006 [US1] Regenerate MIDI and audio reference outputs into `tests/golden/` and `tests/golden/audio/` with the locked environment; verify generated metadata identifies pymatchmaker `v0.5.2`, the resolved python-hiddenmarkov version, and the current lockfile digest.
- [X] T007 [US1] Run `cargo test --workspace --exclude matchmaker-py` from the repository root and investigate every changed `pthmm` or `hmm` golden value against the release diff and lockfile changes; retain existing documented tolerances.

**Checkpoint**: The locked 0.5.2 reference and both HMM methods pass focused and golden
parity checks.

---

## Phase 4: User Story 2 - Find and reconcile every current reference (Priority: P1)

**Goal**: Ensure active versions, source mappings, behavior descriptions, tests, and
user-facing documentation agree with the 0.5.2 reference while preserving historical facts.

**Independent Test**: Search the repository for pymatchmaker and version references; verify
each current reference agrees with 0.5.2 and every older reference is explicitly historical,
attribution-only, or for another dependency.

### Implementation for User Story 2

- [X] T008 [P] [US2] Audit and update current release references in `README.md`, `UPSTREAM.md`, Rust/Python metadata, generator checks, and tests to 0.5.2 where applicable; preserve accurate historical specifications, golden provenance, asset attribution, and unrelated dependency versions.
- [X] T009 [P] [US2] Update the pinned-version table, hiddenmarkov dependency version, module mapping, and 0.5.2 change classification in `UPSTREAM.md`; map each changed upstream file to an implementation, equivalent existing behavior, or a reasoned non-port.
- [X] T010 [US2] Re-run a repository-wide search for `0.5.1`, `v0.5.1`, `0.5.2`, `v0.5.2`, and `python-hiddenmarkov`; review every match in `README.md`, `UPSTREAM.md`, `tools/`, `tests/`, `specs/`, and `NOTICE` to confirm no contradictory active reference remains and no historical record was rewritten.

**Checkpoint**: Current documentation and version declarations agree with the locked
reference; historical records and attribution remain accurate.

---

## Phase 5: User Story 3 - Carry future upstream changes forward efficiently (Priority: P2)

**Goal**: Make later release syncs start from the complete upstream diff instead of only
the current module map, and make each change traceable to code, validation, or an exclusion.

**Independent Test**: Trace all four files changed by the 0.5.1-to-0.5.2 pymatchmaker
release comparison to their port decision and verify the documented procedure would also
require assessment of a changed file not yet listed in the module map.

### Implementation for User Story 3

- [X] T011 [US3] Revise the release-sync procedure in `UPSTREAM.md` to compare the complete upstream release diff before consulting the module map, assess every changed file, map relevant changes to Rust code and tests or an explicit exclusion, regenerate and review goldens, and reconcile active version surfaces.
- [X] T012 [US3] Validate the change record in `UPSTREAM.md` against all four files in `specs/007-pymatchmaker-052-sync/research.md`; confirm each is classified and that a hypothetical changed upstream file absent from the module map must still be reviewed.

**Checkpoint**: A contributor can trace each current release change and follow a complete
procedure for a later release.

---

## Phase 6: Polish & Cross-Cutting Validation

**Purpose**: Confirm performance, platform compatibility, and the complete feature workflow.

- [X] T013 Run five-run benchmarks on `tools/bench/bundled.toml`, compare against pre-change Rust `HEAD`, and record median `pthmm`, `hmm`, and overall deltas; the reference runner requires excluding `pfkorz` because it raises `KeyError: 'ParticleFilterMidiProcessor'`.
- [X] T014 From the repository root, run `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo build --target wasm32-unknown-unknown`, `cargo deny check`, and `cargo +1.88 check -p matchmaker-rs -p matchmaker-cli --all-targets`; repeat the repository-wide reference audit in section 4 of `specs/007-pymatchmaker-052-sync/quickstart.md` and record every result.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No tasks; existing project structure is reused.
- **Foundational (Phase 2)**: No tasks; there is no shared infrastructure blocker.
- **User Story 1 (Phase 3)**: Start immediately. T001, T002, and T003 can run in parallel; T004 depends on T001; T005 can run in parallel with T004; T006 depends on T002–T005; T007 depends on T006.
- **User Story 2 (Phase 4)**: T008 and T009 can run in parallel after the target version and release diff are established; T010 depends on both.
- **User Story 3 (Phase 5)**: T011 follows T009 because both update `UPSTREAM.md`; T012 depends on T011.
- **Polish (Phase 6)**: T013 and T014 require the implementation, generated data, and documentation to be complete.

### User Story Dependencies

- **User Story 1 (P1)**: No dependency on another story.
- **User Story 2 (P1)**: Its repository-wide final audit depends on the active release updates in User Story 1 and the `UPSTREAM.md` version/mapping update in T009.
- **User Story 3 (P2)**: Can begin after the 0.5.2 release changes are recorded in `research.md`; T011 must follow T009 to avoid concurrent edits to `UPSTREAM.md`.

### Parallel Opportunities

- T001, T002, and T003 touch separate files and can begin together.
- T004 and T005 touch separate Rust files after T001 is ready.
- T008 and T009 touch different documentation files and can run together.
- Benchmarking and final quality gates are intentionally not parallelized to avoid benchmark contention and to make failures easier to attribute.

---

## Parallel Example: User Story 1

```text
Task: T001 Add optimized-versus-full forward-step tests in src/prob/hidden_markov.rs
Task: T002 Pin pymatchmaker v0.5.2 in tools/golden/pyproject.toml and update tools/golden/uv.lock
Task: T003 Update reference tags and metadata in tools/golden/generate.py and tools/golden/generate_audio.py
```

After these independent tasks, implement T004; T005 can proceed alongside it. Regenerate
goldens only after the pin, generators, and optimized implementation are ready.

## Implementation Strategy

### MVP First (User Story 1)

1. Complete T001–T005: lock the target, establish focused tests, and implement the shared
   forward optimization.
2. Complete T006–T007: regenerate the reference data and validate HMM parity.
3. Stop and verify the P1 parity outcome before broad documentation cleanup.

### Incremental Delivery

1. Deliver the 0.5.2 reference and HMM parity (User Story 1).
2. Reconcile all current repository references while preserving history (User Story 2).
3. Improve the repeatable release-sync procedure (User Story 3).
4. Finish benchmark and platform gates.

## Notes

- Every task uses the required checkbox, sequential ID, optional parallel marker, story label
  for story tasks, and concrete file paths.
- Tests are included because the specification requires regenerated reference outputs,
  parity tests, and benchmark evidence.
- The untracked `crates/matchmaker-py/uv.lock` is unrelated to the golden reference update
  and must not be modified as part of these tasks unless the audit establishes otherwise.
