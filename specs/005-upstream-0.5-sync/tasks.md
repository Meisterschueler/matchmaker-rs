# Tasks: Follow Upstream pymatchmaker 0.5.1

**Input**: Design documents from `specs/005-upstream-0.5-sync/`
**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md), [data-model.md](data-model.md), [contracts/sync-api.md](contracts/sync-api.md), [quickstart.md](quickstart.md)

**Tests**: Golden and unit tests are mandatory here (constitution principle I); they are listed per story.

**Organization**: Grouped by user story. Commit boundaries from plan.md: **C01** (`chore(upstream): follow pymatchmaker 0.5.1`) = Phases 1–3 plus US3's C01 documentation; **C02** (`feat(midi): add dannenberg follower`) = Phase 4 plus US3's C02 documentation. Obtain explicit authorization before each commit.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: US1 (existing methods match 0.5.1), US2 (dannenberg), US3 (upstream record)

## Phase 1: Setup (re-pin the reference)

**Purpose**: Move the locked golden environment to the released tag v0.5.1 (research R1).

- [X] T001 Bump the pin to `pymatchmaker @ git+https://github.com/pymatchmaker/matchmaker@v0.5.1` in tools/golden/pyproject.toml, then run `uv lock` and `uv sync --locked` in tools/golden/; review the uv.lock diff and note any transitive dependency moves (constrain a mover in pyproject.toml if it would muddy golden tracing, research R1 risk)
- [X] T002 [P] Update `REFERENCE_TAG = "v0.4.0"` to `"v0.5.1"` in tools/golden/generate.py and tools/golden/generate_audio.py

---

## Phase 2: Foundational (regenerate the 0.5.1 golden baseline)

**Purpose**: The regenerated reference data is the arbiter for both user stories. **Blocks all story phases.**

- [X] T003 Regenerate MIDI goldens: `python -m uv run generate.py --data ../../tests/data --out ../../tests/golden` in tools/golden/; verify every meta.json says v0.5.1
- [X] T004 Regenerate audio goldens with the pinned SF2 provisioned (tests/data/audio/README.md): `python -m uv run generate_audio.py --data ../../tests/data/audio --out ../../tests/golden/audio` in tools/golden/
- [X] T005 Review `git diff tests/golden/` number by number: expected changes are (a) every evaluation.json `smoothness` (SPARC 3000 Hz), (b) possible ulp-level moves in `tests/golden/*/hmm/` arrays (upstream forward-step restructuring), (c) lockfile-traced moves; any untraceable change halts the sync (research: numeric risk register)

**Checkpoint**: 0.5.1 golden baseline exists and is fully traced.

---

## Phase 3: User Story 1 — Existing methods match the 0.5.1 reference (Priority: P1) 🎯 MVP

**Goal**: All delivered followers, processors, and evaluation meet the regenerated 0.5.1 goldens; SPARC follows the revised reference; HMM/multi-fold equivalences are recorded.

**Independent Test**: `cargo test --workspace --exclude matchmaker-py` passes against the regenerated goldens, and the golden diff is fully attributed (quickstart C01).

- [X] T006 [US1] Port the SPARC revision in src/utils/eval.rs per contract §2: `SPARC_FS` 50.0 → 3000.0 and a `windowed_spectral_arc_lengths` port with both routes — per-window FFT (existing `spectral_arc_length`) and the sliding nonzero-sample spectrum with per-window one-sided-gradient edge correction — selected by the reference cost heuristic `len(members) * n_fft * log2(n_fft) < 20 * 2 * (in_span + len(members)) * len(frequencies)`; keep `compute_sparc`'s signature and the existing exclusion semantics (uncovered windows and windows with < 2 samples excluded from the mean); dedupe windows by shared sample range as upstream does
- [X] T007 [US1] Update the unit expectations in src/utils/eval.rs tests (`sparc_matches_reference`) to the 0.5.1 value and add one unit test asserting the sliding and per-window routes agree on a fixture that exercises both (research R2 risk table)
- [X] T008 [P] [US1] Add doc-comment equivalence notes (no behavior change, contract §4): in src/prob/hidden_markov.rs that the `Banded` row storage plus `max(sum, 1e-6)` guard realizes 0.5.1's `PitchIOIHMM.step` `_last_col`/nonzero-span fast path; in src/prob/hmm.rs at `PitchIoiHmm::call` naming that fast path; in src/prob/hmm_models.rs and src/prob/pthmm.rs naming `BernoulliPitchObservationModel(inserted_states=True)` (odd rows are the identical uniform profile, values equal)
- [X] T009 [P] [US1] Add a doc-comment note in src/score/unfold.rs at `iter_unfolded_parts` naming upstream 0.5.1's `_iter_first_unfolded_parts(part, limit=4)` as the now-equivalent reference (research R5)
- [X] T010 [US1] Update regenerated expectations in tests/golden_eval.rs (and any other test with a hardcoded smoothness value); run `cargo test --test golden_eval` until green
- [X] T011 [US1] Run the full suite `cargo test --workspace --exclude matchmaker-py`; every golden test must pass within its documented tolerance — an `hmm` tolerance exceedance is a finding to investigate against the upstream `prob/hmm.py` hunk, not to widen away (research R3 risk)

**Checkpoint**: US1 independently complete — parity with 0.5.1 for everything already delivered (commit C01 after US3's T020/T022, with authorization).

---

## Phase 4: User Story 2 — Follow a MIDI performance with `dannenberg` (Priority: P2)

**Goal**: The new windowed DP matcher is selectable everywhere MIDI methods are, with reference defaults and goldens.

**Independent Test**: `cargo test --test golden_dannenberg` replays the fixed MIDI performances against `tests/golden/<piece>/dannenberg/`; CLI and Python bindings accept the method (quickstart C02).

- [X] T012 [US2] Add the `score_cevts` reference builder as a `Score` method in src/score/mod.rs (port of the `score_cevts` builder in `matchmaker/registry.py`): one `u128` pitch-set bitmask per unique `onset_beat` of the note array plus the ascending unique beats as `Vec<f32>`; invariants per data-model.md (strictly increasing positions, non-empty cevts, raw MIDI pitches 0–127, no piano_range)
- [X] T013 [US2] Create src/dp/dannenberg.rs (`Port of matchmaker.dp.dannenberg.DPDannenberg`, `Upstream: matchmaker/dp/dannenberg.py`) and export it in src/dp/mod.rs: state and transitions per data-model.md — initial column `values[r>0] = -c_m * cumsum(sizes)[..r]`; per note: `is_new_cevt` computed and discarded with a comment naming the upstream dead branch; column shift and three-predecessor DP update on `lo = max(0, best_row-10) .. hi = min(n_states, best_row+window)` (same-row extend `-c_e` if pitch ∉ cevt / `+1` and credit if uncredited; diagonal `prev[r-1] + 1 - c_m*missing(r-1)` with credit set `{pitch}`; vertical skip `values[r-1] - c_m*missing(r-1)`, `>=` wins, credit emptied); strict best-value monotone update; tempo smoothing `smoothing*old + (1-smoothing)*rate` gated by `elapsed > 0.05` s and `0 < advanced < 16` beats; position = anchor + drift capped at `score_positions[current_index+1]`; `is_still_following` while `current_index < n_states-1`; `debug_state` exports the rating column and best row; rows outside `lo..hi` keep stale values (upstream behavior); instrument the step with `tracing::debug_span!("inference", method = "dannenberg")` (constitution III) and give the module a runnable doc example (constitution II); unit tests: wrong first note does not lock onto an arbitrary row, single-cevt score, window clamp at the last event, position cap with no new input
- [X] T014 [US2] Wire `Method::Dannenberg` in src/matchmaker.rs: variant doc, `AVAILABLE` gains `"dannenberg"` (appended, upstream declaration order), `name`/`FromStr`, `config_keys` = {`processor`, `polling_period`, `c_m`, `c_e`, `epsilon`, `epsilon_fraction`, `window`, `smoothing`}, `MethodConfig` fields with defaults per contract §1 (`c_m` 0.5, `c_e` 0.0, `epsilon` 0.090, `epsilon_fraction` 0.25, `window` 50 ≥ 1, `smoothing` 0.5), `for_method` defaults (processor `pitch`, `polling_period = None` event-based), `validate` (processor must be `pitch`), construction with `PitchProcessor::new(false, true)` (`piano_range = false`, `return_pitch_list = true`) and the Matchmaker tempo seeding `beats_per_second`; `piano_range`/`return_pitch_list` keys are rejected for this method with a typed `InvalidParameter`
- [X] T015 [US2] Extend tests/method_config.rs: dannenberg accepts its eight keys with valid values, rejects `piano_range`, `return_pitch_list`, unknown keys, `window = 0`, and non-`pitch` processors
- [X] T016 [P] [US2] Python bindings in crates/matchmaker-py/src/lib.rs: `default_kwargs` entry for dannenberg exactly as `methods.yaml` (processor, c_m, c_e, epsilon, epsilon_fraction, window, smoothing), method listed in docstrings/available methods; add a binding test in crates/matchmaker-py/tests/ (run `dannenberg` on the bundled piece, check positions yield and `default_kwargs`)
- [X] T017 [P] [US2] Golden generation for dannenberg: add `"dannenberg"` to `METHODS` in tools/golden/generate.py and extend the pitch-list feature saving (`is_list`) to `method in ("hmm", "dannenberg")`; regenerate; verify `tests/golden/<piece>/dannenberg/` appears for every piece including the wrong-note variant, with meta.json at v0.5.1 (contract §3)
- [X] T018 [US2] Create tests/golden_dannenberg.rs replaying the goldens stage by stage (score_cevts/score_positions, observations, positions, current_index, rating column where bounded, alignment path, evaluation) within documented tolerances, following the pattern of tests/golden_pthmm.rs
- [X] T019 [US2] CLI: verify `--method dannenberg` follow + evaluate round-trip per quickstart C02 §2 and extend the method-coverage test in crates/matchmaker-cli/tests/ (including the rejected `piano_range` kwarg error path)
- [X] T020 [US2] Benchmark: add a dannenberg row to tools/bench/bundled.toml (and any other MIDI bench config), confirm tools/bench/run_reference.py handles it with the 0.5.1 environment, and produce the benchmark delta per quickstart C02 §4 (report shows dannenberg beside the reference; no >10% regression for existing methods)

**Checkpoint**: US2 independently complete (commit C02 after US3's T021/T022, with authorization).

---

## Phase 5: User Story 3 — Updated upstream record and version surfaces (Priority: P3)

**Goal**: A reader finds every 0.4.0→0.5.1 upstream change mapped to a port change or a not-ported reason (SC-005).

**Independent Test**: Cross-read UPSTREAM.md/README.md against the upstream diff summary in research.md; every hunk is accounted for.

- [X] T021 [P] [US3] Update UPSTREAM.md for C01: version table pymatchmaker 0.4.0 → 0.5.1 (partitura/hiddenmarkov rows per regenerated lockfile), not-ported list names the 0.5.1 `wait`-mode feeder thread of `AudioStream.run_offline`, and prose notes for the recorded equivalences (HMM fast path, inserted-states observation, lazy multi-fold enumeration) and the SPARC revision
- [X] T022 [P] [US3] Update README.md (reference tag link → `v0.5.1`) and tests/data/audio/README.md (generator version mentions 0.4.0 → 0.5.1)
- [X] T023 [P] [US3] Update UPSTREAM.md for C02: module-map rows for `src/dp/dannenberg.rs` (`matchmaker/dp/dannenberg.py`, `DPDannenberg`) and the `score_cevts` builder on `src/score/mod.rs` (`matchmaker/registry.py`); not-ported note that Dannenberg's compound-event grouping result is computed but unused upstream and ported as written

**Checkpoint**: All three stories complete; upstream record audits clean.

---

## Phase 6: Polish & Cross-Cutting Gates

**Purpose**: Quality gates per plan.md before each commit; final validation.

- [X] T024 Run the gate set: `cargo fmt --check`, `cargo clippy --all-targets --all-features -- -D warnings`, `cargo build --target wasm32-unknown-unknown`, `cargo deny check`, MSRV 1.88 check, Python binding build/tests (quickstart gates; required before both C01 and C02)
- [X] T025 Run the full quickstart.md validation end to end (C01 §1–4, C02 §1–5) and record outcomes; confirm SC-001…SC-005 from spec.md hold

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)** → **Foundational (Phase 2)** → everything else. T002 can run parallel to T001; T003/T004 need both; T005 needs T003+T004.
- **US1 (Phase 3)** needs Phase 2 (regenerated expectations). T006 → T007 → T010 → T011; T008/T009 parallel anytime after Phase 2.
- **US2 (Phase 4)** needs Phase 2 and **C01 landed** (plan.md: the 0.5.1 baseline defines what C02's goldens join). T012 → T013 → T014 → T015; T016/T017 parallel after T014; T018 needs T017; T019 needs T014; T020 needs T018.
- **US3 (Phase 5)**: T021/T022 belong to commit C01 (complete before committing US1); T023 belongs to commit C02. All three are parallel-safe (different files/sections).
- **Polish (Phase 6)**: T024 runs before each commit; T025 last.

### Commit mapping (authorization required per plan.md)

| Commit | Tasks |
|---|---|
| C01 `chore(upstream): follow pymatchmaker 0.5.1` | T001–T011, T021, T022, T024 |
| C02 `feat(midi): add dannenberg follower` | T012–T020, T023, T024, T025 |

### Parallel Opportunities

- T002 ∥ T001 (different files).
- T008 ∥ T009 ∥ T006 (doc comments vs eval.rs).
- T016 ∥ T017 after T014; T021 ∥ T022 ∥ T023.

---

## Implementation Strategy

**MVP = Phase 1 + 2 + US1 (C01)**: a correctly re-pinned, fully traced 0.5.1 baseline is valuable alone and is the hard prerequisite for everything else. Then US2 as the single new capability (C02), with US3's documentation folded into each commit. Work test-first within each story (regenerated/failing expectations before the port change); each story is one coherent commit; a blocked step pauses the sequence rather than combining stories.
