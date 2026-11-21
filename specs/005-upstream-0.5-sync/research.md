# Research: Follow Upstream pymatchmaker 0.5.1

**Date**: 2026-11-21 · **Upstream diff**: `git diff v0.4.0 v0.5.1 -- matchmaker` in
https://github.com/pymatchmaker/matchmaker (12 files, 633 insertions, 250 deletions)

The diff was inspected file by file against the module map in `UPSTREAM.md`. Pure
reformatting (black-style rewrapping in `particle_filter_korzeniowski.py`, docstring
layout in `skf.py`, citation/docstring additions in `oltw_arzt.py`, a citation fix
ISMIR 2014 → ICMC 2013 that the Rust `pfkorz.rs` header does not repeat) requires no port
change and is not discussed further. Upstream dependency floors (`partitura>=1.9.0`,
`python-hiddenmarkov>=0.1.4`) are unchanged.

## R1: Pin and regeneration procedure

- **Decision**: Pin `pymatchmaker @ git+https://github.com/pymatchmaker/matchmaker@v0.5.1`
  in `tools/golden/pyproject.toml`; `cd tools/golden; uv lock; uv sync --locked`. Update
  `REFERENCE_TAG = "v0.4.0"` → `"v0.5.1"` in `tools/golden/generate.py` and
  `tools/golden/generate_audio.py` (both embed it in every `meta.json`). Regenerate all
  MIDI and audio goldens; review the diff of `tests/golden/` number by number.
- **Rationale**: Exactly the procedure `UPSTREAM.md` prescribes; the tag is a released
  version as principle I requires (0.5.1 ⊃ 0.5.0; 0.5.1 additionally revised
  `io/audio.py`, `prob/hmm.py`, `prob/skf.py` over 0.5.0).
- **Alternatives considered**: Pinning 0.5.0 (rejected: superseded within the same
  series); PyPI sdist instead of the git tag (rejected: the repo already pins the git tag
  and the lockfile records the commit hash).
- **Risk**: `uv lock` may move transitive dependencies (numpy, librosa, partitura point
  releases) within their floors; any resulting golden shifts must be traced to the
  lockfile diff, not silently accepted. If an unrelated transitive move muddies tracing,
  constrain that package in `tools/golden/pyproject.toml` to its previous version.

## R2: SPARC at 3000 Hz and the sliding-spectrum computation

Upstream (`matchmaker/utils/eval.py`): `compute_sparc` default `fs` moves from 50.0 to
`SPARC_SAMPLE_RATE = 3000.0` ("every audio frame rate matchmaker uses (25, 30, 50 and
62.5 Hz) divides it… closely spaced MIDI updates stay apart"). A new
`windowed_spectral_arc_lengths(position, fs, edges, …)` computes per-window values; its
docstring guarantees "the same values as slicing every window and calling
`spectral_arc_length`, without its cost at high rates". It keeps a running complex
spectrum over the nonzero speed samples (held positions make speed zero except at
reports), evaluated only at bins up to `fc`, corrects the two one-sided
`np.gradient` edge samples per window, and falls back to plain per-window FFTs when a
cost heuristic (`len(members) * n_fft * log2(n_fft) < 20 * 2 * (in_span + len(members)) *
len(frequencies)`) says those are cheaper. Windows shorter than two samples report
"insufficient_samples"; "uncovered", "stationary" and "undefined_band" keep their
meanings.

- **Decision**: Port both pieces to `src/utils/eval.rs`: `SPARC_FS` 50.0 → 3000.0 and a
  `windowed_spectral_arc_lengths` port including the reference's cost heuristic and both
  routes (per-window FFT via the existing `spectral_arc_length`, and the sliding
  nonzero-sample spectrum with per-window edge correction). Keep the existing public
  `compute_sparc` signature (`score_beat`, `perf_sec`, `onset_times`, `window_sec`);
  window dedup by sample range mirrors upstream's `keys`/`unique` mapping. The existing
  exclusion semantics (uncovered windows filtered, `< 2` samples skipped) already match
  0.5.1 and stay.
- **Rationale**: At 3000 Hz a 30 s window resamples to 90 000 positions → `n_fft = 2^21`
  per window; dozens of windows per evaluation would make the plain route a measurable
  regression (principle III), which is exactly why upstream wrote the sliding route. The
  cost heuristic must come along because upstream's values are defined by whichever route
  runs — the docstring asserts both routes agree, so the heuristic is pure performance,
  but porting it keeps the code side by side readable (principle II, `UPSTREAM.md`
  convention).
- **Alternatives considered**: Port only the 3000 Hz rate with per-window FFTs (rejected:
  2^21-point FFTs per window; principle III); port only the sliding route (viable since
  both routes agree, but diverges structurally from the reference for no gain — the
  per-window route already exists in `eval.rs`).
- **Consequences**: Every `evaluation.json` golden (`smoothness`) and the unit expectation
  in `eval.rs::tests::sparc_matches_reference` change; regenerate and trace. The Rust port
  returns only the mean (no diagnostics list), unchanged from today.

## R3: HMM revisions are already realized — record, don't restructure

Upstream (`matchmaker/prob/hmm.py`):

1. `BernoulliPitchObservationModel(pitch_profiles, inserted_states=False)`: with
   `inserted_states=True`, odd (inserted) states share one uniform profile, so the model
   evaluates it once and broadcasts (`obs_prob[1::2] = …[0]`). Documented upstream as
   evaluating the same profiles; values identical because all odd rows *are* the same
   profile (see `compute_discrete_pitch_profiles`).
2. `PitchIOIHMM.step`: precomputes the cumulative rightmost nonzero transition column
   (`_last_col`), then computes the prior as `fv[lo:hi] @ T[lo:hi, :m]` over the nonzero
   span of the forward variable, with `fv /= max(fv.sum(), 1e-6)`; first step and
   non-`ConstantTransitionModel` fall back to `forward_algorithm_step`.

Port state: `src/prob/hidden_markov.rs` already stores transition rows as bands of
non-zero entries ("research R2: skipping exact zeros does not change any product or sum")
and already normalizes with `np_sum(&fv).max(1e-6)` — the guard upstream's fast path
copies from python-hiddenmarkov. `bernoulli_pitch_probabilities` evaluates each profile
row; odd rows are the identical uniform profile, so values match item 1 trivially.

- **Decision**: No behavioral code change. Add doc-comment notes: in
  `hidden_markov.rs` that the band storage realizes 0.5.1's `_last_col`/nonzero-span
  restriction, in `hmm.rs` (`PitchIoiHmm::call`) naming the upstream fast path, and in
  `hmm_models.rs`/`pthmm.rs` naming the `inserted_states` flag of the observation model.
  Golden regeneration arbitrates (FR-007); the spec's "adopt the revisions" is satisfied
  the same way FR-008 allows: the port already embodies them, which is recorded.
- **Rationale**: The upstream commits are performance work; the port's banded
  representation predates and subsumes them. Rewriting working, tested code to mimic the
  new Python control flow adds risk without changing a single output (ponytail/deletion
  beats addition; principle II is satisfied by the mapping note).
- **Alternatives considered**: Mirroring `_last_col` and the span-restricted product
  literally (rejected: duplicates what `Banded::transpose_dot` does, no output change);
  adding an `inserted_states` fast path to `bernoulli_pitch_probabilities` (rejected: the
  Rust loop is already O(states × 88) with no numpy row-allocation overhead to save).
- **Risk**: The reference's own restructuring (BLAS slice product, restricted sum) can
  shift regenerated `hmm` goldens at the last-ulp level relative to 0.4.0. The golden
  tests' documented tolerances absorb this; if a regenerated value moves, the trace is
  this upstream hunk. If a tolerance is exceeded, that is a real finding to investigate,
  not to paper over.

## R4: Dannenberg follower design

Upstream adds `matchmaker/dp/dannenberg.py` (`DPDannenberg`, windowed DP matcher after
Bloch & Dannenberg 1985), a `methods.yaml` entry, and the `score_cevts` reference builder
(`registry.py`): one `set(pitches)` per unique `onset_beat` of the score note array.
Method entry: `reference: score_cevts`, `event_based: true` (one MIDI message per frame,
no polling period), `processor_args: {return_pitch_list: true, piano_range: false}`,
`config_passthrough: {exclude: [piano_range, return_pitch_list]}`, `default_kwargs:
{processor: pitch, c_m: 0.5, c_e: 0.0, epsilon: 0.090, epsilon_fraction: 0.25, window:
50, smoothing: 0.5}`; `tempo` flows from the Matchmaker (score tempo or 120 BPM default).

- **Decision**:
  - `src/dp/dannenberg.rs` with header `Port of matchmaker.dp.dannenberg.DPDannenberg`,
    `Upstream: matchmaker/dp/dannenberg.py`. State per [data-model.md](data-model.md).
    Pitch sets as `u128` bitmasks (bit = MIDI pitch 0–127): `missing = (cevt & !used)
    .count_ones()`, union/membership are single instructions; the step keeps the Python
    shape (shift columns, three predecessors y/w/x, rating argmax over `lo..hi`).
  - `score_cevts` as a `Score` method next to `onset_pianoroll` (port of the registry
    builder), returning `(Vec<u128>, Vec<f32>)` — compound events plus their onset beats
    (the `score_positions` the follower receives).
  - `Method::Dannenberg` ("dannenberg", appended to `AVAILABLE` matching upstream
    declaration order). Defaults: processor `pitch` with `return_pitch_list = true`,
    `piano_range = false`, `polling_period = None` (event-based). `MethodConfig` gains
    `c_m`, `c_e`, `epsilon`, `epsilon_fraction`, `window`, `smoothing` (reused `window`
    naming stays method-scoped as with dixon's `window_size`); config keys for dannenberg:
    `{processor, polling_period, c_m, c_e, epsilon, epsilon_fraction, window, smoothing}`.
    `piano_range`/`return_pitch_list` are rejected (upstream excludes them from
    passthrough); processor must be `pitch` (validated like hmm's constraint).
  - `step` ports the upstream exactly: first pitch of the observation
    (`np.atleast_1d(features)[0]`), `_is_new_cevt` computed and discarded with a comment
    naming the upstream dead branch, initial column `-c_m * cumsum(sizes)` so a wrong
    first note cannot lock onto an arbitrary row, best-value monotone update, tempo
    smoothing gated by `elapsed > 0.05 && 0 < advanced < 16`.
  - Position = anchor + smoothed-tempo drift, capped at the next score event;
    `is_still_following` while `current_index < n_states - 1`. `debug_state` exports the
    rating column as `probabilities` and `current_index` (principle IV).
  - Python bindings: `default_kwargs` entry mirroring `methods.yaml`; CLI picks the
    variant up via `Method::AVAILABLE`.
  - Golden generation: add `"dannenberg"` to `METHODS` in `tools/golden/generate.py`; its
    observations are pitch lists, so the `is_list` handling extends to
    `method in ("hmm", "dannenberg")`. Benchmark: add a dannenberg row to the MIDI
    benchmark configuration.
- **Rationale**: Mirrors how every existing event follower is wired (pthmm/hmm pattern for
  processor construction, arzt/dixon pattern for per-method config keys). The `u128`
  bitmask is the one representational deviation; it is documented at the type and changes
  no semantics (sets of small integers ≤ 127).
- **Alternatives considered**: `HashSet<u8>` per compound event (rejected: allocation per
  state per step for `used` copies; the bitmask is simpler and faster); accepting
  arbitrary config keys because the upstream constructor swallows `**kwargs` (rejected:
  the port's existing convention is strict key validation, FR-005 says "consistent with
  existing method-configuration handling").
- **Notes for implementation**: Upstream's `window` counts rows ahead of the best match
  (`hi = best + window`) with a fixed 10-row backtrack (`lo = best - 10`); the DP update
  outside `lo..hi` keeps stale values — port as written. Empty observations (frames
  without a note-on) never reach the follower in the existing pipeline, matching the
  event-based stream upstream; upstream would raise on an empty array, so the port's
  skip is the same observable behavior.

## R5: Multi-fold lazy unfold enumeration — equivalence, no change

Upstream replaces `itertools.islice(pt.score.iter_unfolded_parts(part), 4)` with a local
`_iter_first_unfolded_parts(part, limit=4)` that walks partitura's segment graph lazily
"in Partitura's traversal order, without enumerating the rest".

- **Decision**: No code change. `src/score/unfold.rs::iter_unfolded_parts(part, limit)`
  already enumerates lazily with a work stack and stops at the limit;
  `src/score/mod.rs` already calls it with limit 4. Add a doc-comment note naming the new
  upstream helper. Regenerated multi-fold goldens (candidate lists) verify the traversal
  order still agrees.
- **Rationale**: The port anticipated this change; upstream caught up. Both produce the
  first `limit` variants of the same traversal order, which the existing goldens pin.
- **Alternatives considered**: Restructuring the Rust walker to match the new recursive
  generator (rejected: no output difference; the goldens are the arbiter).

## R6: Out of scope — `wait`-mode feeder thread

Upstream reworks `AudioStream.run_offline`'s `wait=True` path: a feeder thread supplies
blocks at absolute playback deadlines so processing can lag without dropping input, and
`run_offline` consumes a queue.

- **Decision**: Not ported. `src/io/audio.rs` ports only the offline mono
  decode/resample/frame-hop iteration; real-time pacing, devices, and thread/queue
  plumbing are on `UPSTREAM.md`'s "Not ported" list. Extend that list entry to name the
  0.5.1 feeder explicitly.
- **Rationale**: Unchanged scope (spec assumption); the offline (`wait=False`) block
  iteration that the port mirrors is behavior-identical in 0.5.1 (same blocks, same
  padding, same order).

## R7: Documentation, version surfaces, benchmark

- **Decision**: Update `UPSTREAM.md` (version table pymatchmaker 0.4.0 → 0.5.1; new module
  map rows for `src/dp/dannenberg.rs` and the `score_cevts` builder in `src/score/mod.rs`;
  not-ported additions: `wait` feeder, Dannenberg's unused compound-event grouping result;
  equivalence notes for R3/R5). Update `README.md` (reference tag link), and
  `tests/data/audio/README.md` (generator version mentions). `src/matchmaker.rs` method
  docs and `crates/matchmaker-py` docstrings list the sixth method. Benchmark
  configuration gains a `dannenberg` row so the report includes it (SC-004).
- **Rationale**: FR-009/SC-005; the constitution keeps the pinned release in `README.md`.

## Numeric risk register

| Risk | Where it shows | Handling |
|---|---|---|
| Reference's restructured forward step shifts `hmm` goldens at ulp level | `tests/golden/*/hmm/` regeneration diff | Trace to the `prob/hmm.py` hunk; tolerances absorb; exceedance = investigate |
| Lockfile moves transitive deps | Any golden | Trace to lockfile diff; constrain the mover in `tools/golden/pyproject.toml` if tracing is muddied |
| SPARC 3000 Hz `np.arange` float accumulation | `evaluation.json` smoothness | Rust already mirrors `np.arange` fill semantics; regenerated values are compared exactly as before |
| Sliding vs per-window route disagreement beyond float noise | `eval.rs` unit tests | Port both routes and the heuristic; add a unit test asserting both routes agree on a fixture |
