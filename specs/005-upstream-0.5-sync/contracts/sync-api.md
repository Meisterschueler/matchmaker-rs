# Contract: Upstream 0.5.1 Sync Surfaces

Binding reference: pymatchmaker tag `v0.5.1` and the regenerated `tools/golden` lockfile.
Where this contract and the reference at the tag disagree, the reference wins and this
file is corrected.

## 1. `dannenberg` method surface

Available wherever MIDI methods are selected (library `Method`, CLI `--method`, Python
`Matchmaker(method=...)`). Name: `"dannenberg"`, appended to the available-methods list in
upstream declaration order.

### Defaults (from `methods.yaml`)

| Key | Default | Constraint |
|---|---|---|
| `processor` | `pitch` | Only `pitch` accepted (pitch-list observations); others → `InvalidParameter` |
| polling | event-based | `polling_period = None`: one MIDI message per frame |
| `c_m` | `0.5` | Cost per missing score note; finite float |
| `c_e` | `0.0` | Cost per extra performance note; finite float |
| `epsilon` | `0.090` | Seconds; grouping threshold (computed, unused upstream — see §4) |
| `epsilon_fraction` | `0.25` | Fraction of predicted next-event interval |
| `window` | `50` | Rows ahead of the best match; integer ≥ 1 |
| `smoothing` | `0.5` | Tempo smoothing coefficient |
| initial tempo | Matchmaker tempo | Score tempo, user override, or 120 BPM default |

Processor is built with `piano_range = false`, `return_pitch_list = true`; those two keys
are **rejected** for this method (upstream excludes them from config passthrough).
Unknown keys are rejected with the method's allowed-key list (existing convention; the
upstream constructor's `**kwargs` leniency is not reproduced).

### Behavior

- Reference features: score compound events per [data-model.md](../data-model.md); one DP
  column update per incoming note, three predecessors, windowed `lo = best−10`,
  `hi = best+window`.
- Position: anchor + smoothed-tempo drift, capped at the next score event's beat.
  `is_still_following()` is true while the best row is before the last compound event.
- Deterministic: no RNG, no seed parameter.
- Errors: invalid config keys/values are typed `InvalidParameter`; no panics on any MIDI
  input (empty frames never reach the follower; the stream is event-based).
- Export/debug: `debug_state` carries the current DP rating column (`probabilities`) and
  `current_index` (best row); recorded runs export through the existing `export` plumbing.

### Python bindings

`default_kwargs("dannenberg")` returns the table above (processor, c_m, c_e, epsilon,
epsilon_fraction, window, smoothing) exactly as `methods.yaml` spells it. The method is a
drop-in for `matchmaker.Matchmaker(method="dannenberg", input_type="midi")`.

## 2. SPARC smoothness (revised)

- `compute_sparc(score_beat, perf_sec, onset_times, window_sec)` keeps its signature;
  internal resampling rate is `SPARC_FS = 3000.0`.
- Mean over onset-anchored windows; excluded from the mean: windows not fully covered by
  the report span, and windows whose resampled path has fewer than two samples
  (reference reasons `uncovered` / `insufficient_samples`). Stationary and
  band-undefined windows remain excluded as before.
- Per-window values equal `spectral_arc_length` of `|np.gradient(window slice)|` — the
  reference's `windowed_spectral_arc_lengths` guarantee. The port implements both the
  per-window FFT route and the sliding nonzero-sample spectrum with gradient edge
  correction, selected by the reference's cost heuristic; a unit test asserts both routes
  agree on a fixture.
- All `evaluation.json` goldens and the `sparc_matches_reference` unit expectation are
  regenerated from 0.5.1.

## 3. Golden data

- Layout per `specs/001-midi-score-following/contracts/golden-data.md`, unchanged.
- New per-piece directory `dannenberg/` with the standard files (`score_positions.npy`,
  `observations_features.npy` as padded pitch lists, `observations_times.npy`,
  `positions.npy`, `current_index.npy`, `probabilities.npy` when bounded,
  `alignment_path.npy`, `evaluation.json`, `meta.json` with `REFERENCE_TAG` v0.5.1).
- `tools/golden/generate.py`: `METHODS` gains `"dannenberg"`; pitch-list feature saving
  (`is_list`) covers `hmm` and `dannenberg`.
- Regeneration diff review: every changed number traces to (a) the SPARC revision,
  (b) the `prob/hmm.py` forward-step restructuring (ulp-level), or (c) a lockfile move;
  anything else blocks the sync.

## 4. Equivalence records (no code change)

| Upstream 0.5.1 change | Port artifact that already realizes it | Record |
|---|---|---|
| `PitchIOIHMM.step` reachable-state restriction + `max(sum, 1e-6)` guard | `Banded` row storage and guarded normalization in `src/prob/hidden_markov.rs` | Doc comment naming the upstream fast path |
| `BernoulliPitchObservationModel(inserted_states=True)` single uniform-profile evaluation | Odd profile rows are the identical uniform profile; Rust evaluates per row with equal values | Doc comment in `hmm_models.rs`/`pthmm.rs` |
| `_iter_first_unfolded_parts(part, limit=4)` lazy enumeration | `src/score/unfold.rs::iter_unfolded_parts(part, 4)` | Doc comment; multi-fold goldens re-verify traversal order |
| Dannenberg `_is_new_cevt` computed, result unused | Ported as written with a comment naming the dead branch | `UPSTREAM.md` note |
| `AudioStream` `wait` feeder thread | Not ported (live plumbing) | `UPSTREAM.md` "Not ported" list names the feeder |

## 5. Version surfaces

- `UPSTREAM.md`: version table 0.5.1; module-map rows for `src/dp/dannenberg.rs`
  (`matchmaker/dp/dannenberg.py`, `DPDannenberg`) and the `score_cevts` builder
  (`matchmaker/registry.py`) on `src/score/mod.rs`.
- `README.md` reference tag link → `v0.5.1`; `tests/data/audio/README.md` generator
  version mentions updated.
- Benchmark configuration includes `dannenberg`; the published report shows it beside the
  reference (SC-004).
