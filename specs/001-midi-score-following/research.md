# Research: MIDI Score Following (pthmm, hmm, outerhmm)

Phase 0 output of `/speckit-plan`. Every entry: **Decision**, **Rationale**, **Alternatives
considered**. Python paths refer to the reference release (pymatchmaker, tag in `README.md`)
and to the dependency versions pinned by the golden-data lockfile (currently partitura 1.9.0,
python-hiddenmarkov 0.1.4, NumPy 2.x, SciPy).

## R1. What the reference actually runs on the MIDI path

Findings (verified in source, not in docs):

| Method | Class | Processor (as built by `methods.yaml`) | Stream |
|---|---|---|---|
| `pthmm` (MIDI default) | `matchmaker.prob.hmm.PitchHMM`, `has_insertions=True`, `piano_range=True` | `PitchProcessor(piano_range=True, return_pitch_list=False)` → binary 88-vector | windowed, `POLLING_PERIOD = 0.01` s |
| `hmm` | `matchmaker.prob.hmm.PitchIOIHMM`, `has_insertions=True`, `piano_range=True`, `tempo_model=KalmanTempoModel` | `PitchProcessor(piano_range=True, return_pitch_list=True)` → pitch list (pitch − 21) | windowed, 0.01 s |
| `outerhmm` | `matchmaker.prob.outer_product_hmm.OuterProductHMM` | `PitchProcessor(piano_range=True, return_pitch_list=False)` | `event_based: true`, **but** `default_kwargs.polling_period = 0.001` wins → windowed, 1 ms |

- Reference features for all three: `score_part.note_array()` (registry builder `note_array`).
- Score states: `np.unique(note_array["onset_beat"])`; with insertions, a midpoint state is
  inserted between consecutive onsets (`2·n − 1` states).
- Observation time = **first note-on time inside the frame** (`min(note_times)`), not the
  window centre. Frames without note-on yield `None` and are skipped by the follower.
- `chord_onset` (`ChordOnsetProcessor`) is stateful: merges note-ons while
  `m_time − pending_time ≤ 0.035`, uses empty-frame time as heartbeat, has
  `flush_remaining()` called by the stream before `STREAM_END`.
- Run loops (they shape the alignment path and MUST be ported):
  - `OnlineAlignment.run` (pthmm, hmm): stops as soon as `current_index == len(states) − 1`.
  - `OuterProductHMM.run`: `is_still_following()` is always true; stops after `patience = 10`
    consecutive non-continuation observations without state change. A note within
    `IOI_THRESHOLD = 0.035` s of the previous one is folded into the current chord and the step
    is recomputed from the pre-chord distribution; such continuation steps yield a position
    but **do not append to the alignment path**.
- Consequence for the spec: the alignment path has one pair per *consumed, non-continuation*
  observation, not one per observation. US1 scenario 1 was reworded accordingly.

**Decision**: Port exactly this wiring (stream mode, processor flags, run-loop stop rules) as
the per-method defaults of the Rust registry.
**Rationale**: Principle I; these details change positions and path length.
**Alternatives considered**: Porting `Matchmaker`'s generic YAML registry — rejected
(II/readability): a typed Rust table of three methods is clearer and has the same defaults.

## R2. HMM forward step (python-hiddenmarkov)

`HiddenMarkovModel.forward_algorithm_step(obs, log_probabilities=False)`:

```text
if forward_variable is None: prior = init_probabilities            # NOT normalised
else:                        prior = transition_matrix.T @ forward_variable
fv = observation_model(obs) * prior
fv /= max(fv.sum(), 1e-6)
state = argmax(fv)                                                   # first max wins
```

- `ConstantTransitionModel` keeps the matrix as given (no extra normalisation).
- `stable_transition_matrix(n, mp=1, dist=gumbel_l, scale=1.0, inserted_states=True)`:
  row `i` = `gumbel_l.pdf(arange(n), loc=i+2 if i even else i+1, scale)`, then row-normalised.
- `init_dist(n, loc=0, scale=10)`: `gumbel_l.pdf(arange(n), 0, scale if scale < n else n/10)`.
  (Note: `scale < n` is the usual case, so the scale is 10, not `n/10`.)
- `gumbel_l.pdf(x, loc, s) = exp(z − exp(z)) / s`, `z = (x − loc)/s`.

**Decision**: Port `forward_algorithm_step` and `ConstantTransitionModel` 1:1 into
`prob::hidden_markov`, but store each transition row as a **band of its non-zero entries**.
**Rationale**: `gumbel_l.pdf` underflows to exactly `0.0` for `z > ≈6.6` and `z < ≈−745`, so
each row has ≤ ~752 non-zeros. Skipping exact zeros changes no result (0·x adds nothing), cuts
memory from `O(N²)` (≈ 290 MB at N = 6000) to `O(N·752)` and is the main lever for SC-004.
**Alternatives considered**: Dense `N×N` matrix-vector product (simple, but memory-bound and
slower than NumPy's BLAS for large N); truncating at a probability threshold (would change
results — violates I).

## R3. Observation models

- `compute_discrete_pitch_profiles`: chord matrix `(n_chords, 128)` convolved along pitch with
  `[0.02, 0.02, 1, 0.02, 0.02]` (`scipy.signal.convolve(..., mode="same")`), interleave with
  zero rows for inserted states (drop last), `+ 0.01`, slice `[:, 21:109]`, row-normalise.
- Bernoulli (`pthmm`, and the pitch part of `hmm`): `∏_k p_k^x_k (1−p_k)^(1−x_k)` over 88
  pitches. With `x ∈ {0,1}`, `p^1 = p` and `p^0 = 1` exactly, so the product of the selected
  factors is bit-equivalent up to multiplication order.
- `hmm` (`ACCPitchIOIObservationModel`): pitch list → binary vector, times
  `sqrt(1/2π)·exp(−0.5·(beat_period·ioi_matrix[current_state] − ioi_obs)²)` with
  `ioi_precision = 1`. `ioi_obs` = time since previous observation (0 for the first).
  Only row `current_state` of the `N×N` IOI matrix is ever used.
- `outerhmm`: `b_table (N×128)` from pitch-error classes (correct 0.9497, ±1, ±2, ±12, ±3..11
  semitones, else `other_prob = 1e-6`); likelihood
  `exp(log(max(b[:,21:109],1e-300)) @ obs − max)`; Viterbi-style max over band
  `[i−3, i+3]` vs skip term `r_i · max_j(S_j p_j)`; normalise, or uniform if the sum is 0.
  Transition `alpha` = 1e-6 everywhere, deltas −3..3 set, `+ finfo(float).eps`, row-normalise.

**Decision**: Direct convolution; IOI row computed on the fly as `|s_i − s_j|` in `f64`;
`log_b` precomputed once (the reference recomputes it per call with identical values).
**Documented deviation (`hmm`)**: the reference turns the pitch list into a binary vector by
indexing with `pitch − 21`; a pitch below 21 wraps around to the end of the vector (NumPy
negative index) and a pitch above 108 raises `IndexError`. Rust ignores pitches outside the
88-key range instead. Piano performances (all bundled and dataset inputs) never contain
such pitches, so golden results are unaffected; an edge-case test documents the behaviour.
**Rationale**: `scipy.signal.convolve` picks direct convolution for a 5-tap kernel; on-the-fly
`|a − b|` equals `pdist` exactly (`sqrt(RN(x²)) = |x|` in binary floating point).
**Alternatives considered**: Full IOI matrix (`O(N²)` memory for one used row); FFT
convolution (adds rounding noise).

## R4. Numeric types and promotion (parity at the bit level)

- partitura's `note_array()` stores `onset_beat`, `duration_beat`, `onset_quarter`,
  `duration_quarter` as **`float32`**; `onset_div`, `duration_div`, `pitch`, `voice` as
  `int32`. Beat maps are computed in `float64` (`scipy.interpolate.interp1d`) and then cast.
- With NumPy ≥ 2 (NEP 50), Python floats are "weak": `float32_array + 0.5 * diff` stays
  `float32` → score positions incl. inserted midpoints are `float32`. In `KalmanTempoModel`,
  `score_onset` is a NumPy `float32` scalar, so `score_ioi`, the gain denominator and thus
  `kalman_gain` are `float32`, while `performed_ioi` is `float64`. `beat_period` starts as a
  Python float (0.6) and becomes `float64` after the first update, which changes the precision
  of `score_ioi * period_pred` from the second update on. The port annotates every operand.
- Processor outputs are `float32`; Bernoulli / IOI / forward variable are `float64`.
- NumPy's `sum` uses pairwise summation; `argsort` default `quicksort` is not stable and
  SIMD-dispatched (results for ties can depend on the CPU).

**Decision**: Mirror dtypes per operation (`f32` where NumPy computes in `float32`, `f64`
elsewhere) and note the Python expression next to each such line. Use sequential summation
first; switch to a NumPy-compatible pairwise sum only if a golden test shows a flipped argmax.
Golden tolerances: positions `< 1e-6` beats (SC-001), intermediate `f64` arrays `rtol 1e-12`,
`f32` arrays `rtol 1e-6`.
**Rationale**: Positions are argmax decisions; tiny summation-order differences only matter at
exact ties, which the golden tests will reveal (SC-007 names the first differing step).
**Alternatives considered**: Bit-exact emulation of NumPy/BLAS reduction order everywhere —
rejected as unreadable (II) without evidence of need.

## R5. Performance MIDI loading and stream timing

- `partitura.load_performance_midi`: per track, `t += msg.time · mpq/(ppq·1e6)` while tempo
  changes are collected across tracks (`tempo_changes`, only when changed); notes paired per
  `(channel, pitch)` (a new note-on overwrites an unterminated one; unmatched note-offs are
  ignored); notes sorted by `(note_on, pitch, note_off, channel, track)`; finally all times are
  **recomputed from ticks** with `adjust_time(tick, tempo_changes, ppq)`. Note-off time is the
  key-release time (`note_off`), not `sound_off` (pedal is ignored).
- `midi_messages_from_performance`: per note a note-on and a note-off message, then control
  and program changes; sorted with `np.argsort` (unstable).
- `midi_messages_to_framed_midi(polling_period)`: `n = ceil(max_t / pp)`; frame 0 takes
  `t ≤ pp`; frame k takes `k·pp < t ≤ (k+1)·pp`; buffer time = `(k + 0.5)·pp`.
- Event mode (`polling_period=None`): one message per frame, frame time = message time.

**Decision**: Port these three functions 1:1 (same float formulas, same comparisons) in
`performance::midi` and `io::stream`; use a **stable** sort by time over the construction order.
Parse MIDI with `midly`.
**Rationale**: Frame membership depends on the exact float expressions. Tie order cannot
change any default-configuration result: in windowed mode all messages with equal time land
in the same frame and processors aggregate frames order-independently (outerhmm uses 1 ms
windows, so simultaneous notes form one observation). Documented as a deviation only for the
non-default event mode.
**Alternatives considered**: Emulating NumPy's unstable quicksort (CPU-dependent, impossible
to match reliably).

## R6. Score loading (partitura subset)

Calls used: `load_musicxml(ignore_invisible_objects=True, force_note_ids="keep")`,
`load_score` (MIDI scores → `load_score_midi`), `unfold_part_maximal(score,
update_ids=True, ignore_leaps=False)`, `merge_parts(parts)` (fallback without unfolding when
unfolding raises), `note_array()`, `iter_all(Tempo)`.

- `onset_beat = beat_map(onset_div)`; `beat_map` is a piecewise-linear map with keypoints at
  divisions changes and time signatures: `y = cumsum(beat_type/4 · Δdiv / divs_per_quarter)`;
  if the first measure is shorter than the time signature (anacrusis), `y −= actual_dur`.
  `musical_beat` is off by default.
- Tied notes: only the first note of a tie chain (`notes_tied`); grace notes included.
- Unfolded note ids get `-<repetition>` suffixes (`update_ids=True` is the default).
- `merge_parts`: rescales times by `lcm(divs)/divs`, keeps tempos/time signatures of the first
  part only, sorts notes by onset (`mergesort`) then pitch.

**Decision**: Use the `musicxml` crate (MIT, pure Rust, reads `.musicxml` and `.mxl`) for XML
parsing only, and port the partitura semantics on top in `score::{musicxml, timeline, unfold,
merge, midi, tempo}`; each module names its partitura source. Note-array order is reproduced
(stable sort by onset, then pitch) although followers only use the onset→pitch-set grouping.
**Update (implementation)**: switched to the planned fallback, `roxmltree` (generic DOM) + `zip`
(deflate only, pure Rust) for `.mxl`. partitura walks the lxml tree element by element
(`find`, `findall`, attribute lookups); a generic DOM allows a line-by-line port (II) and is
as lenient as lxml, whereas the typed `musicxml` crate would need a different structure.
Text tempo directions: partitura parses `<words>` with a Lark grammar; only its
`tempo_indication` rule (`[qhe]\.* = [0-9]+`, optionally in parentheses after other words)
produces a `Tempo`. The port recognises exactly that rule and does not port the rest of the
grammar (it only yields non-tempo directions); documented deviation for texts the grammar would
reject, affecting only `tempo`, never positions.
**Rationale**: No Rust crate implements partitura's timeline, unfolding or merging; these
define `onset_beat`, which every result depends on (Scope: "MUST match partitura's output").
**Alternatives considered**: Own XML reader on `roxmltree` — kept as fallback if `musicxml`
rejects dataset files that lxml accepts (it is a strictly typed parser); pre-converting scores
to note arrays in Python — rejected (normal use must not need Python).

## R7. Tempo

`get_tempo_from_score`: first partitura `Tempo` (`to_quarter_tempo(unit or "q", bpm)`), else
first MusicXML `<sound tempo>`, else first text marking (`TEMPO_MARKING_TO_BPM`, adjusted for
compound/simple meter), else `None` → 120 BPM. User value overrides. The MIDI followers do not
use the tempo (only audio and `build_score_annotations(return_type="seconds")` do).

**Decision**: Port all three fallbacks in `score::tempo`; expose the tempo on the score.
**Rationale**: FR-003 and Python-API parity (`mm.tempo`).
**Alternatives considered**: None.

## R8. Evaluation

- `evaluate_alignment(aligned_beats, aligned_sec, gt_beats, gt_sec)`:
  `transfer_positions(domain="score", frame_rate=1)` = last path entry per unique perf time,
  then step-hold lookup (`searchsorted(side="right") − 1`, NaN before the first entry).
- Metrics on `gt − predicted`: `mean_b`, `median_b`, `std_b` (NaN-aware, ddof 0),
  `accuracy_{τ}b` = `count(|e| ≤ τ) / len(gt)`, `skewness`, `kurtosis` (SciPy defaults:
  `bias=True`, Fisher), each rounded via `f"{x:.4f}"`; `smoothness` = mean adaptive SPARC over
  30 s GT-onset windows (path resampled at 50 Hz with hold, `np.gradient`, `rfft` zero-padded
  to `2^(ceil(log2 n)+4)`), rounded to 4 decimals.
- Ground truth: `.match` via `gt_from_match` (joins on shared score note ids, otherwise on
  `(pitch, round(onset_beat, 4))` of a score rebuilt from the match file; perf onsets from the
  streamed MIDI if ids/pitches agree), or TSV with header columns `perf_sec`, `score_beat`.
  The bundled `*_annotations.txt` files are headerless audio annotations and are not used by
  the reference for MIDI evaluation.

**Decision**: Port `transfer_positions`, the metrics and SPARC exactly (`realfft` for the
FFT). Ground truth: TSV and `.match` natively; the `.match` fallback join (requires rebuilding a
score from match `snote` lines) is ported in this feature only as far as the bundled examples
and the benchmark datasets need it; files that need an unsupported path raise a typed error
that names `tools/golden/match_to_tsv.py`, which converts them with the reference itself.
**Rationale**: Metric parity is mandatory (I); smoothness differs only by FFT rounding, below
the 4-decimal rounding (tolerance `1e-4`). The rounding helper is tested on halfway cases
against Python.
**Alternatives considered**: Porting partitura's whole match-file importer up front — large,
and only needed for a fallback path.

## R9. Latency measurement and benchmark

- The reference has **no** latency bookkeeping on the MIDI path (`MidiStream` and the HMMs have
  no `latency_stats`; `Matchmaker.get_latency_stats()` fails for MIDI).
- `std::time::Instant` panics on `wasm32-unknown-unknown`.

**Decision**: The core library only emits `tracing` spans (`stream`, `feature`, `inference`,
level `DEBUG`/`TRACE`) and never reads a clock. Timing lives in the harness: the CLI `bench`
command times processor and follower calls per observation with `Instant` (no subscriber
installed). The Python side of the benchmark (`tools/bench/run_reference.py`) wraps the
reference processor `__call__` and follower `__call__` with `time.perf_counter()` without
modifying the reference. Both write the same JSON schema
([contracts/benchmark-report.md](contracts/benchmark-report.md)); `tools/bench/report.py`
merges them into the side-by-side report, flags > 10 % regressions against the last `master`
report, and is published to GitHub Pages.
Reference accuracy results are deterministic and cached by key (reference tag + lockfile hash
+ dataset checksum); Python latency is re-measured on every run.
**Rationale**: III requires identical inputs and per-stage numbers without distortion; the
reference cannot be changed (I).
**Alternatives considered**: Timing inside the library (breaks wasm, distorts measurements);
`criterion` micro-benchmarks — not needed for the Rust-vs-Python comparison; may be added later.

## R10. Golden data

**Decision**: `tools/golden/` holds `pyproject.toml` + `uv.lock` (pymatchmaker at the reference
tag, partitura, NumPy, SciPy, … pinned) and `generate.py`. It writes, per piece and method,
`.npy` arrays and a `meta.json` into `tests/golden/<piece>/<method>/` (layout in
[contracts/golden-data.md](contracts/golden-data.md)). Inputs: the three bundled reference
assets (`simple_mozart_k265_var1`, `mozart_k265_var1`, `Bach-fugue_bwv_858`, MusicXML + MIDI +
`.match`), copied to `tests/data/` (Apache-2.0, attributed in `NOTICE`). Follower golden tests
replay the **reference's recorded observations**, so stream/processor parity and follower
parity are tested independently; end-to-end tests cover the whole pipeline.
The Rust export (`--export`) uses the same layout so `tools/golden/compare.py` can diff a Rust
run against the reference and report the first differing stage and index (US6).
**Rationale**: I (checked-in generator + lockfile), IV (Python-readable), SC-007.
**Alternatives considered**: CSV everywhere (lossy for floats unless printed with 17 digits;
slower); running Python in `cargo test` (normal tests must not need Python).

## R11. Interfaces and packaging

**Decision**: Cargo workspace:
- `matchmaker-rs` (root package, library `matchmaker_rs`): default build, pure Rust, wasm32.
- `crates/matchmaker-cli` (binary `matchmaker`): `clap`, `anyhow`, `tracing-subscriber`.
- `crates/matchmaker-py` (PyO3 + `numpy`, built with maturin, `abi3-py310`): Python module
  `matchmaker_rs` exposing `Matchmaker`. Not a default member; built in a separate CI job.
**Rationale**: V (C/libpython linking only outside the default build), clean wasm build of the
core, PyO3 drop-in for Python 3.10–3.13.
**Alternatives considered**: Cargo features on one crate (`cli`, `python`) — `--all-features`
in CI would then link libpython into every test build.

## R12. Dependencies (V)

| Crate | Where | License | Why |
|---|---|---|---|
| `midly` (`std` only) | core | Unlicense | MIDI file parsing (pure Rust). **Requires adding `Unlicense` to `deny.toml`.** |
| `roxmltree` | core | MIT/Apache-2.0 | MusicXML DOM (see R6 update) |
| `zip` (feature `deflate` only) | core | MIT | `.mxl` containers (pure Rust: `zlib-rs`, `zopfli`) |
| own `npy` module | core | — | `.npy` read/write. `npyz` and `ndarray-npy` were rejected: both pull `py_literal` → `pest` → `stacker` → `psm`, whose `build.rs` compiles C/assembly (violates V) |
| `realfft` (+`rustfft`) | core | MIT (/Apache-2.0) | `rfft` for SPARC |
| `thiserror` | core | MIT/Apache-2.0 | typed errors (IV) |
| `tracing` | core | MIT | stage spans (III) |
| `serde`, `serde_json` | core | MIT/Apache-2.0 | `meta.json`, benchmark JSON |
| `clap`, `anyhow`, `tracing-subscriber` | CLI | MIT/Apache-2.0 | CLI, error wrapping, `RUST_LOG` |
| `pyo3`, `numpy` | Python crate | MIT/Apache-2.0, BSD-2-Clause | bindings |

No other runtime dependencies. All pass `cargo deny` once `Unlicense` is allowed.

## R13. Datasets in CI

**Decision**: `tools/datasets/datasets.toml` lists source URL, version/commit and SHA-256 per
dataset ((n)ASAP, Batik, Vienna4x22); `tools/datasets/fetch.py` downloads into a CI cache
(never committed). Exact URLs/commits and license notes are pinned in the first implementation
task and recorded in `README.md` (volatile fact, see constitution Conventions). The benchmark
job is sharded per dataset to bound CI time.
**Rationale**: FR-016/FR-020, SC-006, no redistribution.
**Alternatives considered**: Committing derived data — not allowed by dataset licenses.
