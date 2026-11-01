# Research: MIDI Online Time Warping (arzt, dixon)

Phase 0 of `/speckit-plan`. Python paths refer to the pinned reference (pymatchmaker v0.4.0,
`tools/golden/uv.lock`). Everything below was read in the installed sources, not assumed.

## R1. What the reference actually runs

| | `arzt` (MIDI) | `dixon` (MIDI) |
|---|---|---|
| Class | `dp/oltw_arzt.py: OnlineTimeWarpingArztEvent` | `dp/oltw_dixon.py: OnlineTimeWarpingDixonEvent` |
| Processor | `chord_onset` (`piano_range=True`) | same |
| Stream | windowed, `polling_period = 0.001` (not `event_based`; the default kwarg wins) | same |
| Reference features | `onset_pianoroll(note_array, "onset_beat", piano_range)` → `float32 [N, 88]`, positions `np.unique(onset_beat)` as `float32` | same |
| Pass-through | everything in `mm.config` except `piano_range`, `window_size`, `start_window_size` | everything except `piano_range` |
| Effective parameters | `window_size = 30`, `start_window_size = 5` (class defaults: **the YAML's 2/2 never arrive**), `step_size = 5`, `distance_func = "cosine"` | `window_size = 30`, `max_run_count = 3`, `distance_func = "euclidean"` |

- `processor`, `polling_period` are popped from `mm.config` by `Matchmaker.__init__`; `piano_range`
  stays in `mm.config` and feeds `onset_pianoroll` (`registry._onset_pianoroll`).
- Run loop: `OnlineAlignment.run` — stops when `current_index == len(score_positions) − 1`; every
  call appends `(perf_time, position)` to the path (no continuation concept). `position =
  float(score_positions[current_index])`, `float32 → float64`.
- Neither class has a state distribution, so there is no `probabilities` golden file.

**Decision**: Rust defaults are the *effective* values above; `MethodConfig::for_method` gets
`processor = ChordOnset`, `polling_period = 0.001` for both.
**Rationale**: Principle I; the YAML values are not what runs.
**Alternatives considered**: Porting the YAML literally (would produce different windows).

## R2. Arzt event algorithm and dtypes

Per call `step(features)` (`dp/oltw_arzt.py:657-`):

1. `start = max(current_index − step_size, 0)`; `w = start_window_size if input_index <
   start_window_size else window_size`; `end = min(start + w, N)`.
2. `window_cost = cdist(ref[start:end], feat, metric)` — inputs `float32`, SciPy computes in
   `float64`.
3. Rolling 2-column DP over `score_index ∈ [start, end)`, `si = score_index + 1`:
   cell `(0, input 0)` = `window_cost`, `min_costs = window_cost`, `min_index = 0`; otherwise
   `d1 = M[si−1,1] + c`, `d2 = M[si,0] + c`, `d3 = M[si−1,0] + c`, `best = min(d1,d2,d3)`
   (Python `min`: keeps the first unless a later value is `<`; differs from `f64::min` for NaN),
   `M[si,1] = best`, `norm = best / (input_index + score_index + 1.0)`, strict `<` update.
4. Shift column 1 → 0, column 1 = `inf`.
5. `current_index = clip(min_index, current_index, current_index + step_size)` if
   `input_index > 0`, else `min_index`; `input_index += 1`.

**dtypes**: `M` is `float32 (N+1, 2)`: every `best` is **rounded to `f32` on store**, while `norm` uses
the unrounded `f64` `best`. `d1..d3` are `f32 + f64 → f64`.
**Decision**: store `M` as `Vec<[f32; 2]>`, do all arithmetic in `f64`, port the `min`/`<` logic
with the same comparisons. Golden tolerance: positions exact (`< 1e-6` beats, SC-001);
no float intermediates are exported beyond `positions`/`current_index`.
**Alternatives considered**: `f64` matrix (simpler, changes tie decisions).

## R3. Dixon event algorithm and dtypes

State: sparse `D: dict[(ref, input)] → float64`, `_inputs` (all `float32` vectors so far),
`_pending`, `t`, `j`, run counts (`dp/oltw_dixon.py:390-`).

- `_evaluate(i, k)`: skip if out of range / `i ≥ N`; cell `(0,0)` = local cost; otherwise
  `min` over predecessors `(i−1,k)` weight 1, `(i,k−1)` weight 1, `(i−1,k−1)` weight 2 of
  `prev + w·d`, only finite `prev`; strict `<`; stored only if finite.
- `_norm_cost = D / (1 + i + k)`; `_argmin_path_cost` scans the last `w` inputs at row `j`, then the
  last `w` rows at column `t`, strict `<`, initial `(inf, j, t)`.
- `get_inc` order: `j ≥ N−1 → INPUT`; `t+1 < w or j+1 < w → BOTH`; `run_count_ref ≥ max →
  INPUT`; `run_count_input ≥ max → REF`; argmin decision.
- `step`: first input sets `t = j = 0`, evaluates `(0,0)`; later inputs run the
  `while True` loop (two `get_inc` calls per round, break when the input should advance but no
  input is pending); then `current_index = min(argmin_ref, N−1)` — it can move backwards.

**dtypes**: all `float64` (`cdist` result, dict values).
**Decision**: port 1:1 with a `HashMap<(usize, usize), f64>` first. Replace it by per-column band
storage **only** if the benchmark shows SC-003 missed.
**Rationale**: II/readability and the project's "smallest change" rule; the dict is the
reference's own structure.
**Alternatives considered**: banded `Vec` from the start (faster, but a second structure to verify
against the dict semantics before any evidence it is needed).
**Measured (T021)**: 63 pieces (3 bundled, 12 Vienna 4x22, 12 (n)ASAP, 36 Batik; Rust 3-5 runs,
reference 1-3 runs, Windows, release build). Inference latency per observation, reference / Rust:
`dixon` 18-27× (median 26.8×, about 0.02 ms vs 0.5 ms), `arzt` 18-25× (median 24.8×, about
0.003 ms vs 0.08 ms); total processing speed-up `dixon` 17-105× (median 40×), `arzt` 22-446×
(median 116×). SC-003 (≥ 5×) holds with the `HashMap`, so the band storage (T022) is not needed.

## R4. Distance functions

`cdist(…, "cosine")` calls SciPy's compiled `cdist_cosine_double_wrap`; `"euclidean"` the
pybind `cdist_euclidean`. Both operate on `float64` copies.

**Decision**: implement `cosine` and `euclidean` in `dp/mod.rs`, reject other names with
`InvalidParameter` listing the two.
**Measured (T001, SciPy 1.18.1, `float32` inputs, result `float64`)**:

- `cosine` with a zero vector on either side (also zero vs zero) is **NaN**, without a warning;
  `euclidean` of zero vs `a` is `sqrt(‖a‖²)`, zero vs zero is `0.0`.
- `1 − clip(dot / (‖u‖·‖v‖))` with `‖x‖ = sqrt(Σ x²)` in `f64` reproduces `cdist` bit for bit
  (0 mismatches in 20 000 random binary pairs); `dot / ‖u‖ / ‖v‖` does not (314 mismatches). Port
  `dot / (nu * nv)`; clip means `|c| > 1 → copysign(1, c)`.
- Python `min(nan, 1.0, 2.0)` is `nan`, `min(1.0, nan, 2.0)` and `min(2.0, 1.0, nan)` are `1.0`
  (R2: port the comparison, not `f64::min`).

The NaN of an all-zero observation therefore flows through the DP like in the reference.
**Rationale**: no bundled configuration uses another metric.
**Alternatives considered**: all SciPy metrics (untestable surface).

## R5. Configuration surface

| Key | `arzt` | `dixon` |
|---|---|---|
| `processor`, `piano_range`, `polling_period` | ✅ (as 001) | ✅ |
| `step_size` (usize) | ✅ default 5 | — |
| `window_size`, `max_run_count` (usize ≥ 1) | `window_size` ✗ | ✅ 30, 3 |
| `start_window_size` | ✗ | — |
| `distance_func` | ✅ `cosine` | ✅ `euclidean` |

- `window_size`/`start_window_size` for `arzt` are rejected by the core (`InvalidParameter`:
  "not used by the reference's MIDI arzt"). The Python binding drops them from `kwargs` as the
  reference's pass-through exclusion does, so `default_kwargs` can stay identical to
  `DEFAULT_KWARGS["midi"]["arzt"]` (incl. `window_size: 2`).
- Unknown names are errors (001 behaviour); the reference swallows them in `**kwargs`.
- Sizes below 1: `MethodConfig::set` rejects negative values, `MethodConfig::validate` (run by
  `Matchmaker::with_config`) rejects 0, so a hand-built `MethodConfig` is checked too.
- Python only: when `kwargs` replaces the defaults, `Matchmaker.__init__` falls back to
  `polling_period = 0.01` (only `event_based` methods fall back to `None`) and to the `pitch`
  processor. `arzt` and `dixon` are not `event_based`, so `kwargs={"piano_range": True}` gives 10 ms
  windows and `pitch`, not the 1 ms / `chord_onset` of their defaults; the binding keeps that rule.

**Decision**: add `Option` fields to `MethodConfig`, extend `Method::config_keys`.
**Alternatives considered**: a generic string map in the core (loses typed validation).

## R6. Score side

`Score::onset_pianoroll(piano_range) -> (Vec<Vec<f32>>, Vec<f32>)`: rows over `unique_onsets()`
(already `np.unique(onset_beat)` as `f32`), pitch `p − offset` in `[0, dim)` set to `1.0`,
`dim/offset = 88/21` or `128/0`. Pitches outside are ignored (as the reference's bounds check).

**Decision**: one function in `score/mod.rs` next to `chords()`, golden files `score/onset_pianoroll.npy` (88 keys), `score/onset_pianoroll_full.npy` (128) and
`score/onset_positions.npy` (new golden entries; the generator calls
`onset_pianoroll` directly).
**Alternatives considered**: reusing `chords()` + the `pitch` processor logic (different grouping
rule: the reference groups by exact `onset_beat`, same as `chords()`, but the vector build is
separate and cheap).

## R7. Wiring checklist (everything a method touches)

- Core: `Method` enum + `AVAILABLE` + `config_keys` + `name` + `FromStr`;
  `MethodConfig::for_method`/`set`/`validate`; `Matchmaker::with_config` follower match;
  `export::write_run` match arm (and `probabilities.npy` only when the steps report them).
- CLI: `--method` and the `bench` default list come from `Method::AVAILABLE`; the bench record
  names the method's processor (`ProcessorKind::name`) and writes tie margins only for followers
  with a state distribution; help text of `--processor` and `--set`.
- Python: `default_kwargs` and the pass-through exclusion for `arzt`; `AVAILABLE_METHODS`,
  `DEFAULT_KWARGS` and the error message follow `Method::AVAILABLE`. The fallbacks for replaced
  `kwargs` (R5) stay as they are.
- Tooling: `tools/golden/{generate,compare}.py`, `tools/bench/{run_reference,report}.py` and
  `METHODS` in `.github/workflows/benchmark.yml`.
- Tests that iterate methods: `edge_cases`, `export`, `golden_eval`, `method_config`,
  `golden_score` (onset pianoroll), the CLI `follow` test and the Python `test_drop_in.py`; the
  001 tests that expected `arzt` to fail (T062 of 001, the Python scenario of the 001
  `quickstart.md`).
- Documents: `README.md`, `UPSTREAM.md` (module map, "Not ported") and the 001 contracts
  `cli.md`, `golden-data.md`, `library-api.md`, `python-api.md`, `data-model.md` (its rule: every
  API change updates the contract).
