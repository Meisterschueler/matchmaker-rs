# Data Model: MIDI Online Time Warping

Only what is new relative to [001](../001-midi-score-following/data-model.md).

## OnsetPianoroll (score side)

| Field | Type | Rule |
|---|---|---|
| `features` | `Vec<Vec<f32>>`, `N × D` | row per unique `onset_beat`; `1.0` at `pitch − offset` for every note starting there; `D/offset` = 88/21 (`piano_range`) or 128/0; out-of-range pitches ignored |
| `positions` | `Vec<f32>`, `N` | `Score::unique_onsets()` (sorted, unique) |

Source: `Score::onset_pianoroll(piano_range)`. Becomes `reference_features`/`score_positions` of both followers.

## ArztEvent (`dp::arzt`)

| Field | Type | Notes |
|---|---|---|
| `reference`, `score_positions` | pianoroll | |
| `window_size` 30, `start_window_size` 5, `step_size` 5 | `usize` | `distance` = `cosine` |
| `cost` | `Vec<[f32; 2]>`, length `N + 1` | `inf` initially; `f32` storage (R2) |
| `current_index`, `input_index` | `usize` | |
| `path` | `AlignmentPath` | one pair per call |

Transitions: `call(obs)` → window → rolling DP → clipped `current_index` → `input_index += 1`.
`is_still_following()` = `current_index < N − 1`.

## DixonEvent (`dp::dixon`)

| Field | Type | Notes |
|---|---|---|
| `window_size` 30, `max_run_count` 3 | `usize` | `distance` = `euclidean` |
| `cost` | `HashMap<(usize, usize), f64>` | only finite cells (R3) |
| `inputs` | `Vec<Vec<f32>>` | every consumed observation (`_inputs`) |
| `started` | `bool` | `t >= 0` in the reference |
| `t`, `j` | `usize` | last consumed observation, last evaluated onset (`-1` before the first input in the reference) |
| `run_count_input`, `run_count_ref` | `usize` | |
| `current_index`, `path` | | as above (`input_index` is not kept: nothing reads it) |

`_pending` is a local of `step`: it holds at most the new observation and is empty between steps.
Transitions: first input → `t = j = 0`; later inputs → direction loop (R3); then
`current_index = min(argmin_ref, N − 1)`.

## Direction

`Ref`, `Input`, `Both` (`dp/direction.py`; `NONE` is only used by the frame-level code and is
not ported).

## MethodConfig additions

`step_size`, `window_size`, `max_run_count`: `Option<usize>`; `distance_func`: `Option<Distance>`
(`Cosine | Euclidean`). `None` = the method's default. Validation: negative values are rejected by
`MethodConfig::set`, sizes of 0 by `MethodConfig::validate` (so a hand-built struct is checked too);
keys per method as in [R5](research.md#r5-configuration-surface).
