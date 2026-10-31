# Contracts: deltas to the 001 interfaces

Everything not listed here is unchanged from [001](../../001-midi-score-following/contracts/).

## Library

- `Method::{Arzt, Dixon}`; `Method::AVAILABLE = ["pthmm", "hmm", "outerhmm", "arzt", "dixon"]`
  (reference `AVAILABLE_METHODS["midi"]` order).
- `MethodConfig::for_method(Arzt | Dixon)`: processor `chord_onset`, `piano_range = true`,
  `polling_period = Some(0.001)`.
- `MethodConfig::set` keys: see [research R5](../research.md#r5-configuration-surface). Invalid
  keys/values → `Error::InvalidParameter` naming key and allowed values.
- `dp::oltw_arzt::ArztEvent`, `dp::oltw_dixon::DixonEvent`: `new(reference_features, score_positions,
  config)` with the arrays of `Score::onset_pianoroll` (the Python constructor's arguments), implement
  `OnlineAlignment`; `debug_state().probabilities` is empty.
- `Score::onset_pianoroll(piano_range)`.

## CLI

`--method arzt|dixon`; `--set step_size=…`, `distance_func=…` (arzt), `window_size=…`,
`max_run_count=…`, `distance_func=…` (dixon); `--processor` defaults to `chord_onset` for both.
`bench --methods` lists them by default.

## Python

- `Matchmaker(..., input_type="midi", method="arzt" | "dixon")` works; `ValueError` for other
  methods lists `['pthmm', 'hmm', 'outerhmm', 'arzt', 'dixon']`.
- `default_kwargs` of `arzt`: `{processor: chord_onset, piano_range: True, polling_period: 0.001,
  window_size: 2, start_window_size: 2, step_size: 5}`; of `dixon`: `{processor: chord_onset,
  piano_range: True, polling_period: 0.001, window_size: 30}` (reference values). For `arzt` the
  binding drops `window_size` and `start_window_size` before building the follower (pass-through
  exclusion).
- `input_type="audio"` keeps raising `ValueError`.

## Golden data

`tests/golden/<piece>/{arzt,dixon}/`: `score_positions.npy`, `observations_features.npy`,
`observations_times.npy`, `positions.npy`, `current_index.npy`, `alignment_path.npy`,
`evaluation.json` (if ground truth exists), `meta.json`; **no** `probabilities.npy`.
`<piece>/score/onset_pianoroll.npy` (88 keys), `onset_pianoroll_full.npy` (128) and
`onset_positions.npy` are new. Tolerances: positions
`< 1e-6` beats, `current_index` exact, observations as in 001.
