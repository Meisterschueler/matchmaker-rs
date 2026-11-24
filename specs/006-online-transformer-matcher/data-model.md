# Data Model: Online Transformer Matcher (OTM)

This feature extends the existing `ScoreNote`, `Observation`, `StepDebug`,
`MethodConfig`, and public `AlignmentPath` types. It does not replace their shared API
contracts.

## Grace-aware score reference

Port of `partitura.utils.music.note_array(..., include_grace_notes=True)` through the
pymatchmaker `note_array_with_grace` builder and Parangonar's
`OnlineTransformerMatcher.prepare_score`.

| Field | Type | Meaning |
|---|---|---|
| `id` | unique string | Identity used to prevent matching one score note more than once. Preserve valid unique IDs; if any are empty or duplicated, replace all with stable `s{row}` IDs as the adapter does. |
| `pitch` | `u8` MIDI note | Note pitch from the score. |
| `onset_beat` | `f32` | Score onset, sorted ascending with the note array. |
| `duration_beat` | `f32` | Score duration; preserved in the score representation, not used as a token. |
| `is_grace` | `bool` | Explicit grace-note identity from the parsed score; never infer it from duration. |

Derived score data:

| Field | Type | Invariant |
|---|---|---|
| `score_positions` | `Vec<f32>` | Strictly ascending unique note onsets; the public score position at index `i`. |
| `pitches_at_onset_by_id` | ordered list of pitch sets | One set per `score_positions` entry, built from **non-grace** notes only. Preserve the reference set's pitch iteration order when filling token slots. |
| `score_by_pitch` | pitch → ordered note rows | Includes grace and non-grace notes; excludes already aligned note IDs when choosing a candidate. |

An empty score is rejected before matcher initialization. The note array, score positions,
and unique-ID policy must be identical to the pinned reference for each fixture.

## Performance observation

One positive-velocity MIDI note-on, in input event order.

| Field | Type | Meaning |
|---|---|---|
| `pitch` | `u8` | The single MIDI note pitch passed to OTM. |
| `onset_sec` | `f64` | Event time in performance seconds. |
| `id` | stable string | Per-run ID assigned by the `parangonar` processor; useful for diagnostics, not used to select the returned score position. |
| `input_index` | `usize` | Zero-based note-event index used by the internal warping path. |

Note-off and non-note messages produce no observation. A frame with multiple positive note-ons
is invalid for this processor; the stream must use `polling_period = None` so simultaneous
note-ons are delivered separately and in stable event order. OTM rejects pitches whose
reference performance token (`pitch - 20`) is outside the configured model vocabulary.

## Token window

The fixed model input is 26 tokens × 7 integer attributes, matching Parangonar's
`tokenize(score_segment, perf_segment, dims=7)`.

| Token region | Count | Contents |
|---|---:|---|
| Performance | up to 8 | Last eight performance pitches, each repeated across seven attributes as `pitch - 20`. |
| Separator | 1 | Seven attributes with value `89` (performance-to-score marker). |
| Score | up to 16 | Score pitch sets from onset indices `[max(current_index - 7, 0), current_index + 9)`, encoded in seven attributes. Only pitches 21–108 map to `pitch - 20`; remaining attributes are zero. |
| Padding | to 26 | Seven attributes with value `90` per token. |

The score pitch-set iteration order and pitch-slot truncation must reproduce the pinned
reference; do not substitute a different set ordering or silently sort tokens. Model token
IDs must be within `[0, token_number)`. Any out-of-vocabulary performance token is an
explicit input error, not a clamp.

## OTM model

One immutable inference model loaded from the versioned converted artifact.

| Property | Default | Invariant |
|---|---:|---|
| Token vocabulary | 91 | Embedding indices `0..90`; shape must match the artifact. |
| Model width | 64 | Seven token attributes are embedded and summed per position. |
| Classes | 2 | Output class dimension must match the artifact. |
| Attention heads | 8 | Must divide model width and match the artifact. |
| Encoder layers | 6 | Post-normalized transformer encoder layers; must match artifact. |
| Feed-forward width | 64 | ReLU feed-forward block, matching the reference model. |
| Dropout | 0.1 | Disabled for evaluation/inference; retained as a reference-compatible config value. |
| Position encoding | fixed 26 rows | Values copied from the released checkpoint; do not regenerate at runtime. |

For each token window, inference returns logits with shape `(26, 2)`. OTM selects the
position with the largest class-1 softmax probability over the sequence axis. The compact
artifact is versioned, checksummed, and never deserialized as a Python pickle at runtime.
Missing, corrupt, or architecture-incompatible model data is an explicit typed error.

## Tempo state

The tempo model is created on the first performance note using `init_beat_period` (default
`0.5` seconds per score beat), the first score onset and the first performed onset. It tracks
the mapping from score onset to performance onset, beat-period estimate, configured
`lookback` (default `1` beat), and the last 100 score/performance onset samples used for
interpolation.

- A model-selected tempo update uses a matched non-grace note at a non-regressing onset.
- Model-selected grace matches do not advance the tempo anchor or current score position.
  The upstream greedy branch has no grace guard and updates tempo on its selected row;
  that exception is preserved.
- Repeated performance matches at one score onset update that onset's median performance
  time, as in the reference.
- The estimated beat period is clamped to `[0.1, 10.0]` seconds per beat.
- Initial state and update ordering must match the pinned implementation; no score-marked
  tempo replaces OTM's `0.5`-second initial value.

`init_beat_period` must be finite and positive; `lookback` must be a positive integer.
The released `__call__` ignores the stored initial-period override and explicitly supplies
`0.5`; the port preserves this. Tempo history, repeated-onset medians, and SciPy's
weighted-endpoint interpolation use NumPy 2's float32 arithmetic; public timestamps remain
float64.

## Online matcher state

| Field | Type | Meaning |
|---|---|---|
| `current_position` | `usize` | Current unique score-onset index, initially zero and always clamped to valid positions. |
| `input_index` | `usize` | Next performance event index. |
| `aligned_score_ids` | set of strings | Score notes already used. |
| `performance_pitches` | bounded history | Last eight pitches needed for the next token window. |
| `time_since_nn_update` | `usize` | Counts events since the last accepted model jump. |
| `stuck_with_no_options` | `usize` | Consecutive unmatched events; reference recovery jumps forward by up to eight on exactly the tenth event. |
| `tempo_model` | tempo state | Initialized on first note; updated by greedy matches and eligible non-grace model matches. |
| `warping_path` | `Vec<(usize, usize)>` | Internal `(score_onset_index, performance_event_index)` path, one entry per consumed observation. |

For each observation, follow the reference order: increment event counters and record its
pitch; initialize tempo if this is the first event; try an unused pitch match at the current
onset; otherwise infer a token window and apply the bounded prediction, next-onset timing,
jump, and stuck-input rules. Append one internal path entry and advance the event index even
when no score note is matched. The reference matcher stops when its current score position
reaches the last unique onset.

## Public alignment path

The adapter maps each returned internal score-onset index through `score_positions`. The
existing score-follower API then exposes the public path as `(2, T)` rows:

1. performance event times in seconds;
2. score positions in beats.

This public path is distinct from the OTM matcher's internal integer warping path. For an
empty performance, the public path is empty with shape `(2, 0)` and no model inference is
performed.
