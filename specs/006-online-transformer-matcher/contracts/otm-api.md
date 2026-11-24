# Contract: Online Transformer Matcher (OTM)

Binding reference: pymatchmaker `v0.5.1`, Parangonar `3.4.1`, and the repository's locked
`tools/golden` environment. The released implementation at that pin takes precedence over
this document if a discrepancy is found; update this contract and its goldens when resolving
one.

## 1. Method selector and supported input

The public method name is exactly `"OTM"` in the Rust library, CLI `--method`, and Python
`Matchmaker(method="OTM")`. It is MIDI-only; audio requests fail with a supported-method
error. It does not become the default MIDI method. `OPTM`, `SLT_OLTW`, and `SL_OLTW` remain
unsupported.

The Rust API exposes the equivalent `Method::Otm` variant. OTM uses the existing
`Stream → Processor → OnlineAlignment` path and returns positions in score beats.

## 2. Defaults and accepted configuration

`DEFAULT_KWARGS["midi"]["OTM"]` and the Python API defaults mirror the `methods.yaml` entry.
The tracker constructor defaults below apply when the user does not pass those keys.

| Key | Default | Constraint |
|---|---:|---|
| `processor` | `"parangonar"` | Fixed for OTM; other processor names are rejected. |
| `polling_period` | `None` | Event mode only; a non-`None` period is rejected. |
| `token_number` | `91` | Must match the supplied artifact. |
| `dim_model` | `64` | Must match the supplied artifact. |
| `dim_class` | `2` | Must match the supplied artifact. |
| `num_heads` | `8` | Must match the supplied artifact and divide `dim_model`. |
| `num_decoder_layers` | `6` | Must match the supplied artifact. |
| `dropout_p` | `0.1` | Finite value in `[0, 1]`; disabled by evaluation mode. |
| `init_beat_period` | `0.5` | Finite positive seconds per score beat. |
| `lookback` | `1` | Positive integer beats. |

`piano_range` and unknown keys are rejected. Model-shape parameters that differ from the
shipped artifact fail explicitly during config/model validation; the runtime must not
silently replace the model or alter its tensor dimensions.

Upstream discrepancy: Parangonar 3.4.1 `__call__` explicitly passes `0.5` to performance
initialization, ignoring its stored `init_beat_period`. This port accepts and validates the
parameter but preserves that behavior. `lookback` does affect tempo estimation.

## 3. Input and step behavior

- Each observation is exactly one positive-velocity note-on with pitch and event time.
- Note-off and non-note messages produce no observation. A frame containing multiple
  positive-velocity note-ons fails with an event-mode error; timestamps do not merge events.
- Score reference construction preserves unique IDs, note pitch/onset/duration and grace
  status. If any IDs are empty or duplicated, regenerate IDs for all score notes as the
  reference adapter does.
- The score pitch sets used as transformer context exclude grace notes. Candidate score
  notes include grace notes; model-selected grace matches do not update the tempo anchor or
  advance the score position. The upstream greedy branch has no grace guard and can update
  tempo for an unmatched grace row at the previous onset when that pitch also belongs to the
  current non-grace context; this exception is preserved.
- Token input has shape `(26, 7)`, uses the reference's last-eight performance pitches,
  at-most-sixteen onset pitch sets, separator `89`, and padding `90`. Preserve the reference
  token order and truncation rules.
- Apply the current-onset greedy match before transformer inference. Otherwise use the
  class-1 sequence argmax, clamp its translated score index, then apply the in-window,
  next-onset tempo check, jump and stuck-input recovery rules in reference order.
- One returned position is emitted per consumed observation, including events that leave the
  current score index unchanged. Stop following at the final score onset.

## 4. Outputs

- `score_positions`: ascending `f32` unique score onset beats.
- `positions`: one score beat per consumed MIDI note event.
- Public `alignment_path`: shape `(2, T)`; row 0 is performance seconds and row 1 is score
  beats, as for the other score-following methods.
- Internal OTM warping path: one `(score_onset_index, performance_event_index)` pair per
  consumed observation. It is not interchangeable with the public time/beat path.
- The empty-performance result has no positions and a public alignment path of shape
  `(2, 0)`.

## 5. Errors

Return a typed library error, surfaced with context by CLI/Python, for:

| Situation | Required behavior |
|---|---|
| Audio input, unknown method, or unsupported Parangonar method | Reject and list supported choices. |
| Empty/invalid score, malformed score or unreadable input | Fail before model inference; never panic. |
| Multiple note-ons in one event frame or non-event polling configuration | Reject with the required event-mode configuration. |
| Pitch maps outside `[0, token_number)` | Reject as an invalid OTM input; do not clamp or wrap. |
| Unknown/invalid option or incompatible model shape | Identify the key/value or expected model shape. |
| Missing, unreadable, corrupt, wrong-version or wrong-checksum model asset | Fail explicitly; never fall back to another method or fetch a model from the network. |

## 6. Diagnostics and golden outputs

The standard export remains Python-readable and compatible with the repository's
`tools/golden/compare.py` conventions. OTM adds, under the exact method directory `OTM/`:

| File | Shape/content |
|---|---|
| `score_positions.npy` | `(N,)`, `float32`; unique score onset beats. |
| `score_notes.npy` | Note rows with IDs, pitch, onset, duration and `is_grace`. |
| `observations_features.npy` | `(T,)`, integer pitch observations in event order. |
| `observations_times.npy` | `(T,)`, `float64` event times in seconds. |
| `token_windows.npy` | `(T, 26, 7)`, integer model inputs; zero-filled when inference is skipped. |
| `token_windows_present.npy` | `(T,)`, bool; identifies steps with a token window. |
| `model_logits.npy` | `(T, 26, 2)`, `float32`; zero-filled where inference was not evaluated. |
| `model_logits_present.npy` | `(T,)`, bool; identifies steps whose logits are meaningful. |
| `tempo_state.npy` | `(T, 2)`, `float64`; current score onset and beat period after each event. |
| `current_index.npy` | `(T,)`, integer; returned score-onset index per event. |
| `alignment_path.npy` | Public `(2, T)` time/beat path. |
| `internal_path.npy` | `(2, T)`, integer score-onset and performance-event indices. |
| `meta.json` | Reference tag, lock hash, model format/version, source, checksum and tolerances. |

When inference is skipped by greedy matching, its token/logit rows are marked absent rather
than presented as valid model output. Golden comparisons cover score notes, processor
observations, tokens, logits where evaluated, tempo state, returned positions, and both path
representations. A small test-only token fixture also compares selected encoder-layer
outputs; those intermediate tensors are not part of routine run exports. A failure identifies
the method, piece, stage and first differing event.

The facade propagates `Result` through checked `Processor::try_process` and
`OnlineAlignment::try_call`, which preserve compatibility with existing implementations.
Direct OTM users should use those checked methods. Legacy infallible calls log invalid input;
`call` returns NaN without adding an alignment entry, rather than fabricating a valid position.

## 7. Distribution and licensing

The runtime has no Python or PyTorch dependency and performs no network download. The
converted checkpoint is shipped only after its redistribution terms and required
attribution are verified. The artifact manifest records the upstream source, checksum,
format version and license/notice. If permission remains unresolved, do not include the
weights in a release; obtain guidance and revise the distribution requirement rather than
substituting weights or returning a success-shaped fallback.
