# Data Model: Audio Online Time Warping

This feature adds audio entities while reusing the existing score, tempo, and alignment-path models. Audio values and state are defined to make reference comparisons stage-specific.

## Audio Clip

The decoded performance file before framing.

| Field | Type | Validation / meaning |
|---|---|---|
| `samples` | sequence of `f32` | Mono PCM normalized to the decoder's float range; empty input is valid and produces no observations. |
| `sample_rate` | positive integer | Output rate after resampling; defaults to 44,100 Hz. The source rate is retained in run metadata. |
| `source_channels` | positive integer | Original channel count; supported multi-channel input is averaged to mono as in the reference loader. |
| `source_format` | supported container | WAV, FLAC, MP3, or Ogg/Vorbis in the initial contract; unsupported containers fail with a typed error. |

## Audio Frame

One fixed-hop slice of the normalized signal, with left context for the processor's analysis window.

| Field | Type | Validation / meaning |
|---|---|---|
| `index` | nonnegative integer | Zero-based emitted frame index. |
| `samples` | sequence of `f32` | Context plus current hop; first context is zero-padded. |
| `perf_time` | `f64` seconds | `index * hop_length / sample_rate`, matching reference emission timestamps. |

At end of file, a partial hop is zero-padded. Empty clips emit no frames. The first processed block primes the overlap cache and is not enqueued; a one-hop-or-shorter file therefore yields no observations, while later emitted frames start at performance time zero. This preserves reference frame counts and timestamps.

## Audio Feature Observation

The processor output consumed by a frame-level follower.

| Field | Type | Validation / meaning |
|---|---|---|
| `features` | `Vec<f32>` | One frame vector: 12 chroma bins or the LSE frequency-map dimension (84 at 44.1 kHz). All values must be finite except where reference semantics explicitly allow a degenerate vector. |
| `perf_time` | `f64` seconds | Timestamp of the frame that produced the observation. |

Processor state (`LSE` previous spectrum) is reset before processing a separate stream or score reference.

## Score Audio Reference

The synthesized symbolic score aligned against the performance audio.

| Field | Type | Validation / meaning |
|---|---|---|
| `features` | sequence of feature vectors | Produced using the same processor/configuration as the performance. Feature dimensions must agree. |
| `frame_to_beat` | sequence of `f64` beats | One entry per score feature frame; nondecreasing and derived from the score beat map, tempo, and frame rate. |
| `score_positions` | sequence of `f32` beats | Ascending unique score onset positions; non-empty scores provide at least one position. |
| `sample_rate` | positive integer | Configured render rate; defaults to 44,100 Hz. |
| `frame_rate` | positive finite `f64` | Derived from explicit `hop_length` if supplied; otherwise method default (30 for Arzt, 50 for Dixon). |

Rendered audio includes the reference-compatible leading silence before the first score onset and a 0.1-second tail after the final onset. RustySynth output is not required to equal FluidSynth waveform samples; alignment behavior is checked against reference metrics.

## Frame-Level Follower State

Each follower stores the score-reference sequence, input frame index, score-frame frontier, method-specific dynamic-programming state, and a reusable alignment path.

- **Arzt**: two rolling cost columns over the active score-frame window; `window_size` and `start_window_size` are seconds converted with frame rate, `step_size` is a maximum reference-frame advance per input observation, and default local distance is Manhattan.
- **Dixon**: sparse accumulated costs over the bounded reference/input frontier; `window_size` is seconds converted to frames, `max_run_count` limits repeated one-sided advances, and default distance is Euclidean.
- **Alignment path**: existing `AlignmentPath`, with performance times in seconds and score beats from the selected reference frame's `frame_to_beat` mapping.

## Error and Empty-State Rules

- Unsupported container, corrupt audio, missing/unreadable SoundFont, and invalid audio parameters return typed errors carrying the path or parameter name.
- Empty decoded audio returns an empty alignment path without panic.
- Silent feature frames use the reference's distance behavior; they do not produce out-of-range indices or invalid beat values.
- A score with fewer than two reference/onset positions follows the reference run-loop stop behavior and returns a valid empty or initialized path.
