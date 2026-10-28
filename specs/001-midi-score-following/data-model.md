# Data Model: MIDI Score Following

Phase 1 output of `/speckit-plan`. Rust names are given with the Python object they port.
Types: `f32`/`f64` follow [research.md R4](research.md#r4-numeric-types-and-promotion-parity-at-the-bit-level).

## Score side

### `ScoreNote` — one row of partitura's `note_array()`

| Field | Type | Source / rule |
|---|---|---|
| `id` | `String` | MusicXML note id (kept, `force_note_ids="keep"`), missing ids filled as partitura does; unfolded copies get `-<repetition>` |
| `pitch` | `u8` | MIDI pitch from step/alter/octave (0–127) |
| `onset_beat` | `f32` | `beat_map(onset_div)` (computed in `f64`, stored as `f32`) |
| `duration_beat` | `f32` | `beat_map(onset_div + duration_div) − onset_beat` |
| `onset_quarter`, `duration_quarter` | — | not ported: no consumer (the golden files still contain them) |
| `onset_div`, `duration_div` | `i32` | merged-part divisions (after `lcm` rescaling) |
| `voice` | `i32` | after `merge_parts` reassignment |

Validation: `pitch ≤ 127`; `duration_div ≥ 0`; ids unique after unfolding (error otherwise).
Only the first note of a tie chain appears; grace notes appear (`duration_div` may be 0).

### `Score` — port of the merged `partitura.score.Part`

| Field | Type | Notes |
|---|---|---|
| `notes` | `Vec<ScoreNote>` | stable-sorted by `onset_beat`, then `pitch` |
| `tempo` | `Option<f64>` | quarter-note BPM from R7; `None` → caller default 120 |
| `source` | `PathBuf` | for error messages |
| `unfolded` | `bool` | whether repeat unfolding was applied (it may fall back, as in the reference) |

Derived (pure functions, no stored state):
- `unique_onsets() -> Vec<f32>` — sorted unique `onset_beat` (`np.unique`).
- `chords() -> Vec<BTreeSet<u8>>` — pitch set per unique onset (`get_chords_from_score`).

Construction: `Score::load(path, unfold)` (`unfold = true` in pymatchmaker) dispatches on extension
(`.musicxml`/`.xml`/`.mxl` → MusicXML; `.mid`/`.midi` → MIDI score); other extensions →
`Error::UnsupportedFormat`.

### `ScoreStates` — follower state space

| Field | Type | Notes |
|---|---|---|
| `positions` | `Vec<f32>` | unique onsets, or with insertions `[o0, o0 + 0.5·(o1 − o0), o1, …]` computed in `f32` (`np.insert` + `np.diff`) |
| `has_insertions` | `bool` | even index = onset state, odd = inserted state |

## Performance side

### `PerformedNote` — port of a `PerformedPart.notes` entry

`id: String ("n<k>")`, `pitch: u8`, `velocity: u8`, `channel: u8`, `track: usize`,
`note_on: f64`, `note_off: f64` (seconds, recomputed from ticks via the tempo map).

### `MidiMessage` — the subset of `mido.Message` the stream carries

```text
enum MidiMessageKind { NoteOn { note, velocity }, NoteOff { note }, ControlChange { control, value }, ProgramChange { program } }
struct MidiMessage { kind, channel: u8, time: f64 }   // time = absolute seconds
```

A `NoteOn` with velocity 0 is treated as note-off by every consumer.

### `Frame` — one stream item (`InputMIDIFrame`)

`messages: Vec<MidiMessage>`, `time: f64` (window centre `(k + 0.5)·pp` in windowed mode;
message time in event mode).

### `StreamMode`

`Event` (one message per frame) or `Windowed { polling_period: f64 }`. Per-method defaults:
`pthmm`, `hmm` → `Windowed(0.01)`; `outerhmm` → `Windowed(0.001)`.

## Features

### `Observation` — processor output (`(features, perf_time)`)

| Field | Type | Notes |
|---|---|---|
| `features` | `Features` | `PitchVector(Vec<f32>)` (88 or 128 binary) or `PitchList(Vec<f32>)` (pitch − shift) |
| `perf_time` | `f64` | first note-on time in the frame (`pitch`), first note-on of the chord (`chord_onset`) |

### `Processor` trait — port of `matchmaker.features.processor.Processor`

```text
fn process(&mut self, frame: &Frame) -> Option<Observation>
fn flush_remaining(&mut self) -> Option<Observation>   // default: None; chord_onset overrides
fn reset(&mut self)
```

| Processor (`key`) | Config | State |
|---|---|---|
| `PitchProcessor` (`pitch`) | `piano_range: bool`, `return_pitch_list: bool` | none |
| `ChordOnsetProcessor` (`chord_onset`) | `piano_range: bool`, `chord_threshold: f64 = 0.035` | `pending_notes: Vec<u8>`, `pending_time: Option<f64>` |

## Followers

### `OnlineAlignment` trait — port of `matchmaker.base.OnlineAlignment`

```text
fn step(&mut self, features: &Features)                 // updates current_index
fn call(&mut self, obs: &Observation) -> f64            // default: step + record path entry
fn is_still_following(&self) -> bool                    // default: current_index < n_states − 1
fn current_position(&self) -> f64                       // score_positions[current_index] as f64
fn alignment_path(&self) -> &AlignmentPath
fn debug_state(&self) -> StepDebug                      // for export (FR-018)
```

`run(observations) -> impl Iterator<Item = f64>` drives `call` with the method's stop rule.

### Follower state per method

| Method | Struct | Parameters (defaults) | Mutable state |
|---|---|---|---|
| `pthmm` | `PitchHmm` | `has_insertions = true`, `piano_range = true`, transition `scale = 1.0`, init `scale = 10` | `forward_variable: Option<Vec<f64>>`, `current_index` |
| `hmm` | `PitchIoiHmm` | as `pthmm` + `ioi_precision = 1`, `KalmanTempoModel { init_beat_period = 0.6, init_score_onset = min onset, trans_par = 1, trans_var = 0.03, obs_var = 0.0213, init_var = 1 }` | as `pthmm` + `prev_perf_time: Option<f64>`, tempo model state |
| `outerhmm` | `OuterProductHmm` | `transitions = [(−3,.00509),(−2,.00516),(−1,.00886),(0,.01342),(1,.94531),(2,.00610),(3,.00073)]`, `D1 = D2 = 3`, pitch-error probs, `S = r = 1/N`, `other_prob = 1e-6`, `patience = 10`, `ioi_threshold = 0.035` | `state_probabilities`, `pre_chord_probs`, `current_chord: Vec<f32>`, `prev_perf_time`, `is_continuation`, `same_state_counter` |

Shared building blocks (in `prob::hidden_markov` / `prob::hmm_models`):
`BandedTransitionMatrix { rows: Vec<(start: usize, values: Vec<f64>)> }` (non-zero band per row,
[R2](research.md#r2-hmm-forward-step-python-hiddenmarkov)), `init_probabilities: Vec<f64>`,
`pitch_profiles: Array2<f64> (N × 88|128)`, `b_table`/`log_b: Array2<f64> (N × 128 | 88)`.

### `KalmanTempoModel` — port of `matchmaker.utils.tempo_models.KalmanTempoModel`

`beat_period`, `var_est`, `est_onset`, `prev_perf_onset`, `prev_score_onset`, `counter`, with
the per-operand precision noted in R4. Transition: first `update_beat_period` only initialises;
afterwards predict/update as in the reference.

### `MethodConfig` — what `Matchmaker(kwargs=…)` controls

`method: Method { Pthmm, Hmm, OuterHmm }`, `processor: ProcessorKind { Pitch, ChordOnset }`,
`piano_range: bool = true`, `stream_mode: StreamMode` (per-method default), method-specific
overrides (e.g. `has_insertions`, `patience`, `transitions`, `pitch_error_probs`, Kalman
parameters). Unknown keys → `Error::InvalidParameter { method, key, allowed }`.

## Results

### `AlignmentPath` — `(2, T)` array of the reference

`perf_time: Vec<f64>`, `score_beat: Vec<f64>`; `perf_time` non-decreasing. Empty for an empty
performance. One entry per consumed non-continuation observation (R1).

### `StepDebug` — exported per step (FR-018)

`perf_time: f64`, `position: f64`, `current_index: usize`, `probabilities: Option<Vec<f64>>`
(forward variable / Viterbi distribution), `beat_period: Option<f64>` (`hmm`),
`is_continuation: Option<bool>` (`outerhmm`).

### `GroundTruth`

`perf_sec: Vec<f64>`, `score_beat: Vec<f64>`, loaded from TSV (`perf_sec`, `score_beat`
columns) or `.match` (R8). Non-finite rows are kept for beat metrics and removed for SPARC
anchors, as in the reference.

### `EvaluationResult` — `evaluate_alignment()` output

`mean_b`, `median_b`, `std_b`, `accuracy_0.05b`, `accuracy_0.1b`, `accuracy_0.3b`,
`accuracy_0.5b`, `accuracy_1b`, `accuracy_2b`, `skewness`, `kurtosis` (all `f64`, 4 decimals),
`smoothness: Option<f64>`. Serialised with exactly these keys.

### `LatencySample` / `BenchmarkRecord` — harness only (CLI / Python), never in the core

Per observation: `feature_s: f64`, `inference_s: f64`. Per run: implementation (`rust` |
`python`), method, piece, dataset, `n_observations`, `performance_duration_s`, latency summary
(mean/median/p95/max per stage), real-time factor = total processing time / performance
duration, `EvaluationResult`. Schema: [contracts/benchmark-report.md](contracts/benchmark-report.md).

## Errors — `matchmaker_rs::Error` (`thiserror`)

| Variant | Raised when |
|---|---|
| `Io { path, source }` | file cannot be read |
| `UnsupportedFormat { path, extension }` | unknown score/performance/GT extension |
| `MusicXml { path, message }` | XML/`.mxl` cannot be parsed or is score-timewise-invalid |
| `Midi { path, message }` | MIDI cannot be parsed |
| `InvalidScore { path, reason }` | e.g. no notes, duplicate ids after unfolding |
| `InvalidMethod { input_type, method, available }` | unknown method (lists options) |
| `InvalidParameter { method, key, allowed }` | unknown or out-of-range parameter |
| `GroundTruth { path, reason }` | missing columns, unsupported `.match` path (names the TSV converter) |
| `Export { path, source }` | export directory/file cannot be written |

No public function panics on user input (IV).

## State transitions

```mermaid
stateDiagram-v2
    [*] --> Built: Matchmaker::new(score, performance, method, config)
    Built --> Following: run() / push(message)
    Following --> Following: observation → position
    Following --> Finished: STREAM_END | last state reached (pthmm, hmm) | patience exceeded (outerhmm)
    Finished --> [*]: alignment_path(), export(), evaluate()
```

`outerhmm` additionally alternates between *new chord* and *continuation* inside `Following`
(gap to previous note `< 0.035 s` → continuation, recomputed from `pre_chord_probs`, no path
entry).
