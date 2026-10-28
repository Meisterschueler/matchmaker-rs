# Contract: Rust Library API (`matchmaker_rs`)

Public surface of the core crate for this feature. Names mirror pymatchmaker (II); types are
defined in [data-model.md](../data-model.md). Everything here builds for
`wasm32-unknown-unknown` (file-reading functions compile there but return `Error::Io`).

## Module map (Rust module ← Python path)

| Rust module | Ports |
|---|---|
| `matchmaker` | `matchmaker.matchmaker.Matchmaker` (+ the MIDI part of `methods.yaml`) |
| `base` | `matchmaker.base.OnlineAlignment` |
| `io::stream` | `matchmaker.io.midi.MidiStream` (file/simulation and push modes), `matchmaker.utils.symbolic.Buffer` / framing |
| `features::pitch` | `matchmaker.features.midi.PitchProcessor` |
| `features::chord_onset` | `matchmaker.features.midi.ChordOnsetProcessor` |
| `prob::hidden_markov` | `hiddenmarkov.HiddenMarkovModel.forward_algorithm_step`, `ConstantTransitionModel` |
| `prob::hmm_models` | `matchmaker.prob.hmm` builders (`stable_transition_matrix`, `init_dist`, `compute_discrete_pitch_profiles`, `compute_ioi_matrix`, Bernoulli / ACC observation models) |
| `prob::pthmm` | `matchmaker.prob.hmm.PitchHMM` |
| `prob::hmm` | `matchmaker.prob.hmm.PitchIOIHMM` |
| `prob::outerhmm` | `matchmaker.prob.outer_product_hmm.OuterProductHMM` |
| `utils::tempo_models` | `matchmaker.utils.tempo_models.KalmanTempoModel` |
| `utils::eval` | `matchmaker.utils.eval` (`transfer_positions`, `evaluate_alignment`, SPARC, `resolve_gt`) |
| `score` (+ `musicxml`, `midi`, `timeline`, `unfold`, `merge`, `tempo`) | partitura subset (R6, R7) |
| `performance` | `partitura.io.importmidi.load_performance_midi`, `midi_messages_from_performance` |
| `export` | (new) Python-readable dumps, layout = [golden-data.md](golden-data.md) |

## High-level entry point

```rust
use matchmaker_rs::matchmaker::{Matchmaker, Method};

// score, method, tempo override (None = score marking, else 120), unfold repeats
let mut mm = Matchmaker::new("tests/data/simple_mozart_k265_var1.musicxml", Method::Pthmm, None, true)?;
let positions = mm.run_file("tests/data/simple_mozart_k265_var1.mid")?;  // every yielded beat
let path = mm.alignment_path();         // &AlignmentPath { perf_time, score_beat }
```

`"pthmm".parse::<Method>()`; unknown names → `Error::InvalidMethod` listing the supported
names (US5-2). Methods are added as they are ported; method parameters and processor choice
(FR-011) are added with the methods that need them. Accessors: `score()`, `tempo()`,
`alignment_path()`.

## Incremental (live-source) use — FR-005

```rust
let mut mm = Matchmaker::new(score_path, Method::Pthmm, None, true)?;
for msg in source {                     // any live source: MidiMessage { kind, channel, time }
    for beat in mm.push(msg)? { /* new position */ }
}
mm.finish();                            // closes the last windows, flushes the processor
```

`run_file` is `push` for every message of the file plus `finish`, so both give identical
results. Windows close with the same comparisons as the reference's framing; timestamps must
be non-decreasing (`Error::InvalidParameter` otherwise).

## Low-level pipeline (Stream → Processor → OnlineAlignment)

```rust
let score = Score::load(score_path, true)?;   // true = unfold repeats
let perf  = Performance::load(perf_path)?;
let frames = io::stream::frames(&perf.messages(), StreamMode::Windowed { polling_period: 0.01 });
let mut processor = PitchProcessor::new(true, false);
let mut follower  = PitchHmm::new(&score, PitchHmmConfig::default())?;
for frame in &frames {
    if let Some(obs) = processor.process(frame) {
        follower.call(&obs);
        if !follower.is_still_following() { break; }
    }
}
```

## Evaluation — FR-012

```rust
let gt = utils::eval::GroundTruth::load("tests/data/simple_mozart_k265_var1.match", &score, Some(perf_path))?;
let result: EvaluationResult = utils::eval::evaluate_alignment(path, &gt);
```

## Export — FR-018

```rust
export::write_run(&mm, "out/simple_mozart/pthmm")?;   // .npy + meta.json, see golden-data.md
```

## Instrumentation — FR-017

Spans (target `matchmaker_rs`): `stream` (per frame, `TRACE`), `feature` (per frame, `DEBUG`),
`inference` (per observation, `DEBUG`, fields `index`, `perf_time`, `position`). The library
never reads a clock and installs no subscriber.

## Stability

Pre-1.0: the API may change between features; every change updates this contract.
