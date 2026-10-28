# Contract: Rust Library API (`matchmaker_rs`)

Public surface of the core crate for this feature. Names mirror pymatchmaker (II); types are
defined in [data-model.md](../data-model.md). Everything here builds for
`wasm32-unknown-unknown`; functions taking paths have `*_from_bytes` siblings for wasm/no-fs use.

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
use matchmaker_rs::{Matchmaker, Method, MatchmakerOptions};

let mut mm = Matchmaker::new(
    "tests/data/simple_mozart_k265_var1.musicxml",
    MatchmakerOptions {
        performance: Some("tests/data/simple_mozart_k265_var1.mid".into()),
        method: Method::Pthmm,          // default for MIDI
        ..Default::default()            // processor/piano_range/polling_period from method defaults
    },
)?;
for beat in mm.run()? {                 // yields f64 score positions in beats
    println!("{beat}");
}
let path = mm.alignment_path();         // &AlignmentPath { perf_time, score_beat }
```

`MatchmakerOptions` fields: `performance: Option<PathBuf>`, `method: Method`,
`processor: Option<ProcessorKind>`, `tempo: Option<f64>`, `unfold_score: bool = true`,
`config: MethodConfig` (overrides; unknown keys rejected with `Error::InvalidParameter`).
`Method::from_str("pthmm" | "hmm" | "outerhmm")`; other names → `Error::InvalidMethod` listing
the supported names (US5-2).

Accessors: `score()`, `tempo()`, `score_positions()`, `alignment_path()`,
`build_score_annotations(level, return_type)`, `follower()` (trait object, for debug state).

## Incremental (live-source) use — FR-005

```rust
let mut mm = Matchmaker::new(score_path, MatchmakerOptions { performance: None, ..Default::default() })?;
for (msg, time_s) in source {           // any live source
    if let Some(beat) = mm.push(msg, time_s)? { /* new position */ }
}
mm.finish();                            // flushes processor (chord_onset) and closes windows
```

`push` applies the method's stream mode (windows close when a message beyond the window end
arrives or on `finish`). Timestamps must be non-decreasing (`Error::InvalidParameter` otherwise).

## Low-level pipeline (Stream → Processor → OnlineAlignment)

```rust
let score = Score::load(score_path, &ScoreOptions::default())?;
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
