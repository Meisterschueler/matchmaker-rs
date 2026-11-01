//! `matchmaker bench`: run each method on each manifest piece several times, timing every
//! processor and follower call with a monotonic clock (contracts/benchmark-report.md). No
//! tracing subscriber is installed for this command.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use matchmaker_rs::base::{AlignmentPath, OnlineAlignment, StepDebug};
use matchmaker_rs::features::{Observation, Processor};
use matchmaker_rs::io::stream::{Frame, MidiMessage};
use matchmaker_rs::matchmaker::{
    AudioMatchmaker, AudioMethod, AudioMethodConfig, AudioProcessorKind, Matchmaker, Method,
    MethodConfig,
};
use matchmaker_rs::performance::Performance;
use matchmaker_rs::score::Score;
use matchmaker_rs::utils::eval::{GroundTruth, evaluate_alignment};
use serde::Deserialize;
use serde_json::{Value, json};

/// `[[pieces]]` of the manifest; paths are relative to the manifest file.
#[derive(Deserialize)]
struct Manifest {
    pieces: Vec<Piece>,
}

#[derive(Deserialize)]
struct Piece {
    dataset: String,
    piece: String,
    score: PathBuf,
    performance: PathBuf,
    ground_truth: Option<PathBuf>,
    input_type: Option<String>,
    soundfont: Option<PathBuf>,
    processor: Option<String>,
}

type Times = Rc<RefCell<Vec<Duration>>>;

/// Times `process` for frames that yield an observation (feature latency).
struct TimedProcessor {
    inner: Box<dyn Processor>,
    times: Times,
}

impl Processor for TimedProcessor {
    fn process(&mut self, frame: &Frame) -> Option<Observation> {
        let start = Instant::now();
        let obs = self.inner.process(frame);
        let elapsed = start.elapsed();
        if obs.is_some() {
            self.times.borrow_mut().push(elapsed);
        }
        obs
    }

    fn flush_remaining(&mut self) -> Option<Observation> {
        self.inner.flush_remaining()
    }
}

/// Times every follower call (inference latency).
struct TimedFollower {
    inner: Box<dyn OnlineAlignment>,
    times: Times,
}

impl OnlineAlignment for TimedFollower {
    fn call(&mut self, obs: &Observation) -> f64 {
        let start = Instant::now();
        let position = self.inner.call(obs);
        self.times.borrow_mut().push(start.elapsed());
        position
    }

    fn is_still_following(&self) -> bool {
        self.inner.is_still_following()
    }

    fn alignment_path(&self) -> &AlignmentPath {
        self.inner.alignment_path()
    }

    fn score_positions(&self) -> &[f32] {
        self.inner.score_positions()
    }

    fn debug_state(&self) -> StepDebug {
        self.inner.debug_state()
    }
}

/// Latency summary in milliseconds; percentiles interpolate linearly like `np.percentile`.
fn latency(times: &[Duration]) -> Value {
    let mut ms: Vec<f64> = times.iter().map(|d| d.as_secs_f64() * 1e3).collect();
    ms.sort_by(f64::total_cmp);
    let percentile = |q: f64| {
        let x = q * (ms.len().max(1) - 1) as f64;
        let (lo, hi) = (x.floor() as usize, x.ceil() as usize);
        ms.get(lo)
            .zip(ms.get(hi))
            .map(|(a, b)| a + (b - a) * (x - lo as f64))
    };
    json!({
        "mean": ms.iter().sum::<f64>() / ms.len() as f64,
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "max": ms.last(),
    })
}

/// Writes the relative margin of the best state over the runner-up per path entry of one
/// untimed run: states of repeated score chords tie exactly, and the reference's BLAS rounding
/// decides which one it reports (tools/bench/report.py treats such steps as equal).
fn write_gaps(file: &Path, score: &Path, method: Method, messages: &[MidiMessage]) -> Result<()> {
    let mut mm = Matchmaker::new(score, method, None, true)?.record();
    for &msg in messages {
        mm.push(msg)?;
    }
    mm.finish();
    let steps = mm.recording().map_or(&[][..], |r| &r.steps[..]);
    // arzt and dixon have no state distribution, hence no ties
    if steps.iter().all(|s| s.debug.probabilities.is_empty()) {
        return Ok(());
    }
    let gaps: String = steps
        .iter()
        // continuations of a chord add no path entry
        .filter(|s| s.debug.is_continuation != Some(true))
        .map(|s| {
            let (mut best, mut second) = (0.0f64, 0.0f64);
            for &p in &s.debug.probabilities {
                if p > best {
                    (best, second) = (p, best);
                } else if p > second {
                    second = p;
                }
            }
            format!("{}\n", (best - second) / best)
        })
        .collect();
    std::fs::write(file, gaps).with_context(|| format!("cannot write {}", file.display()))
}

/// Run the benchmark and write the records (JSON array) to `out`; the alignment path of run 0
/// goes to `<out dir>/paths/<dataset>/<piece>/<method>.tsv` (margins in `<method>.gap.tsv`).
pub fn run(manifest_path: &Path, out: &Path, runs: usize, methods: &[Method]) -> Result<()> {
    let text = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("cannot read {}", manifest_path.display()))?;
    let manifest: Manifest = toml::from_str(&text)
        .with_context(|| format!("invalid manifest {}", manifest_path.display()))?;
    let base = manifest_path.parent().unwrap_or(Path::new("."));
    let out_dir = out.parent().unwrap_or(Path::new("."));
    let environment = json!({
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "commit": std::env::var("GITHUB_SHA").ok(),
        "timestamp": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
    });

    let mut records = Vec::new();
    for p in &manifest.pieces {
        let (score_path, performance) = (base.join(&p.score), base.join(&p.performance));
        if p.input_type.as_deref() == Some("audio") {
            run_audio_piece(
                p,
                &score_path,
                &performance,
                base,
                out_dir,
                runs,
                methods,
                &environment,
                &mut records,
            )?;
            continue;
        }
        let messages = Performance::load(&performance)?.messages();
        let duration = messages.last().map_or(0.0, |m| m.time);
        let gt = match &p.ground_truth {
            Some(gt) => {
                let score = Score::load(&score_path, true)?;
                Some(GroundTruth::load(
                    base.join(gt),
                    &score,
                    Some(&performance),
                )?)
            }
            None => None,
        };
        for &method in methods {
            let processor = MethodConfig::for_method(method).processor.name();
            for run in 0..runs {
                let (feature, inference) = (Times::default(), Times::default());
                let mut mm = Matchmaker::new(&score_path, method, None, true)?.wrap_stages(
                    |processor, follower| {
                        (
                            Box::new(TimedProcessor {
                                inner: processor,
                                times: feature.clone(),
                            }),
                            Box::new(TimedFollower {
                                inner: follower,
                                times: inference.clone(),
                            }),
                        )
                    },
                );
                for &msg in &messages {
                    mm.push(msg)?;
                }
                mm.finish();

                let (feature, inference) = (feature.borrow(), inference.borrow());
                let total: f64 = feature
                    .iter()
                    .chain(inference.iter())
                    .map(Duration::as_secs_f64)
                    .sum();
                if run == 0 {
                    let path = out_dir
                        .join("paths")
                        .join(&p.dataset)
                        .join(&p.piece)
                        .join(format!("{}.tsv", method.name()));
                    crate::write_path_tsv(&path, mm.alignment_path())?;
                    write_gaps(
                        &path.with_extension("gap.tsv"),
                        &score_path,
                        method,
                        &messages,
                    )?;
                }
                records.push(json!({
                    "implementation": "rust",
                    "implementation_version": env!("CARGO_PKG_VERSION"),
                    "method": method.name(),
                    "processor": processor,
                    "dataset": p.dataset,
                    "piece": p.piece,
                    "run": run,
                    "n_observations": inference.len(),
                    "performance_duration_s": duration,
                    "latency_ms": {"feature": latency(&feature), "inference": latency(&inference)},
                    "total_processing_s": total,
                    "real_time_factor": total / duration,
                    "evaluation": gt.as_ref().map(|gt| evaluate_alignment(mm.alignment_path(), gt)),
                    "environment": environment,
                }));
            }
        }
    }
    std::fs::write(out, serde_json::to_string_pretty(&records)?)
        .with_context(|| format!("cannot write {}", out.display()))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_audio_piece(
    piece: &Piece,
    score_path: &Path,
    performance: &Path,
    base: &Path,
    out_dir: &Path,
    runs: usize,
    methods: &[Method],
    environment: &Value,
    records: &mut Vec<Value>,
) -> Result<()> {
    let soundfont = piece
        .soundfont
        .as_ref()
        .map(|path| base.join(path))
        .context("audio benchmark piece requires a soundfont path")?;
    for &method in methods {
        let audio_method: AudioMethod = method
            .name()
            .parse()
            .with_context(|| format!("{} is not an audio method", method.name()))?;
        let processor = piece
            .processor
            .as_deref()
            .unwrap_or_else(|| AudioMethodConfig::for_method(audio_method).processor.name());
        for run in 0..runs {
            let mut config = AudioMethodConfig::for_method(audio_method);
            config.processor = processor.parse::<AudioProcessorKind>()?;
            config.soundfont_path = Some(soundfont.clone());
            let mut mm = match audio_method {
                AudioMethod::Arzt => AudioMatchmaker::new_arzt(score_path, config, None, true)?,
                AudioMethod::Dixon => AudioMatchmaker::new_dixon(score_path, config, None, true)?,
            }
            .record();
            mm.run_file(performance)?;
            let recording = mm.recording().context("audio recording was not enabled")?;
            let feature = &recording.feature_durations;
            let inference = &recording.inference_durations;
            let total: f64 = feature
                .iter()
                .chain(inference.iter())
                .map(Duration::as_secs_f64)
                .sum();
            let hop_length = mm
                .config()
                .hop_length
                .unwrap_or(mm.config().sample_rate as usize / mm.config().frame_rate as usize);
            let duration = recording.frame_times.last().copied().map_or(0.0, |time| {
                time + hop_length as f64 / f64::from(mm.config().sample_rate)
            });
            let path = mm.alignment_path();
            let reference = mm.reference();
            let expected_beats = path
                .perf_time
                .iter()
                .map(|&time| {
                    interpolate_beat(&reference.frame_to_beat, time * reference.frame_rate)
                })
                .collect();
            let evaluation = evaluate_alignment(
                path,
                &GroundTruth {
                    perf_sec: path.perf_time.clone(),
                    score_beat: expected_beats,
                },
            );
            if run == 0 {
                let path = out_dir
                    .join("paths")
                    .join(&piece.dataset)
                    .join(&piece.piece)
                    .join(format!("{}.tsv", method.name()));
                crate::write_path_tsv(&path, mm.alignment_path())?;
            }
            records.push(json!({
                "implementation": "rust",
                "implementation_version": env!("CARGO_PKG_VERSION"),
                "input_type": "audio",
                "renderer": "RustySynth",
                "renderer_version": "1.3.8",
                "method": method.name(),
                "processor": processor,
                "dataset": piece.dataset,
                "piece": piece.piece,
                "run": run,
                "n_observations": inference.len(),
                "performance_duration_s": duration,
                "latency_ms": {"feature": latency(feature), "inference": latency(inference)},
                "total_processing_s": total,
                "real_time_factor": if duration > 0.0 { total / duration } else { 0.0 },
                "evaluation": evaluation,
                "environment": environment,
            }));
        }
    }
    Ok(())
}

fn interpolate_beat(frame_to_beat: &[f64], frame: f64) -> f64 {
    let last = frame_to_beat.len().saturating_sub(1);
    let position = frame.clamp(0.0, last as f64);
    let before = position.floor() as usize;
    let after = position.ceil() as usize;
    let fraction = position - before as f64;
    frame_to_beat[before] + fraction * (frame_to_beat[after] - frame_to_beat[before])
}
