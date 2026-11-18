//! `matchmaker bench`: run each method on each manifest piece several times, timing every
//! processor and follower call with a monotonic clock (contracts/benchmark-report.md). No
//! tracing subscriber is installed for this command.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use matchmaker_rs::base::{AlignmentPath, OnlineAlignment, StepDebug};
use matchmaker_rs::features::korzeniowski::KorzeniowskiProcessor;
use matchmaker_rs::features::{Observation, Processor};
use matchmaker_rs::io::audio::AudioStream;
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
    score: Option<PathBuf>,
    performance: PathBuf,
    ground_truth: Option<PathBuf>,
    input_type: Option<String>,
    soundfont: Option<PathBuf>,
    processor: Option<String>,
    kwargs: Option<AudioKwargs>,
    seeds: Option<Vec<u32>>,
    unfold_score: Option<bool>,
    methods: Option<Vec<String>>,
}

#[derive(Deserialize)]
struct AudioKwargs {
    sample_rate: Option<u32>,
    frame_rate: Option<u32>,
    hop_length: Option<usize>,
    n_fft: Option<usize>,
    win_length: Option<usize>,
    num_particles: Option<usize>,
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
    if ms.is_empty() {
        return json!({"mean": null, "median": null, "p95": null, "max": null});
    }
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
pub fn run(manifest_path: &Path, out: &Path, runs: usize, methods: &[String]) -> Result<()> {
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
        let performance = base.join(&p.performance);
        if p.input_type.as_deref() == Some("audio") {
            let score_path = p.score.as_ref().map(|path| base.join(path));
            run_audio_piece(
                p,
                score_path.as_deref(),
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
        let score_path = base.join(
            p.score
                .as_ref()
                .context("MIDI benchmark piece requires a score")?,
        );
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
        for method_name in methods {
            let method: Method = method_name.parse()?;
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
    score_path: Option<&Path>,
    performance: &Path,
    base: &Path,
    out_dir: &Path,
    runs: usize,
    methods: &[String],
    environment: &Value,
    records: &mut Vec<Value>,
) -> Result<()> {
    for method in methods.iter().filter(|method| {
        piece
            .methods
            .as_ref()
            .is_none_or(|allowed| allowed.contains(method))
    }) {
        if method == "korzeniowski" {
            if piece.processor.as_deref() != Some("korzeniowski") {
                continue;
            }
            run_korzeniowski_piece(piece, performance, runs, method, environment, records)?;
            continue;
        }
        let audio_method: AudioMethod = method
            .parse()
            .with_context(|| format!("{method} is not an audio method"))?;
        anyhow::ensure!(
            !matches!(
                audio_method,
                AudioMethod::Skf | AudioMethod::OuterHmm | AudioMethod::PfKorz
            ) || piece.ground_truth.is_some(),
            "audio skf, outerhmm and pfkorz need a ground_truth TSV"
        );
        let score_path = score_path.context("audio benchmark piece requires a score")?;
        let soundfont = piece.soundfont.as_ref().map(|path| base.join(path));
        if audio_method != AudioMethod::PfKorz && soundfont.is_none() {
            anyhow::bail!("audio benchmark piece requires a soundfont path");
        }
        let processor = piece.processor.as_deref().unwrap_or_else(|| {
            AudioMethodConfig::for_method(audio_method)
                .processor()
                .name()
        });
        let seed_values = if audio_method == AudioMethod::PfKorz {
            let seeds = piece.seeds.clone().unwrap_or_else(|| vec![1984]);
            anyhow::ensure!(!seeds.is_empty(), "audio pfkorz seeds must not be empty");
            seeds.into_iter().map(Some).collect::<Vec<_>>()
        } else {
            vec![None]
        };
        let record_start = records.len();
        let mut pfkorz_seed_paths = Vec::new();
        for run in 0..runs {
            for seed in seed_values.iter().copied() {
                let mut config = AudioMethodConfig::for_method(audio_method);
                config.set_processor(processor.parse::<AudioProcessorKind>()?);
                if let Some(kwargs) = &piece.kwargs {
                    if let Some(sample_rate) = kwargs.sample_rate {
                        config.sample_rate = sample_rate;
                    }
                    if let Some(frame_rate) = kwargs.frame_rate {
                        config.frame_rate = frame_rate;
                    }
                    if let Some(hop_length) = kwargs.hop_length {
                        config.hop_length = Some(hop_length);
                    }
                    if let Some(n_fft) = kwargs.n_fft {
                        config.n_fft = Some(n_fft);
                    }
                    if let Some(win_length) = kwargs.win_length {
                        config.win_length = Some(win_length);
                    }
                    if let Some(num_particles) = kwargs.num_particles {
                        config.num_particles = Some(num_particles);
                    }
                }
                if let Some(seed) = seed {
                    config.random_seed = Some(seed);
                }
                config.soundfont_path = soundfont.clone();
                let mut mm = match audio_method {
                    AudioMethod::Arzt => AudioMatchmaker::new_arzt(
                        score_path,
                        config,
                        None,
                        piece.unfold_score.unwrap_or(true),
                    )?,
                    AudioMethod::Dixon => AudioMatchmaker::new_dixon(
                        score_path,
                        config,
                        None,
                        piece.unfold_score.unwrap_or(true),
                    )?,
                    AudioMethod::ArztTempo => AudioMatchmaker::new_arzt_tempo(
                        score_path,
                        config,
                        None,
                        piece.unfold_score.unwrap_or(true),
                    )?,
                    AudioMethod::ArztMultiFold => AudioMatchmaker::new_arzt_multi_fold(
                        score_path,
                        config,
                        None,
                        piece.unfold_score.unwrap_or(true),
                    )?,
                    AudioMethod::Skf => AudioMatchmaker::new_skf(
                        score_path,
                        config,
                        None,
                        piece.unfold_score.unwrap_or(true),
                    )?,
                    AudioMethod::OuterHmm => AudioMatchmaker::new_outerhmm(
                        score_path,
                        config,
                        None,
                        piece.unfold_score.unwrap_or(true),
                    )?,
                    AudioMethod::PfKorz => AudioMatchmaker::new_pfkorz(
                        score_path,
                        config,
                        None,
                        piece.unfold_score.unwrap_or(true),
                    )?,
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
                let ground_truth = if let Some(ground_truth) = &piece.ground_truth {
                    GroundTruth::load(base.join(ground_truth), mm.score(), None)?
                } else {
                    let expected_beats = path
                        .perf_time
                        .iter()
                        .map(|&time| {
                            interpolate_beat(&reference.frame_to_beat, time * reference.frame_rate)
                        })
                        .collect();
                    GroundTruth {
                        perf_sec: path.perf_time.clone(),
                        score_beat: expected_beats,
                    }
                };
                let evaluation = evaluate_alignment(path, &ground_truth);
                let p95_b = Some(p95_beat_error(path, &ground_truth));
                if let Some(seed) = seed
                    && run == 0
                {
                    pfkorz_seed_paths.push((seed, path.clone(), ground_truth.clone()));
                }
                if run == 0 {
                    let filename = seed.map_or_else(
                        || format!("{method}.tsv"),
                        |seed| format!("{method}-seed-{seed}.tsv"),
                    );
                    let path = out_dir
                        .join("paths")
                        .join(&piece.dataset)
                        .join(&piece.piece)
                        .join(filename);
                    crate::write_path_tsv(&path, mm.alignment_path())?;
                }
                records.push(json!({
                "implementation": "rust",
                "implementation_version": env!("CARGO_PKG_VERSION"),
                "input_type": "audio",
                "renderer": if audio_method == AudioMethod::PfKorz { "none" } else { "RustySynth" },
                "renderer_version": if audio_method == AudioMethod::PfKorz { Value::Null } else { json!("1.3.8") },
                "method": method,
                "random_seed": mm.config().random_seed,
                "seed": seed,
                "unfold_score": piece.unfold_score.unwrap_or(true),
                "candidate_count": mm.multi_fold_candidates().len(),
                "selection_window": mm.config().selection_window,
                "switch_margin": mm.config().switch_margin,
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
                "p95_b": p95_b,
                "ground_truth": piece.ground_truth.is_some(),
                "ground_truth_path": piece
                    .ground_truth
                    .as_ref()
                    .map(|path| path.to_string_lossy().into_owned()),
                "environment": environment,
            }));
            }
        }
        if audio_method == AudioMethod::PfKorz {
            annotate_pfkorz_metrics(records, record_start, &pfkorz_seed_paths)?;
        }
    }
    Ok(())
}

fn annotate_pfkorz_metrics(
    records: &mut [Value],
    record_start: usize,
    seed_paths: &[(u32, AlignmentPath, GroundTruth)],
) -> Result<()> {
    anyhow::ensure!(
        !seed_paths.is_empty(),
        "audio pfkorz has no seeded paths to evaluate"
    );
    let gap = seed_paths
        .iter()
        .flat_map(|(_, path, _)| path.perf_time.iter().copied())
        .fold(0.0f64, f64::max)
        + 10.0;
    let mut pooled_path = AlignmentPath::default();
    let mut pooled_ground_truth = GroundTruth::default();
    let mut per_seed = serde_json::Map::new();
    for (index, (seed, path, ground_truth)) in seed_paths.iter().enumerate() {
        per_seed.insert(
            seed.to_string(),
            json!({
                "evaluation": evaluate_alignment(path, ground_truth),
                "p95_b": p95_beat_error(path, ground_truth),
            }),
        );
        let offset = index as f64 * gap;
        pooled_path
            .perf_time
            .extend(path.perf_time.iter().map(|time| time + offset));
        pooled_path
            .score_beat
            .extend(path.score_beat.iter().copied());
        pooled_ground_truth
            .perf_sec
            .extend(ground_truth.perf_sec.iter().map(|time| time + offset));
        pooled_ground_truth
            .score_beat
            .extend(ground_truth.score_beat.iter().copied());
    }
    let pooled_evaluation = evaluate_alignment(&pooled_path, &pooled_ground_truth);
    let pooled_p95 = p95_beat_error(&pooled_path, &pooled_ground_truth);
    for record in &mut records[record_start..] {
        record["per_seed_metrics"] = json!(per_seed);
        record["pooled_evaluation"] = json!(pooled_evaluation);
        record["pooled_p95_b"] = json!(pooled_p95);
    }
    Ok(())
}

fn run_korzeniowski_piece(
    piece: &Piece,
    performance: &Path,
    runs: usize,
    selector: &str,
    environment: &Value,
    records: &mut Vec<Value>,
) -> Result<()> {
    let (sample_rate, hop_length, win_length, n_fft) =
        piece
            .kwargs
            .as_ref()
            .map_or((44_100, 512, 2_048, 4_096), |kwargs| {
                (
                    kwargs.sample_rate.unwrap_or(44_100),
                    kwargs.hop_length.unwrap_or(512),
                    kwargs.win_length.unwrap_or(2_048),
                    kwargs.n_fft.unwrap_or(4_096),
                )
            });
    for run in 0..runs {
        let mut processor = KorzeniowskiProcessor::new(sample_rate, hop_length, win_length, n_fft)?;
        let mut stream = AudioStream::open(
            performance,
            sample_rate,
            hop_length,
            win_length - hop_length,
        )?;
        let frame_count = stream.len();
        stream.prime_with(|frame| {
            let _ = processor.process(frame);
        });
        let mut feature = Vec::with_capacity(frame_count);
        for frame in stream {
            let start = Instant::now();
            processor
                .process(&frame)
                .context("Korzeniowski frame is shorter than win_length")?;
            feature.push(start.elapsed());
        }
        let total: f64 = feature.iter().map(Duration::as_secs_f64).sum();
        let duration = frame_count as f64 * hop_length as f64 / f64::from(sample_rate);
        records.push(json!({
            "implementation": "rust",
            "implementation_version": env!("CARGO_PKG_VERSION"),
            "input_type": "audio",
            "method": selector,
            "processor": "korzeniowski",
            "dataset": piece.dataset,
            "piece": piece.piece,
            "run": run,
            "n_observations": feature.len(),
            "performance_duration_s": duration,
            "latency_ms": {
                "feature": latency(&feature),
                "inference": latency(&[]),
            },
            "total_processing_s": total,
            "real_time_factor": if duration > 0.0 { total / duration } else { 0.0 },
            "evaluation": null,
            "environment": environment,
        }));
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

fn p95_beat_error(path: &AlignmentPath, ground_truth: &GroundTruth) -> f64 {
    let mut errors: Vec<f64> = ground_truth
        .perf_sec
        .iter()
        .zip(&ground_truth.score_beat)
        .filter(|(time, beat)| time.is_finite() && beat.is_finite())
        .map(|(&time, &beat)| {
            (interpolate_beat(&path.score_beat, time_to_frame(path, time)) - beat).abs()
        })
        .collect();
    if errors.is_empty() {
        return f64::NAN;
    }
    errors.sort_by(f64::total_cmp);
    let index = 0.95 * (errors.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    errors[lower] + (errors[upper] - errors[lower]) * (index - lower as f64)
}

fn time_to_frame(path: &AlignmentPath, time: f64) -> f64 {
    let upper = path
        .perf_time
        .partition_point(|&frame_time| frame_time <= time);
    if upper == 0 {
        return 0.0;
    }
    if upper == path.perf_time.len() {
        return (path.perf_time.len() - 1) as f64;
    }
    let lower = upper - 1;
    let span = path.perf_time[upper] - path.perf_time[lower];
    lower as f64 + (time - path.perf_time[lower]) / span
}
