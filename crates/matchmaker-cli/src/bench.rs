//! `matchmaker bench`: run each method on each manifest piece several times, timing every
//! processor and follower call with a monotonic clock (contracts/benchmark-report.md). No
//! tracing subscriber is installed for this command.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use matchmaker_rs::base::{AlignmentPath, OnlineAlignment};
use matchmaker_rs::features::{Observation, Processor};
use matchmaker_rs::io::stream::Frame;
use matchmaker_rs::matchmaker::{Matchmaker, Method};
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

/// Run the benchmark and write the records (JSON array) to `out`; the alignment path of run 0
/// goes to `<out dir>/paths/<dataset>/<piece>/<method>.tsv`.
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
            let processor = match method {
                Method::Pthmm => "pitch",
            };
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
