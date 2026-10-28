//! `matchmaker` command-line interface (see `specs/001-midi-score-following/contracts/cli.md`).

mod bench;

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use matchmaker_rs::base::AlignmentPath;
use matchmaker_rs::matchmaker::{ConfigValue, Matchmaker, Method, MethodConfig, ProcessorKind};
use matchmaker_rs::performance::Performance;
use matchmaker_rs::score::Score;
use matchmaker_rs::utils::eval::{GroundTruth, evaluate_alignment};
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

#[derive(Parser)]
#[command(version, about = "MIDI score following (Rust port of pymatchmaker)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Follow a performance MIDI file and print `<perf_time>\t<score_beat>` per position.
    Follow {
        /// Score: .musicxml, .xml, .mxl, .mid
        score: PathBuf,
        /// Performance MIDI file
        performance: PathBuf,
        /// Score-following method
        #[arg(long, default_value = "pthmm")]
        method: Method,
        /// Tempo in BPM (default: score marking, else 120)
        #[arg(long)]
        tempo: Option<f64>,
        /// Feature processor: pitch, chord_onset (default: pitch)
        #[arg(long)]
        processor: Option<ProcessorKind>,
        /// Method parameter, repeatable (e.g. --set polling_period=0.02 --set patience=20)
        #[arg(long = "set", value_name = "KEY=VALUE")]
        set: Vec<String>,
        /// Do not unfold repeats
        #[arg(long)]
        no_unfold: bool,
        /// Write the alignment path as TSV (perf_sec, score_beat)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Write intermediate data in the golden layout (compare with tools/golden/compare.py)
        #[arg(long)]
        export: Option<PathBuf>,
        /// Do not print positions
        #[arg(long)]
        quiet: bool,
    },
    /// Evaluate an alignment path against ground truth (reference metrics).
    Evaluate {
        /// Score the alignment refers to (note ids for `.match` ground truth)
        score: PathBuf,
        /// Alignment path TSV written by `follow --output`
        alignment: PathBuf,
        /// Ground truth: .match, or TSV with columns perf_sec and score_beat
        ground_truth: PathBuf,
        /// Performance MIDI the follower received (onset times for `.match`)
        #[arg(long)]
        performance: Option<PathBuf>,
        /// Print JSON instead of a table
        #[arg(long)]
        json: bool,
    },
    /// Time and evaluate methods on the pieces of a TOML manifest (benchmark records).
    Bench {
        /// TOML manifest: [[pieces]] with dataset, piece, score, performance, ground_truth
        /// (paths relative to the manifest)
        manifest: PathBuf,
        /// Output JSON file (one record per method, piece and run)
        #[arg(long)]
        out: PathBuf,
        /// Runs per method and piece
        #[arg(long, default_value_t = 5)]
        runs: usize,
        /// Comma-separated methods (default: all available)
        #[arg(long, value_delimiter = ',')]
        methods: Vec<Method>,
    },
}

/// Write an alignment path as TSV with header `perf_sec\tscore_beat`.
fn write_path_tsv(path: &Path, ap: &AlignmentPath) -> Result<()> {
    let mut tsv = String::from("perf_sec\tscore_beat\n");
    for (t, b) in ap.perf_time.iter().zip(&ap.score_beat) {
        tsv.push_str(&format!("{t}\t{b}\n"));
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, tsv).with_context(|| format!("cannot write {}", path.display()))
}

/// `--set KEY=VALUE` values: `none`, booleans, integers and floats, else a string.
fn parse_config_value(text: &str) -> ConfigValue {
    match text {
        "none" | "None" | "null" => ConfigValue::None,
        "true" | "True" => ConfigValue::Bool(true),
        "false" | "False" => ConfigValue::Bool(false),
        _ => text
            .parse()
            .map(ConfigValue::Int)
            .or_else(|_| text.parse().map(ConfigValue::Float))
            .unwrap_or_else(|_| ConfigValue::Str(text.to_string())),
    }
}

/// Defaults of `method` with `--processor` and the `--set` entries applied.
fn method_config(
    method: Method,
    processor: Option<ProcessorKind>,
    set: &[String],
) -> Result<MethodConfig> {
    let mut config = MethodConfig::for_method(method);
    if let Some(processor) = processor {
        config.processor = processor;
    }
    for entry in set {
        let (key, value) = entry
            .split_once('=')
            .with_context(|| format!("--set expects KEY=VALUE, got {entry:?}"))?;
        config.set(method, key, &parse_config_value(value))?;
    }
    Ok(config)
}

fn main() -> Result<()> {
    let command = Cli::parse().command;
    // RUST_LOG=matchmaker_rs=debug logs positions and per-stage span timings (stderr); the
    // benchmark measures without a subscriber
    if !matches!(command, Command::Bench { .. }) {
        tracing_subscriber::fmt()
            .with_env_filter(EnvFilter::from_default_env())
            .with_span_events(FmtSpan::CLOSE)
            .with_writer(std::io::stderr)
            .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
            .init();
    }
    match command {
        Command::Follow {
            score,
            performance,
            method,
            tempo,
            processor,
            set,
            no_unfold,
            output,
            export,
            quiet,
        } => {
            let config = method_config(method, processor, &set)?;
            let mut mm = Matchmaker::with_config(&score, method, config, tempo, !no_unfold)?;
            if export.is_some() {
                mm = mm.record();
            }
            let mut out = std::io::stdout().lock();
            let mut printed = 0;
            // print alignment-path entries as they are computed (US1-4)
            let mut print_new = |mm: &Matchmaker| -> std::io::Result<()> {
                let path = mm.alignment_path();
                if !quiet {
                    for (t, b) in path.perf_time.iter().zip(&path.score_beat).skip(printed) {
                        writeln!(out, "{t}\t{b}")?;
                    }
                }
                printed = path.perf_time.len();
                Ok(())
            };
            for msg in Performance::load(&performance)?.messages() {
                mm.push(msg)?;
                print_new(&mm)?;
            }
            mm.finish();
            print_new(&mm)?;

            if let Some(path) = output {
                write_path_tsv(&path, mm.alignment_path())?;
            }
            if let Some(dir) = export {
                matchmaker_rs::export::write_run(&mm, dir)?;
            }
            Ok(())
        }
        Command::Evaluate {
            score,
            alignment,
            ground_truth,
            performance,
            json,
        } => {
            let score = Score::load(&score, true)?;
            // the path TSV has the ground-truth TSV layout (perf_sec, score_beat)
            let path = GroundTruth::load(&alignment, &score, None)?;
            let path = AlignmentPath {
                perf_time: path.perf_sec,
                score_beat: path.score_beat,
            };
            let gt = GroundTruth::load(&ground_truth, &score, performance.as_deref())?;
            let result = serde_json::to_value(evaluate_alignment(&path, &gt))?;
            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                for (key, value) in result.as_object().into_iter().flatten() {
                    println!("{key:<16}{value}");
                }
            }
            Ok(())
        }
        Command::Bench {
            manifest,
            out,
            runs,
            methods,
        } => {
            let methods = if methods.is_empty() {
                Method::AVAILABLE
                    .iter()
                    .map(|m| m.parse())
                    .collect::<Result<_, _>>()?
            } else {
                methods
            };
            bench::run(&manifest, &out, runs, &methods)
        }
    }
}
