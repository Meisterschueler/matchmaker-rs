//! `matchmaker` command-line interface (see `specs/001-midi-score-following/contracts/cli.md`).

mod bench;

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use matchmaker_rs::base::AlignmentPath;
use matchmaker_rs::matchmaker::{
    AudioDistance, AudioMatchmaker, AudioMethod, AudioMethodConfig, AudioProcessorKind,
    ConfigValue, Matchmaker, Method, MethodConfig,
};
use matchmaker_rs::performance::Performance;
use matchmaker_rs::score::Score;
use matchmaker_rs::utils::eval::{GroundTruth, evaluate_alignment};
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

#[derive(Parser)]
#[command(
    version,
    about = "MIDI and prerecorded audio score following (Rust port of pymatchmaker)"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Follow a MIDI or prerecorded audio performance and print `<perf_time>\t<score_beat>` per position.
    Follow {
        /// Score: .musicxml, .xml, .mxl, .mid
        score: PathBuf,
        /// Performance MIDI file or supported audio file
        performance: PathBuf,
        /// Performance input type
        #[arg(long, value_enum, default_value_t = InputType::Midi)]
        input_type: InputType,
        /// Score-following method
        #[arg(long, default_value = "pthmm")]
        method: Method,
        /// Local SF2 SoundFont required for audio score rendering
        #[arg(long)]
        soundfont_path: Option<PathBuf>,
        /// Tempo in BPM (default: score marking, else 120)
        #[arg(long)]
        tempo: Option<f64>,
        /// Feature processor: MIDI pitch/chord_onset or audio chroma/lse
        #[arg(long)]
        processor: Option<String>,
        /// Method parameter, repeatable (e.g. --set polling_period=0.02 --set patience=20;
        /// arzt: step_size, distance_func; dixon: window_size, max_run_count, distance_func)
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
        /// Ground truth (.match or TSV): with --export, also writes evaluation.json
        #[arg(long, requires = "export")]
        ground_truth: Option<PathBuf>,
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
        /// TOML manifest: `[[pieces]]` with dataset, piece, score, performance, ground_truth
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum InputType {
    Midi,
    Audio,
}

/// Write an alignment path as TSV with header `perf_sec\tscore_beat`.
fn write_path_tsv(path: &Path, ap: &AlignmentPath) -> Result<()> {
    anyhow::ensure!(
        path.extension().is_none_or(|e| e != "npy"),
        "--output writes TSV, not NPY: {}",
        path.display()
    );
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
fn method_config(method: Method, processor: Option<&str>, set: &[String]) -> Result<MethodConfig> {
    let mut config = MethodConfig::for_method(method);
    if let Some(processor) = processor {
        config.processor = processor.parse()?;
    }
    for entry in set {
        let (key, value) = entry
            .split_once('=')
            .with_context(|| format!("--set expects KEY=VALUE, got {entry:?}"))?;
        config.set(method, key, &parse_config_value(value))?;
    }
    Ok(config)
}

fn audio_config(
    method: AudioMethod,
    processor: Option<&str>,
    soundfont_path: Option<PathBuf>,
    set: &[String],
) -> Result<AudioMethodConfig> {
    let mut config = AudioMethodConfig::for_method(method);
    config.soundfont_path = soundfont_path;
    if let Some(processor) = processor {
        config.processor = processor.parse::<AudioProcessorKind>()?;
    }
    for entry in set {
        let (key, text) = entry
            .split_once('=')
            .with_context(|| format!("--set expects KEY=VALUE, got {entry:?}"))?;
        let value = parse_config_value(text);
        match key {
            "processor" => match value {
                ConfigValue::Str(name) => config.processor = name.parse()?,
                _ => anyhow::bail!("--set processor expects chroma or lse"),
            },
            "sample_rate" => config.sample_rate = audio_u32(key, &value)?,
            "frame_rate" => config.frame_rate = audio_u32(key, &value)?,
            "hop_length" => config.hop_length = audio_optional_usize(key, &value)?,
            "norm" => {
                config.norm = match &value {
                    ConfigValue::None => None,
                    ConfigValue::Str(value)
                        if value.eq_ignore_ascii_case("inf")
                            || value.eq_ignore_ascii_case("infinity") =>
                    {
                        Some(f64::INFINITY)
                    }
                    ConfigValue::Int(value) => Some(*value as f64),
                    ConfigValue::Float(value) => Some(*value),
                    _ => anyhow::bail!("--set norm expects a positive number or none"),
                }
            }
            "window_size" => config.window_size = audio_f64(key, &value)?,
            "start_window_size" => config.start_window_size = audio_optional_f64(key, &value)?,
            "step_size" => config.step_size = audio_optional_usize(key, &value)?,
            "max_run_count" => config.max_run_count = audio_optional_usize(key, &value)?,
            "distance_func" => match value {
                ConfigValue::Str(name) => config.distance_func = name.parse::<AudioDistance>()?,
                _ => anyhow::bail!("--set distance_func expects manhattan or euclidean"),
            },
            _ => anyhow::bail!("unknown audio parameter {key:?}"),
        }
    }
    config.validate(method)?;
    Ok(config)
}

fn audio_u32(key: &str, value: &ConfigValue) -> Result<u32> {
    u32::try_from(audio_usize(key, value)?).with_context(|| format!("{key} is out of range"))
}

fn audio_usize(key: &str, value: &ConfigValue) -> Result<usize> {
    match value {
        ConfigValue::Int(value) => {
            usize::try_from(*value).with_context(|| format!("{key} must be a non-negative integer"))
        }
        _ => anyhow::bail!("{key} expects an integer"),
    }
}

fn audio_optional_usize(key: &str, value: &ConfigValue) -> Result<Option<usize>> {
    match value {
        ConfigValue::None => Ok(None),
        _ => audio_usize(key, value).map(Some),
    }
}

fn audio_f64(key: &str, value: &ConfigValue) -> Result<f64> {
    match value {
        ConfigValue::Int(value) => Ok(*value as f64),
        ConfigValue::Float(value) => Ok(*value),
        _ => anyhow::bail!("{key} expects a number"),
    }
}

fn audio_optional_f64(key: &str, value: &ConfigValue) -> Result<Option<f64>> {
    match value {
        ConfigValue::None => Ok(None),
        _ => audio_f64(key, value).map(Some),
    }
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
            input_type,
            method,
            soundfont_path,
            tempo,
            processor,
            set,
            no_unfold,
            output,
            export,
            ground_truth,
            quiet,
        } => match input_type {
            InputType::Midi => {
                anyhow::ensure!(
                    soundfont_path.is_none(),
                    "--soundfont-path requires --input-type audio"
                );
                let config = method_config(method, processor.as_deref(), &set)?;
                let mut mm = Matchmaker::with_config(&score, method, config, tempo, !no_unfold)?;
                if export.is_some() {
                    mm = mm.record();
                }
                let mut out = std::io::stdout().lock();
                let mut printed = 0;
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
                    matchmaker_rs::export::write_run(&mm, &dir)?;
                    if let Some(ground_truth) = ground_truth {
                        let gt = GroundTruth::load(&ground_truth, mm.score(), Some(&performance))?;
                        let result = evaluate_alignment(mm.alignment_path(), &gt);
                        let file = dir.join(method.name()).join("evaluation.json");
                        std::fs::write(&file, serde_json::to_string_pretty(&result)?)
                            .with_context(|| format!("cannot write {}", file.display()))?;
                    }
                }
                Ok(())
            }
            InputType::Audio => {
                anyhow::ensure!(export.is_none(), "audio export is not implemented yet");
                anyhow::ensure!(
                    ground_truth.is_none(),
                    "audio ground-truth evaluation is not implemented yet"
                );
                let audio_method: AudioMethod = method.name().parse()?;
                let config =
                    audio_config(audio_method, processor.as_deref(), soundfont_path, &set)?;
                let mut mm = match audio_method {
                    AudioMethod::Arzt => {
                        AudioMatchmaker::new_arzt(&score, config, tempo, !no_unfold)?
                    }
                    AudioMethod::Dixon => {
                        AudioMatchmaker::new_dixon(&score, config, tempo, !no_unfold)?
                    }
                };
                mm.run_file(&performance)?;
                let path = mm.alignment_path();
                if !quiet {
                    let mut out = std::io::stdout().lock();
                    for (time, beat) in path.perf_time.iter().zip(&path.score_beat) {
                        writeln!(out, "{time}\t{beat}")?;
                    }
                }
                if let Some(path_output) = output {
                    write_path_tsv(&path_output, path)?;
                }
                Ok(())
            }
        },
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
