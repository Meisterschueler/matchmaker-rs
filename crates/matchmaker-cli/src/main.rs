//! `matchmaker` command-line interface (see `specs/001-midi-score-following/contracts/cli.md`).

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use matchmaker_rs::matchmaker::{Matchmaker, Method};
use matchmaker_rs::performance::Performance;
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
        /// Do not unfold repeats
        #[arg(long)]
        no_unfold: bool,
        /// Write the alignment path as TSV (perf_sec, score_beat)
        #[arg(long)]
        output: Option<PathBuf>,
        /// Do not print positions
        #[arg(long)]
        quiet: bool,
    },
}

fn main() -> Result<()> {
    // RUST_LOG=matchmaker_rs=debug logs positions and per-stage span timings (stderr)
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_span_events(FmtSpan::CLOSE)
        .with_writer(std::io::stderr)
        .init();
    match Cli::parse().command {
        Command::Follow {
            score,
            performance,
            method,
            tempo,
            no_unfold,
            output,
            quiet,
        } => {
            let mut mm = Matchmaker::new(&score, method, tempo, !no_unfold)?;
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
                let ap = mm.alignment_path();
                let mut tsv = String::from("perf_sec\tscore_beat\n");
                for (t, b) in ap.perf_time.iter().zip(&ap.score_beat) {
                    tsv.push_str(&format!("{t}\t{b}\n"));
                }
                std::fs::write(&path, tsv)
                    .with_context(|| format!("cannot write {}", path.display()))?;
            }
            Ok(())
        }
    }
}
