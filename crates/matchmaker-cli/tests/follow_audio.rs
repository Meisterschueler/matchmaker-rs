//! CLI integration tests for prerecorded audio following.

use std::path::PathBuf;
use std::process::{Command, Output};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn data(name: &str) -> PathBuf {
    root().join("tests/data").join(name)
}

fn soundfont() -> PathBuf {
    root()
        .join("target/audio-assets/UprightPianoKW-small-SF2-20190703")
        .join("UprightPianoKW-small-20190703.sf2")
}

fn follow_audio(method: &str, extra: &[&str]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_matchmaker"));
    command
        .arg("follow")
        .arg(data("simple_mozart_k265_var1.musicxml"))
        .arg(data("audio/simple_mozart_k265_var1_sf2.wav"))
        .args(["--input-type", "audio", "--method", method])
        .arg("--soundfont-path")
        .arg(soundfont())
        .args(extra);
    command.output().unwrap()
}

fn positions(output: &Output) -> Vec<(f64, f64)> {
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(|line| {
            let (time, beat) = line.split_once('\t').unwrap();
            (time.parse().unwrap(), beat.parse().unwrap())
        })
        .collect()
}

#[test]
fn audio_arzt_defaults_and_output_path() {
    let output_path = std::env::temp_dir().join("matchmaker_cli_follow_audio.tsv");
    let default = follow_audio("arzt", &["--output", output_path.to_str().unwrap()]);
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );
    let printed = positions(&default);
    assert_eq!(printed.len(), 226);

    let explicit = follow_audio(
        "arzt",
        &[
            "--processor",
            "chroma",
            "--set",
            "sample_rate=44100",
            "--set",
            "frame_rate=30",
        ],
    );
    assert!(
        explicit.status.success(),
        "{}",
        String::from_utf8_lossy(&explicit.stderr)
    );
    assert_eq!(printed, positions(&explicit));

    let written = std::fs::read_to_string(output_path).unwrap();
    assert_eq!(written.lines().next(), Some("perf_sec\tscore_beat"));
    assert_eq!(written.lines().count(), printed.len() + 1);
}

#[test]
fn audio_arzt_reports_soundfont_and_parameter_errors() {
    let mut missing_option = Command::new(env!("CARGO_BIN_EXE_matchmaker"));
    let output = missing_option
        .arg("follow")
        .arg(data("simple_mozart_k265_var1.musicxml"))
        .arg(data("audio/simple_mozart_k265_var1_sf2.wav"))
        .args(["--input-type", "audio", "--method", "arzt"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("soundfont_path"));

    let mut missing_soundfont = Command::new(env!("CARGO_BIN_EXE_matchmaker"));
    let output = missing_soundfont
        .arg("follow")
        .arg(data("simple_mozart_k265_var1.musicxml"))
        .arg(data("audio/simple_mozart_k265_var1_sf2.wav"))
        .args([
            "--input-type",
            "audio",
            "--method",
            "arzt",
            "--soundfont-path",
        ])
        .arg(data("missing.sf2"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("missing.sf2"));

    let invalid = follow_audio("arzt", &["--set", "step_size=0"]);
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("step_size"));
}

#[test]
fn audio_dixon_defaults_and_output_path() {
    let output_path = std::env::temp_dir().join("matchmaker_cli_follow_dixon_audio.tsv");
    let default = follow_audio("dixon", &["--output", output_path.to_str().unwrap()]);
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );
    let printed = positions(&default);
    assert_eq!(printed.len(), 379);

    let explicit = follow_audio(
        "dixon",
        &[
            "--processor",
            "lse",
            "--set",
            "sample_rate=44100",
            "--set",
            "frame_rate=50",
            "--set",
            "window_size=10",
            "--set",
            "max_run_count=3",
        ],
    );
    assert!(
        explicit.status.success(),
        "{}",
        String::from_utf8_lossy(&explicit.stderr)
    );
    assert_eq!(printed, positions(&explicit));

    let written = std::fs::read_to_string(output_path).unwrap();
    assert_eq!(written.lines().next(), Some("perf_sec\tscore_beat"));
    assert_eq!(written.lines().count(), printed.len() + 1);

    let invalid = follow_audio("dixon", &["--set", "max_run_count=0"]);
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("max_run_count"));
}
