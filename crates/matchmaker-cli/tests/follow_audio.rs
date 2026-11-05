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
    follow_audio_files(
        &data("simple_mozart_k265_var1.musicxml"),
        &data("audio/simple_mozart_k265_var1_sf2.wav"),
        method,
        extra,
    )
}

fn follow_audio_files(
    score: &std::path::Path,
    performance: &std::path::Path,
    method: &str,
    extra: &[&str],
) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_matchmaker"));
    command
        .arg("follow")
        .arg(score)
        .arg(performance)
        .args(["--input-type", "audio", "--method", method])
        .arg("--soundfont-path")
        .arg(soundfont())
        .args(extra);
    command.output().unwrap()
}

#[test]
fn audio_arzt_multi_fold_defaults_and_repeat_mode() {
    let defaults = follow_audio("arzt_multi_fold", &[]);
    assert!(
        defaults.status.success(),
        "{}",
        String::from_utf8_lossy(&defaults.stderr)
    );
    assert!(!positions(&defaults).is_empty());

    let export = std::env::temp_dir().join("matchmaker_cli_audio_multi_fold_export");
    let _ = std::fs::remove_dir_all(&export);
    let export_arg = export.to_str().unwrap();
    let repeat = follow_audio_files(
        &data("fixture_repeats_voltas.musicxml"),
        &data("audio/arzt_multi_fold/repeat_taken.wav"),
        "arzt_multi_fold",
        &["--no-unfold", "--export", export_arg],
    );
    assert!(
        repeat.status.success(),
        "{}",
        String::from_utf8_lossy(&repeat.stderr)
    );
    let path = positions(&repeat);
    assert!(!path.is_empty());
    assert!(path.iter().all(|(_, beat)| (0.0..=14.0).contains(beat)));
    for file in [
        "arzt_multi_fold/hypothesis_metadata.json",
        "arzt_multi_fold/selected_hypothesis.npy",
        "arzt_multi_fold/candidate_current_frames.npy",
        "arzt_multi_fold/candidate_local_costs.npy",
        "arzt_multi_fold/candidate_mean_costs.npy",
        "arzt_multi_fold/candidates/0/frame_to_unfolded_beat.npy",
        "arzt_multi_fold/candidates/0/frame_to_notated_beat.npy",
    ] {
        assert!(
            export.join(file).is_file(),
            "missing multi-fold export {file}"
        );
    }
    let metadata = std::fs::read_to_string(export.join("arzt_multi_fold/meta.json")).unwrap();
    assert!(metadata.contains("\"selection_window\":50"));
    assert!(metadata.contains("\"switch_margin\":0.05"));

    for (parameter, value) in [
        ("selection_window", "0"),
        ("selection_window", "none"),
        ("switch_margin", "1.1"),
        ("switch_margin", "none"),
    ] {
        let invalid = follow_audio(
            "arzt_multi_fold",
            &["--set", &format!("{parameter}={value}")],
        );
        assert!(!invalid.status.success());
        assert!(String::from_utf8_lossy(&invalid.stderr).contains(parameter));
    }
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
fn audio_arzt_tempo_defaults_seed_and_parameter_errors() {
    let default = follow_audio("arzt_tempo", &[]);
    assert!(
        default.status.success(),
        "{}",
        String::from_utf8_lossy(&default.stderr)
    );
    assert!(!positions(&default).is_empty());
    let explicit = follow_audio(
        "arzt_tempo",
        &[
            "--processor",
            "chroma",
            "--set",
            "random_seed=1984",
            "--set",
            "frame_rate=30",
            "--set",
            "step_size=3",
        ],
    );
    assert!(
        explicit.status.success(),
        "{}",
        String::from_utf8_lossy(&explicit.stderr)
    );
    assert_eq!(positions(&default), positions(&explicit));
    let invalid = follow_audio("arzt_tempo", &["--set", "random_seed=-1"]);
    assert!(!invalid.status.success());
    assert!(String::from_utf8_lossy(&invalid.stderr).contains("random_seed"));
}

#[test]
fn processor_overrides_use_processor_norm_defaults() {
    let arzt_lse = follow_audio("arzt", &["--processor", "lse"]);
    let arzt_lse_explicit = follow_audio("arzt", &["--processor", "lse", "--set", "norm=2"]);
    assert!(arzt_lse.status.success());
    assert!(arzt_lse_explicit.status.success());
    assert_eq!(positions(&arzt_lse), positions(&arzt_lse_explicit));

    let dixon_chroma = follow_audio("dixon", &["--set", "norm=inf", "--set", "processor=chroma"]);
    let dixon_chroma_explicit =
        follow_audio("dixon", &["--set", "processor=chroma", "--set", "norm=inf"]);
    assert!(dixon_chroma.status.success());
    assert!(dixon_chroma_explicit.status.success());
    assert_eq!(positions(&dixon_chroma), positions(&dixon_chroma_explicit));
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

#[test]
fn audio_export_writes_all_pipeline_stages() {
    let output = std::env::temp_dir().join("matchmaker_cli_audio_export");
    let _ = std::fs::remove_dir_all(&output);
    let run = follow_audio("arzt", &["--quiet", "--export", output.to_str().unwrap()]);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    for file in [
        "audio/meta.json",
        "audio/frame_times.npy",
        "processor/chroma/features.npy",
        "processor/chroma/times.npy",
        "score/audio/features.npy",
        "score/audio/frame_to_beat.npy",
        "arzt/observations_features.npy",
        "arzt/positions.npy",
        "arzt/current_index.npy",
        "arzt/alignment_path.npy",
    ] {
        assert!(output.join(file).is_file(), "missing export stage {file}");
    }
}
