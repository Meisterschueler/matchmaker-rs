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
fn features_command_exports_raw_spectrum_without_a_score_or_soundfont() {
    let output = std::env::temp_dir().join("matchmaker_cli_raw_spectrum_features");
    let _ = std::fs::remove_dir_all(&output);
    let run = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .arg("features")
        .arg(data("audio/simple_mozart_k265_var1_sf2.wav"))
        .args(["--processor", "raw_spectrum", "--export"])
        .arg(&output)
        .args([
            "--sample-rate",
            "44100",
            "--hop-length",
            "128",
            "--n-fft",
            "512",
        ])
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(output.join("processor/raw_spectrum/features.npy").is_file());
    assert!(output.join("processor/raw_spectrum/times.npy").is_file());
    assert!(output.join("audio/meta.json").is_file());
}

#[test]
fn features_command_exports_cqt_with_its_own_defaults_and_rejects_n_fft() {
    let features = |extra: &[&str], output: &std::path::Path| {
        Command::new(env!("CARGO_BIN_EXE_matchmaker"))
            .arg("features")
            .arg(data("audio/simple_mozart_k265_var1_sf2.wav"))
            .args(["--processor", "cqt", "--export"])
            .arg(output)
            .args(extra)
            .output()
            .unwrap()
    };
    let output = std::env::temp_dir().join("matchmaker_cli_cqt_features");
    let _ = std::fs::remove_dir_all(&output);
    let run = features(&[], &output);
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(output.join("processor/cqt/features.npy").is_file());
    assert!(output.join("processor/cqt/times.npy").is_file());
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("audio/meta.json")).unwrap()).unwrap();
    assert_eq!(
        (meta["sample_rate"].as_u64(), meta["hop_length"].as_u64()),
        (Some(44_100), Some(1_470))
    );
    assert!(meta.get("n_fft").is_none());

    let rejected = std::env::temp_dir().join("matchmaker_cli_cqt_features_rejected");
    let _ = std::fs::remove_dir_all(&rejected);
    let run = features(&["--n-fft", "512"], &rejected);
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr).contains("--n-fft"));
    assert!(!rejected.exists());
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

#[test]
fn audio_ground_truth_writes_reference_metric_json() {
    let output = std::env::temp_dir().join("matchmaker_cli_audio_ground_truth");
    let _ = std::fs::remove_dir_all(&output);
    let ground_truth = data("audio/ground_truth/score_onsets.tsv");
    let ground_truth = ground_truth.to_str().unwrap();
    let output_dir = output.to_str().unwrap();
    let run = follow_audio(
        "arzt",
        &["--ground-truth", ground_truth, "--export", output_dir],
    );
    assert!(
        run.status.success(),
        "{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let actual: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("arzt/evaluation.json")).unwrap())
            .unwrap();
    let evaluation_meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("arzt/evaluation_meta.json")).unwrap())
            .unwrap();
    assert_eq!(
        evaluation_meta["ground_truth_format"],
        "perf_sec/score_beat TSV"
    );
    assert_eq!(evaluation_meta["ground_truth_path"], ground_truth);
    let expected: serde_json::Value = serde_json::from_slice(
        &std::fs::read(data("../golden/audio/ground_truth/evaluation.json")).unwrap(),
    )
    .unwrap();
    for key in [
        "median_b",
        "accuracy_0.05b",
        "accuracy_0.1b",
        "accuracy_0.3b",
    ] {
        assert!(
            (actual[key].as_f64().unwrap() - expected[key].as_f64().unwrap()).abs() <= 0.01,
            "{key}: actual {}, expected {}",
            actual[key],
            expected[key]
        );
    }
}

#[test]
fn malformed_audio_ground_truth_does_not_overwrite_existing_output() {
    let directory = std::env::temp_dir().join("matchmaker_cli_bad_audio_ground_truth");
    std::fs::create_dir_all(&directory).unwrap();
    let ground_truth = directory.join("malformed.tsv");
    let output = directory.join("existing.tsv");
    std::fs::write(&ground_truth, "wrong\theader\n0\t1\n").unwrap();
    std::fs::write(&output, "preserve this file\n").unwrap();
    let ground_truth_arg = ground_truth.to_str().unwrap();
    let output_arg = output.to_str().unwrap();
    let export_arg = directory.join("export").to_str().unwrap().to_owned();
    let run = follow_audio(
        "arzt",
        &[
            "--ground-truth",
            ground_truth_arg,
            "--output",
            output_arg,
            "--export",
            &export_arg,
        ],
    );
    assert!(!run.status.success());
    assert_eq!(
        std::fs::read_to_string(output).unwrap(),
        "preserve this file\n"
    );
}

#[test]
fn audio_follow_rejects_note_id_match_ground_truth_without_midi_performance() {
    let ground_truth = data("simple_mozart_k265_var1.match");
    let ground_truth = ground_truth.to_str().unwrap();
    let export = std::env::temp_dir().join("matchmaker_cli_audio_match_gt");
    let export = export.to_str().unwrap().to_owned();
    let run = follow_audio(
        "arzt",
        &["--ground-truth", ground_truth, "--export", &export],
    );
    assert!(!run.status.success());
    assert!(String::from_utf8_lossy(&run.stderr).contains("separate MIDI performance"));
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
fn audio_skf_defaults_template_modes_and_parameter_errors() {
    let defaults = follow_audio("skf", &[]);
    assert!(
        defaults.status.success(),
        "{}",
        String::from_utf8_lossy(&defaults.stderr)
    );
    assert!(!positions(&defaults).is_empty());

    // Gaussian templates need no SoundFont and are recorded as such.
    let export = std::env::temp_dir().join("matchmaker_cli_audio_skf_export");
    let _ = std::fs::remove_dir_all(&export);
    let gaussian = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .arg("follow")
        .arg(data("simple_mozart_k265_var1.musicxml"))
        .arg(data("audio/simple_mozart_k265_var1_sf2.wav"))
        .args(["--input-type", "audio", "--method", "skf", "--quiet"])
        .args(["--set", "template_mode=gaussian", "--export"])
        .arg(&export)
        .output()
        .unwrap();
    assert!(
        gaussian.status.success(),
        "{}",
        String::from_utf8_lossy(&gaussian.stderr)
    );
    for file in ["score/skf/templates.npy", "skf/state_probability.npy"] {
        assert!(export.join(file).is_file(), "missing skf export {file}");
    }
    let meta = std::fs::read_to_string(export.join("score/meta.json")).unwrap();
    assert!(meta.contains("\"template_mode\":\"gaussian\""));

    for (extra, expected) in [
        (vec!["--set", "template_mode=auto"], "template_mode"),
        (vec!["--set", "window_size=5"], "not used by audio skf"),
        (
            vec!["--set", "distance_func=euclidean"],
            "not used by audio skf",
        ),
        (vec!["--set", "step_size=3"], "step_size"),
        (vec!["--set", "n_fft=64"], "n_fft"),
        (vec!["--processor", "chroma"], "processor"),
    ] {
        let invalid = follow_audio("skf", &extra);
        assert!(!invalid.status.success(), "{extra:?}");
        assert!(
            String::from_utf8_lossy(&invalid.stderr).contains(expected),
            "{extra:?}: {}",
            String::from_utf8_lossy(&invalid.stderr)
        );
    }
    let no_soundfont = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .arg("follow")
        .arg(data("simple_mozart_k265_var1.musicxml"))
        .arg(data("audio/simple_mozart_k265_var1_sf2.wav"))
        .args(["--input-type", "audio", "--method", "skf"])
        .output()
        .unwrap();
    assert!(!no_soundfont.status.success());
    assert!(String::from_utf8_lossy(&no_soundfont.stderr).contains("soundfont_path"));
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
    assert_eq!(printed.len(), 378);

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
fn audio_followers_accept_the_cqt_processor_and_reject_incompatible_options() {
    for (method, frames) in [("arzt", 226), ("dixon", 379)] {
        let default = follow_audio(method, &["--processor", "cqt"]);
        assert!(
            default.status.success(),
            "{method}: {}",
            String::from_utf8_lossy(&default.stderr)
        );
        assert_eq!(positions(&default).len(), frames, "{method}");
        let explicit = follow_audio(
            method,
            &["--set", "processor=cqt", "--set", "sample_rate=44100"],
        );
        assert_eq!(positions(&default), positions(&explicit), "{method}");
    }
    for (extra, key) in [
        (vec!["--processor", "cqt", "--set", "norm=2"], "norm"),
        (vec!["--processor", "cqt", "--set", "n_fft=512"], "n_fft"),
    ] {
        let invalid = follow_audio("arzt", &extra);
        assert!(!invalid.status.success(), "{extra:?}");
        assert!(
            String::from_utf8_lossy(&invalid.stderr).contains(key),
            "{extra:?}"
        );
    }
    for method in ["arzt_tempo", "arzt_multi_fold", "skf"] {
        let invalid = follow_audio(method, &["--processor", "cqt"]);
        assert!(!invalid.status.success(), "{method}");
        assert!(String::from_utf8_lossy(&invalid.stderr).contains("processor"));
    }
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
