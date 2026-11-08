//! Smoke tests for audio benchmark records and output paths.

use std::path::Path;
use std::process::Command;

fn root() -> String {
    format!("{}/../..", env!("CARGO_MANIFEST_DIR")).replace('\\', "/")
}

fn npy_f64(path: &Path) -> Vec<f64> {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..6], b"\x93NUMPY");
    let header_len = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    bytes[10 + header_len..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|value| f64::from_le_bytes(*value))
        .collect()
}

#[test]
fn bench_accepts_raw_spectrum_with_explicit_hop_and_fft() {
    let root_dir = root();
    let dir = std::env::temp_dir().join("matchmaker_cli_bench_raw_spectrum");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = dir.join("audio.toml");
    std::fs::write(
        &manifest,
        format!(
            "[[pieces]]\ndataset = \"audio-raw-spectrum\"\npiece = \"mozart_raw_spectrum\"\n\
             methods = [\"arzt\"]\ninput_type = \"audio\"\nprocessor = \"raw_spectrum\"\n\
             kwargs = {{ sample_rate = 44100, hop_length = 128, n_fft = 512 }}\n\
             score = \"{root_dir}/tests/data/simple_mozart_k265_var1.musicxml\"\n\
             performance = \"{root_dir}/tests/data/audio/simple_mozart_k265_var1_sf2.wav\"\n\
             soundfont = \"{root_dir}/target/audio-assets/UprightPianoKW-small-SF2-20190703/UprightPianoKW-small-20190703.sf2\"\n"
        ),
    )
    .unwrap();
    let out = dir.join("records.json");
    let result = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .arg("bench")
        .arg(&manifest)
        .arg("--out")
        .arg(&out)
        .args(["--runs", "1", "--methods", "arzt"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let records: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    let record = &records.as_array().unwrap()[0];
    assert_eq!(record["processor"], "raw_spectrum");
    assert!(record["n_observations"].as_u64().unwrap() > 0);
    assert!(record["latency_ms"]["feature"]["median"].as_f64().unwrap() > 0.0);
}

#[test]
fn bench_accepts_cqt_with_its_default_rates() {
    let root_dir = root();
    let dir = std::env::temp_dir().join("matchmaker_cli_bench_cqt");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = dir.join("audio.toml");
    std::fs::write(
        &manifest,
        format!(
            "[[pieces]]\ndataset = \"audio-cqt\"\npiece = \"mozart_cqt\"\n\
             methods = [\"arzt\"]\ninput_type = \"audio\"\nprocessor = \"cqt\"\n\
             score = \"{root_dir}/tests/data/simple_mozart_k265_var1.musicxml\"\n\
             performance = \"{root_dir}/tests/data/audio/simple_mozart_k265_var1_sf2.wav\"\n\
             soundfont = \"{root_dir}/target/audio-assets/UprightPianoKW-small-SF2-20190703/UprightPianoKW-small-20190703.sf2\"\n"
        ),
    )
    .unwrap();
    let out = dir.join("records.json");
    let result = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .arg("bench")
        .arg(&manifest)
        .arg("--out")
        .arg(&out)
        .args(["--runs", "1", "--methods", "arzt"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let records: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    let record = &records.as_array().unwrap()[0];
    assert_eq!(record["processor"], "cqt");
    assert_eq!(record["n_observations"], 226);
    assert!(record["latency_ms"]["feature"]["median"].as_f64().unwrap() > 0.0);
}

#[test]
fn bench_accepts_tempo_arzt_and_records_seeded_audio_metrics() {
    let output = std::env::temp_dir().join("matchmaker_cli_bench_audio_tempo");
    std::fs::create_dir_all(&output).unwrap();
    let records_path = output.join("records.json");
    let result = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .arg("bench")
        .arg(Path::new(&root()).join("tools/bench/audio.toml"))
        .arg("--out")
        .arg(&records_path)
        .args(["--runs", "1", "--methods", "arzt_tempo"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let records: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&records_path).unwrap()).unwrap();
    let rows = records.as_array().unwrap();
    assert_eq!(rows.len(), 1);
    let record = rows
        .iter()
        .find(|row| row["piece"] == "mozart_k265_sf2")
        .unwrap();
    assert_eq!(record["method"], "arzt_tempo");
    assert_eq!(record["input_type"], "audio");
    assert_eq!(record["random_seed"], 1984);
    assert!(
        record["latency_ms"]["inference"]["median"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(record["evaluation"]["median_b"].as_f64().is_some());
    assert!(
        output
            .join("paths/audio-bundled/mozart_k265_sf2/arzt_tempo.tsv")
            .is_file()
    );
}

#[test]
fn bench_runs_skf_with_a_ground_truth_and_rejects_pieces_without_one() {
    let output = std::env::temp_dir().join("matchmaker_cli_bench_audio_skf");
    std::fs::create_dir_all(&output).unwrap();
    let records_path = output.join("records.json");
    let result = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .arg("bench")
        .arg(Path::new(&root()).join("tools/bench/audio.toml"))
        .arg("--out")
        .arg(&records_path)
        .args(["--runs", "1", "--methods", "skf"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let records: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&records_path).unwrap()).unwrap();
    let rows = records.as_array().unwrap();
    assert_eq!(rows.len(), 1, "only the skf piece runs skf");
    let record = &rows[0];
    assert_eq!(record["method"], "skf");
    assert_eq!(record["processor"], "raw_spectrum");
    assert_eq!(record["ground_truth"], true);
    assert_eq!(record["candidate_count"], 0);
    assert!(record["evaluation"]["median_b"].as_f64().is_some());
    assert!(output.join("paths/audio-skf/mozart_skf/skf.tsv").is_file());

    let manifest = output.join("no_ground_truth.toml");
    let root = root();
    std::fs::write(
        &manifest,
        format!(
            "[[pieces]]\ndataset = \"d\"\npiece = \"p\"\ninput_type = \"audio\"\n\
score = \"{root}/tests/data/simple_mozart_k265_var1.musicxml\"\n\
performance = \"{root}/tests/data/audio/simple_mozart_k265_var1_sf2.wav\"\n\
soundfont = \"{root}/target/audio-assets/UprightPianoKW-small-SF2-20190703/UprightPianoKW-small-20190703.sf2\"\n"
        ),
    )
    .unwrap();
    let rejected = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .arg("bench")
        .arg(&manifest)
        .arg("--out")
        .arg(output.join("rejected.json"))
        .args(["--runs", "1", "--methods", "skf"])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("ground_truth"));
}

#[test]
fn bench_records_direct_audio_ground_truth_identity_and_metrics() {
    let output = std::env::temp_dir().join("matchmaker_cli_bench_audio_ground_truth");
    std::fs::create_dir_all(&output).unwrap();
    let records_path = output.join("records.json");
    let result = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .arg("bench")
        .arg(Path::new(&root()).join("tools/bench/audio.toml"))
        .arg("--out")
        .arg(&records_path)
        .args(["--runs", "1", "--methods", "arzt"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let records: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&records_path).unwrap()).unwrap();
    let rows = records.as_array().unwrap();
    let record = rows
        .iter()
        .find(|row| row["piece"] == "mozart_score_onsets")
        .unwrap();
    assert_eq!(record["ground_truth"], true);
    assert!(
        record["ground_truth_path"]
            .as_str()
            .unwrap()
            .ends_with("score_onsets.tsv")
    );
    assert!(record["evaluation"]["median_b"].as_f64().is_some());
    assert!(record["evaluation"]["accuracy_0.05b"].as_f64().is_some());
}

#[test]
fn bench_runs_outerhmm_with_ground_truth_and_records_p95() {
    let output = std::env::temp_dir().join("matchmaker_cli_bench_audio_outerhmm");
    std::fs::create_dir_all(&output).unwrap();
    let records_path = output.join("records.json");
    let result = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .arg("bench")
        .arg(Path::new(&root()).join("tools/bench/audio.toml"))
        .arg("--out")
        .arg(&records_path)
        .args(["--runs", "1", "--methods", "outerhmm"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let records: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&records_path).unwrap()).unwrap();
    let rows = records.as_array().unwrap();
    assert_eq!(rows.len(), 1);
    let record = &rows[0];
    assert_eq!(record["method"], "outerhmm");
    assert_eq!(record["processor"], "cqt_spectral_flux");
    assert_eq!(record["ground_truth"], true);
    assert!(record["evaluation"]["median_b"].as_f64().is_some());
    assert!(record["p95_b"].as_f64().is_some());
    assert!(
        output
            .join("paths/audio-outerhmm/mozart_outerhmm/outerhmm.tsv")
            .is_file()
    );
}

#[test]
fn bench_accepts_multi_fold_and_records_repeat_candidates() {
    let output = std::env::temp_dir().join("matchmaker_cli_bench_audio_multi_fold");
    std::fs::create_dir_all(&output).unwrap();
    let records_path = output.join("records.json");
    let result = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .arg("bench")
        .arg(Path::new(&root()).join("tools/bench/audio.toml"))
        .arg("--out")
        .arg(&records_path)
        .args(["--runs", "1", "--methods", "arzt_multi_fold"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let records: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&records_path).unwrap()).unwrap();
    let rows = records.as_array().unwrap();
    assert_eq!(rows.len(), 3);
    let bundled = rows
        .iter()
        .find(|row| row["piece"] == "mozart_k265_sf2")
        .unwrap();
    assert_eq!(bundled["candidate_count"], 1);
    for (piece, frames) in [("repeats_taken", 362), ("repeats_skipped", 202)] {
        let record = rows.iter().find(|row| row["piece"] == piece).unwrap();
        assert_eq!(record["candidate_count"], 3);
        assert_eq!(record["unfold_score"], false);
        assert_eq!(record["selection_window"], 50);
        assert_eq!(record["switch_margin"], 0.05);
        assert_eq!(record["n_observations"], frames);
        assert_eq!(record["ground_truth"], true);
        assert!(
            record["ground_truth_path"]
                .as_str()
                .unwrap()
                .ends_with(&format!("{piece}.tsv").replace("repeats_", "repeat_"))
        );
        assert!(record["evaluation"]["median_b"].as_f64().is_some());
    }
}

#[test]
fn bench_writes_audio_records_and_alignment_path() {
    let root_dir = root();
    let dir = std::env::temp_dir().join("matchmaker_cli_bench_audio");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let manifest = dir.join("audio.toml");
    std::fs::write(
        &manifest,
        format!(
            "[[pieces]]\ndataset = \"audio\"\npiece = \"mozart_lse\"\ninput_type = \"audio\"\nprocessor = \"lse\"\n\
             score = \"{root_dir}/tests/data/simple_mozart_k265_var1.musicxml\"\n\
             performance = \"{root_dir}/tests/data/audio/simple_mozart_k265_var1_sf2.wav\"\n\
             soundfont = \"{root_dir}/target/audio-assets/UprightPianoKW-small-SF2-20190703/UprightPianoKW-small-20190703.sf2\"\n"
        ),
    )
    .unwrap();
    let out = dir.join("records.json");
    let result = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .arg("bench")
        .arg(&manifest)
        .arg("--out")
        .arg(&out)
        .args(["--runs", "1", "--methods", "dixon"])
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );

    let records: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    let record = &records.as_array().unwrap()[0];
    assert_eq!(record["input_type"], "audio");
    assert_eq!(record["method"], "dixon");
    assert_eq!(record["processor"], "lse");
    assert_eq!(record["n_observations"], 378);
    assert!(record["real_time_factor"].as_f64().unwrap() > 0.0);
    assert!(record["latency_ms"]["feature"]["median"].as_f64().unwrap() > 0.0);
    assert!(
        record["latency_ms"]["inference"]["median"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(record["evaluation"]["mean_b"].as_f64().is_some());
    assert!(record["evaluation"]["accuracy_0.3b"].as_f64().is_some());
    let path = dir.join("paths/audio/mozart_lse/dixon.tsv");
    assert!(path.is_file());
    let lines = std::fs::read_to_string(&path).unwrap();
    assert_eq!(lines.lines().count(), 379);
    let rows: Vec<(f64, f64)> = lines
        .lines()
        .skip(1)
        .map(|line| {
            let (time, beat) = line.split_once('\t').unwrap();
            (time.parse().unwrap(), beat.parse().unwrap())
        })
        .collect();
    let expected_path =
        npy_f64(&Path::new(&root()).join("tests/golden/audio/dixon_lse/alignment_path.npy"));
    let expected_frame_count = expected_path.len() / 2;
    let shared = rows.len().min(expected_frame_count);
    // reference-faithful score framing gives the reference's step count exactly
    assert_eq!(rows.len(), expected_frame_count);
    let expected_times = &expected_path[..expected_frame_count];
    let expected_beats = &expected_path[expected_frame_count..];
    assert!(
        rows.iter()
            .zip(expected_times)
            .all(|((time, _), expected)| { (time - expected).abs() <= 1e-6 })
    );
    let errors: Vec<f64> = rows[..shared]
        .iter()
        .zip(&expected_beats[..shared])
        .map(|((_, actual), expected)| (actual - expected).abs())
        .collect();
    assert!(errors.iter().copied().fold(0.0, f64::max) <= 0.6);
    assert!(errors.iter().filter(|&&error| error <= 0.6).count() as f64 / shared as f64 >= 0.99);
}
