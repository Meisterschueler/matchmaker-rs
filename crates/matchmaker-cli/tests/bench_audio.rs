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
fn bench_writes_audio_records_and_alignment_path() {
    let dir = std::env::temp_dir().join("matchmaker_cli_bench_audio");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let root_dir = root();
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
    assert_eq!(record["n_observations"], 379);
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
    assert_eq!(lines.lines().count(), 380);
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
    assert!(rows.len().abs_diff(expected_frame_count) <= 1);
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
