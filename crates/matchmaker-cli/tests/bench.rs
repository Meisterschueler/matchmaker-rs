//! CLI `bench` (T046): records per contracts/benchmark-report.md and the run-0 alignment path.

use std::process::Command;

#[test]
fn bench_writes_records_and_path() {
    let dir = std::env::temp_dir().join("matchmaker_cli_bench");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let data = format!("{}/../../tests/data", env!("CARGO_MANIFEST_DIR")).replace('\\', "/");
    let manifest = dir.join("m.toml");
    std::fs::write(
        &manifest,
        format!(
            "[[pieces]]\ndataset = \"bundled\"\npiece = \"simple\"\n\
             score = \"{data}/simple_mozart_k265_var1.musicxml\"\n\
             performance = \"{data}/simple_mozart_k265_var1.mid\"\n\
             ground_truth = \"{data}/simple_mozart_k265_var1.match\"\n"
        ),
    )
    .unwrap();
    let out = dir.join("records.json");
    let status = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .arg("bench")
        .arg(&manifest)
        .arg("--out")
        .arg(&out)
        .args(["--runs", "2", "--methods", "pthmm"])
        .status()
        .unwrap();
    assert!(status.success());

    let records: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&out).unwrap()).unwrap();
    let records = records.as_array().unwrap();
    assert_eq!(records.len(), 2);
    let r = &records[1];
    assert_eq!(
        (r["method"].as_str(), r["run"].as_u64()),
        (Some("pthmm"), Some(1))
    );
    // golden pthmm path of this piece has 67 entries, one per observation
    assert_eq!(r["n_observations"], 67);
    assert!(r["latency_ms"]["inference"]["median"].as_f64().unwrap() > 0.0);
    assert!((r["evaluation"]["smoothness"].as_f64().unwrap() - -8.8134).abs() <= 1e-4);
    let path = std::fs::read_to_string(dir.join("paths/bundled/simple/pthmm.tsv")).unwrap();
    assert_eq!(path.lines().count(), 68);
}
