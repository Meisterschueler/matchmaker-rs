//! CLI `follow` (US1-4, FR-013): prints positions, writes the TSV path, rejects unknown methods.

use std::process::Command;

fn data(name: &str) -> String {
    format!("{}/../../tests/data/{name}", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn follow_prints_and_writes_path() {
    let tsv = std::env::temp_dir().join("matchmaker_cli_follow.tsv");
    let out = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .args(["follow", &data("simple_mozart_k265_var1.musicxml")])
        .arg(data("simple_mozart_k265_var1.mid"))
        .arg("--output")
        .arg(&tsv)
        .output()
        .unwrap();
    assert!(out.status.success());
    let printed = String::from_utf8(out.stdout).unwrap();
    let written = std::fs::read_to_string(&tsv).unwrap();
    assert_eq!(written.lines().next(), Some("perf_sec\tscore_beat"));
    // golden pthmm path of this piece has 67 entries
    assert_eq!(printed.lines().count(), 67);
    assert_eq!(
        written.lines().skip(1).collect::<Vec<_>>(),
        printed.lines().collect::<Vec<_>>()
    );
}

#[test]
fn npy_output_is_rejected() {
    let out = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .args(["follow", &data("simple_mozart_k265_var1.musicxml")])
        .arg(data("simple_mozart_k265_var1.mid"))
        .args(["--quiet", "--output", "path.npy"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("TSV"));
}

#[test]
fn unknown_method_lists_available() {
    let out = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .args(["follow", "score.musicxml", "perf.mid", "--method", "arzt"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("pthmm"));
}

#[test]
fn export_with_ground_truth_writes_evaluation() {
    let dir = std::env::temp_dir().join("matchmaker_cli_export");
    let _ = std::fs::remove_dir_all(&dir);
    let status = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .args(["follow", &data("simple_mozart_k265_var1.musicxml")])
        .arg(data("simple_mozart_k265_var1.mid"))
        .arg("--quiet")
        .arg("--export")
        .arg(&dir)
        .arg("--ground-truth")
        .arg(data("simple_mozart_k265_var1.match"))
        .status()
        .unwrap();
    assert!(status.success());
    let evaluation = std::fs::read_to_string(dir.join("pthmm/evaluation.json")).unwrap();
    assert!(evaluation.contains("\"accuracy_0.1b\""), "{evaluation}");
}
