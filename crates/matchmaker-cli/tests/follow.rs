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
        .args(["follow", "score.musicxml", "perf.mid", "--method", "foo"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("pthmm") && stderr.contains("dixon"),
        "{stderr}"
    );
}

/// `(perf_time, score_beat)` of `tests/golden/simple_mozart_k265_var1/<method>/alignment_path.npy`.
fn golden_path(method: &str) -> Vec<(f64, f64)> {
    let file = data(&format!(
        "../golden/simple_mozart_k265_var1/{method}/alignment_path.npy"
    ));
    let bytes = std::fs::read(file).unwrap();
    // npy 1.0: magic (6), version (2), header length (2, little endian), header, f64 data
    let header = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let values: Vec<f64> = bytes[10 + header..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_le_bytes(*c))
        .collect();
    let (perf_time, score_beat) = values.split_at(values.len() / 2);
    perf_time
        .iter()
        .copied()
        .zip(score_beat.iter().copied())
        .collect()
}

fn follow(method: &str, set: &[&str]) -> std::process::Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_matchmaker"));
    command
        .args(["follow", &data("simple_mozart_k265_var1.musicxml")])
        .arg(data("simple_mozart_k265_var1.mid"))
        .args(["--method", method]);
    for entry in set {
        command.args(["--set", entry]);
    }
    command.output().unwrap()
}

#[test]
fn oltw_methods_print_the_golden_path() {
    for method in ["arzt", "dixon"] {
        let out = follow(method, &[]);
        assert!(out.status.success(), "{method}");
        let printed: Vec<(f64, f64)> = String::from_utf8(out.stdout)
            .unwrap()
            .lines()
            .map(|line| {
                let (t, b) = line.split_once('\t').unwrap();
                (t.parse().unwrap(), b.parse().unwrap())
            })
            .collect();
        let golden = golden_path(method);
        assert_eq!(printed.len(), golden.len(), "{method}");
        for (i, (p, g)) in printed.iter().zip(&golden).enumerate() {
            assert!(
                (p.0 - g.0).abs() < 1e-6 && (p.1 - g.1).abs() < 1e-6,
                "{method} entry {i}: {p:?} vs {g:?}"
            );
        }
    }
}

#[test]
fn oltw_parameters_are_set_per_method() {
    assert!(
        follow("arzt", &["step_size=3", "distance_func=euclidean"])
            .status
            .success()
    );
    assert!(
        follow("dixon", &["window_size=20", "max_run_count=2"])
            .status
            .success()
    );
    let out = follow("arzt", &["window_size=5"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("window_size"));
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
