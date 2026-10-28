//! CLI `evaluate` (FR-012): `follow --output` then `evaluate --json` prints the reference's keys.

use std::process::Command;

fn data(name: &str) -> String {
    format!("{}/../../tests/data/{name}", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn evaluate_prints_reference_keys() {
    let tsv = std::env::temp_dir().join("matchmaker_cli_evaluate.tsv");
    let score = data("simple_mozart_k265_var1.musicxml");
    let perf = data("simple_mozart_k265_var1.mid");
    let follow = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .args(["follow", &score, &perf, "--quiet", "--output"])
        .arg(&tsv)
        .status()
        .unwrap();
    assert!(follow.success());
    let out = Command::new(env!("CARGO_BIN_EXE_matchmaker"))
        .args(["evaluate", &score])
        .arg(&tsv)
        .args([
            &data("simple_mozart_k265_var1.match"),
            "--performance",
            &perf,
            "--json",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let keys: Vec<&str> = json
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    assert_eq!(
        keys,
        [
            "mean_b",
            "median_b",
            "std_b",
            "accuracy_0.05b",
            "accuracy_0.1b",
            "accuracy_0.3b",
            "accuracy_0.5b",
            "accuracy_1b",
            "accuracy_2b",
            "skewness",
            "kurtosis",
            "smoothness"
        ]
    );
    // reference evaluation.json of this piece (pthmm)
    assert!((json["smoothness"].as_f64().unwrap() - -8.8134).abs() <= 1e-4);
}
