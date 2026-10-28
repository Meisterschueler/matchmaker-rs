//! `RUST_LOG=matchmaker_rs=debug` logs the position per observation and the span timings of
//! each stage (US6-2, FR-016).

use std::process::Command;

fn data(name: &str) -> String {
    format!("{}/../../tests/data/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn follow(log: Option<&str>) -> String {
    let mut command = Command::new(env!("CARGO_BIN_EXE_matchmaker"));
    command
        .args(["follow", &data("simple_mozart_k265_var1.musicxml")])
        .arg(data("simple_mozart_k265_var1.mid"))
        .arg("--quiet")
        .env_remove("RUST_LOG");
    if let Some(log) = log {
        command.env("RUST_LOG", log);
    }
    String::from_utf8(command.output().unwrap().stderr).unwrap()
}

#[test]
fn debug_log_has_positions_and_stage_timings() {
    let log = follow(Some("matchmaker_rs=debug"));
    // the golden pthmm path of this piece has 67 entries, one position line each
    assert_eq!(log.lines().filter(|l| l.contains("position=")).count(), 67);
    for stage in ["inference", "feature"] {
        assert!(
            log.lines()
                .any(|l| l.contains(stage) && l.contains("time.busy")),
            "no {stage} span timing in the log"
        );
    }
}

#[test]
fn trace_log_has_stream_span() {
    let log = follow(Some("matchmaker_rs=trace"));
    assert!(
        log.lines()
            .any(|l| l.contains("stream") && l.contains("time.busy"))
    );
}

#[test]
fn silent_by_default() {
    assert_eq!(follow(None), "");
}
