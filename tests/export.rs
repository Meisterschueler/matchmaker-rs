//! `export::write_run` (T066): every method writes the files of contracts/golden-data.md, and
//! each file equals its golden counterpart where the reference has one.
//!
//! Tolerances as in the contract: positions and alignment path `< 1e-6`, probabilities
//! `rtol 1e-12`, `beat_period` `rtol 1e-9`, everything else exact. Messages are compared as a
//! multiset (the reference sorts them with an unstable `argsort`, research R5).

mod common;

use std::path::{Path, PathBuf};

use common::{Ctx, NpyArray, assert_close, data, golden, json, npy};
use matchmaker_rs::Error;
use matchmaker_rs::export::write_run;
use matchmaker_rs::matchmaker::{Matchmaker, Method};

const PIECE: &str = "simple_mozart_k265_var1";

fn files(dir: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            found.extend(files(&path));
        } else {
            found.push(path);
        }
    }
    found.sort();
    found
}

/// Records sorted by (time, rest) for the fields in `names`, time last in the record.
fn sorted_rows(a: &NpyArray, names: &[&str]) -> Vec<Vec<f64>> {
    let columns: Vec<Vec<f64>> = names.iter().map(|n| a.field(n)).collect();
    let mut rows: Vec<Vec<f64>> = (0..a.len())
        .map(|r| columns.iter().map(|c| c[r]).collect())
        .collect();
    rows.sort_by(|a, b| a.partial_cmp(b).unwrap());
    rows
}

fn compare(ctx: &Ctx, rel: &str, exported: &Path, expected: &Path) {
    if rel.ends_with(".json") {
        if rel.ends_with("note_ids.json") {
            assert_eq!(json(exported), json(expected), "{rel}: note ids differ");
        }
        return;
    }
    let (actual, reference) = (npy(exported), npy(expected));
    assert_eq!(
        actual.shape, reference.shape,
        "[{} / {rel}] shape",
        ctx.method
    );
    if rel.ends_with("messages.npy") {
        let all = [
            "kind", "channel", "note", "velocity", "control", "value", "program", "time",
        ];
        let (a, e) = (sorted_rows(&actual, &all), sorted_rows(&reference, &all));
        assert!(a == e, "[{} / {rel}] messages differ", ctx.method);
    } else if rel.ends_with("note_array.npy") {
        for field in [
            "onset_beat",
            "duration_beat",
            "onset_div",
            "duration_div",
            "pitch",
            "voice",
        ] {
            let file = format!("{rel}:{field}");
            let ctx = Ctx::new(&ctx.method, &ctx.piece, &file);
            assert_close(
                &ctx,
                &reference.field(field),
                &actual.field(field),
                0.0,
                0.0,
            );
        }
    } else {
        let (rtol, atol) = match rel.rsplit('/').next().unwrap() {
            "probabilities.npy" => (1e-12, 1e-300),
            "beat_period.npy" => (1e-9, 0.0),
            "positions.npy" | "alignment_path.npy" => (0.0, 1e-6),
            _ => (0.0, 0.0),
        };
        let ctx = Ctx::new(&ctx.method, &ctx.piece, rel);
        assert_close(&ctx, &reference.values(), &actual.values(), rtol, atol);
    }
}

#[test]
fn export_equals_golden_for_every_method() {
    for method in [Method::Pthmm, Method::Hmm, Method::OuterHmm] {
        let name = method.name();
        let out = std::env::temp_dir().join(format!("matchmaker_export_{name}"));
        let _ = std::fs::remove_dir_all(&out);
        let mut mm = Matchmaker::new(data(&format!("{PIECE}.musicxml")), method, None, true)
            .unwrap()
            .record();
        mm.run_file(data(&format!("{PIECE}.mid"))).unwrap();
        write_run(&mm, &out).unwrap();

        let written = files(&out);
        let relative = |p: &Path| {
            p.strip_prefix(&out)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/")
        };
        for expected in [
            "score/note_array.npy",
            "score/note_ids.json",
            "score/meta.json",
            "stream",
            &format!("{name}/positions.npy"),
            &format!("{name}/alignment_path.npy"),
            &format!("{name}/probabilities.npy"),
            &format!("{name}/meta.json"),
            "processor",
        ] {
            assert!(
                written.iter().any(|p| relative(p).starts_with(expected)),
                "[{name}] missing {expected}"
            );
        }
        match method {
            Method::Hmm => assert!(out.join("hmm/beat_period.npy").exists()),
            Method::OuterHmm => assert!(out.join("outerhmm/is_continuation.npy").exists()),
            Method::Pthmm => {}
        }

        let ctx = Ctx::new(name, PIECE, "");
        let mut compared = 0;
        for path in &written {
            let rel = relative(path);
            let counterpart = golden(PIECE).join(&rel);
            if counterpart.exists() {
                compare(&ctx, &rel, path, &counterpart);
                compared += 1;
            }
        }
        assert!(
            compared >= 10,
            "[{name}] only {compared} files had a golden counterpart"
        );
    }
}

#[test]
fn export_requires_a_recorded_run() {
    let mut mm = Matchmaker::new(
        data(&format!("{PIECE}.musicxml")),
        Method::Pthmm,
        None,
        true,
    )
    .unwrap();
    mm.run_file(data(&format!("{PIECE}.mid"))).unwrap();
    let err = write_run(&mm, std::env::temp_dir().join("matchmaker_export_none")).unwrap_err();
    assert!(matches!(err, Error::InvalidParameter { .. }), "{err}");
}
