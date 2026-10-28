//! Shared helpers for golden tests: loading reference data and comparing it with a report of
//! the first differing element (SC-007).

#![allow(dead_code)]

mod npy;

use std::path::{Path, PathBuf};

pub use npy::NpyArray;

/// `tests/data/<name>`.
pub fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join(name)
}

/// `tests/golden/<piece>`.
pub fn golden(piece: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("golden")
        .join(piece)
}

/// All golden piece directories.
pub fn golden_pieces() -> Vec<String> {
    let mut pieces: Vec<String> = std::fs::read_dir(golden(""))
        .expect("tests/golden exists")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    pieces.sort();
    pieces
}

/// Load a golden `.npy` file (panics with the path on failure).
pub fn npy(path: impl AsRef<Path>) -> NpyArray {
    let path = path.as_ref();
    NpyArray::read(path)
        .unwrap_or_else(|e| panic!("cannot read golden file {}: {e}", path.display()))
}

/// Observations of `<piece>/<method>/observations_*.npy`; `hmm` pitch lists are `-1`-padded.
pub fn golden_observations(dir: &Path) -> Vec<matchmaker_rs::features::Observation> {
    let features = npy(dir.join("observations_features.npy"));
    let width = features.shape[1];
    let times = npy(dir.join("observations_times.npy")).values();
    features
        .values()
        .chunks(width)
        .zip(times)
        .map(|(row, perf_time)| matchmaker_rs::features::Observation {
            features: row
                .iter()
                .filter(|&&v| v >= 0.0)
                .map(|&v| v as f32)
                .collect(),
            perf_time,
        })
        .collect()
}

/// Load a golden JSON file; Python's `NaN` becomes `null`.
pub fn json(path: impl AsRef<Path>) -> serde_json::Value {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text.replace("NaN", "null"))
        .unwrap_or_else(|e| panic!("invalid JSON {}: {e}", path.display()))
}

/// Where a comparison happens: printed on failure.
#[derive(Debug, Clone)]
pub struct Ctx {
    pub method: String,
    pub piece: String,
    pub file: String,
}

impl Ctx {
    pub fn new(method: &str, piece: &str, file: &str) -> Self {
        Ctx {
            method: method.into(),
            piece: piece.into(),
            file: file.into(),
        }
    }
}

/// Compare sequences element by element with `|a - e| <= atol + rtol * |e|`; panic at the first
/// difference with method, piece, file, index, expected and actual value.
pub fn assert_close(ctx: &Ctx, expected: &[f64], actual: &[f64], rtol: f64, atol: f64) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "[{} / {} / {}] length differs: expected {}, actual {}",
        ctx.method,
        ctx.piece,
        ctx.file,
        expected.len(),
        actual.len()
    );
    for (i, (e, a)) in expected.iter().zip(actual).enumerate() {
        let ok = (e.is_nan() && a.is_nan()) || (a - e).abs() <= atol + rtol * e.abs();
        assert!(
            ok,
            "[{} / {} / {}] first difference at index {i}: expected {e:?}, actual {a:?}",
            ctx.method, ctx.piece, ctx.file
        );
    }
}

/// Exact comparison (bit-equal values, NaN == NaN).
pub fn assert_exact(ctx: &Ctx, expected: &[f64], actual: &[f64]) {
    assert_close(ctx, expected, actual, 0.0, 0.0);
}
