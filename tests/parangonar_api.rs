//! Golden/API coverage for the remaining Parangonar MIDI matchers.

mod common;

use matchmaker_rs::matchmaker::{Matchmaker, Method};

fn golden_path(method: &str) -> Vec<(f64, f64)> {
    let file = common::data(&format!(
        "../golden/simple_mozart_k265_var1/{method}/alignment_path.npy"
    ));
    let bytes = std::fs::read(file).unwrap();
    let header = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let values: Vec<f64> = bytes[10 + header..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_le_bytes(*chunk))
        .collect();
    let (time, beat) = values.split_at(values.len() / 2);
    time.iter().copied().zip(beat.iter().copied()).collect()
}

#[test]
fn remaining_parangonar_methods_match_goldens() {
    let score = common::data("simple_mozart_k265_var1.musicxml");
    let performance = common::data("simple_mozart_k265_var1.mid");
    for name in ["OPTM", "SLT_OLTW", "SL_OLTW"] {
        let method = name.parse::<Method>().unwrap();
        let mut mm = Matchmaker::new(&score, method, None, true).unwrap();
        let positions = mm.run_file(&performance).unwrap();
        assert!(!positions.is_empty());
        let expected = golden_path(name);
        assert_eq!(mm.alignment_path().perf_time.len(), expected.len());
        for ((actual_time, actual_beat), (expected_time, expected_beat)) in mm
            .alignment_path()
            .perf_time
            .iter()
            .zip(mm.alignment_path().score_beat.iter())
            .zip(expected)
        {
            assert!((*actual_time - expected_time).abs() < 1e-6, "{name}");
            assert!((*actual_beat - expected_beat).abs() < 1e-6, "{name}");
        }
    }
}
