//! US2: evaluation parity. `evaluate_alignment` on the reference's alignment paths with the
//! `.match` ground truth reproduces `evaluation.json` (abs 1e-4, research R8). The paths of
//! `hmm` and `outerhmm` come from the reference, so this test does not depend on those ports.

mod common;

use common::{data, golden, json, npy};
use matchmaker_rs::base::AlignmentPath;
use matchmaker_rs::score::Score;
use matchmaker_rs::utils::eval::{GroundTruth, evaluate_alignment};

#[test]
fn evaluation_matches_reference() {
    for piece in [
        "simple_mozart_k265_var1",
        "mozart_k265_var1",
        "Bach-fugue_bwv_858",
    ] {
        let score = Score::load(data(&format!("{piece}.musicxml")), true).unwrap();
        let perf = data(&format!("{piece}.mid"));
        let gt = GroundTruth::load(data(&format!("{piece}.match")), &score, Some(&perf)).unwrap();
        for method in ["pthmm", "hmm", "outerhmm"] {
            let dir = golden(piece).join(method);
            let values = npy(dir.join("alignment_path.npy")).values();
            let (perf_time, score_beat) = values.split_at(values.len() / 2);
            let path = AlignmentPath {
                perf_time: perf_time.to_vec(),
                score_beat: score_beat.to_vec(),
            };
            let actual = serde_json::to_value(evaluate_alignment(&path, &gt)).unwrap();
            let expected = json(dir.join("evaluation.json"));
            let (actual, expected) = (actual.as_object().unwrap(), expected.as_object().unwrap());
            let mut keys: Vec<_> = actual.keys().collect();
            let mut expected_keys: Vec<_> = expected.keys().collect();
            keys.sort();
            expected_keys.sort();
            assert_eq!(keys, expected_keys, "[{method} / {piece}] keys");
            for (key, e) in expected {
                let a = &actual[key];
                let ok = match (e.as_f64(), a.as_f64()) {
                    (Some(e), Some(a)) => (e - a).abs() <= 1e-4,
                    (None, None) => true,
                    _ => false,
                };
                assert!(
                    ok,
                    "[{method} / {piece} / evaluation.json] {key}: expected {e}, actual {a}"
                );
            }
        }
    }
}

/// Without the streamed MIDI the onsets come from the match file's ticks; for the bundled
/// piece both store the same notes, so they agree to within a tick.
#[test]
fn match_ticks_without_performance() {
    let piece = "simple_mozart_k265_var1";
    let score = Score::load(data(&format!("{piece}.musicxml")), true).unwrap();
    let gt = |perf: Option<&std::path::Path>| {
        GroundTruth::load(data(&format!("{piece}.match")), &score, perf).unwrap()
    };
    let (ticks, streamed) = (gt(None), gt(Some(&data(&format!("{piece}.mid")))));
    assert_eq!(ticks.score_beat, streamed.score_beat);
    for (t, s) in ticks.perf_sec.iter().zip(&streamed.perf_sec) {
        assert!((t - s).abs() < 2e-3, "{t} vs {s}");
    }
}
