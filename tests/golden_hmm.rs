//! Golden test: `hmm` (FR-008, SC-002).
//!
//! `replayed_observations_equal_reference` feeds the reference's own observations (pitch lists,
//! `-1`-padded in the files), so only the follower is tested; `end_to_end_equals_reference` runs
//! the whole pipeline via `Matchmaker`. Tolerances: positions and alignment path `< 1e-6`
//! beats, `beat_period` rtol 1e-9, forward variable rtol 1e-12 (plus `atol 1e-300` for
//! subnormals), evaluation metrics within 0.01.

mod common;

use common::{Ctx, assert_close, assert_exact, data, golden, golden_observations, json, npy};
use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::matchmaker::{Matchmaker, Method};
use matchmaker_rs::prob::hmm::{PitchIoiHmm, PitchIoiHmmConfig};
use matchmaker_rs::score::Score;
use matchmaker_rs::utils::eval::{GroundTruth, evaluate_alignment};

const PIECES: [(&str, &str); 5] = [
    (
        "simple_mozart_k265_var1",
        "simple_mozart_k265_var1.musicxml",
    ),
    ("mozart_k265_var1", "mozart_k265_var1.musicxml"),
    ("Bach-fugue_bwv_858", "Bach-fugue_bwv_858.musicxml"),
    (
        "simple_mozart_k265_var1_midiscore",
        "simple_mozart_k265_var1.mid",
    ),
    (
        "simple_mozart_k265_var1_wrongnote",
        "simple_mozart_k265_var1.musicxml",
    ),
];

#[test]
fn replayed_observations_equal_reference() {
    for (piece, score_file) in PIECES {
        let dir = golden(piece).join("hmm");
        let ctx = |file: &str| Ctx::new("hmm", piece, file);
        let score = Score::load(data(score_file), true).unwrap();
        let mut hmm = PitchIoiHmm::new(&score, PitchIoiHmmConfig::default());

        let expected_states = npy(dir.join("score_positions.npy")).values();
        let states: Vec<f64> = hmm
            .score_positions()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        assert_exact(&ctx("score_positions.npy"), &expected_states, &states);

        let probabilities = dir.join("probabilities.npy");
        let expected_probs = probabilities.exists().then(|| npy(&probabilities).values());
        let expected_periods = npy(dir.join("beat_period.npy")).values();
        let (mut positions, mut indices, mut periods) = (Vec::new(), Vec::new(), Vec::new());
        for (k, obs) in golden_observations(&dir).iter().enumerate() {
            if !hmm.is_still_following() {
                break;
            }
            positions.push(hmm.call(obs));
            indices.push(hmm.current_index() as f64);
            periods.push(hmm.beat_period());
            if let Some(all) = &expected_probs {
                let n = states.len();
                assert_close(
                    &ctx(&format!("probabilities.npy step {k}")),
                    &all[k * n..(k + 1) * n],
                    hmm.forward_variable().unwrap(),
                    1e-12,
                    1e-300,
                );
            }
        }
        assert_close(
            &ctx("beat_period.npy"),
            &expected_periods,
            &periods,
            1e-9,
            0.0,
        );
        assert_exact(
            &ctx("current_index.npy"),
            &npy(dir.join("current_index.npy")).values(),
            &indices,
        );
        assert_close(
            &ctx("positions.npy"),
            &npy(dir.join("positions.npy")).values(),
            &positions,
            0.0,
            1e-6,
        );
        let path = hmm.alignment_path();
        let actual_path = [path.perf_time.clone(), path.score_beat.clone()].concat();
        assert_close(
            &ctx("alignment_path.npy"),
            &npy(dir.join("alignment_path.npy")).values(),
            &actual_path,
            0.0,
            1e-6,
        );
    }
}

#[test]
fn end_to_end_equals_reference() {
    for (piece, score_file) in PIECES {
        let dir = golden(piece).join("hmm");
        let ctx = |file: &str| Ctx::new("hmm", piece, file);
        let mut mm = Matchmaker::new(data(score_file), Method::Hmm, None, true).unwrap();
        let performance = format!("{}.mid", piece.trim_end_matches("_midiscore"));
        let positions = mm.run_file(data(&performance)).unwrap();
        assert_close(
            &ctx("positions.npy"),
            &npy(dir.join("positions.npy")).values(),
            &positions,
            0.0,
            1e-6,
        );

        let evaluation = dir.join("evaluation.json");
        let match_file = data(&format!("{piece}.match"));
        if !evaluation.exists() || !match_file.exists() {
            continue;
        }
        let gt = GroundTruth::load(&match_file, mm.score(), Some(&data(&performance))).unwrap();
        let actual = serde_json::to_value(evaluate_alignment(mm.alignment_path(), &gt)).unwrap();
        for (key, e) in json(&evaluation).as_object().unwrap() {
            if let (Some(e), Some(a)) = (e.as_f64(), actual[key].as_f64()) {
                assert!(
                    (e - a).abs() <= 0.01,
                    "[hmm / {piece} / evaluation.json] {key}: expected {e}, actual {a}"
                );
            }
        }
    }
}
