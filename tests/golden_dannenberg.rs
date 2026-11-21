//! Golden test: `dannenberg` (005 FR-004, SC-002).
//!
//! `replayed_observations_equal_reference` feeds the reference's own observations, so only
//! the follower is tested; `end_to_end_equals_reference` runs the whole pipeline via
//! `Matchmaker`. Tolerances: score positions exact (`float32`), the DP rating column exact
//! (every update adds exactly representable multiples of `c_m`/`c_e` and `1`), current
//! index exact, positions and alignment path `< 1e-6` beats (the tempo-drift division).

mod common;

use common::{Ctx, assert_close, assert_exact, data, golden, golden_observations, npy};
use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::dp::dannenberg::{DannenbergConfig, DannenbergEvent};
use matchmaker_rs::matchmaker::{DEFAULT_TEMPO, Matchmaker, Method};
use matchmaker_rs::score::Score;

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
        let dir = golden(piece).join("dannenberg");
        let ctx = |file: &str| Ctx::new("dannenberg", piece, file);
        let score = Score::load(data(score_file), true).unwrap();
        let tempo = score.tempo.unwrap_or(DEFAULT_TEMPO);
        let (cevts, onsets) = score.score_cevts();
        let mut follower = DannenbergEvent::new(cevts, onsets, DannenbergConfig::default(), tempo);

        let expected_states = npy(dir.join("score_positions.npy")).values();
        let states: Vec<f64> = follower
            .score_positions()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        assert_exact(&ctx("score_positions.npy"), &expected_states, &states);

        let probabilities = dir.join("probabilities.npy");
        let expected_probs = probabilities.exists().then(|| npy(&probabilities).values());
        let (mut positions, mut indices) = (Vec::new(), Vec::new());
        for (k, obs) in golden_observations(&dir).iter().enumerate() {
            if !follower.is_still_following() {
                break;
            }
            positions.push(follower.call(obs));
            let debug = follower.debug_state();
            indices.push(debug.current_index as f64);
            if let Some(all) = &expected_probs {
                let n = states.len();
                assert_exact(
                    &ctx(&format!("probabilities.npy step {k}")),
                    &all[k * n..(k + 1) * n],
                    &debug.probabilities,
                );
            }
        }
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
        let path = follower.alignment_path();
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
        let dir = golden(piece).join("dannenberg");
        let ctx = |file: &str| Ctx::new("dannenberg", piece, file);
        let mut mm = Matchmaker::new(data(score_file), Method::Dannenberg, None, true).unwrap();
        let performance = format!("{}.mid", piece.trim_end_matches("_midiscore"));
        let positions = mm.run_file(data(&performance)).unwrap();
        assert_close(
            &ctx("positions.npy"),
            &npy(dir.join("positions.npy")).values(),
            &positions,
            0.0,
            1e-6,
        );
        let path = mm.alignment_path();
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
