//! Golden test: `dixon` (002 FR-004, SC-001).
//!
//! `replayed_observations_equal_reference` feeds the reference's own observations, so only the
//! follower is tested; `end_to_end_equals_reference` runs the whole pipeline via `Matchmaker`.
//! Positions and alignment path `< 1e-6` beats, current index exact. The reference has no state
//! distribution for `dixon`, so there is no `probabilities.npy`.

mod common;

use common::{Ctx, assert_close, assert_exact, data, golden, golden_observations, npy};
use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::dp::dixon::{DixonEvent, DixonEventConfig};
use matchmaker_rs::matchmaker::{Matchmaker, Method};
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

fn alignment_path_values(follower: &dyn OnlineAlignment) -> Vec<f64> {
    let path = follower.alignment_path();
    [path.perf_time.clone(), path.score_beat.clone()].concat()
}

#[test]
fn replayed_observations_equal_reference() {
    for (piece, score_file) in PIECES {
        let dir = golden(piece).join("dixon");
        let ctx = |file: &str| Ctx::new("dixon", piece, file);
        let score = Score::load(data(score_file), true).unwrap();
        let (features, onsets) = score.onset_pianoroll(true);
        let mut dixon = DixonEvent::new(features, onsets, DixonEventConfig::default());

        let expected_states = npy(dir.join("score_positions.npy")).values();
        let states: Vec<f64> = dixon
            .score_positions()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        assert_exact(&ctx("score_positions.npy"), &expected_states, &states);

        let (mut positions, mut indices) = (Vec::new(), Vec::new());
        for obs in golden_observations(&dir) {
            if !dixon.is_still_following() {
                break;
            }
            positions.push(dixon.call(&obs));
            indices.push(dixon.current_index() as f64);
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
        assert_close(
            &ctx("alignment_path.npy"),
            &npy(dir.join("alignment_path.npy")).values(),
            &alignment_path_values(&dixon),
            0.0,
            1e-6,
        );
    }
}

#[test]
fn end_to_end_equals_reference() {
    for (piece, score_file) in PIECES {
        let dir = golden(piece).join("dixon");
        let ctx = |file: &str| Ctx::new("dixon", piece, file);
        let mut mm = Matchmaker::new(data(score_file), Method::Dixon, None, true).unwrap();
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
        assert_close(
            &ctx("alignment_path.npy"),
            &npy(dir.join("alignment_path.npy")).values(),
            &[path.perf_time.clone(), path.score_beat.clone()].concat(),
            0.0,
            1e-6,
        );
    }
}
