//! Golden test: `outerhmm` (FR-009, SC-001).
//!
//! `replayed_observations_equal_reference` feeds the reference's own observations, so only the
//! follower is tested; `end_to_end_equals_reference` runs the whole pipeline via `Matchmaker`.
//! Every call (chord starts and continuations) is compared: positions and alignment path
//! `< 1e-6` beats, current index and `is_continuation` exact, state probabilities `rtol 1e-12`
//! (plus `atol 1e-300` for subnormals).

mod common;

use common::{Ctx, assert_close, assert_exact, data, golden, golden_observations, npy};
use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::matchmaker::{Matchmaker, Method};
use matchmaker_rs::prob::outerhmm::{OuterProductHmm, OuterProductHmmConfig};
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
        let dir = golden(piece).join("outerhmm");
        let ctx = |file: &str| Ctx::new("outerhmm", piece, file);
        let score = Score::load(data(score_file), true).unwrap();
        let mut hmm = OuterProductHmm::new(&score, OuterProductHmmConfig::default());

        let expected_states = npy(dir.join("score_positions.npy")).values();
        let states: Vec<f64> = hmm
            .score_positions()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        assert_exact(&ctx("score_positions.npy"), &expected_states, &states);

        let probabilities = dir.join("probabilities.npy");
        let expected_probs = probabilities.exists().then(|| npy(&probabilities).values());
        let (mut positions, mut indices, mut continuations) = (Vec::new(), Vec::new(), Vec::new());
        for (k, obs) in golden_observations(&dir).iter().enumerate() {
            if !hmm.is_still_following() {
                break;
            }
            positions.push(hmm.call(obs));
            indices.push(hmm.current_index() as f64);
            continuations.push(f64::from(u8::from(hmm.is_continuation())));
            if let Some(all) = &expected_probs {
                let n = states.len();
                assert_close(
                    &ctx(&format!("probabilities.npy step {k}")),
                    &all[k * n..(k + 1) * n],
                    hmm.state_probabilities(),
                    1e-12,
                    1e-300,
                );
            }
        }
        assert_exact(
            &ctx("current_index.npy"),
            &npy(dir.join("current_index.npy")).values(),
            &indices,
        );
        assert_exact(
            &ctx("is_continuation.npy"),
            &npy(dir.join("is_continuation.npy")).values(),
            &continuations,
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
            &alignment_path_values(&hmm),
            0.0,
            1e-6,
        );
    }
}

#[test]
fn end_to_end_equals_reference() {
    for (piece, score_file) in PIECES {
        let dir = golden(piece).join("outerhmm");
        let ctx = |file: &str| Ctx::new("outerhmm", piece, file);
        let mut mm = Matchmaker::new(data(score_file), Method::OuterHmm, None, true).unwrap();
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
