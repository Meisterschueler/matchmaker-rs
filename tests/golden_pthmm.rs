//! Golden test: `pthmm` (FR-007, SC-001).
//!
//! `replayed_observations_equal_reference` feeds the reference's own observations, so only the
//! follower is tested; `end_to_end_equals_reference` runs the whole pipeline via `Matchmaker`.
//! Tolerances: score positions exact (`float32`), positions and alignment path `< 1e-6` beats,
//! current index exact, forward variable `rtol 1e-12` (summation order, research R4) plus
//! `atol 1e-300` because subnormal values carry fewer significant bits.

mod common;

use common::{Ctx, assert_close, assert_exact, data, golden, npy};
use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::features::Observation;
use matchmaker_rs::matchmaker::{Matchmaker, Method};
use matchmaker_rs::prob::pthmm::PitchHmm;
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

/// Observations of `<piece>/<method>/observations_*.npy` (rows of `width` values).
pub fn golden_observations(dir: &std::path::Path) -> Vec<Observation> {
    let features = npy(dir.join("observations_features.npy"));
    let width = features.shape[1];
    let times = npy(dir.join("observations_times.npy")).values();
    features
        .values()
        .chunks(width)
        .zip(times)
        .map(|(row, perf_time)| Observation {
            features: row.iter().map(|&v| v as f32).collect(),
            perf_time,
        })
        .collect()
}

#[test]
fn replayed_observations_equal_reference() {
    for (piece, score_file) in PIECES {
        let dir = golden(piece).join("pthmm");
        let ctx = |file: &str| Ctx::new("pthmm", piece, file);
        let score = Score::load(data(score_file), true).unwrap();
        let mut hmm = PitchHmm::new(&score, true, true);

        let expected_states = npy(dir.join("score_positions.npy")).values();
        let states: Vec<f64> = hmm
            .score_positions()
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        assert_exact(&ctx("score_positions.npy"), &expected_states, &states);

        let probabilities = dir.join("probabilities.npy");
        let expected_probs = probabilities.exists().then(|| npy(&probabilities).values());
        let (mut positions, mut indices) = (Vec::new(), Vec::new());
        for (k, obs) in golden_observations(&dir).iter().enumerate() {
            if !hmm.is_still_following() {
                break;
            }
            positions.push(hmm.call(obs));
            indices.push(hmm.current_index() as f64);
            if let Some(all) = &expected_probs {
                let n = states.len();
                let fv = hmm.forward_variable().unwrap();
                assert_close(
                    &ctx(&format!("probabilities.npy step {k}")),
                    &all[k * n..(k + 1) * n],
                    fv,
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
        let dir = golden(piece).join("pthmm");
        let ctx = |file: &str| Ctx::new("pthmm", piece, file);
        let mut mm = Matchmaker::new(data(score_file), Method::Pthmm, None, true).unwrap();
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
