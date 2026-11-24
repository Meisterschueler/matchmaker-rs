//! Event-by-event parity with pinned Parangonar CPU inference.
mod common;

use common::{Ctx, assert_close, data, golden, golden_observations, npy};
use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::dp::otm::{OnlineTransformerMatcher, OtmConfig};
use matchmaker_rs::matchmaker::{Matchmaker, Method};
use matchmaker_rs::score::Score;

const PIECES: [&str; 6] = [
    "simple_mozart_k265_var1",
    "mozart_k265_var1",
    "Bach-fugue_bwv_858",
    "simple_mozart_k265_var1_midiscore",
    "simple_mozart_k265_var1_wrongnote",
    "otm_grace",
];

fn score_path(piece: &str) -> std::path::PathBuf {
    match piece {
        "simple_mozart_k265_var1_midiscore" => data("simple_mozart_k265_var1.mid"),
        "simple_mozart_k265_var1_wrongnote" => data("simple_mozart_k265_var1.musicxml"),
        "otm_grace" => data("otm/grace.musicxml"),
        _ => data(&format!("{piece}.musicxml")),
    }
}

fn performance_path(piece: &str) -> std::path::PathBuf {
    match piece {
        "simple_mozart_k265_var1_midiscore" => data("simple_mozart_k265_var1.mid"),
        "otm_grace" => data("otm/grace.mid"),
        _ => data(&format!("{piece}.mid")),
    }
}

#[test]
fn replay_matches_reference_at_every_event_and_stage() {
    for piece in PIECES {
        let dir = golden(piece).join("OTM");
        let score = Score::load(score_path(piece), true).unwrap();
        let notes = score.otm_notes();
        let expected_notes = npy(dir.join("score_notes.npy"));
        assert_eq!(
            expected_notes.len(),
            notes.len(),
            "{piece}: score note count"
        );
        assert_eq!(
            expected_notes.string_field("id"),
            notes.iter().map(|note| note.id.clone()).collect::<Vec<_>>(),
            "{piece}: score IDs"
        );
        for (field, actual) in [
            (
                "pitch",
                notes
                    .iter()
                    .map(|note| f64::from(note.pitch))
                    .collect::<Vec<_>>(),
            ),
            (
                "onset_beat",
                notes
                    .iter()
                    .map(|note| f64::from(note.onset_beat))
                    .collect(),
            ),
            (
                "duration_beat",
                notes
                    .iter()
                    .map(|note| f64::from(note.duration_beat))
                    .collect(),
            ),
            (
                "is_grace",
                notes.iter().map(|note| f64::from(note.is_grace)).collect(),
            ),
        ] {
            assert_close(
                &Ctx::new("OTM", piece, field),
                &expected_notes.field(field),
                &actual,
                0.0,
                0.0,
            );
        }
        let mut follower = OnlineTransformerMatcher::new(notes, OtmConfig::default()).unwrap();
        let expected_onsets = npy(dir.join("score_positions.npy")).values();
        assert_close(
            &Ctx::new("OTM", piece, "score_positions"),
            &expected_onsets,
            &follower
                .score_positions()
                .iter()
                .map(|value| f64::from(*value))
                .collect::<Vec<_>>(),
            0.0,
            0.0,
        );
        let observations = golden_observations(&dir);
        let expected_positions = npy(dir.join("positions.npy")).values();
        let expected_indices = npy(dir.join("current_index.npy")).values();
        let expected_tokens = npy(dir.join("token_windows.npy")).values();
        let tokens_present = npy(dir.join("token_windows_present.npy")).values();
        let expected_logits = npy(dir.join("model_logits.npy")).values();
        let logits_present = npy(dir.join("model_logits_present.npy")).values();
        let expected_tempo = npy(dir.join("tempo_state.npy")).values();
        let internal_path = npy(dir.join("internal_path.npy")).values();
        let mut positions = Vec::new();
        for (event, observation) in observations.iter().enumerate() {
            positions.push(follower.try_call(observation).unwrap());
            let debug = follower.debug_state();
            let state = debug.otm.unwrap();
            assert_eq!(
                debug.current_index, expected_indices[event] as usize,
                "{piece} event {event}: current index"
            );
            assert_eq!(
                state.tokens.is_some(),
                tokens_present[event] != 0.0,
                "{piece} event {event}: token presence"
            );
            assert_eq!(
                state.logits.is_some(),
                logits_present[event] != 0.0,
                "{piece} event {event}: logit presence"
            );
            if let Some(tokens) = state.tokens {
                let actual: Vec<_> = tokens
                    .into_iter()
                    .flatten()
                    .map(|value| value as f64)
                    .collect();
                assert_close(
                    &Ctx::new("OTM", piece, &format!("tokens event {event}")),
                    &expected_tokens[event * 182..(event + 1) * 182],
                    &actual,
                    0.0,
                    0.0,
                );
            }
            if let Some(logits) = state.logits {
                assert_close(
                    &Ctx::new("OTM", piece, &format!("logits event {event}")),
                    &expected_logits[event * 52..(event + 1) * 52],
                    &logits.into_iter().map(f64::from).collect::<Vec<_>>(),
                    1e-4,
                    1e-4,
                );
            }
            assert_close(
                &Ctx::new("OTM", piece, &format!("tempo event {event}")),
                &expected_tempo[event * 2..(event + 1) * 2],
                &state.tempo,
                0.0,
                1e-6,
            );
            assert_eq!(
                state.warping,
                [
                    internal_path[event] as usize,
                    internal_path[observations.len() + event] as usize
                ]
            );
        }
        assert_close(
            &Ctx::new("OTM", piece, "positions"),
            &expected_positions,
            &positions,
            0.0,
            1e-6,
        );
        let path = follower.alignment_path();
        assert_close(
            &Ctx::new("OTM", piece, "alignment_path"),
            &npy(dir.join("alignment_path.npy")).values(),
            &path
                .perf_time
                .iter()
                .chain(&path.score_beat)
                .copied()
                .collect::<Vec<_>>(),
            0.0,
            1e-6,
        );
    }
}

#[test]
fn full_pipeline_matches_reference_positions_and_path() {
    for piece in PIECES {
        let dir = golden(piece).join("OTM");
        let mut follower = Matchmaker::new(score_path(piece), Method::Otm, None, true).unwrap();
        let positions = follower.run_file(performance_path(piece)).unwrap();
        assert_close(
            &Ctx::new("OTM", piece, "end-to-end positions"),
            &npy(dir.join("positions.npy")).values(),
            &positions,
            0.0,
            1e-6,
        );
        let path = follower.alignment_path();
        assert_close(
            &Ctx::new("OTM", piece, "end-to-end alignment_path"),
            &npy(dir.join("alignment_path.npy")).values(),
            &path
                .perf_time
                .iter()
                .chain(&path.score_beat)
                .copied()
                .collect::<Vec<_>>(),
            0.0,
            1e-6,
        );
    }
}
