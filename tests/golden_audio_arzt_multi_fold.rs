//! Golden replay for parallel repeat hypotheses and notated-beat selection.

mod common;

use common::{Ctx, NpyArray, assert_close, assert_exact, golden, npy};
use matchmaker_rs::base::{OnlineAlignment, StepDebug};
use matchmaker_rs::dp::arzt_multi_fold::{ArztMultiFoldCandidate, ArztMultiFoldFrame};
use matchmaker_rs::features::Observation;
use matchmaker_rs::matchmaker::{AudioMethod, AudioMethodConfig};

fn rows(array: NpyArray) -> Vec<Vec<f32>> {
    let width = array.shape[1];
    array
        .values()
        .chunks(width)
        .map(|row| row.iter().map(|&value| value as f32).collect())
        .collect()
}

fn repeat_follower(directory: &std::path::Path) -> ArztMultiFoldFrame {
    let mut candidates = Vec::new();
    for index in 0..3 {
        let candidate_dir = directory.join(format!("candidate_{index}"));
        let notes = npy(candidate_dir.join("notes.npy"));
        let note_values = notes.values();
        let max_onset_beat = note_values
            .chunks(3)
            .map(|row| row[1] as f32)
            .fold(0.0, f32::max);
        candidates.push(ArztMultiFoldCandidate {
            index,
            note_count: notes.shape[0],
            max_onset_beat,
            features: rows(npy(candidate_dir.join("score_features.npy"))),
            score_positions: npy(candidate_dir.join("score_positions.npy"))
                .values()
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            unfolded_frame_to_beat: npy(candidate_dir.join("frame_to_unfolded_beat.npy")).values(),
            notated_frame_to_beat: npy(candidate_dir.join("frame_to_notated_beat.npy")).values(),
        });
    }
    let folded_notes = npy(golden("audio")
        .join("multi_fold/fixture_repeats_voltas/folded_input/candidate_1_notes.npy"))
    .values();
    let mut score_positions: Vec<f32> = folded_notes.chunks(3).map(|row| row[1] as f32).collect();
    score_positions.sort_by(f32::total_cmp);
    score_positions.dedup();
    ArztMultiFoldFrame::new(
        candidates,
        score_positions,
        AudioMethodConfig::for_method(AudioMethod::ArztMultiFold),
    )
    .unwrap()
}

fn replay(
    follower: &mut ArztMultiFoldFrame,
    directory: &std::path::Path,
) -> (Vec<f64>, Vec<StepDebug>) {
    let features = rows(npy(directory.join("performance_features.npy")));
    let times = npy(directory.join("performance_times.npy")).values();
    let mut positions = Vec::new();
    let mut states = Vec::new();
    for (features, perf_time) in features.into_iter().zip(times) {
        positions.push(follower.call(&Observation {
            features,
            perf_time,
        }));
        states.push(follower.debug_state());
    }
    (positions, states)
}

#[test]
fn taken_and_skipped_repeats_match_reference_candidates_switches_and_reset() {
    for name in ["repeat_taken", "repeat_skipped"] {
        let directory = golden("audio").join("multi_fold").join(name);
        let mut follower = repeat_follower(&directory);
        let (positions, states) = replay(&mut follower, &directory);
        let expected = npy(directory.join("positions.npy")).values();
        assert_eq!(
            positions.len(),
            expected.len(),
            "{name}: all audio frames consumed"
        );
        assert_close(
            &Ctx::new("arzt_multi_fold", name, "positions.npy"),
            &expected,
            &positions,
            0.0,
            1e-9,
        );

        let selected: Vec<f64> = states
            .iter()
            .map(|state| state.multi_fold.as_ref().unwrap().selected_hypothesis as f64)
            .collect();
        assert_exact(
            &Ctx::new("arzt_multi_fold", name, "selected_hypothesis.npy"),
            &npy(directory.join("selected_hypothesis.npy")).values(),
            &selected,
        );
        let frames: Vec<f64> = states
            .iter()
            .flat_map(|state| {
                state
                    .multi_fold
                    .as_ref()
                    .unwrap()
                    .current_frames
                    .iter()
                    .map(|&frame| frame as f64)
            })
            .collect();
        assert_exact(
            &Ctx::new("arzt_multi_fold", name, "candidate_frames.npy"),
            &npy(directory.join("candidate_frames.npy")).values(),
            &frames,
        );
        let means: Vec<f64> = states
            .iter()
            .flat_map(|state| {
                state
                    .multi_fold
                    .as_ref()
                    .unwrap()
                    .mean_costs
                    .iter()
                    .copied()
            })
            .collect();
        assert_close(
            &Ctx::new("arzt_multi_fold", name, "candidate_mean_costs.npy"),
            &npy(directory.join("candidate_mean_costs.npy")).values(),
            &means,
            1e-7,
            1e-9,
        );

        follower.reset();
        let (repeated_positions, repeated_states) = replay(&mut follower, &directory);
        assert_eq!(positions, repeated_positions, "{name}: reset positions");
        assert_eq!(states, repeated_states, "{name}: reset candidate state");
    }
}

#[test]
fn short_single_hypothesis_path_stays_in_notated_beats() {
    let candidate = ArztMultiFoldCandidate {
        index: 0,
        note_count: 2,
        max_onset_beat: 1.0,
        features: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        score_positions: vec![0.0, 1.0],
        unfolded_frame_to_beat: vec![0.0, 1.0],
        notated_frame_to_beat: vec![0.0, 0.5],
    };
    let mut follower = ArztMultiFoldFrame::new(
        vec![candidate],
        vec![0.0, 1.0],
        AudioMethodConfig::for_method(AudioMethod::ArztMultiFold),
    )
    .unwrap();
    let position = follower.call(&Observation {
        features: vec![1.0, 0.0],
        perf_time: 0.0,
    });
    assert!((0.0..=1.0).contains(&position));
    assert_eq!(follower.alignment_path().score_beat, [position]);
}
