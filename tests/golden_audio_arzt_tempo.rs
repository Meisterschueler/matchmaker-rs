//! Golden replay for seeded tempo-aware audio Arzt, without renderer differences.

mod common;

use common::{Ctx, assert_close, assert_exact, golden, npy};
use matchmaker_rs::base::AlignmentPath;
use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::dp::arzt_tempo::ArztTempoFrame;
use matchmaker_rs::features::Observation;
use matchmaker_rs::matchmaker::AudioMatchmaker;
use matchmaker_rs::matchmaker::{AudioMethod, AudioMethodConfig};
use matchmaker_rs::utils::eval::{GroundTruth, evaluate_alignment};
use std::path::PathBuf;

fn rows(path: PathBuf) -> Vec<Vec<f32>> {
    let array = npy(path);
    array
        .values()
        .chunks(array.shape[1])
        .map(|row| row.iter().map(|&value| value as f32).collect())
        .collect()
}

#[test]
fn seeded_tempo_replay_matches_reference_states_and_score_changes() {
    for name in ["steady", "accelerating", "slowing"] {
        let directory = golden("audio").join("tempo_replay").join(name);
        let mut follower = ArztTempoFrame::new(
            rows(directory.join("score_features.npy")),
            npy(directory.join("score_positions.npy"))
                .values()
                .iter()
                .map(|&value| value as f32)
                .collect(),
            npy(directory.join("score_frame_to_beat.npy")).values(),
            AudioMethodConfig::for_method(AudioMethod::Arzt),
        )
        .unwrap();
        let observations = rows(directory.join("performance_features.npy"));
        let times = npy(directory.join("performance_times.npy")).values();
        let mut positions = Vec::new();
        let mut states = Vec::new();
        for (features, perf_time) in observations.into_iter().zip(times) {
            positions.push(follower.call(&Observation {
                features,
                perf_time,
            }));
            states.extend(follower.tempo_state());
        }
        assert_close(
            &Ctx::new("arzt_tempo", name, "positions.npy"),
            &npy(directory.join("positions.npy")).values(),
            &positions,
            0.0,
            1e-9,
        );
        assert_close(
            &Ctx::new("arzt_tempo", name, "tempo_states.npy"),
            &npy(directory.join("tempo_states.npy")).values(),
            &states,
            1e-7,
            1e-9,
        );
        let actual_features = follower
            .reference_features()
            .iter()
            .flatten()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>();
        assert_exact(
            &Ctx::new("arzt_tempo", name, "final_score_features.npy"),
            &npy(directory.join("final_score_features.npy")).values(),
            &actual_features,
        );
        assert_close(
            &Ctx::new("arzt_tempo", name, "final_score_frame_to_beat.npy"),
            &npy(directory.join("final_score_frame_to_beat.npy")).values(),
            follower.frame_to_beat(),
            0.0,
            1e-9,
        );
        follower.reset();
        let mut repeated_positions = Vec::new();
        let mut repeated_states = Vec::new();
        for (features, perf_time) in rows(directory.join("performance_features.npy"))
            .into_iter()
            .zip(npy(directory.join("performance_times.npy")).values())
        {
            repeated_positions.push(follower.call(&Observation {
                features,
                perf_time,
            }));
            repeated_states.extend(follower.tempo_state());
        }
        assert_eq!(positions, repeated_positions, "{name}: reset positions");
        assert_eq!(states, repeated_states, "{name}: reset tempo states");
    }
}

#[test]
fn audio_feature_replays_match_reference_tempo_states() {
    for processor in ["chroma", "lse"] {
        let directory = golden("audio")
            .join("tempo_pipeline")
            .join(format!("arzt_tempo_{processor}"));
        let mut config = AudioMethodConfig::for_method(AudioMethod::ArztTempo);
        config.set_processor(processor.parse().unwrap());
        let mut follower = ArztTempoFrame::new(
            rows(directory.join("score_features.npy")),
            npy(directory.join("score_positions.npy"))
                .values()
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            npy(directory.join("score_frame_to_beat.npy")).values(),
            config,
        )
        .unwrap();
        let mut positions = Vec::new();
        let mut states = Vec::new();
        for (features, perf_time) in rows(directory.join("performance_features.npy"))
            .into_iter()
            .zip(npy(directory.join("performance_times.npy")).values())
        {
            positions.push(follower.call(&Observation {
                features,
                perf_time,
            }));
            states.extend(follower.tempo_state());
        }
        assert_close(
            &Ctx::new("arzt_tempo", processor, "positions.npy"),
            &npy(directory.join("positions.npy")).values(),
            &positions,
            0.0,
            1e-9,
        );
        assert_close(
            &Ctx::new("arzt_tempo", processor, "tempo_states.npy"),
            &npy(directory.join("tempo_states.npy")).values(),
            &states,
            1e-7,
            1e-9,
        );
    }
}

#[test]
fn full_audio_pipeline_meets_declared_accuracy_limits() {
    let soundfont = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "target/audio-assets/UprightPianoKW-small-SF2-20190703/UprightPianoKW-small-20190703.sf2",
    );
    for processor in ["chroma", "lse"] {
        let directory = golden("audio")
            .join("tempo_pipeline")
            .join(format!("arzt_tempo_{processor}"));
        let expected = npy(directory.join("alignment_path.npy"));
        let count = expected.shape[1];
        let values = expected.values();
        let expected_path = AlignmentPath {
            perf_time: values[..count].to_vec(),
            score_beat: values[count..].to_vec(),
        };
        let mapping = npy(directory.join("score_frame_to_beat.npy")).values();
        let truth = GroundTruth {
            perf_sec: expected_path.perf_time.clone(),
            score_beat: expected_path
                .perf_time
                .iter()
                .map(|time| {
                    let frame = (time * 30.0).clamp(0.0, (mapping.len() - 1) as f64);
                    let lower = frame.floor() as usize;
                    let upper = (lower + 1).min(mapping.len() - 1);
                    mapping[lower] + (mapping[upper] - mapping[lower]) * (frame - lower as f64)
                })
                .collect(),
        };
        let mut config = AudioMethodConfig::for_method(AudioMethod::ArztTempo);
        config.set_processor(processor.parse().unwrap());
        config.soundfont_path = Some(soundfont.clone());
        let mut follower = AudioMatchmaker::new_arzt_tempo(
            common::data("simple_mozart_k265_var1.musicxml"),
            config,
            None,
            true,
        )
        .unwrap();
        follower
            .run_file(common::data("audio/simple_mozart_k265_var1_sf2.wav"))
            .unwrap();
        let actual_path = follower.alignment_path();
        assert_eq!(actual_path.perf_time.len(), truth.perf_sec.len());
        assert_close(
            &Ctx::new("arzt_tempo", processor, "alignment_path.perf_time"),
            &truth.perf_sec,
            &actual_path.perf_time,
            0.0,
            1e-9,
        );
        let p95 = |beats: &[f64]| {
            let mut errors = beats
                .iter()
                .zip(&truth.score_beat)
                .map(|(actual, expected)| (actual - expected).abs())
                .collect::<Vec<_>>();
            errors.sort_by(f64::total_cmp);
            let index = 0.95 * (errors.len() - 1) as f64;
            let lower = index.floor() as usize;
            let upper = index.ceil() as usize;
            errors[lower] + (errors[upper] - errors[lower]) * (index - lower as f64)
        };
        assert!(
            (p95(&actual_path.score_beat) - p95(&expected_path.score_beat)).abs() <= 0.05,
            "{processor}: p95 beat-error difference exceeds 0.05 beats"
        );
        let actual = evaluate_alignment(follower.alignment_path(), &truth);
        let reference = evaluate_alignment(&expected_path, &truth);
        assert!(
            (actual.median_b - reference.median_b).abs() <= 0.01,
            "{processor}: median beat-error limit exceeded: {actual:?} vs {reference:?}"
        );
        for (name, actual, reference) in [
            (
                "accuracy_0.05b",
                actual.accuracy_0_05b,
                reference.accuracy_0_05b,
            ),
            (
                "accuracy_0.1b",
                actual.accuracy_0_1b,
                reference.accuracy_0_1b,
            ),
            (
                "accuracy_0.3b",
                actual.accuracy_0_3b,
                reference.accuracy_0_3b,
            ),
            (
                "accuracy_0.5b",
                actual.accuracy_0_5b,
                reference.accuracy_0_5b,
            ),
            ("accuracy_1b", actual.accuracy_1b, reference.accuracy_1b),
            ("accuracy_2b", actual.accuracy_2b, reference.accuracy_2b),
        ] {
            assert!(
                (actual - reference).abs() <= 0.01,
                "{processor}: {name} delivery limit exceeded: Rust={actual}, reference={reference}"
            );
        }
    }
}
