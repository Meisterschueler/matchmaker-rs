//! Golden replay tests for frame-level audio `dixon`.

mod common;

use common::{Ctx, assert_close, assert_exact, golden, npy};
use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::dp::dixon_frame::DixonFrame;
use matchmaker_rs::features::Observation;
use matchmaker_rs::matchmaker::{AudioMatchmaker, AudioMethod, AudioMethodConfig};
use std::path::PathBuf;

const RENDERER_ALIGNMENT_TOLERANCE_BEATS: f64 = 0.5;
const RENDERER_PATH_TOLERANCE_BEATS: f64 = 0.6;
const MIN_ACCURACY_0_3B: f64 = 0.99;
const MIN_ACCURACY_0_5B: f64 = 0.99;

fn feature_rows(path: PathBuf) -> Vec<Vec<f32>> {
    let array = npy(path);
    let width = array.shape[1];
    array
        .values()
        .chunks(width)
        .map(|row| row.iter().map(|&value| value as f32).collect())
        .collect()
}

fn pinned_soundfont() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/audio-assets/UprightPianoKW-small-SF2-20190703")
        .join("UprightPianoKW-small-20190703.sf2")
}

fn report_renderer_deviation(processor: &str, expected: &[f64], actual: &[f64]) -> (f64, f64, f64) {
    let compared_frames = expected.len().min(actual.len());
    let frame_count_delta = expected.len().abs_diff(actual.len());
    assert!(
        frame_count_delta <= 1,
        "frame count differs by {frame_count_delta}"
    );
    let expected = &expected[..compared_frames];
    let actual = &actual[..compared_frames];
    let first_differing_frame = expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| (expected - actual).abs() > 1e-6);
    let max_error = expected
        .iter()
        .zip(actual)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0f64, f64::max);
    let accuracy_0_3b = expected
        .iter()
        .zip(actual)
        .filter(|(expected, actual)| (*expected - *actual).abs() <= 0.3)
        .count() as f64
        / expected.len() as f64;
    let accuracy_0_5b = expected
        .iter()
        .zip(actual)
        .filter(|(expected, actual)| (*expected - *actual).abs() <= 0.5)
        .count() as f64
        / expected.len() as f64;
    println!(
        "RENDERER_DEVIATION\tdixon\t{processor}\tfirst_frame={first_differing_frame:?}\tmax_abs_beats={max_error}\taccuracy_0.3b={accuracy_0_3b}\taccuracy_0.5b={accuracy_0_5b}\tframe_count_delta={frame_count_delta}"
    );
    (max_error, accuracy_0_3b, accuracy_0_5b)
}

#[test]
fn replayed_audio_observations_match_reference() {
    for (processor, norm) in [("chroma", f64::INFINITY), ("lse", 2.0)] {
        let case = format!("dixon_{processor}");
        let dir = golden("audio").join(&case);
        let score_features = feature_rows(dir.join("score_features.npy"));
        let score_positions = npy(dir.join("score_positions.npy")).values();
        let score_positions_f32 = score_positions.iter().map(|&value| value as f32).collect();
        let frame_to_beat = npy(dir.join("score_frame_to_beat.npy")).values();
        let mut config = AudioMethodConfig::for_method(AudioMethod::Dixon);
        config.set_processor(processor.parse().unwrap());
        config.norm = Some(norm);
        let mut follower =
            DixonFrame::new(score_features, score_positions_f32, frame_to_beat, config).unwrap();

        let performance_features = feature_rows(dir.join("performance_features.npy"));
        let performance_times = npy(dir.join("performance_times.npy")).values();
        assert_eq!(performance_features.len(), performance_times.len());
        let mut positions = Vec::new();
        for (features, perf_time) in performance_features.into_iter().zip(performance_times) {
            if !follower.is_still_following() {
                break;
            }
            positions.push(follower.call(&Observation {
                features,
                perf_time,
            }));
        }

        assert_close(
            &Ctx::new("dixon", &case, "positions.npy"),
            &npy(dir.join("positions.npy")).values(),
            &positions,
            0.0,
            1e-6,
        );
        let path = follower.alignment_path();
        assert_close(
            &Ctx::new("dixon", &case, "alignment_path.npy"),
            &npy(dir.join("alignment_path.npy")).values(),
            &[path.perf_time.clone(), path.score_beat.clone()].concat(),
            0.0,
            1e-6,
        );
        assert_exact(
            &Ctx::new("dixon", &case, "score_positions.npy"),
            &score_positions,
            &follower
                .score_positions()
                .iter()
                .map(|&value| f64::from(value))
                .collect::<Vec<_>>(),
        );
    }
}

#[test]
fn rust_audio_pipeline_matches_reference_alignment() {
    let soundfont = pinned_soundfont();
    if !soundfont.is_file() {
        eprintln!("skipping audio Dixon end-to-end test: local pinned SF2 is unavailable");
        return;
    }

    for (name, norm) in [("chroma", f64::INFINITY), ("lse", 2.0)] {
        let mut config = AudioMethodConfig::for_method(AudioMethod::Dixon);
        config.set_processor(name.parse().unwrap());
        config.norm = Some(norm);
        config.soundfont_path = Some(soundfont.clone());
        let mut matchmaker = AudioMatchmaker::new_dixon(
            common::data("simple_mozart_k265_var1.musicxml"),
            config,
            None,
            true,
        )
        .unwrap();
        let positions = matchmaker
            .run_file(common::data("audio/simple_mozart_k265_var1_sf2.wav"))
            .unwrap();
        println!(
            "DIXON_RENDERER_FRAMES\t{name}\tscore={}",
            matchmaker.reference().features.len()
        );
        let dir = golden("audio").join(format!("dixon_{name}"));
        let expected_positions = npy(dir.join("positions.npy")).values();
        let (position_max_error, position_accuracy_0_3b, _) =
            report_renderer_deviation(name, &expected_positions, &positions);
        assert!(position_max_error <= RENDERER_ALIGNMENT_TOLERANCE_BEATS);
        assert!(position_accuracy_0_3b >= MIN_ACCURACY_0_3B);
        let compared_frames = expected_positions.len().min(positions.len());
        assert_close(
            &Ctx::new("dixon", name, "positions.npy"),
            &expected_positions[..compared_frames],
            &positions[..compared_frames],
            0.0,
            RENDERER_ALIGNMENT_TOLERANCE_BEATS,
        );
        let frame_to_beat = &matchmaker.reference().frame_to_beat;
        assert!(positions.iter().all(|beat| {
            *beat >= frame_to_beat[0]
                && *beat <= *frame_to_beat.last().expect("score frame beat map")
        }));

        let expected_path = npy(dir.join("alignment_path.npy"));
        let expected_frame_count = expected_path.shape[1];
        let expected_path_values = expected_path.values();
        let actual_path = matchmaker.alignment_path();
        let path_frames = expected_frame_count.min(actual_path.perf_time.len());
        assert!(expected_frame_count.abs_diff(actual_path.perf_time.len()) <= 1);
        let expected_path_beats =
            &expected_path_values[expected_frame_count..expected_frame_count + path_frames];
        let actual_path_beats = &actual_path.score_beat[..path_frames];
        let (path_max_error, _, path_accuracy_0_5b) = report_renderer_deviation(
            &format!("{name}_path"),
            expected_path_beats,
            actual_path_beats,
        );
        assert!(path_max_error <= RENDERER_PATH_TOLERANCE_BEATS);
        assert!(path_accuracy_0_5b >= MIN_ACCURACY_0_5B);
        assert_close(
            &Ctx::new("dixon", name, "alignment_path.perf_time"),
            &expected_path_values[..path_frames],
            &actual_path.perf_time[..path_frames],
            0.0,
            1e-6,
        );
        assert_close(
            &Ctx::new("dixon", name, "alignment_path.score_beat"),
            expected_path_beats,
            actual_path_beats,
            0.0,
            RENDERER_PATH_TOLERANCE_BEATS,
        );
    }
}
