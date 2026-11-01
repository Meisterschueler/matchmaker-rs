//! Golden replay tests for frame-level audio `arzt`.

mod common;

use common::{Ctx, assert_close, assert_exact, golden, npy};
use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::dp::arzt_frame::ArztFrame;
use matchmaker_rs::features::Observation;
use matchmaker_rs::matchmaker::{AudioMatchmaker, AudioMethod, AudioMethodConfig};
use std::path::PathBuf;

const RENDERER_ALIGNMENT_TOLERANCE_BEATS: f64 = 0.3;

fn feature_rows(path: std::path::PathBuf) -> Vec<Vec<f32>> {
    let array = npy(path);
    let width = array.shape[1];
    array
        .values()
        .chunks(width)
        .map(|row| row.iter().map(|&value| value as f32).collect())
        .collect()
}

#[test]
fn replayed_audio_observations_match_reference() {
    for processor in ["chroma", "lse"] {
        let case = format!("arzt_{processor}");
        let dir = golden("audio").join(&case);
        let score_features = feature_rows(dir.join("score_features.npy"));
        let score_positions = npy(dir.join("score_positions.npy")).values();
        let score_positions_f32 = score_positions.iter().map(|&value| value as f32).collect();
        let frame_to_beat = npy(dir.join("score_frame_to_beat.npy")).values();
        let config = AudioMethodConfig::for_method(AudioMethod::Arzt);
        let mut follower =
            ArztFrame::new(score_features, score_positions_f32, frame_to_beat, config).unwrap();

        let expected_score_positions = npy(dir.join("score_positions.npy")).values();
        let actual_score_positions = follower
            .score_positions()
            .iter()
            .map(|&value| f64::from(value))
            .collect::<Vec<_>>();
        assert_exact(
            &Ctx::new("arzt", &case, "score_positions.npy"),
            &expected_score_positions,
            &actual_score_positions,
        );

        let performance_features = feature_rows(dir.join("performance_features.npy"));
        let performance_times = npy(dir.join("performance_times.npy")).values();
        assert_eq!(performance_features.len(), performance_times.len());
        let positions = performance_features
            .into_iter()
            .zip(performance_times)
            .map(|(features, perf_time)| {
                follower.call(&Observation {
                    features,
                    perf_time,
                })
            })
            .collect::<Vec<_>>();
        assert_close(
            &Ctx::new("arzt", &case, "positions.npy"),
            &npy(dir.join("positions.npy")).values(),
            &positions,
            0.0,
            1e-6,
        );

        let path = follower.alignment_path();
        assert_close(
            &Ctx::new("arzt", &case, "alignment_path.npy"),
            &npy(dir.join("alignment_path.npy")).values(),
            &[path.perf_time.clone(), path.score_beat.clone()].concat(),
            0.0,
            1e-6,
        );
    }
}

fn pinned_soundfont() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/audio-assets/UprightPianoKW-small-SF2-20190703")
        .join("UprightPianoKW-small-20190703.sf2")
}

fn report_renderer_deviation(processor: &str, expected: &[f64], actual: &[f64]) {
    assert_eq!(expected.len(), actual.len());
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
    println!(
        "RENDERER_DEVIATION\tarzt\t{processor}\tfirst_frame={first_differing_frame:?}\tmax_abs_beats={max_error}\taccuracy_0.3b={accuracy_0_3b}"
    );
}

#[test]
fn rust_audio_pipeline_matches_reference_alignment() {
    let soundfont = pinned_soundfont();
    if !soundfont.is_file() {
        eprintln!("skipping audio Arzt end-to-end test: local pinned SF2 is unavailable");
        return;
    }

    for (name, norm) in [("chroma", f64::INFINITY), ("lse", 2.0)] {
        let mut config = AudioMethodConfig::for_method(AudioMethod::Arzt);
        config.processor = name.parse().unwrap();
        config.norm = Some(norm);
        config.soundfont_path = Some(soundfont.clone());
        let mut matchmaker = AudioMatchmaker::new_arzt(
            common::data("simple_mozart_k265_var1.musicxml"),
            config,
            None,
            true,
        )
        .unwrap();
        let positions = matchmaker
            .run_file(common::data("audio/simple_mozart_k265_var1_sf2.wav"))
            .unwrap();

        let dir = golden("audio").join(format!("arzt_{name}"));
        let expected_positions = npy(dir.join("positions.npy")).values();
        report_renderer_deviation(name, &expected_positions, &positions);
        assert_close(
            &Ctx::new("arzt", name, "positions.npy"),
            &expected_positions,
            &positions,
            0.0,
            RENDERER_ALIGNMENT_TOLERANCE_BEATS,
        );
        let expected_path = npy(dir.join("alignment_path.npy"));
        let frame_count = expected_path.shape[1];
        let expected_path_values = expected_path.values();
        let actual_path = matchmaker.alignment_path();
        assert_close(
            &Ctx::new("arzt", name, "alignment_path.perf_time"),
            &expected_path_values[..frame_count],
            &actual_path.perf_time,
            0.0,
            1e-6,
        );
        assert_close(
            &Ctx::new("arzt", name, "alignment_path.score_beat"),
            &expected_path_values[frame_count..],
            &actual_path.score_beat,
            0.0,
            RENDERER_ALIGNMENT_TOLERANCE_BEATS,
        );
    }
}
