//! Golden score-model and seeded Korzeniowski particle-filter fixtures.

mod common;

use common::{Ctx, assert_close, data, golden, json, npy};
use matchmaker_rs::matchmaker::{AudioMatchmaker, AudioMethod, AudioMethodConfig};
use matchmaker_rs::score::{Score, korzeniowski::KorzeniowskiScoreModel};

const SAMPLE_RATE: u32 = 44_100;
const N_FFT: usize = 4_096;

#[test]
fn analytical_score_model_matches_reference_arrays() {
    let score = Score::load(data("simple_mozart_k265_var1.musicxml"), true).unwrap();
    let model = KorzeniowskiScoreModel::from_score(&score, SAMPLE_RATE, N_FFT).unwrap();
    let reference = golden("audio/pfkorz/score_model");

    for (name, actual) in [
        ("beat_grid.npy", model.beat_grid.clone()),
        (
            "onset_positions.npy",
            model
                .onset_positions
                .iter()
                .map(|&value| f64::from(value))
                .collect(),
        ),
        (
            "nearest_onsets.npy",
            model
                .nearest_onsets
                .iter()
                .map(|&value| f64::from(value))
                .collect(),
        ),
        (
            "rest_mask.npy",
            model
                .rest_mask
                .iter()
                .map(|&rest| if rest { 1.0 } else { 0.0 })
                .collect(),
        ),
    ] {
        let expected = npy(reference.join(name));
        assert_close(
            &Ctx::new("pfkorz", "score-model", name),
            &expected.values(),
            &actual,
            1e-7,
            1e-9,
        );
    }

    let expected_templates = npy(reference.join("templates.npy"));
    let actual_templates: Vec<f64> = model
        .templates
        .iter()
        .flatten()
        .map(|&value| f64::from(value))
        .collect();
    assert_close(
        &Ctx::new("pfkorz", "score-model", "templates.npy"),
        &expected_templates.values(),
        &actual_templates,
        1e-5,
        1e-5,
    );

    let offsets = npy(reference.join("active_note_offsets.npy")).values();
    let notes = npy(reference.join("active_notes.npy")).values();
    let actual_offsets: Vec<f64> = model
        .active_notes
        .iter()
        .scan(0usize, |offset, active| {
            *offset += active.len();
            Some(*offset as f64)
        })
        .collect();
    let actual_notes: Vec<f64> = model
        .active_notes
        .iter()
        .flatten()
        .map(|&pitch| pitch as f64)
        .collect();
    assert_eq!(offsets.len(), actual_offsets.len() + 1);
    assert_eq!(offsets[1..], actual_offsets);
    assert_eq!(notes, actual_notes);
}

#[test]
fn reference_contains_controlled_draws_and_all_five_seed_records() {
    let reference = golden("audio/pfkorz");
    let meta = json(reference.join("meta.json"));
    assert_eq!(
        meta["seeds"],
        serde_json::json!([1984, 1985, 1986, 1987, 1988])
    );
    assert_eq!(meta["numpy_rng_identity_claim"], false);
    assert_eq!(npy(reference.join("performance_times.npy")).shape, [227]);
    assert_eq!(
        npy(reference.join("performance_spectrum.npy")).shape,
        [227, N_FFT / 2 + 1]
    );
    for name in [
        "fast_choice_uniforms.npy",
        "standard_normals.npy",
        "resample_offsets.npy",
    ] {
        assert!(
            !npy(reference.join("fixed_draws").join(name))
                .values()
                .is_empty()
        );
    }
    for seed in 1984..=1988 {
        let directory = reference.join(format!("seed-{seed}"));
        assert_eq!(
            npy(directory.join("alignment_path.npy")).shape,
            [2, 227],
            "missing or malformed seed record {seed}"
        );
        assert!(json(directory.join("evaluation.json"))["median_b"].is_number());
    }
}

#[test]
fn facade_runs_without_a_soundfont_and_repeats_exactly_for_a_seed() {
    let make = || {
        let mut config = AudioMethodConfig::for_method(AudioMethod::PfKorz);
        config.num_particles = Some(32);
        AudioMatchmaker::new_pfkorz(data("simple_mozart_k265_var1.musicxml"), config, None, true)
            .unwrap()
            .record()
    };
    let performance = data("audio/simple_mozart_k265_var1_sf2.wav");
    let mut first = make();
    let mut second = make();
    assert_eq!(
        first.config().processor(),
        matchmaker_rs::matchmaker::AudioProcessorKind::Korzeniowski
    );
    assert_eq!(first.config().random_seed, Some(1984));
    assert_eq!(first.config().num_particles, Some(32));

    let first_positions = first.run_file(&performance).unwrap();
    let second_positions = second.run_file(&performance).unwrap();
    assert_eq!(first_positions.len(), 227);
    assert_eq!(first_positions, second_positions);
    assert_eq!(first.alignment_path(), second.alignment_path());
    let recording = first.recording().unwrap();
    let second_recording = second.recording().unwrap();
    assert_eq!(recording.korzeniowski_outputs.len(), 227);
    assert_eq!(recording.pfkorz_steps.len(), 227);
    assert_eq!(recording.pfkorz_steps, second_recording.pfkorz_steps);
    assert!(
        recording
            .pfkorz_steps
            .iter()
            .all(|step| step.positions.len() == 32 && step.weights.len() == 32)
    );
}
