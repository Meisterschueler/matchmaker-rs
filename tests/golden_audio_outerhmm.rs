//! Golden tests for audio `outerhmm`: fixed-template replay against pymatchmaker 0.4.0.

mod common;

use common::{Ctx, assert_close, assert_exact, data, golden, json, npy};
use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::features::Observation;
use matchmaker_rs::matchmaker::{
    AudioMatchmaker, AudioMethod, AudioMethodConfig, AudioProcessorKind,
};
use matchmaker_rs::prob::outerhmm_audio::{
    AudioOuterHmm, AudioOuterHmmConfig, LEARNED_TEMPLATE_SHA256, build_pitch_templates,
};
use matchmaker_rs::score::ScoreNote;
use std::path::Path;

const CASES: [&str; 4] = ["weighted", "repeats", "one_chord", "bundled"];

fn case_dir(case: &str) -> std::path::PathBuf {
    golden(&format!("audio/outerhmm/{case}"))
}

fn notes(dir: &Path) -> Vec<ScoreNote> {
    let beats = npy(dir.join("note_onset_beat.npy")).values();
    let quarters = npy(dir.join("note_onset_quarter.npy")).values();
    let durations = npy(dir.join("note_duration_beat.npy")).values();
    let pitches = npy(dir.join("note_pitch.npy")).values();
    (0..beats.len())
        .map(|index| ScoreNote {
            id: format!("n{index}"),
            pitch: pitches[index] as i32,
            onset_beat: beats[index] as f32,
            duration_beat: durations[index] as f32,
            onset_quarter: quarters[index] as f32,
            duration_quarter: durations[index] as f32,
            onset_div: 0,
            duration_div: 1,
            voice: 1,
        })
        .collect()
}

fn matrix(path: impl AsRef<Path>) -> Vec<Vec<f64>> {
    let array = npy(path);
    array
        .values()
        .chunks(array.shape[1])
        .map(<[f64]>::to_vec)
        .collect()
}

#[test]
fn fixed_template_replay_matches_weighted_chords_and_sound_pause_state() {
    for case in CASES {
        let dir = case_dir(case);
        let meta = json(dir.join("meta.json"));
        let follower = &meta["follower"];
        let config = AudioOuterHmmConfig {
            tempo: meta["tempo"].as_f64().unwrap(),
            sample_rate: follower["sample_rate"].as_u64().unwrap() as u32,
            hop_length: follower["hop_length"].as_u64().unwrap() as usize,
            s_j: follower["s_j"].as_f64().unwrap(),
        };
        let pitch_templates = matrix(golden("audio/outerhmm/pitch_templates.npy"));
        let mut follower = AudioOuterHmm::new(&notes(&dir), pitch_templates, config).unwrap();
        let expected_chords = npy(dir.join("chord_templates.npy"));
        let model = follower.model();
        assert_close(
            &Ctx::new("outerhmm", case, "chord_templates.npy"),
            &expected_chords.values(),
            &model.chord_templates().concat(),
            1e-7,
            1e-9,
        );

        let observations = npy(dir.join("observations.npy"));
        let times = npy(dir.join("perf_times.npy")).values();
        let expected_likelihoods = npy(dir.join("sound_likelihoods.npy"));
        let expected_pauses = npy(dir.join("pause_likelihood.npy")).values();
        let mut positions = Vec::new();
        let mut indices = Vec::new();
        let mut states = Vec::new();
        let mut likelihoods = Vec::new();
        let mut pauses = Vec::new();
        for (row, &perf_time) in observations
            .values()
            .chunks(observations.shape[1])
            .zip(&times)
        {
            let features: Vec<f32> = row.iter().map(|&value| value as f32).collect();
            likelihoods.extend(follower.compute_obs_likelihood(&features).unwrap());
            pauses.push(follower.compute_pause_emission(&features).unwrap());
            positions.push(follower.call(&Observation {
                features,
                perf_time,
            }));
            let debug = follower.debug_state();
            indices.push(debug.current_index as f64);
            states.push(debug.probabilities);
        }
        assert!(
            follower.is_still_following(),
            "{case}: audio outerhmm consumes until stream end"
        );
        let tolerances = &meta["tolerances"];
        assert_close(
            &Ctx::new("outerhmm", case, "sound_likelihoods.npy"),
            &expected_likelihoods.values(),
            &likelihoods,
            tolerances["model_rtol"].as_f64().unwrap(),
            tolerances["model_atol"].as_f64().unwrap(),
        );
        assert_close(
            &Ctx::new("outerhmm", case, "pause_likelihood.npy"),
            &expected_pauses,
            &pauses,
            tolerances["model_rtol"].as_f64().unwrap(),
            tolerances["model_atol"].as_f64().unwrap(),
        );
        assert_close(
            &Ctx::new("outerhmm", case, "positions.npy"),
            &npy(dir.join("positions.npy")).values(),
            &positions,
            0.0,
            tolerances["position_atol"].as_f64().unwrap(),
        );
        assert_exact(
            &Ctx::new("outerhmm", case, "current_index.npy"),
            &npy(dir.join("current_index.npy")).values(),
            &indices,
        );
        assert_close(
            &Ctx::new("outerhmm", case, "state_probabilities.npy"),
            &npy(dir.join("state_probabilities.npy")).values(),
            &states.concat(),
            tolerances["model_rtol"].as_f64().unwrap(),
            tolerances["model_atol"].as_f64().unwrap(),
        );
    }
}

#[test]
fn template_archive_support_and_overlay_are_pinned() {
    let base = golden("audio/outerhmm");
    let meta = json(base.join("template_meta.json"));
    assert_eq!(meta["license"], "Apache-2.0");
    assert_eq!(meta["support_rows"], 77);
    assert_eq!(meta["archive_sha256"], LEARNED_TEMPLATE_SHA256);
    assert_eq!(
        npy(base.join("template_support.npy"))
            .values()
            .iter()
            .filter(|&&value| value != 0.0)
            .count(),
        77
    );
    assert_eq!(npy(base.join("pitch_templates.npy")).shape, [88, 88]);
}

#[test]
fn synthesized_pitch_templates_match_reference_and_learned_support() {
    let soundfont = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/audio-assets/UprightPianoKW-small-SF2-20190703")
        .join("UprightPianoKW-small-20190703.sf2");
    let base = golden("audio/outerhmm");
    let meta = json(base.join("template_meta.json"));
    let expected = matrix(base.join("pitch_templates.npy"));
    let (actual, support) = build_pitch_templates(&soundfont, 16_000, 640).unwrap();
    assert_eq!(
        build_pitch_templates(&soundfont, 16_000, 640).unwrap(),
        (actual.clone(), support.clone())
    );
    assert_eq!(support.iter().filter(|&&supported| supported).count(), 77);
    let mut minimum_cosine: f64 = 1.0;
    let mut maximum_variation: f64 = 0.0;
    for (pitch, (rust, reference)) in actual.iter().zip(&expected).enumerate() {
        if support[pitch] {
            assert_eq!(
                rust,
                reference,
                "learned row {} is overlaid unchanged",
                pitch + 21
            );
            continue;
        }
        let dot = rust.iter().zip(reference).map(|(a, b)| a * b).sum::<f64>();
        let norm = |row: &[f64]| row.iter().map(|value| value * value).sum::<f64>().sqrt();
        let cosine = dot / (norm(rust) * norm(reference));
        let variation = 0.5
            * rust
                .iter()
                .zip(reference)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
        minimum_cosine = minimum_cosine.min(cosine);
        maximum_variation = maximum_variation.max(variation);
        assert!(
            (rust.iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "pitch {} template is normalized",
            pitch + 21
        );
    }
    println!(
        "TEMPLATE\tmin cosine {minimum_cosine:.5}\tmax total variation {maximum_variation:.4}"
    );
    assert_eq!(
        meta["support_rows"],
        support.iter().filter(|&&value| value).count()
    );
}

#[test]
fn audio_matchmaker_uses_note_array_and_stream_flux_with_reference_rate_override() {
    let soundfont = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/audio-assets/UprightPianoKW-small-SF2-20190703")
        .join("UprightPianoKW-small-20190703.sf2");
    let mut config = AudioMethodConfig::for_method(AudioMethod::OuterHmm);
    config.soundfont_path = Some(soundfont.clone());
    assert_eq!(config.processor(), AudioProcessorKind::CqtSpectralFlux);
    let mut matchmaker =
        AudioMatchmaker::new_outerhmm(data("simple_mozart_k265_var1.musicxml"), config, None, true)
            .unwrap()
            .record();
    assert!(matchmaker.reference().features.is_empty());
    assert_eq!(
        matchmaker
            .outerhmm_model()
            .unwrap()
            .learned_support()
            .iter()
            .filter(|&&value| value)
            .count(),
        77
    );
    let positions = matchmaker
        .run_file(data("audio/simple_mozart_k265_var1_sf2.wav"))
        .unwrap();
    assert!(!positions.is_empty());
    assert!(positions.iter().all(|position| position.is_finite()));
    let recording = matchmaker.recording().unwrap();
    assert_eq!(recording.steps.len(), positions.len());
    assert!(
        recording
            .processor_outputs
            .iter()
            .all(|row| row.features.len() == 89)
    );

    let mut overridden = AudioMethodConfig::for_method(AudioMethod::OuterHmm);
    overridden.sample_rate = 22_050;
    overridden.hop_length = Some(512);
    overridden.soundfont_path = Some(soundfont);
    let matchmaker = AudioMatchmaker::new_outerhmm(
        data("simple_mozart_k265_var1.musicxml"),
        overridden,
        None,
        true,
    )
    .unwrap();
    assert_eq!(matchmaker.config().sample_rate, 22_050);
    assert_eq!(
        matchmaker
            .outerhmm_model()
            .unwrap()
            .learned_support()
            .iter()
            .filter(|&&value| value)
            .count(),
        0
    );
}

#[test]
fn rejects_empty_scores_bad_template_shapes_and_invalid_observations() {
    let dir = case_dir("one_chord");
    let score_notes = notes(&dir);
    assert!(
        AudioOuterHmm::new(&[], vec![vec![0.0; 88]; 88], AudioOuterHmmConfig::default()).is_err()
    );
    assert!(
        AudioOuterHmm::new(
            &score_notes,
            vec![vec![0.0; 87]; 88],
            AudioOuterHmmConfig::default()
        )
        .is_err()
    );

    let mut invalid_config = AudioOuterHmmConfig {
        s_j: 1.1,
        ..AudioOuterHmmConfig::default()
    };
    assert!(
        AudioOuterHmm::new(
            &score_notes,
            vec![vec![1.0 / 88.0; 88]; 88],
            invalid_config.clone()
        )
        .is_err()
    );
    invalid_config.s_j = 0.0;
    let follower =
        AudioOuterHmm::new(&score_notes, vec![vec![1.0 / 88.0; 88]; 88], invalid_config).unwrap();
    assert!(follower.compute_obs_likelihood(&[0.0; 87]).is_err());
    assert!(follower.compute_pause_emission(&[f32::NAN; 88]).is_err());
}
