//! Score-to-audio reference rendering and feature mapping tests.

mod common;

use common::data;
use matchmaker_rs::features::{AudioProcessor, lse::LseProcessor};
use matchmaker_rs::io::audio::AudioFrame;
use matchmaker_rs::matchmaker::{AudioMethod, AudioMethodConfig};
use matchmaker_rs::score::Score;
use matchmaker_rs::score::audio::{build_reference, render_score_audio};
use std::path::PathBuf;

fn pinned_soundfont() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/audio-assets/UprightPianoKW-small-SF2-20190703")
        .join("UprightPianoKW-small-20190703.sf2")
}

#[test]
fn score_lse_uses_unprimed_center_false_windows() {
    let score = Score::load(data("simple_mozart_k265_var1.musicxml"), true).unwrap();
    for frame_rate in [30, 50] {
        let method = AudioMethod::ArztTempo;
        let mut config = AudioMethodConfig::for_method(method);
        config.frame_rate = frame_rate;
        config.set_processor("lse".parse().unwrap());
        let reference = build_reference(&score, pinned_soundfont(), &config, method).unwrap();
        let hop = (config.sample_rate / config.frame_rate) as usize;
        let mut processor = LseProcessor::new(config.sample_rate, hop, config.norm);
        let n_fft = processor.n_fft();
        let expected_count = reference.samples.windows(n_fft).step_by(hop).count();
        assert_eq!(reference.features.len(), expected_count);
        let expected_first = processor
            .process(&AudioFrame {
                index: 0,
                perf_time: 0.0,
                samples: reference.samples[..n_fft].to_vec(),
            })
            .unwrap();
        assert_eq!(
            reference.features[0], expected_first.features,
            "score features must not inherit performance-stream priming"
        );
    }
}

#[test]
fn score_audio_includes_onset_padding_tail_features_and_beat_map() {
    let soundfont = pinned_soundfont();
    let score = Score::load(data("simple_mozart_k265_var1.musicxml"), true).unwrap();
    let config = AudioMethodConfig::for_method(AudioMethod::Arzt);
    let reference = build_reference(&score, &soundfont, &config, AudioMethod::Arzt).unwrap();

    assert_eq!(reference.sample_rate, config.sample_rate);
    assert_eq!(reference.features.len(), reference.frame_to_beat.len());
    assert!(
        reference
            .features
            .iter()
            .all(|features| features.len() == 12)
    );
    assert!(
        reference
            .frame_to_beat
            .windows(2)
            .all(|pair| pair[1] >= pair[0])
    );
    assert!(reference.frame_to_beat.iter().all(|beat| beat.is_finite()));
    assert_eq!(reference.score_positions, score.unique_onsets());

    let tempo = score.tempo.unwrap_or(120.0);
    let last_onset_quarter = score
        .notes
        .iter()
        .map(|note| f64::from(note.onset_quarter))
        .reduce(f64::max)
        .unwrap();
    let last_onset_seconds = last_onset_quarter * 60.0 / tempo;
    let expected_len = ((last_onset_seconds + 0.1) * f64::from(config.sample_rate)) as usize;
    assert_eq!(reference.samples.len(), expected_len);

    let first_onset_seconds = score
        .notes
        .iter()
        .map(|note| f64::from(note.onset_quarter) * 60.0 / tempo)
        .fold(f64::INFINITY, f64::min)
        .max(0.0);
    let padding = (first_onset_seconds * f64::from(config.sample_rate)) as usize;
    assert!(
        reference.samples[..padding]
            .iter()
            .all(|&sample| sample == 0.0)
    );
}

#[test]
fn score_audio_renders_a_leading_rest_only_once() {
    let soundfont = pinned_soundfont();
    let score = Score::load(data("audio/leading_rest.musicxml"), false).unwrap();
    let tempo = score.tempo.unwrap_or(120.0);
    let padding = (f64::from(score.notes[0].onset_quarter) * 60.0 / tempo
        * f64::from(AudioMethodConfig::for_method(AudioMethod::Arzt).sample_rate))
        as usize;
    let samples = render_score_audio(
        &score,
        &soundfont,
        AudioMethodConfig::for_method(AudioMethod::Arzt).sample_rate,
        tempo,
    )
    .unwrap();

    assert!(padding > 0);
    assert!(samples[..padding].iter().all(|&sample| sample == 0.0));
    assert!(samples[padding..].iter().any(|&sample| sample.abs() > 1e-5));
}

#[test]
fn score_audio_processor_state_resets_between_builds() {
    let soundfont = pinned_soundfont();
    let score = Score::load(data("simple_mozart_k265_var1.musicxml"), true).unwrap();
    let config = AudioMethodConfig::for_method(AudioMethod::Dixon);
    let first = build_reference(&score, &soundfont, &config, AudioMethod::Dixon).unwrap();
    let second = build_reference(&score, &soundfont, &config, AudioMethod::Dixon).unwrap();
    assert_eq!(first.features, second.features);
    assert_eq!(first.frame_to_beat, second.frame_to_beat);
}
