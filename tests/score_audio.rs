//! Score-to-audio reference rendering and feature mapping tests.

mod common;

use common::data;
use matchmaker_rs::matchmaker::{AudioMethod, AudioMethodConfig};
use matchmaker_rs::score::Score;
use matchmaker_rs::score::audio::build_reference;
use std::path::PathBuf;

fn pinned_soundfont() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/audio-assets/UprightPianoKW-small-SF2-20190703")
        .join("UprightPianoKW-small-20190703.sf2")
}

#[test]
fn score_audio_includes_onset_padding_tail_features_and_beat_map() {
    let soundfont = pinned_soundfont();
    if !soundfont.is_file() {
        eprintln!("skipping score audio renderer test: local pinned SF2 is unavailable");
        return;
    }
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
fn score_audio_processor_state_resets_between_builds() {
    let soundfont = pinned_soundfont();
    if !soundfont.is_file() {
        eprintln!("skipping score audio reset test: local pinned SF2 is unavailable");
        return;
    }
    let score = Score::load(data("simple_mozart_k265_var1.musicxml"), true).unwrap();
    let config = AudioMethodConfig::for_method(AudioMethod::Dixon);
    let first = build_reference(&score, &soundfont, &config, AudioMethod::Dixon).unwrap();
    let second = build_reference(&score, &soundfont, &config, AudioMethod::Dixon).unwrap();
    assert_eq!(first.features, second.features);
    assert_eq!(first.frame_to_beat, second.frame_to_beat);
}
