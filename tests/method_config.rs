//! `MethodConfig` (what `Matchmaker(kwargs=...)` controls), event-based streaming and score
//! annotations of the `Matchmaker` facade.

mod common;

use common::data;
use matchmaker_rs::Error;
use matchmaker_rs::matchmaker::{
    AnnotationLevel, ConfigValue, Matchmaker, Method, MethodConfig, ProcessorKind,
};

const SCORE: &str = "simple_mozart_k265_var1.musicxml";
const PERFORMANCE: &str = "simple_mozart_k265_var1.mid";

fn invalid_key(err: Error) -> String {
    match err {
        Error::InvalidParameter { key, .. } => key,
        other => panic!("expected InvalidParameter, got {other}"),
    }
}

#[test]
fn defaults_match_the_reference_default_kwargs() {
    let pthmm = MethodConfig::for_method(Method::Pthmm);
    assert_eq!(
        (pthmm.processor, pthmm.piano_range, pthmm.polling_period),
        (ProcessorKind::Pitch, true, Some(0.01))
    );
    assert_eq!(
        MethodConfig::for_method(Method::OuterHmm).polling_period,
        Some(0.001)
    );
}

#[test]
fn unknown_keys_and_bad_values_are_rejected() {
    let mut config = MethodConfig::for_method(Method::Pthmm);
    let err = config
        .set(Method::Pthmm, "patience", &ConfigValue::Int(3))
        .unwrap_err();
    assert!(err.to_string().contains("allowed"), "{err}");
    assert_eq!(invalid_key(err), "patience");

    let bad = config.set(Method::Pthmm, "processor", &ConfigValue::Str("mfcc".into()));
    assert_eq!(invalid_key(bad.unwrap_err()), "processor");
    let bad = config.set(Method::Pthmm, "piano_range", &ConfigValue::Int(1));
    assert_eq!(invalid_key(bad.unwrap_err()), "piano_range");

    let mut outer = MethodConfig::for_method(Method::OuterHmm);
    outer
        .set(Method::OuterHmm, "patience", &ConfigValue::Int(20))
        .unwrap();
    assert_eq!(outer.patience, Some(20));
    let negative = outer.set(Method::OuterHmm, "patience", &ConfigValue::Int(-1));
    assert_eq!(invalid_key(negative.unwrap_err()), "patience");
}

#[test]
fn incompatible_combinations_are_rejected() {
    let build = |method, edit: &dyn Fn(&mut MethodConfig)| {
        let mut config = MethodConfig::for_method(method);
        edit(&mut config);
        Matchmaker::with_config(data(SCORE), method, config, None, true).err()
    };
    let hmm_chords = build(Method::Hmm, &|c| c.processor = ProcessorKind::ChordOnset);
    assert_eq!(invalid_key(hmm_chords.unwrap()), "processor");
    let outer_full_range = build(Method::OuterHmm, &|c| c.piano_range = false);
    assert_eq!(invalid_key(outer_full_range.unwrap()), "piano_range");
    let zero_window = build(Method::Pthmm, &|c| c.polling_period = Some(0.0));
    assert_eq!(invalid_key(zero_window.unwrap()), "polling_period");
}

#[test]
fn chord_onset_processor_and_full_range_follow_the_performance() {
    let mut config = MethodConfig::for_method(Method::Pthmm);
    config.processor = ProcessorKind::ChordOnset;
    let mut mm = Matchmaker::with_config(data(SCORE), Method::Pthmm, config, None, true).unwrap();
    let positions = mm.run_file(data(PERFORMANCE)).unwrap();
    assert!(!positions.is_empty());

    let mut config = MethodConfig::for_method(Method::Pthmm);
    config.piano_range = false;
    let mut mm = Matchmaker::with_config(data(SCORE), Method::Pthmm, config, None, true).unwrap();
    // 128-pitch profiles reproduce the piano-range decisions on piano music
    let full = mm.run_file(data(PERFORMANCE)).unwrap();
    let mut reference = Matchmaker::new(data(SCORE), Method::Pthmm, None, true).unwrap();
    assert_eq!(full, reference.run_file(data(PERFORMANCE)).unwrap());
}

#[test]
fn event_based_streaming_gives_one_observation_per_note() {
    let mut config = MethodConfig::for_method(Method::OuterHmm);
    config
        .set(Method::OuterHmm, "polling_period", &ConfigValue::None)
        .unwrap();
    let mut mm =
        Matchmaker::with_config(data(SCORE), Method::OuterHmm, config, None, true).unwrap();
    assert_eq!(mm.polling_period(), None);
    let positions = mm.run_file(data(PERFORMANCE)).unwrap();
    assert!(!positions.is_empty());
    let last_onset = f64::from(*mm.score_positions().last().unwrap());
    assert!(positions.iter().all(|p| (0.0..=last_onset).contains(p)));
}

#[test]
fn score_annotations_cover_the_onsets() {
    let mm = Matchmaker::new(data(SCORE), Method::Pthmm, None, true).unwrap();
    let onsets = mm.score_positions();
    assert_eq!(mm.score_annotations(AnnotationLevel::Note), onsets);

    let beats = mm.score_annotations(AnnotationLevel::Beat);
    let (first, last) = (onsets[0], *onsets.last().unwrap());
    assert_eq!(beats.first().copied(), Some(first.ceil()));
    assert_eq!(beats.last().copied(), Some(last.floor()));
    assert!(beats.windows(2).all(|w| w[1] - w[0] == 1.0));

    // inserted states sit between onsets in the follower, not in the score positions
    assert_eq!(mm.follower_score_positions().len(), 2 * onsets.len() - 1);
}
