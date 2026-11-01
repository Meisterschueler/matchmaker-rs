//! `MethodConfig` (what `Matchmaker(kwargs=...)` controls), event-based streaming and score
//! annotations of the `Matchmaker` facade.

mod common;

use common::data;
use matchmaker_rs::Error;
use matchmaker_rs::dp::Distance;
use matchmaker_rs::matchmaker::{ConfigValue, Matchmaker, Method, MethodConfig, ProcessorKind};
use matchmaker_rs::score::AnnotationLevel;

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
fn arzt_defaults_and_keys() {
    let mut config = MethodConfig::for_method(Method::Arzt);
    assert_eq!(
        (config.processor, config.piano_range, config.polling_period),
        (ProcessorKind::ChordOnset, true, Some(0.001))
    );
    config
        .set(Method::Arzt, "step_size", &ConfigValue::Int(3))
        .unwrap();
    assert_eq!(config.step_size, Some(3));
    let euclidean = ConfigValue::Str("euclidean".into());
    config
        .set(Method::Arzt, "distance_func", &euclidean)
        .unwrap();
    assert_eq!(config.distance_func, Some(Distance::Euclidean));

    let manhattan = ConfigValue::Str("manhattan".into());
    let bad = config
        .set(Method::Arzt, "distance_func", &manhattan)
        .unwrap_err();
    assert!(bad.to_string().contains("cosine"), "{bad}");
    assert_eq!(invalid_key(bad), "distance_func");
    // the reference's pass-through drops these two for MIDI arzt
    for key in ["window_size", "start_window_size"] {
        let err = config.set(Method::Arzt, key, &ConfigValue::Int(5));
        let err = err.unwrap_err();
        assert!(err.to_string().contains("not used by"), "{err}");
        assert_eq!(invalid_key(err), key);
    }
    let unknown = config.set(Method::Arzt, "patience", &ConfigValue::Int(3));
    assert!(unknown.unwrap_err().to_string().contains("allowed"));
}

#[test]
fn dixon_defaults_and_keys() {
    let mut config = MethodConfig::for_method(Method::Dixon);
    assert_eq!(
        (config.processor, config.piano_range, config.polling_period),
        (ProcessorKind::ChordOnset, true, Some(0.001))
    );
    for key in ["window_size", "max_run_count"] {
        config
            .set(Method::Dixon, key, &ConfigValue::Int(7))
            .unwrap();
    }
    assert_eq!(
        (config.window_size, config.max_run_count),
        (Some(7), Some(7))
    );
    let cosine = ConfigValue::Str("cosine".into());
    config.set(Method::Dixon, "distance_func", &cosine).unwrap();
    assert_eq!(config.distance_func, Some(Distance::Cosine));
    let err = config
        .set(Method::Dixon, "step_size", &ConfigValue::Int(3))
        .unwrap_err();
    assert!(err.to_string().contains("allowed"), "{err}");
    assert_eq!(invalid_key(err), "step_size");
}

#[test]
fn sizes_must_be_at_least_one() {
    for (method, key) in [
        (Method::Arzt, "step_size"),
        (Method::Dixon, "window_size"),
        (Method::Dixon, "max_run_count"),
    ] {
        let mut config = MethodConfig::for_method(method);
        let negative = config.set(method, key, &ConfigValue::Int(-1));
        assert_eq!(invalid_key(negative.unwrap_err()), key);
        // zero is caught where the whole configuration is checked, also for a struct built by hand
        config.set(method, key, &ConfigValue::Int(0)).unwrap();
        let err = Matchmaker::with_config(data(SCORE), method, config, None, true).err();
        assert_eq!(invalid_key(err.unwrap()), key);
    }
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
fn oltw_full_range_follows_like_the_piano_range() {
    // the 40 extra pitches are zero in score and performance: neither metric can change
    for method in [Method::Arzt, Method::Dixon] {
        let mut config = MethodConfig::for_method(method);
        config.piano_range = false;
        let mut full = Matchmaker::with_config(data(SCORE), method, config, None, true).unwrap();
        let mut piano = Matchmaker::new(data(SCORE), method, None, true).unwrap();
        let expected = piano.run_file(data(PERFORMANCE)).unwrap();
        assert_eq!(
            full.run_file(data(PERFORMANCE)).unwrap(),
            expected,
            "{method:?}"
        );
    }
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
    assert_eq!(mm.score().annotations(AnnotationLevel::Note, false), onsets);

    let beats = mm.score().annotations(AnnotationLevel::Beat, false);
    let (first, last) = (onsets[0], *onsets.last().unwrap());
    assert_eq!(beats.first().copied(), Some(first.ceil()));
    assert_eq!(beats.last().copied(), Some(last.floor()));
    assert!(beats.windows(2).all(|w| w[1] - w[0] == 1.0));

    // inserted states sit between onsets in the follower, not in the score positions
    assert_eq!(mm.follower_score_positions().len(), 2 * onsets.len() - 1);
}
