//! Invalid OTM inputs do not mutate a successful alignment.
mod common;

use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::dp::otm::{OnlineTransformerMatcher, OtmConfig};
use matchmaker_rs::features::Observation;
use matchmaker_rs::matchmaker::{AudioMethod, ConfigValue, Matchmaker, Method, MethodConfig};
use matchmaker_rs::score::Score;

#[test]
fn malformed_inputs_are_rejected_before_updating_tracker_state() {
    let score = Score::load(common::data("simple_mozart_k265_var1.musicxml"), true).unwrap();
    assert!(OnlineTransformerMatcher::new(Vec::new(), OtmConfig::default()).is_err());
    let mut follower =
        OnlineTransformerMatcher::new(score.otm_notes(), OtmConfig::default()).unwrap();
    for (features, perf_time) in [
        (vec![19.0], 0.0),
        (vec![111.0], 0.0),
        (vec![60.5], 0.0),
        (vec![], 0.0),
        (vec![60.0, 64.0], 0.0),
        (vec![60.0], f64::NAN),
    ] {
        assert!(
            follower
                .try_call(&Observation {
                    features,
                    perf_time
                })
                .is_err()
        );
        assert!(follower.alignment_path().perf_time.is_empty());
    }
}

#[test]
fn incompatible_configuration_is_rejected_at_construction() {
    assert!("OTM".parse::<AudioMethod>().is_err());
    let score = common::data("simple_mozart_k265_var1.musicxml");
    for (key, value) in [
        ("lookback", ConfigValue::Int(0)),
        ("init_beat_period", ConfigValue::Float(-1.0)),
        ("polling_period", ConfigValue::Float(0.01)),
        ("processor", ConfigValue::Str("pitch".into())),
    ] {
        let mut config = MethodConfig::for_method(Method::Otm);
        config.set(Method::Otm, key, &value).unwrap();
        let result = Matchmaker::with_config(&score, Method::Otm, config, None, true);
        assert!(result.is_err(), "{key}");
    }
    let mut empty = Matchmaker::new(score, Method::Otm, None, true).unwrap();
    assert!(empty.finish().is_empty());
    assert!(empty.alignment_path().perf_time.is_empty());
    assert!(empty.alignment_path().score_beat.is_empty());
}
