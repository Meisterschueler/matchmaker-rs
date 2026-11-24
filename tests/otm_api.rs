//! MIDI-only OTM method registration and public pipeline behavior.
mod common;

use matchmaker_rs::matchmaker::{ConfigValue, Matchmaker, Method, MethodConfig};

#[test]
fn otm_defaults_and_invalid_configuration_are_method_specific() {
    let method: Method = "OTM".parse().unwrap();
    assert_eq!(method, Method::Otm);
    assert_eq!(method.name(), "OTM");
    let mut config = MethodConfig::for_method(method);
    assert_eq!(config.processor.name(), "parangonar");
    assert_eq!(config.polling_period, None);
    assert!(config.set(method, "unknown", &ConfigValue::Int(1)).is_err());
    assert!(
        config
            .set(method, "dim_model", &ConfigValue::Int(32))
            .is_err()
    );
    for deferred in ["OPTM", "SLT_OLTW", "SL_OLTW"] {
        assert!(deferred.parse::<Method>().is_err());
    }
}

#[test]
fn file_and_push_interfaces_return_the_same_time_beat_path() {
    let score = common::data("simple_mozart_k265_var1.musicxml");
    let performance = common::data("simple_mozart_k265_var1.mid");
    let mut file = Matchmaker::new(&score, Method::Otm, None, true).unwrap();
    let expected = file.run_file(&performance).unwrap();
    assert!(!expected.is_empty());
    let mut stream = Matchmaker::new(&score, Method::Otm, None, true).unwrap();
    let mut actual = Vec::new();
    for message in matchmaker_rs::performance::Performance::load(&performance)
        .unwrap()
        .messages()
    {
        actual.extend(stream.push(message).unwrap());
    }
    actual.extend(stream.finish());
    assert_eq!(expected, actual);
    assert_eq!(file.alignment_path(), stream.alignment_path());
    for _ in 0..3 {
        let mut repeated = Matchmaker::new(&score, Method::Otm, None, true).unwrap();
        assert_eq!(repeated.run_file(&performance).unwrap(), expected);
        assert_eq!(repeated.alignment_path(), file.alignment_path());
    }
}
