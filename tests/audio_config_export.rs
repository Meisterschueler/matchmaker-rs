//! Audio method validation and full stage-export coverage.

mod common;

use common::{data, npy};
use matchmaker_rs::Error;
use matchmaker_rs::export::{write_audio_evaluation, write_audio_run};
use matchmaker_rs::matchmaker::{
    AudioMatchmaker, AudioMethod, AudioMethodConfig, AudioProcessorKind,
};
use matchmaker_rs::prob::skf::TemplateMode;
use matchmaker_rs::utils::eval::EvaluationResult;
use std::path::PathBuf;

fn pinned_soundfont() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/audio-assets/UprightPianoKW-small-SF2-20190703")
        .join("UprightPianoKW-small-20190703.sf2")
}

#[test]
fn tempo_arzt_exports_state_and_accumulated_cost_windows() {
    let mut config = AudioMethodConfig::for_method(AudioMethod::ArztTempo);
    config.soundfont_path = Some(pinned_soundfont());
    let mut follower = AudioMatchmaker::new_arzt_tempo(
        data("simple_mozart_k265_var1.musicxml"),
        config,
        None,
        true,
    )
    .unwrap()
    .record();
    follower
        .run_file(data("audio/simple_mozart_k265_var1_sf2.wav"))
        .unwrap();
    let output = std::env::temp_dir().join("matchmaker_audio_tempo_export");
    write_audio_run(&follower, &output).unwrap();
    let method = output.join("arzt_tempo");
    let count = follower.alignment_path().perf_time.len();
    assert_eq!(npy(method.join("tempo_states.npy")).shape, [count, 8]);
    assert_eq!(
        npy(method.join("cost_window_offsets.npy")).shape,
        [count + 1]
    );
    assert_eq!(npy(method.join("cost_window_start.npy")).shape, [count]);
    assert!(
        !npy(method.join("cost_window_values.npy"))
            .values()
            .is_empty()
    );
}

#[test]
fn audio_evaluation_export_records_metrics_and_ground_truth_identity() {
    let output = std::env::temp_dir().join("matchmaker_audio_evaluation_export");
    let evaluation = EvaluationResult {
        mean_b: 0.1,
        median_b: 0.05,
        std_b: 0.02,
        accuracy_0_05b: 0.5,
        accuracy_0_1b: 1.0,
        accuracy_0_3b: 1.0,
        accuracy_0_5b: 1.0,
        accuracy_1b: 1.0,
        accuracy_2b: 1.0,
        skewness: 0.0,
        kurtosis: 0.0,
        smoothness: None,
    };
    let ground_truth = data("audio/ground_truth/score_onsets.tsv");
    write_audio_evaluation(AudioMethod::Arzt, &output, &ground_truth, &evaluation).unwrap();
    let method_dir = output.join("arzt");
    let metrics: serde_json::Value =
        serde_json::from_slice(&std::fs::read(method_dir.join("evaluation.json")).unwrap())
            .unwrap();
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(method_dir.join("evaluation_meta.json")).unwrap())
            .unwrap();
    assert_eq!(metrics["accuracy_0.05b"], 0.5);
    assert_eq!(
        metadata["ground_truth_path"],
        ground_truth.to_string_lossy().as_ref()
    );
    assert_eq!(metadata["ground_truth_format"], "perf_sec/score_beat TSV");
}

#[test]
fn audio_method_configs_allow_processors_and_reject_incompatible_parameters() {
    for (method, processor) in [
        (AudioMethod::Arzt, AudioProcessorKind::Chroma),
        (AudioMethod::Arzt, AudioProcessorKind::Lse),
        (AudioMethod::Dixon, AudioProcessorKind::Chroma),
        (AudioMethod::Dixon, AudioProcessorKind::Lse),
        (AudioMethod::ArztMultiFold, AudioProcessorKind::Chroma),
        (AudioMethod::ArztMultiFold, AudioProcessorKind::Lse),
        (AudioMethod::Arzt, AudioProcessorKind::RawSpectrum),
    ] {
        let mut config = AudioMethodConfig::for_method(method);
        config.set_processor(processor);
        if processor == AudioProcessorKind::RawSpectrum {
            config.hop_length = Some(128);
        }
        config.validate(method).unwrap();
    }

    let mut arzt = AudioMethodConfig::for_method(AudioMethod::Arzt);
    arzt.max_run_count = Some(3);
    assert!(matches!(
        arzt.validate(AudioMethod::Arzt),
        Err(Error::AudioConfig { .. })
    ));

    let mut dixon = AudioMethodConfig::for_method(AudioMethod::Dixon);
    dixon.step_size = Some(3);
    assert!(matches!(
        dixon.validate(AudioMethod::Dixon),
        Err(Error::AudioConfig { .. })
    ));

    let mut multi_fold = AudioMethodConfig::for_method(AudioMethod::ArztMultiFold);
    multi_fold.selection_window = None;
    assert!(matches!(
        multi_fold.validate(AudioMethod::ArztMultiFold),
        Err(Error::AudioConfig { .. })
    ));

    let mut raw_spectrum = AudioMethodConfig::for_method(AudioMethod::Arzt);
    raw_spectrum.set_processor(AudioProcessorKind::RawSpectrum);
    raw_spectrum.hop_length = Some(128);
    raw_spectrum.n_fft = Some(127);
    assert!(matches!(
        raw_spectrum.validate(AudioMethod::Arzt),
        Err(Error::AudioConfig { key, .. }) if key == "n_fft"
    ));
    raw_spectrum.n_fft = Some(512);
    raw_spectrum.norm = Some(2.0);
    assert!(matches!(
        raw_spectrum.validate(AudioMethod::Arzt),
        Err(Error::AudioConfig { key, .. }) if key == "norm"
    ));

    let mut raw_tempo = AudioMethodConfig::for_method(AudioMethod::ArztTempo);
    raw_tempo.set_processor(AudioProcessorKind::RawSpectrum);
    raw_tempo.hop_length = Some(128);
    assert!(matches!(
        raw_tempo.validate(AudioMethod::ArztTempo),
        Err(Error::AudioConfig { key, .. }) if key == "processor"
    ));
}

#[test]
fn raw_spectrum_audio_follow_exports_257_feature_bins() {
    let mut config = AudioMethodConfig::for_method(AudioMethod::Arzt);
    config.set_processor(AudioProcessorKind::RawSpectrum);
    config.sample_rate = 44_100;
    config.hop_length = Some(128);
    config.soundfont_path = Some(pinned_soundfont());
    let mut matchmaker =
        AudioMatchmaker::new_arzt(data("simple_mozart_k265_var1.musicxml"), config, None, true)
            .unwrap()
            .record();
    matchmaker
        .run_file(data("audio/simple_mozart_k265_var1_sf2.wav"))
        .unwrap();

    let output = std::env::temp_dir().join("matchmaker_audio_raw_spectrum_export");
    let _ = std::fs::remove_dir_all(&output);
    write_audio_run(&matchmaker, &output).unwrap();
    let features = npy(output.join("processor/raw_spectrum/features.npy"));
    assert_eq!(
        features.shape[0],
        matchmaker.alignment_path().perf_time.len()
    );
    assert_eq!(features.shape[1], 257);
}

#[test]
fn recorded_audio_run_exports_all_stages_as_numpy_arrays() {
    let soundfont = pinned_soundfont();
    let mut config = AudioMethodConfig::for_method(AudioMethod::Arzt);
    config.set_processor(AudioProcessorKind::Lse);
    config.norm = Some(2.0);
    config.soundfont_path = Some(soundfont);
    let mut matchmaker =
        AudioMatchmaker::new_arzt(data("simple_mozart_k265_var1.musicxml"), config, None, true)
            .unwrap()
            .record();
    matchmaker
        .run_file(data("audio/simple_mozart_k265_var1_sf2.wav"))
        .unwrap();

    let output = std::env::temp_dir().join("matchmaker_audio_export");
    let _ = std::fs::remove_dir_all(&output);
    write_audio_run(&matchmaker, &output).unwrap();

    for file in [
        "audio/meta.json",
        "audio/frame_times.npy",
        "processor/lse/features.npy",
        "processor/lse/times.npy",
        "score/audio/features.npy",
        "score/audio/frame_to_beat.npy",
        "arzt/observations_features.npy",
        "arzt/observations_times.npy",
        "arzt/positions.npy",
        "arzt/current_index.npy",
        "arzt/alignment_path.npy",
    ] {
        assert!(output.join(file).is_file(), "missing export stage {file}");
    }

    let performance_features = npy(output.join("processor/lse/features.npy"));
    let score_features = npy(output.join("score/audio/features.npy"));
    let frame_to_beat = npy(output.join("score/audio/frame_to_beat.npy"));
    let alignment_path = npy(output.join("arzt/alignment_path.npy"));
    assert_eq!(performance_features.shape[1], 84);
    assert_eq!(score_features.shape[0], frame_to_beat.shape[0]);
    assert_eq!(alignment_path.shape[0], 2);
    assert_eq!(npy(output.join("audio/frame_times.npy")).shape, [226]);
}

#[test]
fn skf_config_has_reference_defaults_and_rejects_incompatible_parameters() {
    let skf = AudioMethodConfig::for_method(AudioMethod::Skf);
    assert_eq!(skf.processor(), AudioProcessorKind::RawSpectrum);
    assert_eq!(
        (skf.sample_rate, skf.hop_length, skf.n_fft),
        (8_000, Some(128), Some(512))
    );
    assert_eq!(skf.template_mode, Some(TemplateMode::Synthesized));
    skf.validate(AudioMethod::Skf).unwrap();

    let key_of = |config: &AudioMethodConfig, method| match config.validate(method) {
        Err(Error::AudioConfig { key, .. }) => key,
        other => panic!("expected an audio config error, got {other:?}"),
    };
    let mut chroma = skf.clone();
    chroma.set_processor(AudioProcessorKind::Chroma);
    assert_eq!(key_of(&chroma, AudioMethod::Skf), "processor");
    let mut stepped = skf.clone();
    stepped.step_size = Some(3);
    assert_eq!(
        key_of(&stepped, AudioMethod::Skf),
        "max_run_count/start_window_size/step_size"
    );
    let mut unset = skf.clone();
    unset.template_mode = None;
    assert_eq!(key_of(&unset, AudioMethod::Skf), "template_mode");
    let mut foreign = AudioMethodConfig::for_method(AudioMethod::Arzt);
    foreign.template_mode = Some(TemplateMode::Gaussian);
    assert_eq!(key_of(&foreign, AudioMethod::Arzt), "template_mode");
    let mut short_fft = skf;
    short_fft.n_fft = Some(64);
    assert_eq!(key_of(&short_fft, AudioMethod::Skf), "n_fft");
}

#[test]
fn skf_export_records_model_mode_and_hypothesis_state_without_score_audio() {
    let mut config = AudioMethodConfig::for_method(AudioMethod::Skf);
    config.template_mode = Some(TemplateMode::Gaussian);
    let mut follower =
        AudioMatchmaker::new_skf(data("simple_mozart_k265_var1.musicxml"), config, None, true)
            .unwrap()
            .record();
    follower
        .run_file(data("audio/simple_mozart_k265_var1_sf2.wav"))
        .unwrap();
    let model = follower.skf_model().unwrap();
    assert_eq!(model.chords.len(), 54);
    assert!(follower.reference().features.is_empty());

    let output = std::env::temp_dir().join("matchmaker_audio_skf_export");
    let _ = std::fs::remove_dir_all(&output);
    write_audio_run(&follower, &output).unwrap();
    let count = follower.alignment_path().perf_time.len();
    assert!(count > 0);
    assert!(!output.join("score/audio/features.npy").exists());
    assert_eq!(npy(output.join("score/skf/templates.npy")).shape, [54, 257]);
    assert_eq!(npy(output.join("score/skf/lengths.npy")).shape, [54]);
    assert_eq!(npy(output.join("score/skf/chord_offsets.npy")).shape, [55]);
    let method = output.join("skf");
    assert_eq!(npy(method.join("state_frames.npy")).shape, [count]);
    assert_eq!(npy(method.join("state_offsets.npy")).shape, [count + 1]);
    assert_eq!(npy(method.join("candidates.npy")).shape, [count]);
    let hypotheses = npy(method.join("state_chord.npy")).shape[0];
    for file in [
        "state_age.npy",
        "state_probability.npy",
        "state_tempo_mean.npy",
        "state_tempo_variance.npy",
    ] {
        assert_eq!(npy(method.join(file)).shape, [hypotheses], "{file}");
    }
    let read = |file: &str| -> serde_json::Value {
        serde_json::from_slice(&std::fs::read(output.join(file)).unwrap()).unwrap()
    };
    assert_eq!(read("score/meta.json")["template_mode"], "gaussian");
    assert!(read("score/meta.json")["soundfont"].is_null());
    assert_eq!(read("audio/meta.json")["n_fft"], 512);
    assert_eq!(read("skf/meta.json")["config"]["template_mode"], "gaussian");
}
