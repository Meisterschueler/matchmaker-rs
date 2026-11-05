//! Audio method validation and full stage-export coverage.

mod common;

use common::{data, npy};
use matchmaker_rs::Error;
use matchmaker_rs::export::{write_audio_evaluation, write_audio_run};
use matchmaker_rs::matchmaker::{
    AudioMatchmaker, AudioMethod, AudioMethodConfig, AudioProcessorKind,
};
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
    ] {
        let mut config = AudioMethodConfig::for_method(method);
        config.set_processor(processor);
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
