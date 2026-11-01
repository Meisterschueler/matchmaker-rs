//! Audio method validation and full stage-export coverage.

mod common;

use common::{data, npy};
use matchmaker_rs::Error;
use matchmaker_rs::export::write_audio_run;
use matchmaker_rs::matchmaker::{
    AudioMatchmaker, AudioMethod, AudioMethodConfig, AudioProcessorKind,
};
use std::path::PathBuf;

fn pinned_soundfont() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/audio-assets/UprightPianoKW-small-SF2-20190703")
        .join("UprightPianoKW-small-20190703.sf2")
}

#[test]
fn audio_method_configs_allow_processors_and_reject_incompatible_parameters() {
    for (method, processor) in [
        (AudioMethod::Arzt, AudioProcessorKind::Chroma),
        (AudioMethod::Arzt, AudioProcessorKind::Lse),
        (AudioMethod::Dixon, AudioProcessorKind::Chroma),
        (AudioMethod::Dixon, AudioProcessorKind::Lse),
    ] {
        let mut config = AudioMethodConfig::for_method(method);
        config.processor = processor;
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
}

#[test]
fn recorded_audio_run_exports_all_stages_as_numpy_arrays() {
    let soundfont = pinned_soundfont();
    if !soundfont.is_file() {
        eprintln!("skipping audio export test: local pinned SF2 is unavailable");
        return;
    }

    let mut config = AudioMethodConfig::for_method(AudioMethod::Arzt);
    config.processor = AudioProcessorKind::Lse;
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
