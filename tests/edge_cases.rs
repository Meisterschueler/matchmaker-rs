//! Edge cases of the US1 pipeline (spec.md "Edge Cases"): empty and oversized performances,
//! velocity-0 note-ons, missing tempo, unreadable files and out-of-range pitches. None of them
//! may panic; broken files give typed errors that name the file. The `oltw_*` tests are the edge
//! cases of `arzt` and `dixon` (specs/002-midi-oltw/spec.md).

mod common;

use std::path::PathBuf;

use common::data;
use matchmaker_rs::Error;
use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::export::write_audio_features;
use matchmaker_rs::features::Observation;
use matchmaker_rs::io::stream::MidiMessageKind;
use matchmaker_rs::matchmaker::{
    AudioMatchmaker, AudioMethod, AudioMethodConfig, AudioProcessorKind, Matchmaker, Method,
};
use matchmaker_rs::performance::Performance;
use matchmaker_rs::prob::hmm::{PitchIoiHmm, PitchIoiHmmConfig};
use matchmaker_rs::prob::skf::TemplateMode;
use matchmaker_rs::score::Score;
use midly::num::{u4, u7, u15, u28};
use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind};

const SCORE: &str = "simple_mozart_k265_var1.musicxml";
const ALL_METHODS: [Method; 5] = [
    Method::Pthmm,
    Method::Hmm,
    Method::OuterHmm,
    Method::Arzt,
    Method::Dixon,
];

/// `(delta ticks, key, velocity)` note-on events (velocity 0 = note off) as a type-0 MIDI file.
fn midi_file(name: &str, events: &[(u32, u8, u8)]) -> PathBuf {
    let mut track: Vec<TrackEvent> = events
        .iter()
        .map(|&(delta, key, vel)| TrackEvent {
            delta: u28::new(delta),
            kind: TrackEventKind::Midi {
                channel: u4::new(0),
                message: MidiMessage::NoteOn {
                    key: u7::new(key),
                    vel: u7::new(vel),
                },
            },
        })
        .collect();
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    let header = Header::new(Format::SingleTrack, Timing::Metrical(u15::new(480)));
    let path = std::env::temp_dir().join(format!("matchmaker_edge_{name}.mid"));
    Smf {
        header,
        tracks: vec![track],
    }
    .save(&path)
    .unwrap();
    path
}

fn temp_file(name: &str, bytes: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(format!("matchmaker_edge_{name}"));
    std::fs::write(&path, bytes).unwrap();
    path
}

fn audio_wav(name: &str, channels: u16, samples: &[i16]) -> PathBuf {
    let data_len = (samples.len() * 2) as u32;
    let block_align = channels * 2;
    let mut bytes = Vec::with_capacity(data_len as usize + 44);
    bytes.extend(b"RIFF");
    bytes.extend((36 + data_len).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(channels.to_le_bytes());
    bytes.extend(44_100u32.to_le_bytes());
    bytes.extend((44_100u32 * u32::from(block_align)).to_le_bytes());
    bytes.extend(block_align.to_le_bytes());
    bytes.extend(16u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend(data_len.to_le_bytes());
    bytes.extend(samples.iter().flat_map(|sample| sample.to_le_bytes()));
    temp_file(&format!("{name}.wav"), &bytes)
}

fn audio_melody_wav(name: &str, duration_seconds: usize) -> PathBuf {
    let sample_rate = 44_100usize;
    let note_length = sample_rate / 2;
    let frequencies = [261.63, 293.66, 329.63, 349.23, 392.0, 440.0, 493.88, 523.25];
    let samples: Vec<i16> = (0..sample_rate * duration_seconds)
        .map(|index| {
            let frequency = frequencies[(index / note_length) % frequencies.len()];
            let value =
                (std::f64::consts::TAU * frequency * index as f64 / sample_rate as f64).sin() * 0.4;
            (value * f64::from(i16::MAX)) as i16
        })
        .collect();
    audio_wav(name, 1, &samples)
}

#[test]
fn empty_performance_gives_an_empty_path() {
    let performance = midi_file("empty", &[]);
    for method in ALL_METHODS {
        let mut mm = Matchmaker::new(data(SCORE), method, None, true).unwrap();
        let positions = mm.run_file(&performance).unwrap();
        assert!(positions.is_empty(), "{method:?}");
        assert!(mm.alignment_path().perf_time.is_empty(), "{method:?}");
    }
}

#[test]
fn performance_longer_than_the_score_stays_within_the_score() {
    // far more notes than the score has chords
    let events: Vec<(u32, u8, u8)> = (0..300)
        .flat_map(|i| {
            [
                (if i == 0 { 0 } else { 10 }, 60 + (i % 12) as u8, 80),
                (10, 60 + (i % 12) as u8, 0),
            ]
        })
        .collect();
    let performance = midi_file("long", &events);
    for method in ALL_METHODS {
        let mut mm = Matchmaker::new(data(SCORE), method, None, true).unwrap();
        let last = f64::from(*mm.score_positions().last().unwrap());
        let positions = mm.run_file(&performance).unwrap();
        assert!(!positions.is_empty(), "{method:?}");
        assert!(
            positions.iter().all(|p| (0.0..=last).contains(p)),
            "{method:?}: position outside the score"
        );
    }
}

#[test]
fn velocity_zero_note_on_is_a_note_off() {
    let performance = midi_file("vel0", &[(0, 60, 80), (480, 60, 0), (0, 62, 0)]);
    let messages = Performance::load(&performance).unwrap().messages();
    let on = messages
        .iter()
        .filter(|m| m.note_on_pitch().is_some())
        .count();
    let off = messages
        .iter()
        .filter(|m| matches!(m.kind, MidiMessageKind::NoteOff { note: 60 }))
        .count();
    // the lone velocity-0 on of pitch 62 has no note to end and is ignored
    assert_eq!((on, off, messages.len()), (1, 1, 2), "{messages:?}");
}

#[test]
fn score_without_tempo_uses_120_bpm() {
    let mm = Matchmaker::new(data(SCORE), Method::Pthmm, None, true).unwrap();
    assert_eq!(mm.score().tempo, None);
    assert_eq!(mm.tempo(), 120.0);
    let mm = Matchmaker::new(data(SCORE), Method::Pthmm, Some(90.0), true).unwrap();
    assert_eq!(mm.tempo(), 90.0);
}

#[test]
fn broken_files_give_typed_errors_naming_the_file() {
    let text = temp_file("broken.musicxml", b"this is not xml");
    let err = Score::load(&text, true).unwrap_err();
    assert!(
        matches!(err, Error::MusicXml { .. } | Error::InvalidScore { .. }),
        "{err}"
    );
    assert!(
        err.to_string().contains("matchmaker_edge_broken.musicxml"),
        "{err}"
    );

    let other = temp_file("score.txt", b"x");
    let err = Score::load(&other, true).unwrap_err();
    assert!(matches!(err, Error::UnsupportedFormat { .. }), "{err}");
    assert!(
        err.to_string().contains("matchmaker_edge_score.txt"),
        "{err}"
    );

    let err = Score::load(
        std::env::temp_dir().join("matchmaker_edge_missing.musicxml"),
        true,
    )
    .unwrap_err();
    assert!(matches!(err, Error::Io { .. }), "{err}");

    let midi = temp_file("broken.mid", b"MThd but not really");
    let err = Performance::load(&midi).unwrap_err();
    assert!(matches!(err, Error::Midi { .. }), "{err}");
    assert!(
        err.to_string().contains("matchmaker_edge_broken.mid"),
        "{err}"
    );
}

#[test]
fn truncated_files_do_not_panic() {
    let xml = std::fs::read(data(SCORE)).unwrap();
    let mid = std::fs::read(data("simple_mozart_k265_var1.mid")).unwrap();
    for cut in [0, 1, 10, xml.len() / 2, xml.len() - 20] {
        let path = temp_file("cut.musicxml", &xml[..cut]);
        assert!(Score::load(&path, true).is_err(), "xml cut at {cut}");
    }
    for cut in [0, 5, 14, mid.len() / 2] {
        let path = temp_file("cut.mid", &mid[..cut]);
        // a truncated track may still hold complete events; it must not panic either way
        let _ = Performance::load(&path);
    }
}

#[test]
fn hmm_ignores_pitches_outside_the_piano_range() {
    let score = Score::load(data(SCORE), true).unwrap();
    let call = |features: Vec<f32>| {
        let mut hmm = PitchIoiHmm::new(&score, PitchIoiHmmConfig::default());
        let position = hmm.call(&Observation {
            features,
            perf_time: 0.1,
        });
        (position, hmm.forward_variable().unwrap().to_vec())
    };
    // documented deviation (research R3): the reference wraps or raises for these
    assert_eq!(call(vec![5.0, -3.0, 200.0]), call(vec![5.0]));
}

#[test]
fn oltw_handles_scores_smaller_than_the_window() {
    let chord = |key: u8| [(0, key, 80), (480, key, 0)];
    let three: Vec<_> = [60, 62, 64].into_iter().flat_map(chord).collect();
    let one_onset = midi_file("one_onset", &chord(60));
    let three_onsets = midi_file("three_onsets", &three);
    let performance = midi_file("short", &three[..4]);
    for method in [Method::Arzt, Method::Dixon] {
        // one onset is already the last state: the reference's run loop never calls the follower
        let mut mm = Matchmaker::new(&one_onset, method, None, true).unwrap();
        assert!(mm.run_file(&performance).unwrap().is_empty(), "{method:?}");

        let mut mm = Matchmaker::new(&three_onsets, method, None, true).unwrap();
        // the MIDI score starts with a pickup measure, so its beats are negative
        let (first, last) = (
            mm.score_positions()[0],
            *mm.score_positions().last().unwrap(),
        );
        let positions = mm.run_file(&performance).unwrap();
        assert!(!positions.is_empty(), "{method:?}");
        assert!(
            positions
                .iter()
                .all(|p| (f64::from(first)..=f64::from(last)).contains(p)),
            "{method:?}: {positions:?} outside the score {:?}",
            mm.score_positions()
        );
    }
}

#[test]
fn oltw_zero_vector_observations_do_not_panic() {
    // pitches below the piano range leave the chord vector empty: cosine distance is NaN
    let performance = midi_file(
        "out_of_range",
        &[(0, 10, 80), (480, 10, 0), (0, 12, 80), (480, 12, 0)],
    );
    for method in [Method::Arzt, Method::Dixon] {
        let mut mm = Matchmaker::new(data(SCORE), method, None, true).unwrap();
        let last = f64::from(*mm.score_positions().last().unwrap());
        let positions = mm.run_file(&performance).unwrap();
        assert!(
            positions.iter().all(|p| (0.0..=last).contains(p)),
            "{method:?}: position outside the score"
        );
    }
}

#[test]
fn audio_followers_handle_short_silent_overrun_and_one_onset_scores() {
    let soundfont = data(
        "../../target/audio-assets/UprightPianoKW-small-SF2-20190703/UprightPianoKW-small-20190703.sf2",
    );
    let short = audio_wav("audio_short", 1, &[0; 100]);
    let silent = audio_wav("audio_silent", 1, &vec![0; 44_100]);
    let empty = audio_wav("audio_empty", 1, &[]);
    let overrun = audio_melody_wav("audio_overrun", 30);
    let one_onset = midi_file("audio_one_onset_score", &[(0, 60, 80), (480, 60, 0)]);

    for method in [
        AudioMethod::Arzt,
        AudioMethod::Dixon,
        AudioMethod::ArztTempo,
        AudioMethod::ArztMultiFold,
    ] {
        let make = |score: &PathBuf| {
            let mut config = AudioMethodConfig::for_method(method);
            config.soundfont_path = Some(soundfont.clone());
            match method {
                AudioMethod::Arzt => AudioMatchmaker::new_arzt(score, config, None, true).unwrap(),
                AudioMethod::ArztTempo => {
                    AudioMatchmaker::new_arzt_tempo(score, config, None, true).unwrap()
                }
                AudioMethod::ArztMultiFold => {
                    AudioMatchmaker::new_arzt_multi_fold(score, config, None, true).unwrap()
                }
                AudioMethod::Dixon => {
                    AudioMatchmaker::new_dixon(score, config, None, true).unwrap()
                }
                AudioMethod::Skf => unreachable!("skf has its own edge-case test"),
            }
        };

        let mut short_run = make(&data(SCORE));
        assert!(short_run.run_file(&short).unwrap().is_empty(), "{method:?}");

        let mut silent_run = make(&data(SCORE));
        let silent_positions = silent_run.run_file(&silent).unwrap();
        assert!(
            silent_positions.iter().all(|position| position.is_finite()),
            "{method:?} produced a non-finite position"
        );

        let mut long_run = make(&data(SCORE)).record();
        let long_positions = long_run.run_file(&overrun).unwrap();
        let frame_to_beat = &long_run.reference().frame_to_beat;
        let reference_frames = long_run
            .recording()
            .and_then(|recording| recording.steps.last())
            .and_then(|step| step.debug.tempo.as_ref())
            .map_or(frame_to_beat.len(), |tempo| tempo.state[7] as usize);
        if method == AudioMethod::ArztMultiFold {
            assert_eq!(
                long_positions.len(),
                long_run.recording().unwrap().frame_times.len(),
                "{method:?} must consume the complete performance"
            );
        } else {
            assert!(
                long_positions.len() <= reference_frames,
                "{method:?}: {} positions for {} score frames; final beat {}",
                long_positions.len(),
                reference_frames,
                frame_to_beat.last().unwrap()
            );
        }
        assert!(long_positions.iter().all(|position| {
            *position >= frame_to_beat[0]
                && *position <= *frame_to_beat.last().expect("score frame beat map")
        }));

        let mut one_onset_run = make(&one_onset);
        assert!(
            one_onset_run.run_file(&empty).unwrap().is_empty(),
            "{method:?}"
        );
        assert!(one_onset_run.alignment_path().perf_time.is_empty());
    }
}

#[test]
fn skf_handles_short_silent_overrun_and_one_onset_scores() {
    let short = audio_wav("skf_short", 1, &[0; 100]);
    let silent = audio_wav("skf_silent", 1, &vec![0; 44_100]);
    let empty = audio_wav("skf_empty", 1, &[]);
    let overrun = audio_melody_wav("skf_overrun", 30);
    let one_onset = midi_file("skf_one_onset_score", &[(0, 60, 80), (480, 60, 0)]);
    let make = |score: &PathBuf| {
        let mut config = AudioMethodConfig::for_method(AudioMethod::Skf);
        config.template_mode = Some(TemplateMode::Gaussian);
        AudioMatchmaker::new_skf(score, config, None, true).unwrap()
    };

    assert!(make(&data(SCORE)).run_file(&short).unwrap().is_empty());
    let last = f64::from(*make(&data(SCORE)).score_positions().last().unwrap());
    for performance in [&silent, &overrun] {
        let positions = make(&data(SCORE)).run_file(performance).unwrap();
        assert!(!positions.is_empty());
        assert!(
            positions.iter().all(|p| (0.0..=last).contains(p)),
            "position outside the score"
        );
    }
    let first = make(&data(SCORE)).run_file(&overrun).unwrap();
    assert_eq!(
        first,
        make(&data(SCORE)).run_file(&overrun).unwrap(),
        "fresh runs share no state"
    );

    let mut one = make(&one_onset);
    assert!(one.run_file(&empty).unwrap().is_empty());
    assert!(one.alignment_path().perf_time.is_empty());
    assert!(make(&one_onset).run_file(&overrun).unwrap().is_empty());
}

#[test]
fn skf_reports_template_asset_errors_instead_of_falling_back() {
    let missing = data("audio/does_not_exist.sf2");
    let corrupt = temp_file("skf_corrupt.sf2", b"not a soundfont");
    let build = |soundfont: Option<PathBuf>| {
        let mut config = AudioMethodConfig::for_method(AudioMethod::Skf);
        config.soundfont_path = soundfont;
        AudioMatchmaker::new_skf(data(SCORE), config, None, true)
    };
    assert!(matches!(build(Some(missing)), Err(Error::Io { .. })));
    assert!(matches!(
        build(Some(corrupt)),
        Err(Error::AudioRender { .. })
    ));
    assert!(matches!(build(None), Err(Error::AudioConfig { .. })));
}

#[test]
fn cqt_handles_short_silent_overrun_and_one_onset_scores() {
    let soundfont = data(
        "../../target/audio-assets/UprightPianoKW-small-SF2-20190703/UprightPianoKW-small-20190703.sf2",
    );
    let short = audio_wav("cqt_short", 1, &[0; 100]);
    let silent = audio_wav("cqt_silent", 1, &vec![0; 44_100]);
    let empty = audio_wav("cqt_empty", 1, &[]);
    let overrun = audio_melody_wav("cqt_overrun", 9);
    let one_onset = midi_file("cqt_one_onset_score", &[(0, 60, 80), (480, 60, 0)]);
    for method in [AudioMethod::Arzt, AudioMethod::Dixon] {
        let make = |score: &PathBuf| {
            let mut config = AudioMethodConfig::for_method(method);
            config.set_processor(AudioProcessorKind::Cqt);
            config.soundfont_path = Some(soundfont.clone());
            if method == AudioMethod::Arzt {
                AudioMatchmaker::new_arzt(score, config, None, true)
            } else {
                AudioMatchmaker::new_dixon(score, config, None, true)
            }
            .unwrap()
        };
        assert!(make(&data(SCORE)).run_file(&short).unwrap().is_empty());
        let silent_positions = make(&data(SCORE)).run_file(&silent).unwrap();
        assert!(!silent_positions.is_empty());
        assert!(silent_positions.iter().all(|position| position.is_finite()));

        let mut long_run = make(&data(SCORE));
        let positions = long_run.run_file(&overrun).unwrap();
        let last = *long_run.reference().frame_to_beat.last().unwrap();
        assert!(
            positions
                .iter()
                .all(|position| (0.0..=last).contains(position))
        );
        assert_eq!(long_run.reference().features[0].len(), 88);

        let mut one = make(&one_onset);
        assert!(one.run_file(&empty).unwrap().is_empty(), "{method:?}");
    }
}

#[test]
fn invalid_raw_spectrum_export_preserves_existing_output() {
    let output = std::env::temp_dir().join("matchmaker_invalid_raw_spectrum_export");
    let _ = std::fs::remove_dir_all(&output);
    std::fs::create_dir_all(&output).unwrap();
    let sentinel = output.join("sentinel.txt");
    std::fs::write(&sentinel, b"preserve").unwrap();

    for (sample_rate, hop_length, n_fft) in [(0, 128, 512), (8_000, 0, 512), (8_000, 256, 128)] {
        assert!(matches!(
            write_audio_features(
                data("audio/simple_mozart_k265_var1_sf2.wav"),
                &output,
                sample_rate,
                hop_length,
                n_fft,
            ),
            Err(Error::AudioConfig { .. })
        ));
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"preserve");
    }
    assert!(!output.join("audio/meta.json").exists());
}
