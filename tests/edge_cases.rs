//! Edge cases of the US1 pipeline (spec.md "Edge Cases"): empty and oversized performances,
//! velocity-0 note-ons, missing tempo, unreadable files and out-of-range pitches. None of them
//! may panic; broken files give typed errors that name the file.

mod common;

use std::path::PathBuf;

use common::data;
use matchmaker_rs::Error;
use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::features::Observation;
use matchmaker_rs::io::stream::MidiMessageKind;
use matchmaker_rs::matchmaker::{Matchmaker, Method};
use matchmaker_rs::performance::Performance;
use matchmaker_rs::prob::hmm::{PitchIoiHmm, PitchIoiHmmConfig};
use matchmaker_rs::score::Score;
use midly::num::{u4, u7, u15, u28};
use midly::{Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind};

const SCORE: &str = "simple_mozart_k265_var1.musicxml";

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

#[test]
fn empty_performance_gives_an_empty_path() {
    let performance = midi_file("empty", &[]);
    for method in [Method::Pthmm, Method::Hmm, Method::OuterHmm] {
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
    for method in [Method::Pthmm, Method::Hmm, Method::OuterHmm] {
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
