//! Event-based Parangonar observation contract.
use matchmaker_rs::features::parangonar::ParangonarProcessor;
use matchmaker_rs::io::stream::{Frame, MidiMessage, MidiMessageKind};

fn on(note: u8, velocity: u8, time: f64) -> MidiMessage {
    MidiMessage {
        kind: MidiMessageKind::NoteOn { note, velocity },
        channel: 0,
        time,
    }
}

#[test]
fn event_processor_preserves_each_note_and_ignores_zero_velocity() {
    let mut processor = ParangonarProcessor;
    for note in [60, 64] {
        let frame = Frame {
            messages: vec![on(note, 64, 0.123456789)],
            time: 0.123456789,
        };
        let obs = processor.process_checked(&frame).unwrap().unwrap();
        assert_eq!(obs.features, vec![f32::from(note)]);
        assert_eq!(obs.perf_time, frame.time);
    }
    let frame = Frame {
        messages: vec![on(60, 0, 0.0)],
        time: 0.0,
    };
    assert!(processor.process_checked(&frame).unwrap().is_none());
    for kind in [
        MidiMessageKind::NoteOff { note: 60 },
        MidiMessageKind::ControlChange {
            control: 64,
            value: 127,
        },
    ] {
        let frame = Frame {
            messages: vec![MidiMessage {
                kind,
                channel: 0,
                time: 0.0,
            }],
            time: 0.0,
        };
        assert!(processor.process_checked(&frame).unwrap().is_none());
    }
}

#[test]
fn chord_frames_are_rejected_instead_of_merged() {
    let frame = Frame {
        messages: vec![on(60, 64, 0.0), on(64, 64, 0.0)],
        time: 0.0,
    };
    assert!(
        ParangonarProcessor
            .process_checked(&frame)
            .unwrap_err()
            .to_string()
            .contains("polling_period")
    );
}
