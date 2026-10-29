//! MIDI stream simulation: port of `matchmaker.io.midi.MidiStream` (file mode) and of the
//! framing in `matchmaker.utils.symbolic` (`Buffer`, `midi_messages_to_framed_midi`).
//!
//! Upstream: `matchmaker/io/midi.py`, `matchmaker/utils/symbolic.py`.
//!
//! ```
//! use matchmaker_rs::io::stream::{frames, MidiMessage, MidiMessageKind, StreamMode};
//! let msgs = vec![
//!     MidiMessage { kind: MidiMessageKind::NoteOn { note: 60, velocity: 80 }, channel: 0, time: 0.004 },
//!     MidiMessage { kind: MidiMessageKind::NoteOff { note: 60 }, channel: 0, time: 0.5 },
//! ];
//! let windows = frames(&msgs, StreamMode::Windowed { polling_period: 0.01 });
//! assert_eq!(windows.len(), 50);
//! assert_eq!(windows[0].messages.len(), 1);
//! ```

/// What a [`MidiMessage`] carries (the subset of `mido.Message` the stream uses).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MidiMessageKind {
    /// Note on; velocity 0 is treated as note off by every consumer.
    NoteOn {
        /// MIDI pitch.
        note: u8,
        /// Velocity.
        velocity: u8,
    },
    /// Note off.
    NoteOff {
        /// MIDI pitch.
        note: u8,
    },
    /// Control change.
    ControlChange {
        /// Controller number.
        control: u8,
        /// Value.
        value: u8,
    },
    /// Program change.
    ProgramChange {
        /// Program number.
        program: u8,
    },
}

/// A timestamped MIDI message.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MidiMessage {
    /// Message type and data.
    pub kind: MidiMessageKind,
    /// MIDI channel.
    pub channel: u8,
    /// Absolute time in seconds.
    pub time: f64,
}

impl MidiMessage {
    /// Pitch of a note-on with velocity > 0, else `None`.
    pub fn note_on_pitch(&self) -> Option<u8> {
        match self.kind {
            MidiMessageKind::NoteOn { note, velocity } if velocity > 0 => Some(note),
            _ => None,
        }
    }
}

/// One stream item (`InputMIDIFrame`): messages plus the frame time.
#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    /// Messages in the frame, in stream order.
    pub messages: Vec<MidiMessage>,
    /// Window centre `start + 0.5 * polling_period` (`Buffer.time`) in windowed mode, the
    /// message time in event mode.
    pub time: f64,
}

/// How the stream groups messages.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StreamMode {
    /// One message per frame (`polling_period=None`).
    Event,
    /// Windows of `polling_period` seconds.
    Windowed {
        /// Window length in seconds.
        polling_period: f64,
    },
}

/// Group time-ordered messages into frames like `MidiStream.run_offline_single` /
/// `run_offline_windowed`. Windowed mode ports `midi_messages_to_framed_midi` exactly:
/// `n = ceil(max_time / pp)` frames; frame 0 holds `t <= pp`, frame k holds
/// `k·pp < t <= (k+1)·pp`.
pub fn frames(messages: &[MidiMessage], mode: StreamMode) -> Vec<Frame> {
    match mode {
        StreamMode::Event => messages
            .iter()
            .map(|m| Frame {
                messages: vec![*m],
                time: m.time,
            })
            .collect(),
        StreamMode::Windowed { polling_period: pp } => {
            if messages.is_empty() {
                return Vec::new();
            }
            let max_time = messages
                .iter()
                .map(|m| m.time)
                .fold(f64::NEG_INFINITY, f64::max);
            let n_frames = (max_time / pp).ceil() as usize;
            let mut out = Vec::with_capacity(n_frames);
            // messages are time-sorted, so each frame is the contiguous run lower < t <= upper
            let first_after = |bound: f64| messages.partition_point(|m| m.time <= bound);
            for cursor in 0..n_frames {
                let lower = cursor as f64 * pp;
                let upper = (cursor + 1) as f64 * pp;
                let from = if cursor == 0 { 0 } else { first_after(lower) };
                out.push(Frame {
                    messages: messages[from..first_after(upper)].to_vec(),
                    time: lower + 0.5 * pp,
                });
            }
            out
        }
    }
}
