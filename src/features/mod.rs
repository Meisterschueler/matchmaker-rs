//! Feature processors (port of `matchmaker.features`): turn a stream [`Frame`] into an
//! [`Observation`] for the score follower.
//!
//! ```
//! use matchmaker_rs::features::{pitch::PitchProcessor, Processor};
//! use matchmaker_rs::io::stream::{Frame, MidiMessage, MidiMessageKind};
//! let on = |note, time| MidiMessage { kind: MidiMessageKind::NoteOn { note, velocity: 64 }, channel: 0, time };
//! let frame = Frame { messages: vec![on(60, 0.003), on(64, 0.001)], time: 0.005 };
//! let obs = PitchProcessor::new(true, true).process(&frame).unwrap();
//! assert_eq!((obs.features, obs.perf_time), (vec![39.0, 43.0], 0.001));
//! ```

pub mod chord_onset;
pub mod pitch;

use crate::io::stream::Frame;

/// Processor output for one frame: `(features, perf_time)` in the reference.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// Binary pitch vector (88 or 128 values) or pitch list, depending on the processor.
    pub features: Vec<f32>,
    /// Performance time in seconds the observation refers to.
    pub perf_time: f64,
}

/// Port of `matchmaker.features.processor.Processor` (the part the MIDI pipeline uses).
pub trait Processor {
    /// Process one frame; `None` if the frame yields no observation.
    fn process(&mut self, frame: &Frame) -> Option<Observation>;

    /// Emit what is still buffered at the end of the stream (`flush_remaining`).
    fn flush_remaining(&mut self) -> Option<Observation> {
        None
    }
}
