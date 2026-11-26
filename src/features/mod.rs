//! Feature processors (port of `matchmaker.features`): turn a stream [`Frame`] into an
//! [`Observation`] for the score follower.
//!
//! Upstream: `matchmaker/features/processor.py`, `matchmaker/features/midi.py`, and
//! `matchmaker/features/audio.py`.
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
pub mod chroma;
pub mod cqt;
pub mod cqt_spectral_flux;
pub mod korzeniowski;
pub mod lse;
pub mod mel;
pub mod mfcc;
pub mod parangonar;
pub mod pitch;
pub mod raw_spectrum;

use crate::io::stream::Frame;

/// Processor output for one frame: `(features, perf_time)` in the reference.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    /// Processor feature vector (MIDI pitches or audio chroma/LSE values).
    pub features: Vec<f32>,
    /// Performance time in seconds the observation refers to.
    pub perf_time: f64,
}

/// Structured audio output from the Korzeniowski processor.
#[derive(Debug, Clone, PartialEq)]
pub struct KorzeniowskiObservation {
    /// L2-normalized magnitude spectrum.
    pub spectrum: Vec<f64>,
    /// Active MIDI pitches, absent for audio observations.
    pub active_notes: Option<Vec<usize>>,
    /// Causal positive spectral flux.
    pub onset: f64,
    /// Newly active MIDI pitches, absent for audio observations.
    pub onset_notes: Option<Vec<usize>>,
    /// RMS loudness in dBFS.
    pub loudness: f64,
    /// Performance time in seconds.
    pub perf_time: f64,
}

/// Port of `matchmaker.features.processor.Processor` (the part the MIDI pipeline uses).
pub trait Processor {
    /// Fallible processing for processors with input validation.
    fn try_process(&mut self, frame: &Frame) -> crate::error::Result<Option<Observation>> {
        Ok(self.process(frame))
    }

    /// Process one frame; `None` if the frame yields no observation.
    fn process(&mut self, frame: &Frame) -> Option<Observation>;

    /// Emit what is still buffered at the end of the stream (`flush_remaining`).
    fn flush_remaining(&mut self) -> Option<Observation> {
        None
    }
}

/// Processes one decoded audio frame into a feature observation.
pub trait AudioProcessor {
    /// Process one mono audio frame; `None` if its sample block is too short.
    fn process(&mut self, frame: &crate::io::audio::AudioFrame) -> Option<Observation>;

    /// Reset state between separate performance and score streams.
    fn reset(&mut self) {}
}
