//! Port of `matchmaker.features.midi.ChordOnsetProcessor` (processor key `chord_onset`).
//!
//! Upstream: `matchmaker/features/midi.py`.
//!
//! ```
//! use matchmaker_rs::features::{Processor, chord_onset::ChordOnsetProcessor};
//! use matchmaker_rs::io::stream::{Frame, MidiMessage, MidiMessageKind};
//! let on = |note, time| MidiMessage {
//!     kind: MidiMessageKind::NoteOn { note, velocity: 64 },
//!     channel: 0,
//!     time,
//! };
//! // 60 and 64 are 10 ms apart and form a chord; the 0.5 s gap before 67 emits it
//! let frame = Frame { messages: vec![on(60, 0.0), on(64, 0.01), on(67, 0.5)], time: 0.5 };
//! let chord = ChordOnsetProcessor::new(true).process(&frame).unwrap();
//! assert_eq!(chord.perf_time, 0.0);
//! assert_eq!(chord.features.iter().filter(|&&v| v == 1.0).count(), 2);
//! ```

use super::{Observation, Processor};
use crate::io::stream::Frame;

/// `CHORD_THRESHOLD`: note-ons closer than this (seconds) form one chord.
pub const CHORD_THRESHOLD: f64 = 0.035;

/// Emits one binary pitch vector per chord, timed at its first note-on.
#[derive(Debug, Clone)]
pub struct ChordOnsetProcessor {
    piano_range: bool,
    chord_threshold: f64,
    pending_notes: Vec<u8>,
    pending_time: Option<f64>,
}

impl ChordOnsetProcessor {
    /// Create a processor with the reference threshold of 35 ms.
    pub fn new(piano_range: bool) -> Self {
        ChordOnsetProcessor {
            piano_range,
            chord_threshold: CHORD_THRESHOLD,
            pending_notes: Vec::new(),
            pending_time: None,
        }
    }

    /// `_flush`: the pending chord as a vector; `None` if nothing is pending.
    fn flush(&mut self) -> Option<Observation> {
        let time = self.pending_time.take()?;
        let (dim, offset) = if self.piano_range { (88, 21) } else { (128, 0) };
        let mut features = vec![0.0f32; dim];
        for note in self.pending_notes.drain(..) {
            if let Some(i) = usize::from(note).checked_sub(offset).filter(|&i| i < dim) {
                features[i] = 1.0;
            }
        }
        Some(Observation {
            features,
            perf_time: time,
        })
    }

    fn gap_exceeded(&self, t: f64) -> bool {
        self.pending_time
            .is_some_and(|p| t - p > self.chord_threshold && !self.pending_notes.is_empty())
    }
}

impl Processor for ChordOnsetProcessor {
    fn process(&mut self, frame: &Frame) -> Option<Observation> {
        let _span = tracing::debug_span!("feature", processor = "chord_onset").entered();
        let mut result = None;
        for m in &frame.messages {
            if let Some(p) = m.note_on_pitch() {
                if self.gap_exceeded(m.time) {
                    result = self.flush();
                }
                self.pending_notes.push(p);
                self.pending_time.get_or_insert(m.time);
            }
        }
        // empty windows act as a heartbeat that releases a pending chord
        if result.is_none() && self.gap_exceeded(frame.time) {
            result = self.flush();
        }
        result
    }

    fn flush_remaining(&mut self) -> Option<Observation> {
        if self.pending_notes.is_empty() {
            return None;
        }
        self.flush()
    }
}
