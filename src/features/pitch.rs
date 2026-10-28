//! Port of `matchmaker.features.midi.PitchProcessor` (processor key `pitch`).

use super::{Observation, Processor};
use crate::io::stream::Frame;

/// Merges all note-ons of a frame into one observation.
#[derive(Debug, Clone)]
pub struct PitchProcessor {
    /// Restrict to the 88 piano keys (MIDI 21–108).
    pub piano_range: bool,
    /// Emit pitches (minus 21 with `piano_range`) instead of a binary vector.
    pub return_pitch_list: bool,
}

impl PitchProcessor {
    /// Create a processor (`PitchProcessor(piano_range, return_pitch_list)`).
    pub fn new(piano_range: bool, return_pitch_list: bool) -> Self {
        PitchProcessor {
            piano_range,
            return_pitch_list,
        }
    }
}

impl Processor for PitchProcessor {
    fn process(&mut self, frame: &Frame) -> Option<Observation> {
        let _span = tracing::debug_span!("feature", processor = "pitch").entered();
        let shift = if self.piano_range { 21 } else { 0 };
        let mut vector = vec![0.0f32; 128];
        let mut list = Vec::new();
        let mut first_time = f64::INFINITY;
        for m in &frame.messages {
            if let Some(p) = m.note_on_pitch() {
                vector[usize::from(p)] = 1.0;
                list.push(f32::from(p) - shift as f32);
                first_time = first_time.min(m.time);
            }
        }
        // the reference tests the full 128-vector, so out-of-range notes still emit
        if list.is_empty() {
            return None;
        }
        let features = if self.return_pitch_list {
            list
        } else if self.piano_range {
            vector[21..109].to_vec()
        } else {
            vector
        };
        Some(Observation {
            features,
            perf_time: first_time,
        })
    }
}
