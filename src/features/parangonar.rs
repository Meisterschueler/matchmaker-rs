//! Port of `matchmaker.external.ParangonarProcessor`: event-based MIDI notes.
//! Upstream: `matchmaker/external.py`.
//!
//! ```
//! use matchmaker_rs::features::parangonar::ParangonarProcessor;
//! use matchmaker_rs::io::stream::Frame;
//! let frame = Frame { messages: vec![], time: 0.0 };
//! assert!(ParangonarProcessor.process_checked(&frame)?.is_none());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use crate::error::{Error, Result};
use crate::features::{Observation, Processor};
use crate::io::stream::{Frame, MidiMessageKind};

/// One note per observation, without polling-window chord aggregation.
pub struct ParangonarProcessor;

impl ParangonarProcessor {
    /// Process one frame, rejecting chords and invalid event times.
    pub fn process_checked(&mut self, frame: &Frame) -> Result<Option<Observation>> {
        let notes: Vec<_> = frame
            .messages
            .iter()
            .filter_map(|message| match message.kind {
                MidiMessageKind::NoteOn { note, velocity } if velocity > 0 => {
                    Some((note, message.time))
                }
                _ => None,
            })
            .collect();
        if notes.len() > 1 {
            return Err(Error::InvalidParameter {
                key: "polling_period".into(),
                reason: "parangonar requires event-based MIDI input with polling_period=None"
                    .into(),
            });
        }
        if let Some(&(pitch, time)) = notes.first() {
            if !time.is_finite() || !(20..=110).contains(&pitch) {
                return Err(Error::InvalidParameter {
                    key: "OTM observation".into(),
                    reason: format!(
                        "invalid time {time} or pitch {pitch}; expected a finite time and pitch 20..=110"
                    ),
                });
            }
            return Ok(Some(Observation {
                features: vec![f32::from(pitch)],
                perf_time: time,
            }));
        }
        Ok(None)
    }
}

impl Processor for ParangonarProcessor {
    fn try_process(&mut self, frame: &Frame) -> Result<Option<Observation>> {
        self.process_checked(frame)
    }

    fn process(&mut self, frame: &Frame) -> Option<Observation> {
        match self.process_checked(frame) {
            Ok(observation) => observation,
            Err(error) => {
                tracing::error!(%error, "parangonar frame rejected; use try_process to propagate errors");
                None
            }
        }
    }
}
