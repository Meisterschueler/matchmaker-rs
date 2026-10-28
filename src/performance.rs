//! Performance MIDI: port of `partitura.io.importmidi.load_performance_midi` (defaults:
//! `default_bpm=120`, `merge_tracks=False`) and of
//! `matchmaker.utils.symbolic.midi_messages_from_performance`.
//!
//! ```no_run
//! use matchmaker_rs::performance::Performance;
//! let perf = Performance::load("tests/data/simple_mozart_k265_var1.mid")?;
//! let messages = perf.messages(); // time-ordered note on/off, control and program changes
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::collections::HashMap;
use std::path::Path;

use midly::{MetaMessage, MidiMessage as MidlyMessage, Smf, Timing, TrackEventKind};

use crate::error::{Error, Result};
use crate::io::stream::{MidiMessage, MidiMessageKind};

/// A performed note (`PerformedPart.notes` entry).
#[derive(Debug, Clone, PartialEq)]
pub struct PerformedNote {
    /// MIDI pitch.
    pub pitch: u8,
    /// Note-on velocity.
    pub velocity: u8,
    /// MIDI channel.
    pub channel: u8,
    /// Onset in seconds.
    pub note_on: f64,
    /// Key release in seconds (pedal is ignored, like the reference's `note_off`).
    pub note_off: f64,
}

/// A control change (`PerformedPart.controls` entry).
#[derive(Debug, Clone, PartialEq)]
pub struct Control {
    /// Time in seconds.
    pub time: f64,
    /// Controller number.
    pub number: u8,
    /// Controller value.
    pub value: u8,
    /// MIDI channel.
    pub channel: u8,
}

/// A program change (`PerformedPart.programs` entry).
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    /// Time in seconds.
    pub time: f64,
    /// Program number.
    pub program: u8,
    /// MIDI channel.
    pub channel: u8,
}

/// One `PerformedPart`: the events of one MIDI track.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PerformedPart {
    /// Notes sorted by (onset, pitch, offset, channel, track).
    pub notes: Vec<PerformedNote>,
    /// Control changes in file order.
    pub controls: Vec<Control>,
    /// Program changes in file order.
    pub programs: Vec<Program>,
}

/// A loaded performance (`partitura.performance.Performance`).
#[derive(Debug, Clone, PartialEq)]
pub struct Performance {
    /// Parts for tracks that contain notes, controls or programs.
    pub parts: Vec<PerformedPart>,
}

/// Port of `adjust_time(tick, tempo_changes, ppq)`.
fn adjust_time(tick: i64, tempo_changes: &[(i64, i64)], ppq: i64) -> f64 {
    let mut time = 0.0;
    let mut last_tick = 0;
    let mut last_mpq = tempo_changes[0].1;
    for &(change_tick, mpq) in tempo_changes {
        if tick < change_tick {
            break;
        }
        // midi_ticks_to_seconds: (mpq * ticks) / float(1e6 * ppq)
        time += (last_mpq as f64 * (change_tick - last_tick) as f64) / (1e6 * ppq as f64);
        last_tick = change_tick;
        last_mpq = mpq;
    }
    time + (tick - last_tick) as f64 * (last_mpq as f64 / (ppq * 1_000_000) as f64)
}

impl Performance {
    /// Load a performance MIDI file (`.mid`, `.midi`).
    pub fn load(path: impl AsRef<Path>) -> Result<Performance> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        let midi_err = |message: String| Error::Midi {
            path: path.to_path_buf(),
            message,
        };
        let smf = Smf::parse(&bytes).map_err(|e| midi_err(e.to_string()))?;
        let ppq = match smf.header.timing {
            Timing::Metrical(t) => i64::from(t.as_int()),
            Timing::Timecode(..) => return Err(midi_err("SMPTE timing is not supported".into())),
        };
        let default_mpq = (60.0 * (1e6 / 120.0)) as i64;
        let mut time_conversion_factor = default_mpq as f64 / (ppq * 1_000_000) as f64;
        let mut tempo_changes: Vec<(i64, i64)> = vec![(0, default_mpq)];
        let mut parts = Vec::new();

        for track in &smf.tracks {
            // (note_on t, note_on tick, note_off t, note_off tick, pitch, channel, velocity)
            let mut notes: Vec<(f64, i64, f64, i64, u8, u8, u8)> = Vec::new();
            let mut controls: Vec<(i64, u8, u8, u8)> = Vec::new();
            let mut programs: Vec<(i64, u8, u8)> = Vec::new();
            let mut sounding: HashMap<(u8, u8), (f64, i64, u8)> = HashMap::new();
            let (mut t, mut ttick) = (0.0f64, 0i64);
            for event in track {
                let delta = i64::from(event.delta.as_int());
                t += delta as f64 * time_conversion_factor;
                ttick += delta;
                match event.kind {
                    TrackEventKind::Meta(MetaMessage::Tempo(us)) => {
                        let mpq = i64::from(us.as_int());
                        if tempo_changes.last().is_some_and(|&(_, last)| last != mpq) {
                            tempo_changes.push((ttick, mpq));
                        }
                        time_conversion_factor = mpq as f64 / (ppq * 1_000_000) as f64;
                    }
                    TrackEventKind::Midi { channel, message } => {
                        let ch = channel.as_int();
                        match message {
                            MidlyMessage::Controller { controller, value } => {
                                controls.push((ttick, controller.as_int(), value.as_int(), ch));
                            }
                            MidlyMessage::ProgramChange { program } => {
                                programs.push((ttick, program.as_int(), ch));
                            }
                            MidlyMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                                sounding.insert((ch, key.as_int()), (t, ttick, vel.as_int()));
                            }
                            MidlyMessage::NoteOn { key, .. }
                            | MidlyMessage::NoteOff { key, .. } => {
                                if let Some((on_t, on_tick, vel)) =
                                    sounding.remove(&(ch, key.as_int()))
                                {
                                    notes.push((on_t, on_tick, t, ttick, key.as_int(), ch, vel));
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            // sort by (note_on, pitch, note_off, channel, track) on the accumulated times
            notes.sort_by(|a, b| {
                a.0.total_cmp(&b.0)
                    .then(a.4.cmp(&b.4))
                    .then(a.2.total_cmp(&b.2))
                    .then(a.5.cmp(&b.5))
            });
            if notes.is_empty() && controls.is_empty() && programs.is_empty() {
                continue;
            }
            let part = PerformedPart {
                notes: notes
                    .iter()
                    .map(|n| PerformedNote {
                        pitch: n.4,
                        velocity: n.6,
                        channel: n.5,
                        note_on: adjust_time(n.1, &tempo_changes, ppq),
                        note_off: adjust_time(n.3, &tempo_changes, ppq),
                    })
                    .collect(),
                controls: controls
                    .iter()
                    .map(|&(tick, number, value, channel)| Control {
                        time: adjust_time(tick, &tempo_changes, ppq),
                        number,
                        value,
                        channel,
                    })
                    .collect(),
                programs: programs
                    .iter()
                    .map(|&(tick, program, channel)| Program {
                        time: adjust_time(tick, &tempo_changes, ppq),
                        program,
                        channel,
                    })
                    .collect(),
            };
            parts.push(part);
        }
        Ok(Performance { parts })
    }

    /// Port of `midi_messages_from_performance`: per note a note-on and a note-off message,
    /// then control and program changes, all sorted by time. The sort is stable (the
    /// reference uses NumPy's unstable `argsort`; see research R5).
    pub fn messages(&self) -> Vec<MidiMessage> {
        let mut messages = Vec::new();
        for part in &self.parts {
            for n in &part.notes {
                messages.push(MidiMessage {
                    kind: MidiMessageKind::NoteOn {
                        note: n.pitch,
                        velocity: n.velocity,
                    },
                    channel: n.channel,
                    time: n.note_on,
                });
                messages.push(MidiMessage {
                    kind: MidiMessageKind::NoteOff { note: n.pitch },
                    channel: n.channel,
                    time: n.note_off,
                });
            }
            for c in &part.controls {
                messages.push(MidiMessage {
                    kind: MidiMessageKind::ControlChange {
                        control: c.number,
                        value: c.value,
                    },
                    channel: c.channel,
                    time: c.time,
                });
            }
            for p in &part.programs {
                messages.push(MidiMessage {
                    kind: MidiMessageKind::ProgramChange { program: p.program },
                    channel: p.channel,
                    time: p.time,
                });
            }
        }
        messages.sort_by(|a, b| a.time.total_cmp(&b.time));
        messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjust_time_with_tempo_change() {
        // 480 ppq, 120 BPM until tick 960, then 60 BPM
        let changes = [(0, 500_000), (960, 1_000_000)];
        assert_eq!(adjust_time(480, &changes, 480), 0.5);
        assert_eq!(adjust_time(1440, &changes, 480), 2.0);
    }
}
