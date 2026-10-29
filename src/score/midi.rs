//! Port of `partitura.io.importmidi.load_score_midi` with its defaults
//! (`part_voice_assign_mode=0`, no quantization, `assign_note_ids=True`).
//!
//! Upstream: `partitura/io/importmidi.py`.
//!
//! One part per track that contains notes, voices by channel (in order of first use), note
//! ids `n<i>` over all notes in `(track, channel)` order, `set_tempo` events as tempos of the
//! first part, time signatures from tracks or global tracks. Notes of zero duration become
//! grace notes. Splitting notes at bar lines (`tie_notes`) does not change the note array
//! (onset and summed tied duration stay the same) and is therefore not ported.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};

use super::part::{Note, Part, Span, Tempo, Time, TimeSignature};
use super::timeline::{TimeMap, Unit};
use crate::error::{Error, Result};

/// A note collected from a track: `(onset, pitch, duration)` in ticks.
type RawNote = (Time, i32, Time);

/// Parse a MIDI file as a score (see module docs).
pub(crate) fn load_score_midi(path: &Path) -> Result<Vec<Part>> {
    let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
    let midi_err = |message: String| Error::Midi {
        path: path.to_path_buf(),
        message,
    };
    let smf = Smf::parse(&bytes).map_err(|e| midi_err(e.to_string()))?;
    let divs = match smf.header.timing {
        Timing::Metrical(t) => i64::from(t.as_int()),
        Timing::Timecode(..) => return Err(midi_err("SMPTE timing is not supported".into())),
    };

    let mut global_time_sigs: Vec<(Time, i64, i64)> = Vec::new();
    let mut global_tempos: Vec<(Time, f64)> = Vec::new();
    let mut time_sigs_by_track: BTreeMap<usize, Vec<(Time, i64, i64)>> = BTreeMap::new();
    let mut notes_by_track_ch: BTreeMap<(usize, u8), Vec<RawNote>> = BTreeMap::new();

    for (track_nr, track) in smf.tracks.iter().enumerate() {
        let mut time_sigs = Vec::new();
        let mut notes: BTreeMap<u8, Vec<RawNote>> = BTreeMap::new();
        let mut sounding: HashMap<(u8, u8), Time> = HashMap::new();
        let mut t: Time = 0;
        for event in track {
            t += i64::from(event.delta.as_int());
            match event.kind {
                TrackEventKind::Meta(MetaMessage::TimeSignature(num, den_pow, _, _)) => {
                    time_sigs.push((t, i64::from(num), 1i64 << den_pow));
                }
                TrackEventKind::Meta(MetaMessage::Tempo(us)) => {
                    global_tempos.push((t, 60.0 * 1e6 / f64::from(us.as_int())));
                }
                TrackEventKind::Midi { channel, message } => {
                    let ch = channel.as_int();
                    let (key, on) = match message {
                        MidiMessage::NoteOn { key, vel } => (key.as_int(), vel.as_int() > 0),
                        MidiMessage::NoteOff { key, .. } => (key.as_int(), false),
                        _ => continue,
                    };
                    if on {
                        sounding.insert((ch, key), t);
                    } else if let Some(onset) = sounding.remove(&(ch, key)) {
                        notes
                            .entry(ch)
                            .or_default()
                            .push((onset, i32::from(key), t - onset));
                    }
                }
                _ => {}
            }
        }
        if notes.is_empty() {
            global_time_sigs.extend(time_sigs);
        } else {
            time_sigs_by_track.insert(track_nr, time_sigs);
            for (ch, ch_notes) in notes {
                if !ch_notes.is_empty() {
                    notes_by_track_ch.insert((track_nr, ch), ch_notes);
                }
            }
        }
    }
    if notes_by_track_ch.is_empty() {
        return Err(midi_err("no notes".into()));
    }

    // assign_group_part_voice(mode=0): part per track (in order), voice per channel (1-based)
    let mut part_of_track: BTreeMap<usize, usize> = BTreeMap::new();
    let mut voices_of_track: HashMap<usize, Vec<u8>> = HashMap::new();
    let mut part_voice: Vec<(usize, i32)> = Vec::new();
    for &(tr, ch) in notes_by_track_ch.keys() {
        let n_parts = part_of_track.len();
        let part = *part_of_track.entry(tr).or_insert(n_parts);
        let chans = voices_of_track.entry(tr).or_default();
        if !chans.contains(&ch) {
            chans.push(ch);
        }
        let voice = chans
            .iter()
            .position(|c| *c == ch)
            .map_or(1, |i| i as i32 + 1);
        part_voice.push((part, voice));
    }

    // sanitize time signatures present in only some tracks
    let counts: Vec<usize> = time_sigs_by_track.values().map(Vec::len).collect();
    if global_time_sigs.is_empty()
        && counts.iter().min() == Some(&0)
        && counts.iter().max().is_some_and(|&m| m != 0)
    {
        for ts_list in time_sigs_by_track.values() {
            global_time_sigs.extend(ts_list.iter().copied());
        }
        time_sigs_by_track.clear();
    }
    let n_parts = part_of_track.len();
    let mut time_sigs_by_part: Vec<Vec<(Time, i64, i64)>> = vec![Vec::new(); n_parts];
    for (tr, ts_list) in &time_sigs_by_track {
        if let Some(&p) = part_of_track.get(tr) {
            time_sigs_by_part[p].extend(ts_list.iter().copied());
        }
    }
    for ts in &global_time_sigs {
        for list in &mut time_sigs_by_part {
            list.push(*ts);
        }
    }

    // notes with ids n<i> in (track, channel) order
    let mut notes_by_part: Vec<Vec<(RawNote, i32, String)>> = vec![Vec::new(); n_parts];
    let mut i = 0;
    for (notes, &(part, voice)) in notes_by_track_ch.values().zip(&part_voice) {
        for &note in notes {
            notes_by_part[part].push((note, voice, format!("n{i}")));
            i += 1;
        }
    }

    let mut parts = Vec::with_capacity(n_parts);
    for (p, notes) in notes_by_part.into_iter().enumerate() {
        let mut ts = time_sigs_by_part[p].clone();
        ts.sort_by_key(|t| t.0);
        parts.push(create_part(p, divs, notes, ts));
    }
    if let Some(first) = parts.first_mut() {
        for (t, qpm) in global_tempos {
            first.add_point(t);
            first.tempos.push(Tempo {
                start: t,
                bpm: qpm,
                unit: Some("q".into()),
            });
        }
    }
    Ok(parts)
}

/// Port of `create_part` (without pitch spelling, clefs, key signatures, ties, tuplets).
fn create_part(
    index: usize,
    divs: Time,
    notes: Vec<(RawNote, i32, String)>,
    mut time_sigs: Vec<(Time, i64, i64)>,
) -> Part {
    let mut part = Part::new(format!("P{}", index + 1));
    part.set_quarter_duration(0, divs);
    part.add_point(0); // clef at 0
    for ((onset, pitch, duration), voice, id) in notes {
        part.add_point(onset);
        part.add_point(onset + duration);
        part.notes.push(Note {
            id: Some(id),
            midi_pitch: pitch,
            voice,
            start: onset,
            end: onset + duration,
            tie_next: None,
            tie_prev: None,
        });
    }
    if time_sigs.is_empty() {
        time_sigs.push((0, 4, 4));
    }
    for &(start, beats, beat_type) in &time_sigs {
        part.add_point(start);
        part.time_signatures.push(TimeSignature {
            start,
            beats,
            beat_type,
        });
    }
    // timeline order of notes (by start, stable)
    part.notes.sort_by_key(|n| n.start);
    add_first_measure(&mut part);
    part
}

/// The first measure that `add_measures` creates; only it matters (anacrusis correction).
fn add_first_measure(part: &mut Part) {
    let (Some(start), Some(end)) = (part.first_point(), part.last_point()) else {
        return;
    };
    if start == end {
        return;
    }
    // add_measures prepends a 4-beat signature if the first one starts later
    let ts = part
        .time_signatures
        .iter()
        .find(|t| t.start <= start)
        .copied()
        .unwrap_or(TimeSignature {
            start,
            beats: 4,
            beat_type: 4,
        });
    let ts_end = part
        .time_signatures
        .iter()
        .map(|t| t.start)
        .filter(|&t| t > ts.start)
        .min()
        .unwrap_or(end)
        .min(end);
    let Some(map) = TimeMap::new(part, Unit::Beat) else {
        return;
    };
    let (Some(b_start), Some(b_end)) = (map.eval(start as f64), map.eval(end as f64)) else {
        return;
    };
    let measure_end_beats = (b_start + ts.beats as f64).min(b_end);
    // inv_beat_map is linear within the first time-signature segment
    let divs_per_beat = (part.quarter_durations[0] as f64) * 4.0 / ts.beat_type as f64;
    let measure_end =
        ((measure_end_beats - b_start) * divs_per_beat + start as f64).min(ts_end as f64);
    let measure_end = measure_end as Time;
    part.add_point(measure_end);
    part.measures.push(Span {
        start: Some(start),
        end: Some(measure_end),
        number: Some("1".into()),
    });
}
