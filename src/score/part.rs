//! The subset of `partitura.score.Part` that score following needs.
//!
//! partitura keeps every musical object on a timeline of integer "divs". This port keeps only
//! the objects that influence the note array (`onset_beat`, ids, voices), the repeat structure
//! used for unfolding, and the tempo: notes, measures, time signatures, tempos, divisions,
//! repeats, endings and navigation markers. Timepoints of objects that are not modelled can
//! only lie inside measures and do not change any of these results (see research R6).

use std::collections::BTreeSet;

/// Score time in divisions (partitura `TimePoint.t`).
pub(crate) type Time = i64;

/// A pitched note (`partitura.score.Note` or `GraceNote`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Note {
    pub id: Option<String>,
    pub midi_pitch: i32,
    pub voice: i32,
    pub start: Time,
    pub end: Time,
    /// Index of the next note in a tie chain (`Note.tie_next`).
    pub tie_next: Option<usize>,
    /// Index of the previous note in a tie chain (`Note.tie_prev`).
    pub tie_prev: Option<usize>,
}

impl Note {
    pub fn duration(&self) -> Time {
        self.end - self.start
    }
}

/// `partitura.score.TimeSignature`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct TimeSignature {
    pub start: Time,
    pub beats: i64,
    pub beat_type: i64,
}

/// `partitura.score.Tempo` (bpm and optional unit such as `"h."`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Tempo {
    pub start: Time,
    pub bpm: f64,
    pub unit: Option<String>,
}

/// An object with optional start and end (`Repeat`, `Ending`, `Measure`).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Span {
    pub start: Option<Time>,
    pub end: Option<Time>,
    /// Ending number (`Ending.number`), unused for other spans.
    pub number: Option<String>,
}

/// Navigation markers used by `add_segments`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Marker {
    Fine,
    DaCapo,
    DalSegno,
    Segno,
    Coda,
    ToCoda,
}

/// Port of `partitura.score.Part` (subset).
#[derive(Debug, Clone)]
pub(crate) struct Part {
    pub id: String,
    /// `Part._quarter_times` (sorted).
    pub quarter_times: Vec<Time>,
    /// `Part._quarter_durations`.
    pub quarter_durations: Vec<Time>,
    /// Pitched notes in timeline iteration order (`iter_all(Note, include_subclasses=True)`).
    pub notes: Vec<Note>,
    pub measures: Vec<Span>,
    pub time_signatures: Vec<TimeSignature>,
    pub tempos: Vec<Tempo>,
    pub repeats: Vec<Span>,
    pub endings: Vec<Span>,
    pub markers: Vec<(Time, Marker)>,
    /// Times of all timepoints (`Part._points`).
    pub points: BTreeSet<Time>,
}

impl Part {
    /// `Part(id)`: quarter duration 1 at time 0, empty timeline.
    pub fn new(id: impl Into<String>) -> Self {
        Part {
            id: id.into(),
            quarter_times: vec![0],
            quarter_durations: vec![1],
            notes: Vec::new(),
            measures: Vec::new(),
            time_signatures: Vec::new(),
            tempos: Vec::new(),
            repeats: Vec::new(),
            endings: Vec::new(),
            markers: Vec::new(),
            points: BTreeSet::new(),
        }
    }

    /// Register a timepoint at `t` (`Part.get_or_add_point`).
    pub fn add_point(&mut self, t: Time) {
        self.points.insert(t);
    }

    /// `Part.first_point.t`.
    pub fn first_point(&self) -> Option<Time> {
        self.points.first().copied()
    }

    /// `Part.last_point.t`.
    pub fn last_point(&self) -> Option<Time> {
        self.points.last().copied()
    }

    /// Port of `Part.set_quarter_duration` (including its "redundant value" rule).
    pub fn set_quarter_duration(&mut self, t: Time, quarter: Time) {
        let times = &mut self.quarter_times;
        let quarters = &mut self.quarter_durations;
        let i = times.partition_point(|&x| x < t);
        if i == 0 || quarters[i - 1] != quarter {
            if i == times.len() || times[i] != t {
                times.insert(i, t);
                quarters.insert(i, quarter);
            } else if quarters[i] != quarter {
                quarters[i] = quarter;
            }
        }
    }

    /// Port of `Part.quarter_durations(start, end)`: entries with `start <= t < end`.
    pub fn quarter_durations_in(&self, start: Time, end: Time) -> Vec<(Time, Time)> {
        self.quarter_times
            .iter()
            .zip(&self.quarter_durations)
            .filter(|(t, _)| **t >= start && **t < end)
            .map(|(t, q)| (*t, *q))
            .collect()
    }

    /// Notes that are not the continuation of a tie (`Part.notes_tied`).
    pub fn notes_tied(&self) -> impl Iterator<Item = (usize, &Note)> {
        self.notes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.tie_prev.is_none())
    }

    /// Put notes in timeline iteration order (stable by start time), keeping tie links.
    pub fn sort_notes_by_start(&mut self) {
        let mut order: Vec<usize> = (0..self.notes.len()).collect();
        order.sort_by_key(|&i| self.notes[i].start);
        let mut position = vec![0; order.len()];
        for (new_i, &old_i) in order.iter().enumerate() {
            position[old_i] = new_i;
        }
        self.notes = order
            .iter()
            .map(|&i| {
                let mut n = self.notes[i].clone();
                n.tie_next = n.tie_next.map(|j| position[j]);
                n.tie_prev = n.tie_prev.map(|j| position[j]);
                n
            })
            .collect();
    }

    /// `Note.duration_tied`: own duration plus the tied continuation(s).
    pub fn duration_tied(&self, index: usize) -> Time {
        let mut total = 0;
        let mut current = Some(index);
        let mut guard = 0;
        while let Some(i) = current {
            total += self.notes[i].duration();
            current = self.notes[i].tie_next;
            guard += 1;
            if guard > self.notes.len() {
                break; // malformed cyclic tie; partitura would recurse forever
            }
        }
        total
    }

    /// First measure starting at the first timepoint (`next(first_point.iter_starting(Measure))`).
    pub fn first_measure(&self) -> Option<(Time, Time)> {
        let first = self.first_point()?;
        self.measures
            .iter()
            .find(|m| m.start == Some(first))
            .and_then(|m| Some((m.start?, m.end?)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_duration_rules() {
        let mut p = Part::new("P1");
        p.set_quarter_duration(0, 4);
        assert_eq!(
            (p.quarter_times.clone(), p.quarter_durations.clone()),
            (vec![0], vec![4])
        );
        p.set_quarter_duration(8, 4); // redundant
        assert_eq!(p.quarter_times, vec![0]);
        p.set_quarter_duration(8, 2);
        assert_eq!(
            (p.quarter_times.clone(), p.quarter_durations.clone()),
            (vec![0, 8], vec![4, 2])
        );
        p.set_quarter_duration(8, 4); // previous value equal -> partitura ignores it
        assert_eq!(p.quarter_durations, vec![4, 2]);
    }
}
