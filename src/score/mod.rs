//! Score loading: the partitura subset pymatchmaker uses to build its reference features.
//!
//! Mirrors `matchmaker.matchmaker.Matchmaker.__init__`: load MusicXML
//! (`partitura.load_musicxml(..., ignore_invisible_objects=True, force_note_ids="keep")`) or a
//! MIDI score (`partitura.load_score`), unfold repeats maximally
//! (`partitura.score.unfold_part_maximal(score, ignore_leaps=False)`, falling back to the folded
//! score when unfolding fails), merge all parts (`partitura.score.merge_parts`), and produce
//! the note array (`Part.note_array()`).
//!
//! ```no_run
//! use matchmaker_rs::score::Score;
//! let score = Score::load("tests/data/simple_mozart_k265_var1.musicxml", true)?;
//! println!("{} notes, {} onsets", score.notes.len(), score.unique_onsets().len());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

mod merge;
mod midi;
mod musicxml;
mod part;
mod tempo;
mod timeline;
mod unfold;

use std::collections::BTreeSet;
use std::path::Path;

use crate::error::{Error, Result};
use part::Part;
use timeline::TimeMap;

/// One row of partitura's `note_array()` (the fields score following and its tests use).
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreNote {
    /// Note id (`-<repetition>` suffix after unfolding).
    pub id: String,
    /// MIDI pitch.
    pub pitch: i32,
    /// Onset in beats of the time-signature denominator (`float32` like partitura).
    pub onset_beat: f32,
    /// Duration in beats, including tied continuations.
    pub duration_beat: f32,
    /// Onset in quarter notes.
    pub onset_quarter: f32,
    /// Duration in quarter notes (tied).
    pub duration_quarter: f32,
    /// Onset in divisions of the merged part.
    pub onset_div: i64,
    /// Duration in divisions (tied).
    pub duration_div: i64,
    /// Voice after `merge_parts` reassignment.
    pub voice: i32,
}

/// A loaded score: the merged, (unfolded) part's note array and its tempo.
#[derive(Debug, Clone)]
pub struct Score {
    /// Notes sorted by onset, then pitch.
    pub notes: Vec<ScoreNote>,
    /// Tempo marked in the score (quarter-note BPM), if any.
    pub tempo: Option<f64>,
    /// Whether repeat unfolding was applied (it falls back to the folded score on failure).
    pub unfolded: bool,
}

impl Score {
    /// Load a score file (`.musicxml`, `.xml`, `.mxl`, `.mid`, `.midi`); `unfold` is
    /// pymatchmaker's `unfold_score` (default `true`).
    pub fn load(path: impl AsRef<Path>, unfold: bool) -> Result<Score> {
        let path = path.as_ref();
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        let (parts, xml_text) = match extension.as_str() {
            "musicxml" | "xml" | "mxl" => {
                let parts = musicxml::load_musicxml(path)?;
                // ElementTree in the tempo fallback only reads plain XML files
                let text = if extension == "mxl" {
                    None
                } else {
                    std::fs::read_to_string(path).ok()
                };
                (parts, text)
            }
            "mid" | "midi" => (midi::load_score_midi(path)?, None),
            _ => {
                return Err(Error::UnsupportedFormat {
                    path: path.to_path_buf(),
                    extension,
                });
            }
        };
        let invalid = |reason: String| Error::InvalidScore {
            path: path.to_path_buf(),
            reason,
        };
        let (part, unfolded) = build_part(parts, unfold).map_err(invalid)?;
        let notes = note_array(&part).map_err(invalid)?;
        let tempo = tempo::get_tempo_from_score(&part, xml_text.as_deref());
        Ok(Score {
            notes,
            tempo,
            unfolded,
        })
    }

    /// Sorted unique onsets in beats (`np.unique(note_array["onset_beat"])`).
    pub fn unique_onsets(&self) -> Vec<f32> {
        let mut onsets: Vec<f32> = self.notes.iter().map(|n| n.onset_beat).collect();
        onsets.sort_by(f32::total_cmp);
        onsets.dedup();
        onsets
    }

    /// Pitch set per unique onset (`get_chords_from_score`).
    pub fn chords(&self) -> Vec<BTreeSet<i32>> {
        let onsets = self.unique_onsets();
        let mut chords = vec![BTreeSet::new(); onsets.len()];
        for n in &self.notes {
            if let Ok(i) = onsets.binary_search_by(|o| o.total_cmp(&n.onset_beat)) {
                chords[i].insert(n.pitch);
            }
        }
        chords
    }
}

/// Unfold (with fallback) and merge, as in `Matchmaker.__init__`.
fn build_part(parts: Vec<Part>, unfold: bool) -> std::result::Result<(Part, bool), String> {
    if unfold {
        match parts
            .iter()
            .map(unfold::unfold_part_maximal)
            .collect::<std::result::Result<Vec<Part>, _>>()
        {
            Ok(unfolded) => match merge::merge_parts(unfolded) {
                Ok(merged) => return Ok((merged, true)),
                Err(reason) => {
                    tracing::debug!(%reason, "merging unfolded parts failed; using folded score")
                }
            },
            Err(unfold::UnfoldError(reason)) => {
                tracing::debug!(%reason, "unfolding failed; using folded score")
            }
        }
    }
    Ok((merge::merge_parts(parts)?, false))
}

/// Port of `note_array_from_note_list` for the default fields, followed by the sort by
/// pitch and then (stably) by onset.
fn note_array(part: &Part) -> std::result::Result<Vec<ScoreNote>, String> {
    let (beat_map, quarter_map) = (TimeMap::new(part, false), TimeMap::new(part, true));
    let eval = |map: &Option<TimeMap>, t: i64| -> std::result::Result<f64, String> {
        match map {
            None => Ok(0.0),
            Some(m) => m.eval(t as f64).ok_or_else(|| {
                format!("A value ({t}) in x_new is outside the interpolation range.")
            }),
        }
    };
    let mut notes = Vec::new();
    for (i, n) in part.notes_tied() {
        let on = n.start;
        let off = on + part.duration_tied(i);
        let (on_beat, off_beat) = (eval(&beat_map, on)?, eval(&beat_map, off)?);
        let (on_quarter, off_quarter) = (eval(&quarter_map, on)?, eval(&quarter_map, off)?);
        notes.push(ScoreNote {
            id: n.id.clone().unwrap_or_default(),
            pitch: n.midi_pitch,
            onset_beat: on_beat as f32,
            duration_beat: (off_beat - on_beat) as f32,
            onset_quarter: on_quarter as f32,
            duration_quarter: (off_quarter - on_quarter) as f32,
            onset_div: on,
            duration_div: off - on,
            voice: n.voice,
        });
    }
    notes.sort_by_key(|n| n.pitch);
    notes.sort_by(|a, b| a.onset_beat.total_cmp(&b.onset_beat));
    Ok(notes)
}
