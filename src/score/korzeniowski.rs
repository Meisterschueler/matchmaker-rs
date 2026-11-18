//! Analytical score model for the Korzeniowski audio particle filter.
//!
//! Upstream: `matchmaker.features.processor.KorzeniowskiScoreProcessor`.

use crate::error::{Error, Result};
use crate::score::Score;
use crate::utils::np_sum::np_sum_squares_f32;

const BEAT_RESOLUTION: f64 = 0.1;
const NUM_HARMONICS: usize = 10;
const HARMONIC_BANDWIDTH_HZ: f64 = 15.0;

/// Beat-grid templates and lookup tables used by the Korzeniowski particle filter.
#[derive(Debug, Clone, PartialEq)]
pub struct KorzeniowskiScoreModel {
    /// Analysis beat grid, at 0.1-beat resolution.
    pub beat_grid: Vec<f64>,
    /// Unique score-note onset beats.
    pub onset_positions: Vec<f32>,
    /// Closest score onset for each beat-grid entry.
    pub nearest_onsets: Vec<f32>,
    /// Whether each beat-grid entry is a rest.
    pub rest_mask: Vec<bool>,
    /// L2-normalized harmonic spectrum for each beat-grid entry.
    pub templates: Vec<Vec<f32>>,
    /// Active MIDI pitches for each beat-grid entry.
    pub active_notes: Vec<Vec<i32>>,
}

impl KorzeniowskiScoreModel {
    /// Build the reference's harmonic-Gaussian score model.
    pub fn from_score(score: &Score, sample_rate: u32, n_fft: usize) -> Result<Self> {
        if sample_rate == 0 {
            return Err(invalid("sample_rate", "must be at least 1 Hz"));
        }
        if n_fft == 0 {
            return Err(invalid("n_fft", "must be at least 1 sample"));
        }
        if score.notes.is_empty() {
            return Err(invalid("score", "must contain at least one note"));
        }
        if score
            .notes
            .iter()
            .any(|note| !(0..=127).contains(&note.pitch))
        {
            return Err(invalid("score", "MIDI pitches must be between 0 and 127"));
        }

        let last_note_end = score
            .notes
            .iter()
            .map(|note| f64::from(note.onset_beat + note.duration_beat))
            .fold(f64::NEG_INFINITY, f64::max);
        let grid_len = ((last_note_end + BEAT_RESOLUTION) / BEAT_RESOLUTION).ceil() as usize;
        if grid_len == 0 {
            return Err(invalid("score", "must have a positive beat-grid length"));
        }
        let beat_grid: Vec<f64> = (0..grid_len)
            .map(|index| index as f64 * BEAT_RESOLUTION)
            .collect();
        let onset_positions = score.unique_onsets();
        if onset_positions.is_empty() {
            return Err(invalid("score", "must contain at least one onset"));
        }

        let note_templates = (0..128)
            .map(|pitch| build_note_template(pitch, sample_rate, n_fft))
            .collect::<Vec<_>>();
        let mut active_notes = Vec::with_capacity(beat_grid.len());
        let mut templates = Vec::with_capacity(beat_grid.len());
        let mut rest_mask = Vec::with_capacity(beat_grid.len());
        let mut nearest_onsets = Vec::with_capacity(beat_grid.len());

        for &beat in &beat_grid {
            let active: Vec<i32> = score
                .notes
                .iter()
                .filter(|note| {
                    f64::from(note.onset_beat) <= beat
                        && beat < f64::from(note.onset_beat + note.duration_beat)
                })
                .map(|note| note.pitch)
                .collect();
            let mut template = vec![0.0f32; n_fft / 2 + 1];
            for &pitch in &active {
                for (value, &note_value) in template.iter_mut().zip(&note_templates[pitch as usize])
                {
                    *value += note_value;
                }
            }
            normalize(&mut template);
            rest_mask.push(active.is_empty());
            active_notes.push(active);
            templates.push(template);
            nearest_onsets.push(nearest_onset(beat, &onset_positions));
        }

        if templates.len() != beat_grid.len()
            || active_notes.len() != beat_grid.len()
            || rest_mask.len() != beat_grid.len()
            || nearest_onsets.len() != beat_grid.len()
            || templates.iter().any(|row| row.len() != n_fft / 2 + 1)
        {
            return Err(invalid(
                "score",
                "generated model arrays have inconsistent dimensions",
            ));
        }

        Ok(Self {
            beat_grid,
            onset_positions,
            nearest_onsets,
            rest_mask,
            templates,
            active_notes,
        })
    }

    /// Find the closest beat-grid row, preferring the earlier row on a tie.
    pub fn beat_index(&self, beat: f64) -> usize {
        let upper = self.beat_grid.partition_point(|value| *value < beat);
        if upper == 0 {
            0
        } else if upper == self.beat_grid.len()
            || beat - self.beat_grid[upper - 1] <= self.beat_grid[upper] - beat
        {
            upper - 1
        } else {
            upper
        }
    }
}

fn build_note_template(pitch: usize, sample_rate: u32, n_fft: usize) -> Vec<f32> {
    let mut template = vec![0.0f32; n_fft / 2 + 1];
    let fundamental = 440.0 * 2.0f64.powf((pitch as f64 - 69.0) / 12.0);
    let frequency_resolution = 1.0 / ((1.0 / f64::from(sample_rate)) * n_fft as f64);
    for harmonic in 1..=NUM_HARMONICS {
        let frequency = harmonic as f64 * fundamental;
        if frequency > f64::from(sample_rate) / 2.0 {
            break;
        }
        let amplitude = 1.0 / (harmonic * harmonic) as f64;
        let sigma = HARMONIC_BANDWIDTH_HZ * harmonic as f64;
        for (index, value) in template.iter_mut().enumerate() {
            let bin_frequency = index as f64 * frequency_resolution;
            let distance = (bin_frequency - frequency) / sigma;
            *value = (f64::from(*value) + amplitude * (-0.5 * distance.powi(2)).exp()) as f32;
        }
    }
    normalize(&mut template);
    template
}

fn normalize(values: &mut [f32]) {
    let norm = np_sum_squares_f32(values).sqrt();
    if norm > 0.0 {
        for value in values {
            *value /= norm;
        }
    }
}

fn nearest_onset(beat: f64, onsets: &[f32]) -> f32 {
    let mut nearest = onsets[0];
    let mut distance = (f64::from(nearest) - beat).abs();
    for &onset in &onsets[1..] {
        let candidate_distance = (f64::from(onset) - beat).abs();
        if candidate_distance < distance {
            nearest = onset;
            distance = candidate_distance;
        }
    }
    nearest
}

fn invalid(key: &str, reason: &str) -> Error {
    Error::InvalidParameter {
        key: key.into(),
        reason: reason.into(),
    }
}
