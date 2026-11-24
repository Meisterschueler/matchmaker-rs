//! Port of Parangonar 3.4.1's `OnlineTransformerMatcher` and `TempoModel`.
//! Upstream: `parangonar/match/online_matchers.py`.
//!
//! ```
//! use matchmaker_rs::{base::OnlineAlignment, features::Observation, score::Score};
//! use matchmaker_rs::dp::otm::{OtmConfig, OnlineTransformerMatcher};
//! # fn example(score: &Score) -> matchmaker_rs::Result<()> {
//! let mut follower = OnlineTransformerMatcher::new(score.otm_notes(), OtmConfig::default())?;
//! let beat = follower.try_call(&Observation { features: vec![60.0], perf_time: 0.0 })?;
//! # Ok(()) }
//! ```

use super::otm_model::{OtmModel, tokenize};
use crate::base::{AlignmentPath, OnlineAlignment, OtmStepDebug, StepDebug};
use crate::error::{Error, Result};
use crate::features::Observation;
use crate::score::ScoreNote;

type InferenceStep = (Option<[[usize; 7]; 26]>, Option<Vec<f32>>);

// Float32 softmax can tie distinct logits; upstream chooses the first tied probability.
fn sequence_argmax(logits: &[f32]) -> usize {
    let maximum = logits
        .iter()
        .skip(1)
        .step_by(2)
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let scores: Vec<_> = logits
        .iter()
        .skip(1)
        .step_by(2)
        .map(|value| libm::expf(value - maximum))
        .collect();
    let inverse_sum = 1.0 / scores.iter().sum::<f32>();
    let probabilities: Vec<_> = scores.iter().map(|score| score * inverse_sum).collect();
    let mut prediction = 0;
    for index in 1..probabilities.len() {
        if probabilities[index] > probabilities[prediction] {
            prediction = index;
        }
    }
    prediction
}

fn invalid(key: &str, reason: impl Into<String>) -> Error {
    Error::InvalidParameter {
        key: key.into(),
        reason: reason.into(),
    }
}

/// Runtime parameters of the fixed pretrained tracker.
#[derive(Debug, Clone)]
pub struct OtmConfig {
    /// Accepted for upstream compatibility; upstream `__call__` starts at 0.5.
    pub init_beat_period: f64,
    /// Positive integer tempo lookback in beats.
    pub lookback: usize,
}

impl Default for OtmConfig {
    fn default() -> Self {
        Self {
            init_beat_period: 0.5,
            lookback: 1,
        }
    }
}

/// Median-at-onset, piecewise-linear tempo model from Parangonar.
#[derive(Debug, Clone)]
struct TempoModel {
    lookback: f32,
    scores: Vec<f32>,
    times: Vec<f32>,
    samples: Vec<(f32, Vec<f32>)>,
    /// Seconds per beat, clipped to `[0.1, 10.0]`.
    beat_period: f64,
}

impl TempoModel {
    /// Initialize with two synthetic observations before the first score onset.
    fn new(beat_period: f64, score: f64, time: f64, lookback: usize) -> Result<Self> {
        if !beat_period.is_finite()
            || beat_period <= 0.0
            || lookback == 0
            || !score.is_finite()
            || !time.is_finite()
        {
            return Err(invalid(
                "tempo_model",
                "finite onsets, positive beat period and lookback required",
            ));
        }
        let lookback = lookback as f32;
        let (score, time, beat_period) = (score as f32, time as f32, beat_period as f32);
        let first_score = score - 2.0 * lookback;
        let first_time = time - 2.0 * lookback * beat_period;
        let mut model = Self {
            lookback,
            scores: vec![first_score],
            times: vec![first_time],
            samples: vec![(first_score, vec![first_time])],
            beat_period: f64::from(beat_period),
        };
        model.update(
            f64::from(time - lookback * beat_period),
            f64::from(score - lookback),
        );
        Ok(model)
    }

    fn interpolate(&self, query: f32) -> f32 {
        let start = self.scores.len().saturating_sub(100);
        let mut points: Vec<_> = self.scores[start..]
            .iter()
            .copied()
            .zip(self.times[start..].iter().copied())
            .collect();
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        let upper = points
            .partition_point(|(x, _)| *x < query)
            .clamp(1, points.len() - 1);
        let (x0, y0) = points[upper - 1];
        let (x1, y1) = points[upper];
        // SciPy's de Boor expression and NumPy 2's float32 scalar promotion.
        ((query - x0) / (x1 - x0)) * y1 + ((x1 - query) / (x1 - x0)) * y0
    }

    /// Add a matched onset, replacing repeated-onset times by their median.
    fn update(&mut self, time: f64, score: f64) {
        let (time, score) = (time as f32, score as f32);
        let index = self
            .samples
            .iter()
            .position(|(onset, _)| *onset == score)
            .unwrap_or_else(|| {
                self.samples.push((score, Vec::new()));
                self.samples.len() - 1
            });
        let samples = &mut self.samples[index].1;
        samples.push(time);
        if self.scores.last() == Some(&score) {
            let mut sorted = samples.clone();
            sorted.sort_by(f32::total_cmp);
            let middle = sorted.len() / 2;
            *self.times.last_mut().expect("initialized tempo samples") =
                if sorted.len().is_multiple_of(2) {
                    (sorted[middle - 1] + sorted[middle]) / 2.0
                } else {
                    sorted[middle]
                };
        } else {
            self.scores.push(score);
            self.times.push(time);
        }
        self.beat_period = f64::from(
            ((self.interpolate(score) - self.interpolate(score - self.lookback)) / self.lookback)
                .clamp(0.1, 10.0),
        );
    }

    /// Extrapolate the performance time of a score onset.
    fn predict(&self, score: f64) -> f64 {
        let offset = self.lookback + 1.0;
        f64::from(self.interpolate(score as f32 - offset) + offset * self.beat_period as f32)
    }
}

/// Fixed-graph online MIDI follower, exposing beats rather than note indices.
pub struct OnlineTransformerMatcher {
    notes: Vec<ScoreNote>,
    positions: Vec<f32>,
    pitches: Vec<Vec<i32>>,
    aligned: Vec<bool>,
    model: OtmModel,
    config: OtmConfig,
    tempo: Option<TempoModel>,
    previous_onset: f64,
    performance: Vec<i32>,
    current: usize,
    input: usize,
    since_update: usize,
    stuck: usize,
    path: AlignmentPath,
    debug: Option<OtmStepDebug>,
}

impl OnlineTransformerMatcher {
    /// Construct from `Score::otm_notes()`; reject empty or malformed scores.
    pub fn new(notes: Vec<ScoreNote>, config: OtmConfig) -> Result<Self> {
        if notes.is_empty() {
            return Err(invalid("score", "OTM requires at least one score note"));
        }
        let mut ids = std::collections::HashSet::new();
        if notes
            .iter()
            .any(|note| note.id.is_empty() || !ids.insert(note.id.as_str()))
        {
            return Err(invalid(
                "score.id",
                "requires valid unique IDs; construct notes with Score::otm_notes",
            ));
        }
        if !config.init_beat_period.is_finite() || config.init_beat_period <= 0.0 {
            return Err(invalid("init_beat_period", "must be finite and positive"));
        }
        if config.lookback == 0 {
            return Err(invalid("lookback", "must be at least 1"));
        }
        if notes.iter().any(|note| !note.onset_beat.is_finite())
            || notes.windows(2).any(|p| p[0].onset_beat > p[1].onset_beat)
        {
            return Err(invalid(
                "score",
                "OTM score onsets must be finite and sorted",
            ));
        }
        let mut positions: Vec<_> = notes.iter().map(|note| note.onset_beat).collect();
        positions.dedup();
        let pitches = positions
            .iter()
            .map(|onset| {
                python_pitch_set(
                    notes
                        .iter()
                        .filter(|note| !note.is_grace && note.onset_beat == *onset)
                        .map(|note| note.pitch),
                )
            })
            .collect();
        Ok(Self {
            previous_onset: f64::from(positions[0]),
            aligned: vec![false; notes.len()],
            notes,
            positions,
            pitches,
            model: OtmModel::bundled()?,
            config,
            tempo: None,
            performance: Vec::with_capacity(8),
            current: 0,
            input: 0,
            since_update: 0,
            stuck: 0,
            path: AlignmentPath::default(),
            debug: None,
        })
    }

    fn candidate(&self, pitch: i32, onset: f64) -> Option<usize> {
        self.notes.iter().enumerate().find_map(|(index, note)| {
            (note.pitch == pitch && f64::from(note.onset_beat) == onset && !self.aligned[index])
                .then_some(index)
        })
    }

    fn matched(&mut self, note: usize, position: usize, time: f64, greedy: bool) {
        self.aligned[note] = true;
        let onset = f64::from(self.notes[note].onset_beat);
        if greedy || (!self.notes[note].is_grace && onset >= self.previous_onset) {
            self.tempo
                .as_mut()
                .expect("tempo initialized on first event")
                .update(time, onset);
            if !greedy {
                self.previous_onset = onset;
                self.current = position;
            }
        }
        self.stuck = 0;
    }

    fn step(&mut self, pitch: i32, time: f64) -> Result<InferenceStep> {
        self.since_update += 1;
        if self.performance.len() == 8 {
            self.performance.remove(0);
        }
        self.performance.push(pitch);
        if self.input == 0 {
            // Upstream __call__ overrides init_beat_period even when configured.
            self.tempo = Some(TempoModel::new(
                0.5,
                self.previous_onset,
                time,
                self.config.lookback,
            )?);
        }
        let current = self.current;
        if self.pitches[current].contains(&pitch)
            && let Some(note) = self.candidate(pitch, self.previous_onset)
        {
            self.matched(note, current, time, true);
            return Ok((None, None));
        }
        let start = current.saturating_sub(7);
        let end = (current + 9).min(self.pitches.len());
        let performance_start = self.performance.len().saturating_sub(8);
        let performance = &self.performance[performance_start..];
        let tokens = tokenize(&self.pitches[start..end], performance)?;
        let logits = self.model.forward(&tokens)?;
        let prediction = sequence_argmax(&logits);
        let predicted = (start as isize + prediction as isize - performance.len() as isize - 1)
            .clamp(0, self.positions.len() as isize - 1) as usize;
        let relative = predicted as isize - current as isize;
        if relative > -5 && relative < 2 {
            if let Some(note) = self.candidate(pitch, f64::from(self.positions[predicted])) {
                self.matched(note, predicted, time, false);
                return Ok((Some(tokens), Some(logits)));
            }
        } else if relative >= 2 {
            let jumped: usize = self.pitches[current..predicted].iter().map(Vec::len).sum();
            if self.pitches[current + 1].contains(&pitch)
                && let Some(note) = self.candidate(pitch, f64::from(self.positions[current + 1]))
            {
                let expected = self
                    .tempo
                    .as_ref()
                    .expect("initialized tempo")
                    .predict(f64::from(self.notes[note].onset_beat));
                if (expected as f32 - time as f32).abs() < 1.0 {
                    self.matched(note, current + 1, time, false);
                    return Ok((Some(tokens), Some(logits)));
                }
            }
            if self.since_update > 2
                && jumped <= 10
                && let Some(note) = self.candidate(pitch, f64::from(self.positions[predicted]))
            {
                self.since_update = 0;
                self.matched(note, predicted, time, false);
                return Ok((Some(tokens), Some(logits)));
            }
        }
        self.stuck += 1;
        if self.stuck == 10 {
            self.current = (self.current + 8).min(self.positions.len() - 1);
            tracing::warn!(
                current_index = self.current,
                "OTM stuck for ten events; advancing eight onsets"
            );
        } else if self.stuck == 30 {
            tracing::warn!("OTM stuck for thirty events");
        }
        Ok((Some(tokens), Some(logits)))
    }
}

impl OnlineAlignment for OnlineTransformerMatcher {
    fn try_call(&mut self, observation: &Observation) -> Result<f64> {
        let pitch = observation.features.first().copied().unwrap_or(f32::NAN);
        if observation.features.len() != 1
            || !pitch.is_finite()
            || pitch.fract() != 0.0
            || !(20.0..=110.0).contains(&pitch)
            || !(observation.perf_time as f32).is_finite()
        {
            return Err(invalid(
                "OTM observation",
                "requires one integral pitch 20..=110 and a finite time",
            ));
        }
        let (tokens, logits) = self.step(pitch as i32, f64::from(observation.perf_time as f32))?;
        let position = f64::from(self.positions[self.current]);
        tracing::debug!(
            current_index = self.current,
            ?tokens,
            ?logits,
            previous_onset = self.previous_onset,
            beat_period = self.tempo.as_ref().map(|tempo| tempo.beat_period),
            "OTM event"
        );
        self.path.push(observation.perf_time, position);
        self.debug = Some(OtmStepDebug {
            tokens,
            logits,
            tempo: [
                self.previous_onset,
                self.tempo.as_ref().expect("initialized tempo").beat_period,
            ],
            warping: [self.current, self.input],
        });
        self.input += 1;
        Ok(position)
    }

    fn call(&mut self, observation: &Observation) -> f64 {
        match self.try_call(observation) {
            Ok(position) => position,
            Err(error) => {
                tracing::error!(%error, "OTM observation rejected; use try_call to propagate errors");
                f64::NAN
            }
        }
    }

    fn is_still_following(&self) -> bool {
        self.current < self.positions.len() - 1
    }
    fn alignment_path(&self) -> &AlignmentPath {
        &self.path
    }
    fn score_positions(&self) -> &[f32] {
        &self.positions
    }
    fn debug_state(&self) -> StepDebug {
        StepDebug {
            current_index: self.current,
            otm: self.debug.clone(),
            ..StepDebug::default()
        }
    }
}

// CPython's small-integer set iteration order is part of the upstream tokenizer:
// collisions, growth, and truncation to seven slots make sorted pitches inequivalent.
fn python_pitch_set(pitches: impl Iterator<Item = i32>) -> Vec<i32> {
    fn hash(pitch: i32) -> usize {
        if pitch == -1 {
            (-2isize) as usize
        } else {
            pitch as isize as usize
        }
    }
    fn insert(table: &mut [Option<i32>], pitch: i32) -> bool {
        let mask = table.len() - 1;
        let mut perturb = hash(pitch);
        let mut index = perturb & mask;
        loop {
            let probes = if index + 9 <= mask { 9 } else { 0 };
            for slot in &mut table[index..=index + probes] {
                match slot {
                    Some(value) if *value == pitch => return false,
                    None => {
                        *slot = Some(pitch);
                        return true;
                    }
                    _ => {}
                }
            }
            perturb >>= 5;
            index = (index.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb)) & mask;
        }
    }
    let mut table = vec![None; 8];
    let mut used = 0;
    for pitch in pitches {
        if insert(&mut table, pitch) {
            used += 1;
            if used * 5 >= (table.len() - 1) * 3 {
                let mut size = 8;
                while size <= used * 4 {
                    size *= 2;
                }
                let mut new = vec![None; size];
                for value in table.iter().flatten() {
                    insert(&mut new, *value);
                }
                table = new;
            }
        }
    }
    table.into_iter().flatten().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_softmax_ties_keep_the_first_token() {
        let mut logits = vec![0.0; 52];
        logits[3] = 1e-8;
        assert_eq!(sequence_argmax(&logits), 0);
    }

    #[test]
    fn tempo_uses_medians_and_clips_extrapolated_periods() {
        let mut tempo = TempoModel::new(0.5, 0.0, 0.0, 1).unwrap();
        tempo.update(0.0, 0.0);
        tempo.update(1.0, 0.0);
        tempo.update(0.1, 0.0);
        assert_eq!(tempo.times.last(), Some(&0.1));
        tempo.update(100.0, 1.0);
        assert_eq!(tempo.beat_period, 10.0);
        tempo.update(100.001, 2.0);
        assert_eq!(tempo.beat_period, f64::from(0.1f32));
    }

    #[test]
    fn pitch_set_matches_python_collision_and_growth_order() {
        assert_eq!(
            python_pitch_set([60, 64, 67, 72].into_iter()),
            [64, 72, 67, 60]
        );
        assert_eq!(python_pitch_set([60, 60, 64].into_iter()), [64, 60]);
        assert_eq!(
            python_pitch_set([-1, -2, 31, 63, 95, 127].into_iter()),
            [127, -2, 31, 63, -1, 95]
        );
    }
}
