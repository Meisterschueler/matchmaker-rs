//! Ports of Parangonar's symbolic OLTW and pure-transformer MIDI matchers.
//! Upstream: `parangonar/match/online_matchers.py` (`OLTWMatcher`, `TOLTWMatcher`,
//! `OnlinePureTransformerMatcher`).
//!
//! ```
//! use matchmaker_rs::{base::OnlineAlignment, features::Observation, score::Score};
//! use matchmaker_rs::dp::parangonar::{SlOltwConfig, SlOltwMatcher};
//! let score = Score::load("tests/data/simple_mozart_k265_var1.musicxml", true)?;
//! let mut follower = SlOltwMatcher::new(&score.otm_notes(), SlOltwConfig::default())?;
//! let beat = follower.try_call(&Observation { features: vec![60.0], perf_time: 0.0 })?;
//! assert!(beat.is_finite());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::collections::{HashSet, VecDeque};

use super::otm::python_pitch_set;
use super::otm_model::{OtmModel, tokenize};
use crate::base::{AlignmentPath, OnlineAlignment, StepDebug};
use crate::error::{Error, Result};
use crate::features::Observation;
use crate::score::ScoreNote;

fn invalid(key: &str, reason: impl Into<String>) -> Error {
    Error::InvalidParameter {
        key: key.into(),
        reason: reason.into(),
    }
}

fn observation_pitch(obs: &Observation) -> Result<i32> {
    let pitch = obs.features.first().copied().unwrap_or(f32::NAN);
    if obs.features.len() != 1
        || !pitch.is_finite()
        || pitch.fract() != 0.0
        || !(0.0..=127.0).contains(&pitch)
        || !obs.perf_time.is_finite()
    {
        return Err(invalid(
            "observation",
            "requires one integral MIDI pitch 0..=127 and a finite time",
        ));
    }
    Ok(pitch as i32)
}

fn validate_score(notes: &[ScoreNote]) -> Result<()> {
    if notes.is_empty()
        || notes.iter().any(|note| !note.onset_beat.is_finite())
        || notes
            .windows(2)
            .any(|pair| pair[0].onset_beat > pair[1].onset_beat)
    {
        return Err(invalid(
            "score",
            "onsets must be nonempty, finite and sorted",
        ));
    }
    Ok(())
}

fn unique_score(notes: &[ScoreNote]) -> (Vec<f32>, Vec<HashSet<i32>>) {
    let mut positions = Vec::<f32>::new();
    let mut pitches = Vec::<HashSet<i32>>::new();
    for note in notes {
        if positions.last().is_none_or(|last| *last != note.onset_beat) {
            positions.push(note.onset_beat);
            pitches.push(HashSet::new());
        }
        pitches.last_mut().unwrap().insert(note.pitch);
    }
    (positions, pitches)
}

/// Runtime parameters for Parangonar's symbolic-level `SL_OLTW`.
#[derive(Debug, Clone)]
pub struct SlOltwConfig {
    /// Score-side search window.
    pub window_size: usize,
    /// Maximum score advance per performance note.
    pub max_run_count: usize,
}

impl Default for SlOltwConfig {
    fn default() -> Self {
        Self {
            window_size: 40,
            max_run_count: 10,
        }
    }
}

/// Port of Parangonar's single-loop symbolic OLTW matcher.
pub struct SlOltwMatcher {
    positions: Vec<f32>,
    reference: Vec<HashSet<i32>>,
    input: Vec<i32>,
    cost: Vec<Vec<f32>>,
    length: Vec<Vec<f32>>,
    current: usize,
    input_index: usize,
    config: SlOltwConfig,
    path: AlignmentPath,
}

impl SlOltwMatcher {
    /// Construct an `SL_OLTW` follower from the unloaded score note array.
    pub fn new(notes: &[ScoreNote], config: SlOltwConfig) -> Result<Self> {
        validate_score(notes)?;
        if config.window_size == 0 || config.max_run_count == 0 {
            return Err(invalid("window_size/max_run_count", "must be at least 1"));
        }
        let (positions, reference) = unique_score(notes);
        let mut cost = vec![vec![f32::INFINITY; 2]; positions.len() + 1];
        let length = vec![vec![0.0; 2]; positions.len() + 1];
        cost[0][0] = 0.0;
        Ok(Self {
            positions,
            reference,
            input: Vec::new(),
            cost,
            length,
            current: 0,
            input_index: 0,
            config,
            path: AlignmentPath::default(),
        })
    }
}

impl OnlineAlignment for SlOltwMatcher {
    fn try_call(&mut self, obs: &Observation) -> Result<f64> {
        let pitch = observation_pitch(obs)?;
        self.input.push(pitch);
        let start = self.current.saturating_sub(self.config.window_size);
        let end = self
            .current
            .saturating_add(self.config.window_size)
            .min(self.reference.len());
        let mut min_cost = f64::INFINITY;
        let mut min_index = start;
        if start == 0 && self.input_index == 0 {
            self.cost[1][1] = 0.0;
            self.length[1][1] = 0.0;
            min_cost = 0.0;
            min_index = 0;
        }
        for i in start..end {
            if i == 0 && self.input_index == 0 {
                continue;
            }
            let mut local_cost = f64::INFINITY;
            let mut local_length = 1.0;
            for (d_index, &(istep, jstep)) in [(1usize, 0usize), (1, 1), (0, 1)].iter().enumerate()
            {
                let prev_row = i + 1 - istep;
                let jlocal = 1usize.saturating_sub(jstep);
                let prev_cost = self.cost[prev_row][jlocal];
                if prev_cost.is_finite() {
                    let prev_length = self.length[prev_row][jlocal];
                    let distance = if self.reference[i].contains(&pitch) {
                        0.2
                    } else {
                        1.0
                    };
                    let weights = [2.0, 1.0, 1.0];
                    // NumPy promotes the weighted distance to f64 before comparing costs.
                    let cost = (f64::from(prev_cost * prev_length) + distance * weights[d_index])
                        / f64::from(prev_length + 1.0);
                    if cost < local_cost {
                        local_cost = cost;
                        local_length = 1.0 + prev_length;
                    }
                }
            }
            self.cost[i + 1][1] = local_cost as f32;
            self.length[i + 1][1] = local_length;
            if local_cost < min_cost {
                min_cost = local_cost;
                min_index = i;
            }
        }
        for i in 0..=self.reference.len() {
            self.cost[i][0] = self.cost[i][1];
            self.cost[i][1] = f32::INFINITY;
            self.length[i][0] = self.length[i][1];
            self.length[i][1] = f32::INFINITY;
        }
        if self.input_index != 0 {
            self.current = self
                .current
                .max(min_index)
                .min(self.current.saturating_add(self.config.max_run_count));
        }
        self.input_index += 1;
        self.path
            .push(obs.perf_time, f64::from(self.positions[self.current]));
        Ok(f64::from(self.positions[self.current]))
    }

    fn call(&mut self, obs: &Observation) -> f64 {
        match self.try_call(obs) {
            Ok(position) => position,
            Err(error) => {
                tracing::error!(%error, "SL_OLTW observation rejected; use try_call to propagate errors");
                f64::NAN
            }
        }
    }

    fn is_still_following(&self) -> bool {
        self.current < self.positions.len().saturating_sub(1)
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
            ..StepDebug::default()
        }
    }
}

/// Runtime parameters for Parangonar's tempo-aware `SLT_OLTW`.
#[derive(Debug, Clone)]
pub struct SltOltwConfig {
    /// Score-side search window.
    pub window_size: usize,
    /// Maximum score advance per performance note.
    pub max_run_count: usize,
    /// Initial seconds per score beat.
    pub init_tempo: f64,
    /// Exponential tempo smoothing factor.
    pub tempo_factor: f64,
    /// Relative weight of onset deviation.
    pub time_weight: f64,
}

impl Default for SltOltwConfig {
    fn default() -> Self {
        Self {
            window_size: 40,
            max_run_count: 10,
            init_tempo: 60.0 / 90.0,
            tempo_factor: 0.1,
            time_weight: 2.0,
        }
    }
}

/// Port of Parangonar's single-loop tempo-aware OLTW matcher.
///
/// Observation times must fit in `f32` for reference-compatible tempo calculations;
/// the alignment path retains the original `f64` times.
pub struct SltOltwMatcher {
    positions: Vec<f32>,
    reference: Vec<(f32, HashSet<i32>)>,
    input: Vec<(f32, i32)>,
    cost: Vec<Vec<f32>>,
    length: Vec<Vec<f32>>,
    tempo: Vec<Vec<f32>>,
    current: usize,
    input_index: usize,
    config: SltOltwConfig,
    path: AlignmentPath,
}

impl SltOltwMatcher {
    /// Construct an `SLT_OLTW` follower from the unloaded score note array.
    pub fn new(notes: &[ScoreNote], config: SltOltwConfig) -> Result<Self> {
        validate_score(notes)?;
        if config.window_size == 0
            || config.max_run_count == 0
            || !config.init_tempo.is_finite()
            || config.init_tempo <= 0.0
            || !(config.init_tempo as f32).is_finite()
            || config.init_tempo as f32 == 0.0
            || !config.tempo_factor.is_finite()
            || !(config.tempo_factor as f32).is_finite()
            || !config.time_weight.is_finite()
            || !(config.time_weight as f32).is_finite()
        {
            return Err(invalid(
                "SLT_OLTW",
                "window, run count and tempo parameters must be valid",
            ));
        }
        let (positions, pitch_sets) = unique_score(notes);
        let reference = positions
            .iter()
            .cloned()
            .zip(pitch_sets)
            .collect::<Vec<_>>();
        let n = reference.len();
        let mut cost = vec![vec![f32::INFINITY; 2]; n + 1];
        let length = vec![vec![0.0; 2]; n + 1];
        let tempo = vec![vec![config.init_tempo as f32; 2]; n + 1];
        cost[0][0] = 0.0;
        Ok(Self {
            positions,
            reference,
            input: Vec::new(),
            cost,
            length,
            tempo,
            current: 0,
            input_index: 0,
            config,
            path: AlignmentPath::default(),
        })
    }
}

impl OnlineAlignment for SltOltwMatcher {
    fn try_call(&mut self, obs: &Observation) -> Result<f64> {
        let pitch = observation_pitch(obs)?;
        let time = obs.perf_time as f32;
        if !time.is_finite() {
            return Err(invalid("observation", "time must fit in float32"));
        }
        self.input.push((time, pitch));
        let start = self.current.saturating_sub(self.config.window_size);
        let end = self
            .current
            .saturating_add(self.config.window_size)
            .min(self.reference.len());
        let mut min_cost = f64::INFINITY;
        let mut min_index = start;
        let j = self.input_index;
        if start == 0 && j == 0 {
            self.cost[1][1] = 0.0;
            self.length[1][1] = 0.0;
            min_cost = 0.0;
            min_index = 0;
        }
        for i in start..end {
            if i == 0 && j == 0 {
                continue;
            }
            let mut local_cost = f64::INFINITY;
            let mut local_tempo = self.config.init_tempo as f32;
            let mut local_length = 1.0;
            for (d_index, &(istep, jstep)) in [(1usize, 0usize), (1, 1), (0, 1)].iter().enumerate()
            {
                let previ = i.saturating_sub(istep);
                let prev_row = i + 1 - istep;
                let prevj = j.saturating_sub(jstep);
                let jlocal = 1usize.saturating_sub(jstep);
                let prev_cost = self.cost[prev_row][jlocal];
                if prev_cost.is_finite() && prevj < self.input.len() {
                    let prev_length = self.length[prev_row][jlocal];
                    let tempo = self.tempo[prev_row][jlocal];
                    let pitch_dist = if self.reference[i].1.contains(&pitch) {
                        0.0
                    } else {
                        1.0
                    };
                    let onset_s = self.reference[i].0;
                    let prev_onset_s = self.reference[previ].0;
                    let onset_p = time;
                    let prev_onset_p = self.input[prevj].0;
                    let estimated = prev_onset_p + (onset_s - prev_onset_s) * tempo;
                    let onset_dist = (onset_p - estimated).abs() / tempo;
                    let current_tempo = if onset_s - prev_onset_s > 0.0 {
                        (onset_p - prev_onset_p) / (onset_s - prev_onset_s)
                    } else {
                        tempo + (onset_p - prev_onset_p)
                    };
                    let new_tempo = ((self.config.tempo_factor as f32) * current_tempo
                        + ((1.0 - self.config.tempo_factor) as f32) * tempo)
                        .clamp(0.05, 10.0);
                    let distance = pitch_dist + self.config.time_weight as f32 * onset_dist;
                    let weights = [2.0, 1.0, 1.0];
                    // The reference uses float32 onsets and tempo, but float64 direction weights.
                    let cost = (f64::from(prev_cost * prev_length)
                        + f64::from(distance) * weights[d_index])
                        / f64::from(prev_length + 1.0);
                    if cost < local_cost {
                        local_cost = cost;
                        local_tempo = new_tempo;
                        local_length = 1.0 + prev_length;
                    }
                }
            }
            self.cost[i + 1][1] = local_cost as f32;
            self.length[i + 1][1] = local_length;
            self.tempo[i + 1][1] = local_tempo;
            if local_cost < min_cost {
                min_cost = local_cost;
                min_index = i;
            }
        }
        for i in 0..=self.reference.len() {
            self.cost[i][0] = self.cost[i][1];
            self.cost[i][1] = f32::INFINITY;
            self.length[i][0] = self.length[i][1];
            self.length[i][1] = 0.0;
            self.tempo[i][0] = self.tempo[i][1];
            self.tempo[i][1] = self.config.init_tempo as f32;
        }
        if j != 0 {
            self.current = self
                .current
                .saturating_sub(self.config.max_run_count)
                .max(min_index)
                .min(self.current.saturating_add(self.config.max_run_count));
        }
        self.input_index += 1;
        self.path
            .push(obs.perf_time, f64::from(self.positions[self.current]));
        Ok(f64::from(self.positions[self.current]))
    }

    fn call(&mut self, obs: &Observation) -> f64 {
        match self.try_call(obs) {
            Ok(position) => position,
            Err(error) => {
                tracing::error!(%error, "SLT_OLTW observation rejected; use try_call to propagate errors");
                f64::NAN
            }
        }
    }

    fn is_still_following(&self) -> bool {
        self.current < self.positions.len().saturating_sub(1)
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
            ..StepDebug::default()
        }
    }
}

/// Runtime parameters for Parangonar's `OnlinePureTransformerMatcher`.
#[derive(Debug, Clone)]
pub struct OptmConfig {
    /// Enable lostness-triggered jumps.
    pub allow_jump: bool,
    /// Backup gain recursion depth.
    pub backup_steps: usize,
    /// Lostness trigger threshold.
    pub jump_trigger: usize,
}

impl Default for OptmConfig {
    fn default() -> Self {
        Self {
            allow_jump: true,
            backup_steps: 24,
            jump_trigger: 24,
        }
    }
}

/// Pure-transformer event follower ported over the bundled OTM architecture weights.
pub struct OptmMatcher {
    positions: Vec<f32>,
    pitches: Vec<Vec<i32>>,
    performance: Vec<i32>,
    model: OtmModel,
    config: OptmConfig,
    backup: Vec<f64>,
    unmatched: VecDeque<bool>,
    unmatched_count: usize,
    current: usize,
    input_index: usize,
    path: AlignmentPath,
}

impl OptmMatcher {
    /// Construct an `OPTM` follower from the unloaded score note array.
    pub fn new(notes: &[ScoreNote], config: OptmConfig) -> Result<Self> {
        if notes.is_empty() {
            return Err(invalid("score", "OPTM requires at least one score note"));
        }
        if notes.iter().any(|note| !note.onset_beat.is_finite())
            || notes
                .windows(2)
                .any(|pair| pair[0].onset_beat > pair[1].onset_beat)
        {
            return Err(invalid(
                "score",
                "OPTM score onsets must be finite and sorted",
            ));
        }
        let buffer_len = config
            .jump_trigger
            .checked_add(4)
            .filter(|_| config.backup_steps > 0 && config.jump_trigger > 0)
            .ok_or_else(|| invalid("OPTM", "backup_steps and jump_trigger must be positive"))?;
        let mut positions = Vec::new();
        let mut pitches = Vec::new();
        for onset in notes.chunk_by(|a, b| a.onset_beat == b.onset_beat) {
            positions.push(onset[0].onset_beat);
            pitches.push(python_pitch_set(
                onset
                    .iter()
                    .filter(|note| !note.is_grace)
                    .map(|note| note.pitch),
            ));
        }
        let mut unmatched = VecDeque::new();
        unmatched.try_reserve_exact(buffer_len).map_err(|error| {
            invalid("jump_trigger", format!("buffer allocation failed: {error}"))
        })?;
        unmatched.resize(buffer_len, false);
        Ok(Self {
            backup: vec![0.0; positions.len()],
            positions,
            pitches,
            performance: Vec::new(),
            model: OtmModel::bundled()?,
            config,
            unmatched,
            unmatched_count: 0,
            current: 0,
            input_index: 0,
            path: AlignmentPath::default(),
        })
    }

    fn update_backup(&mut self, pitch: i32) {
        let max = self.config.backup_steps as f64;
        let mut previous = self.backup[0];
        for (index, pitches) in self.pitches.iter().enumerate() {
            let value = self.backup[index];
            if pitches.contains(&pitch) {
                let filtered = previous.max(value);
                let normalized = (filtered - max) / -max;
                let next = (1.0 - 1.0 / max) * normalized;
                self.backup[index] = -max * next + max;
            } else {
                self.backup[index] = (value - 1.0).clamp(0.0, max);
            }
            previous = value;
        }
    }

    fn track_lostness(&mut self, predicted: usize, pitch: i32) -> usize {
        let unmatched = !self.pitches[predicted].contains(&pitch);
        self.unmatched_count -= usize::from(self.unmatched.pop_front().expect("nonempty buffer"));
        self.unmatched.push_back(unmatched);
        self.unmatched_count += usize::from(unmatched);
        if self.unmatched_count < self.config.jump_trigger {
            return predicted;
        }
        let start = predicted.saturating_sub(4);
        let end = predicted.saturating_add(16).min(self.backup.len());
        let mut best = start;
        for index in start + 1..end {
            if self.backup[index] > self.backup[best] {
                best = index;
            }
        }
        self.backup.fill(0.0);
        self.unmatched.iter_mut().for_each(|value| *value = false);
        self.unmatched_count = 0;
        best
    }
}

impl OnlineAlignment for OptmMatcher {
    fn try_call(&mut self, obs: &Observation) -> Result<f64> {
        let pitch = obs.features.first().copied().unwrap_or(f32::NAN);
        if obs.features.len() != 1
            || !pitch.is_finite()
            || pitch.fract() != 0.0
            || !(20.0..=110.0).contains(&pitch)
            || !(obs.perf_time as f32).is_finite()
        {
            return Err(invalid(
                "OPTM observation",
                "requires one integral pitch 20..=110 and a finite time",
            ));
        }
        let pitch = pitch as i32;
        let start = self.current.saturating_sub(7);
        let end = (self.current + 9).min(self.positions.len());
        let mut performance = self.performance.clone();
        if performance.len() == 8 {
            performance.remove(0);
        }
        performance.push(pitch);
        let tokens = tokenize(&self.pitches[start..end], &performance)?;
        let logits = self.model.forward(&tokens)?;
        let probability = |index: usize| {
            let a = logits[index * 2];
            let b = logits[index * 2 + 1];
            let max = a.max(b);
            let ea = libm::expf(a - max);
            let eb = libm::expf(b - max);
            eb * (1.0 / (ea + eb))
        };
        let mut pred = 0usize;
        for index in 1..26 {
            if probability(index) > probability(pred) {
                pred = index;
            }
        }
        let predicted = (start as isize + pred as isize - performance.len() as isize - 1)
            .clamp(0, self.positions.len() as isize - 1) as usize;
        self.performance = performance;
        self.update_backup(pitch);
        self.current = if self.config.allow_jump {
            self.track_lostness(predicted, pitch)
        } else {
            predicted
        };
        self.input_index += 1;
        self.path
            .push(obs.perf_time, f64::from(self.positions[self.current]));
        Ok(f64::from(self.positions[self.current]))
    }

    fn call(&mut self, obs: &Observation) -> f64 {
        match self.try_call(obs) {
            Ok(position) => position,
            Err(error) => {
                tracing::error!(%error, "OPTM observation rejected; use try_call to propagate errors");
                f64::NAN
            }
        }
    }

    fn is_still_following(&self) -> bool {
        self.current < self.positions.len().saturating_sub(1)
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
            ..StepDebug::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn notes() -> Vec<ScoreNote> {
        (0..5)
            .map(|index| ScoreNote {
                id: index.to_string(),
                is_grace: false,
                pitch: 60 + index,
                onset_beat: index as f32,
                duration_beat: 1.0,
                onset_quarter: index as f32,
                duration_quarter: 1.0,
                onset_div: i64::from(index),
                duration_div: 1,
                voice: 1,
            })
            .collect()
    }

    #[test]
    fn optm_backup_and_jump_match_reference() {
        let mut matcher = OptmMatcher::new(
            &notes(),
            OptmConfig {
                allow_jump: true,
                backup_steps: 4,
                jump_trigger: 2,
            },
        )
        .unwrap();
        for (pitch, expected) in [(62, 0), (63, 3), (64, 0), (62, 2), (63, 0)] {
            matcher.update_backup(pitch);
            matcher.current = matcher.track_lostness(0, pitch);
            assert_eq!(matcher.current, expected);
        }
    }

    #[test]
    fn optm_rejects_invalid_input_without_changing_path() {
        let mut matcher = OptmMatcher::new(&notes(), OptmConfig::default()).unwrap();
        for features in [vec![], vec![19.0], vec![60.5], vec![60.0, 61.0]] {
            assert!(
                matcher
                    .try_call(&Observation {
                        features,
                        perf_time: 0.0
                    })
                    .is_err()
            );
        }
        assert!(matcher.alignment_path().perf_time.is_empty());
        assert_eq!(matcher.input_index, 0);
    }
}
