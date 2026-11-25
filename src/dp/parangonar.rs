//! Ports of Parangonar's symbolic OLTW and pure-transformer MIDI matchers.

use std::collections::HashSet;

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
        if notes.is_empty() {
            return Err(invalid("score", "SL_OLTW requires at least one score note"));
        }
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
    fn call(&mut self, obs: &Observation) -> f64 {
        let pitch = obs
            .features
            .first()
            .copied()
            .map(|v| v as i32)
            .unwrap_or(-1);
        self.input.push(pitch);
        let start = self.current.saturating_sub(self.config.window_size);
        let end = (self.current + self.config.window_size).min(self.reference.len());
        let mut min_cost = f32::INFINITY;
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
            let mut local_cost = f32::INFINITY;
            let mut local_length = 1.0;
            for (d_index, &(istep, jstep)) in [(1usize, 0usize), (1, 1), (0, 1)].iter().enumerate()
            {
                let previ = i.saturating_sub(istep);
                let jlocal = 1usize.saturating_sub(jstep);
                let prev_cost = self.cost[previ + 1][jlocal];
                if prev_cost.is_finite() {
                    let prev_length = self.length[previ + 1][jlocal];
                    let distance = if self.reference[i].contains(&pitch) {
                        0.2
                    } else {
                        1.0
                    };
                    let weights = [2.0, 1.0, 1.0];
                    let cost = (prev_cost * prev_length + distance * weights[d_index])
                        / (prev_length + 1.0);
                    if cost < local_cost {
                        local_cost = cost;
                        local_length = 1.0 + prev_length;
                    }
                }
            }
            self.cost[i + 1][1] = local_cost;
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
        f64::from(self.positions[self.current])
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
pub struct SltOltwMatcher {
    positions: Vec<f32>,
    reference: Vec<(f64, HashSet<i32>)>,
    input: Vec<(f64, i32)>,
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
        if notes.is_empty() {
            return Err(invalid(
                "score",
                "SLT_OLTW requires at least one score note",
            ));
        }
        if config.window_size == 0
            || config.max_run_count == 0
            || !config.init_tempo.is_finite()
            || config.init_tempo <= 0.0
        {
            return Err(invalid(
                "SLT_OLTW",
                "window, run count and tempo must be valid",
            ));
        }
        let (positions, pitch_sets) = unique_score(notes);
        let reference = positions
            .iter()
            .cloned()
            .zip(pitch_sets)
            .map(|(onset, pitches)| (f64::from(onset), pitches))
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
    fn call(&mut self, obs: &Observation) -> f64 {
        let pitch = obs
            .features
            .first()
            .copied()
            .map(|v| v as i32)
            .unwrap_or(-1);
        self.input.push((obs.perf_time, pitch));
        let start = self.current.saturating_sub(self.config.window_size);
        let end = (self.current + self.config.window_size).min(self.reference.len());
        let mut min_cost = f32::INFINITY;
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
            let mut local_cost = f32::INFINITY;
            let mut local_tempo = self.config.init_tempo as f32;
            let mut local_length = 1.0;
            for (d_index, &(istep, jstep)) in [(1usize, 0usize), (1, 1), (0, 1)].iter().enumerate()
            {
                let previ = i.saturating_sub(istep);
                let prevj = j.saturating_sub(jstep);
                let jlocal = 1usize.saturating_sub(jstep);
                let prev_cost = self.cost[previ + 1][jlocal];
                if prev_cost.is_finite() && prevj < self.input.len() {
                    let prev_length = self.length[previ + 1][jlocal];
                    let tempo = f64::from(self.tempo[previ + 1][jlocal]);
                    let pitch_dist = if self.reference[i].1.contains(&pitch) {
                        0.0
                    } else {
                        1.0
                    };
                    let onset_s = self.reference[i].0;
                    let prev_onset_s = self.reference[previ].0;
                    let onset_p = obs.perf_time;
                    let prev_onset_p = self.input[prevj].0;
                    let estimated = prev_onset_p + (onset_s - prev_onset_s) * tempo;
                    let onset_dist = (onset_p - estimated).abs() / tempo;
                    let current_tempo = if onset_s - prev_onset_s > 0.0 {
                        (onset_p - prev_onset_p) / (onset_s - prev_onset_s)
                    } else {
                        tempo + (onset_p - prev_onset_p)
                    };
                    let new_tempo = (self.config.tempo_factor * current_tempo
                        + (1.0 - self.config.tempo_factor) * tempo)
                        .clamp(0.05, 10.0);
                    let distance = pitch_dist + self.config.time_weight * onset_dist;
                    let weights = [2.0, 1.0, 1.0];
                    let cost = (prev_cost * prev_length + distance as f32 * weights[d_index])
                        / (prev_length + 1.0);
                    if cost < local_cost {
                        local_cost = cost;
                        local_tempo = new_tempo as f32;
                        local_length = 1.0 + prev_length;
                    }
                }
            }
            self.cost[i + 1][1] = local_cost;
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
        f64::from(self.positions[self.current])
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
    current: usize,
    input_index: usize,
    path: AlignmentPath,
}

impl OptmMatcher {
    /// Construct an `OPTM` follower from the unloaded score note array.
    pub fn new(notes: &[ScoreNote], _config: OptmConfig) -> Result<Self> {
        if notes.is_empty() {
            return Err(invalid("score", "OPTM requires at least one score note"));
        }
        let (positions, pitch_sets) = unique_score(notes);
        let pitches = pitch_sets
            .into_iter()
            .map(|set| set.into_iter().collect())
            .collect();
        Ok(Self {
            positions,
            pitches,
            performance: Vec::new(),
            model: OtmModel::bundled()?,
            current: 0,
            input_index: 0,
            path: AlignmentPath::default(),
        })
    }
}

impl OnlineAlignment for OptmMatcher {
    fn call(&mut self, obs: &Observation) -> f64 {
        let pitch = obs
            .features
            .first()
            .copied()
            .map(|v| v as i32)
            .unwrap_or(-1);
        self.performance.push(pitch);
        let start = self.current.saturating_sub(7);
        let end = (self.current + 9).min(self.positions.len());
        let score: Vec<Vec<i32>> = self.pitches[start..end].to_vec();
        let performance: Vec<i32> = self
            .performance
            .iter()
            .rev()
            .take(8)
            .copied()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if let Ok(tokens) = tokenize(&score, &performance) {
            if let Ok(logits) = self.model.forward(&tokens) {
                let mut pred = 0usize;
                for index in 1..26 {
                    if logits[index * 2 + 1] > logits[pred * 2 + 1] {
                        pred = index;
                    }
                }
                let relative = pred as i64
                    - performance.len() as i64
                    - 1
                    - (self.current as i64 - start as i64);
                let absolute = (relative + self.current as i64)
                    .clamp(0, self.positions.len().saturating_sub(1) as i64);
                self.current = absolute as usize;
            }
        }
        self.input_index += 1;
        self.path
            .push(obs.perf_time, f64::from(self.positions[self.current]));
        f64::from(self.positions[self.current])
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
