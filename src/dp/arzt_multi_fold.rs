//! Parallel repeat-hypothesis OLTW, based on `matchmaker.dp.oltw_arzt_multi_fold`.
//!
//! Each candidate owns an independent Arzt frame tracker. The lowest trailing local-cost mean
//! wins, with a relative hysteresis margin before switching.
//!
//! ```
//! use matchmaker_rs::base::OnlineAlignment;
//! use matchmaker_rs::dp::arzt_multi_fold::{ArztMultiFoldCandidate, ArztMultiFoldFrame};
//! use matchmaker_rs::matchmaker::{AudioMethod, AudioMethodConfig};
//! let candidate = ArztMultiFoldCandidate {
//!     index: 0,
//!     note_count: 2,
//!     max_onset_beat: 1.0,
//!     features: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
//!     score_positions: vec![0.0, 1.0],
//!     unfolded_frame_to_beat: vec![0.0, 1.0],
//!     notated_frame_to_beat: vec![0.0, 1.0],
//! };
//! let follower = ArztMultiFoldFrame::new(
//!     vec![candidate], vec![0.0, 1.0],
//!     AudioMethodConfig::for_method(AudioMethod::ArztMultiFold),
//! )?;
//! assert_eq!(follower.score_positions(), &[0.0, 1.0]);
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::collections::{HashSet, VecDeque};

use crate::Result;
use crate::base::{AlignmentPath, MultiFoldStepDebug, OnlineAlignment, StepDebug};
use crate::dp::arzt_frame::ArztFrame;
use crate::error::Error;
use crate::features::Observation;
use crate::matchmaker::{AudioMethod, AudioMethodConfig};

/// One reference score and its unfolded-to-notated beat mapping.
#[derive(Debug, Clone)]
pub struct ArztMultiFoldCandidate {
    /// Stable index in the reference candidate order.
    pub index: usize,
    /// Number of score notes in this candidate.
    pub note_count: usize,
    /// Final candidate onset in beats.
    pub max_onset_beat: f32,
    /// Score audio feature frames.
    pub features: Vec<Vec<f32>>,
    /// Candidate-specific score onset beats.
    pub score_positions: Vec<f32>,
    /// Unfolded beat associated with each feature frame.
    pub unfolded_frame_to_beat: Vec<f64>,
    /// Input-score (notated) beat associated with each feature frame.
    pub notated_frame_to_beat: Vec<f64>,
}

#[derive(Debug, Clone)]
struct CandidateState {
    reference: ArztMultiFoldCandidate,
    tracker: ArztFrame,
    recent_costs: VecDeque<f64>,
    mean_recent_cost: f64,
}

/// Frame-level Arzt follower over parallel repeat hypotheses.
#[derive(Debug, Clone)]
pub struct ArztMultiFoldFrame {
    candidates: Vec<CandidateState>,
    initial: Vec<CandidateState>,
    score_positions: Vec<f32>,
    selection_window: usize,
    switch_margin: f64,
    active_hypothesis: usize,
    input_index: usize,
    path: AlignmentPath,
}

impl ArztMultiFoldFrame {
    /// Create a follower from ordered repeat candidates and the input score's notated onsets.
    pub fn new(
        candidates: Vec<ArztMultiFoldCandidate>,
        score_positions: Vec<f32>,
        config: AudioMethodConfig,
    ) -> Result<Self> {
        config.validate(AudioMethod::ArztMultiFold)?;
        if candidates.is_empty() {
            return Err(audio_config(
                "candidates",
                "must contain at least one hypothesis",
            ));
        }
        if score_positions.is_empty()
            || score_positions
                .windows(2)
                .any(|window| window[0] > window[1])
        {
            return Err(audio_config(
                "score_positions",
                "must be non-empty and sorted in ascending order",
            ));
        }
        let selection_window = config.selection_window.unwrap_or(50);
        let switch_margin = config.switch_margin.unwrap_or(0.05);
        let mut ids = HashSet::new();
        let mut feature_width = None;
        let mut follower_config = config;
        follower_config.selection_window = None;
        follower_config.switch_margin = None;

        let mut states = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            if !ids.insert(candidate.index) {
                return Err(audio_config("candidate.index", "must be unique"));
            }
            if candidate.note_count == 0
                || !candidate.max_onset_beat.is_finite()
                || candidate.notated_frame_to_beat.len() != candidate.features.len()
                || candidate
                    .notated_frame_to_beat
                    .iter()
                    .any(|beat| !beat.is_finite())
            {
                return Err(audio_config(
                    "candidate",
                    "must have notes and one finite notated beat per feature frame",
                ));
            }
            let width = candidate.features.first().map_or(0, Vec::len);
            if feature_width.is_some_and(|expected| expected != width) {
                return Err(audio_config(
                    "candidate.features",
                    "all hypotheses must use the same feature dimension",
                ));
            }
            feature_width = Some(width);
            let tracker = ArztFrame::new(
                candidate.features.clone(),
                candidate.score_positions.clone(),
                candidate.unfolded_frame_to_beat.clone(),
                follower_config.clone(),
            )?;
            states.push(CandidateState {
                reference: candidate,
                tracker,
                recent_costs: VecDeque::with_capacity(selection_window),
                mean_recent_cost: f64::INFINITY,
            });
        }
        Ok(Self {
            initial: states.clone(),
            candidates: states,
            score_positions,
            selection_window,
            switch_margin,
            active_hypothesis: 0,
            input_index: 0,
            path: AlignmentPath::default(),
        })
    }

    /// Reset every candidate, cost history, selection and alignment path.
    pub fn reset(&mut self) {
        self.candidates.clone_from(&self.initial);
        self.active_hypothesis = 0;
        self.input_index = 0;
        self.path = AlignmentPath::default();
    }
}

impl OnlineAlignment for ArztMultiFoldFrame {
    fn call(&mut self, observation: &Observation) -> f64 {
        let _span = tracing::debug_span!("inference", method = "arzt_multi_fold").entered();
        for candidate in &mut self.candidates {
            candidate.tracker.call(observation);
            let cost = f64::from(candidate.tracker.current_local_cost(&observation.features));
            if candidate.recent_costs.len() == self.selection_window {
                candidate.recent_costs.pop_front();
            }
            candidate.recent_costs.push_back(cost);
            candidate.mean_recent_cost =
                candidate.recent_costs.iter().sum::<f64>() / candidate.recent_costs.len() as f64;
        }
        let means: Vec<f64> = self
            .candidates
            .iter()
            .map(|candidate| candidate.mean_recent_cost)
            .collect();
        self.active_hypothesis =
            select_hypothesis(&means, self.active_hypothesis, self.switch_margin);
        let selected = &self.candidates[self.active_hypothesis];
        let frame = selected.tracker.current_frame();
        let position = selected.reference.notated_frame_to_beat
            [frame.min(selected.reference.notated_frame_to_beat.len() - 1)];
        self.path.push(observation.perf_time, position);
        tracing::debug!(
            perf_time = observation.perf_time,
            position,
            hypothesis = selected.reference.index
        );
        self.input_index += 1;
        position
    }

    fn is_still_following(&self) -> bool {
        self.score_positions.len() > 1
    }

    fn alignment_path(&self) -> &AlignmentPath {
        &self.path
    }

    fn score_positions(&self) -> &[f32] {
        &self.score_positions
    }

    fn debug_state(&self) -> StepDebug {
        let selected = &self.candidates[self.active_hypothesis];
        let position = self.path.score_beat.last().copied().unwrap_or(0.0);
        StepDebug {
            current_index: self
                .score_positions
                .partition_point(|beat| f64::from(*beat) <= position)
                .saturating_sub(1)
                .min(self.score_positions.len().saturating_sub(1)),
            multi_fold: Some(MultiFoldStepDebug {
                selected_hypothesis: selected.reference.index,
                current_frames: self
                    .candidates
                    .iter()
                    .map(|candidate| candidate.tracker.current_frame())
                    .collect(),
                local_costs: self
                    .candidates
                    .iter()
                    .map(|candidate| {
                        candidate
                            .recent_costs
                            .back()
                            .copied()
                            .unwrap_or(f64::INFINITY)
                    })
                    .collect(),
                mean_costs: self
                    .candidates
                    .iter()
                    .map(|candidate| candidate.mean_recent_cost)
                    .collect(),
            }),
            ..StepDebug::default()
        }
    }
}

fn audio_config(key: &str, reason: &str) -> Error {
    Error::AudioConfig {
        key: key.into(),
        reason: reason.into(),
    }
}

fn select_hypothesis(costs: &[f64], active: usize, switch_margin: f64) -> usize {
    let mut best = 0;
    for index in 1..costs.len() {
        if costs[index] < costs[best] {
            best = index;
        }
    }
    if costs[best] < costs[active] * (1.0 - switch_margin) {
        best
    } else {
        active
    }
}

#[cfg(test)]
mod tests {
    use super::select_hypothesis;

    #[test]
    fn switch_margin_is_strict_and_equal_costs_keep_first_candidate() {
        assert_eq!(select_hypothesis(&[75.0, 100.0], 1, 0.25), 1);
        assert_eq!(select_hypothesis(&[74.0, 100.0], 1, 0.25), 0);
        assert_eq!(select_hypothesis(&[1.0, 1.0, 2.0], 2, 0.05), 0);
        assert_eq!(select_hypothesis(&[1.0, 1.0], 1, 0.05), 1);
    }
}
