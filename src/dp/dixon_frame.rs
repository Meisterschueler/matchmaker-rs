//! Frame-level `dixon` online time warping for audio input.
//!
//! Upstream: `matchmaker.dp.oltw_dixon.OnlineTimeWarpingDixonFrame`.
//!
//! ```
//! use matchmaker_rs::base::OnlineAlignment;
//! use matchmaker_rs::dp::dixon_frame::DixonFrame;
//! use matchmaker_rs::features::Observation;
//! use matchmaker_rs::matchmaker::{AudioMethod, AudioMethodConfig};
//! let config = AudioMethodConfig::for_method(AudioMethod::Dixon);
//! let mut follower = DixonFrame::new(
//!     vec![vec![1.0, 0.0], vec![0.0, 1.0]],
//!     vec![0.0, 1.0],
//!     vec![0.0, 1.0],
//!     config,
//! )?;
//! let beat = follower.call(&Observation { features: vec![1.0, 0.0], perf_time: 0.0 });
//! assert_eq!(beat, 0.0);
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::collections::HashMap;

use crate::base::{AlignmentPath, OnlineAlignment, StepDebug};
use crate::error::{Error, Result};
use crate::features::Observation;
use crate::matchmaker::{AudioDistance, AudioMethod, AudioMethodConfig};

/// Frame-level audio Dixon follower with sparse accumulated path costs.
#[derive(Debug, Clone)]
pub struct DixonFrame {
    reference: Vec<Vec<f32>>,
    score_positions: Vec<f32>,
    frame_to_beat: Vec<f64>,
    frame_rate: f64,
    window_frames: usize,
    max_run_count: usize,
    distance: AudioDistance,
    accumulated_costs: HashMap<(usize, usize), f64>,
    input_features: Vec<Vec<f32>>,
    history: Vec<(usize, f64)>,
    ref_pointer: usize,
    input_pointer: usize,
    run_count_ref: usize,
    run_count_input: usize,
    best_ref: usize,
    best_input: usize,
    current_index: usize,
    input_index: usize,
    path: AlignmentPath,
}

impl DixonFrame {
    /// Create a frame-level follower from score features, onset beats and a beat per score frame.
    pub fn new(
        reference_features: Vec<Vec<f32>>,
        score_positions: Vec<f32>,
        frame_to_beat: Vec<f64>,
        config: AudioMethodConfig,
    ) -> Result<Self> {
        config.validate(AudioMethod::Dixon)?;
        if reference_features.is_empty() || reference_features[0].is_empty() {
            return Err(audio_config(
                "reference_features",
                "must contain at least one non-empty feature frame",
            ));
        }
        if reference_features
            .iter()
            .any(|frame| frame.len() != reference_features[0].len())
        {
            return Err(audio_config(
                "reference_features",
                "all feature frames must have the same dimension",
            ));
        }
        if score_positions.is_empty() || score_positions.windows(2).any(|w| w[0] > w[1]) {
            return Err(audio_config(
                "score_positions",
                "must be non-empty and sorted in ascending order",
            ));
        }
        if frame_to_beat.len() != reference_features.len()
            || frame_to_beat.iter().any(|beat| !beat.is_finite())
        {
            return Err(audio_config(
                "frame_to_beat",
                "must contain one finite beat value per reference frame",
            ));
        }

        let hop_length = config
            .hop_length
            .unwrap_or((config.sample_rate / config.frame_rate) as usize);
        if hop_length == 0 {
            return Err(audio_config("frame_rate", "must not exceed sample_rate"));
        }
        let frame_rate = f64::from(config.sample_rate) / hop_length as f64;
        let window_frames = (config.window_size * frame_rate) as usize;
        if window_frames == 0 {
            return Err(audio_config(
                "window_size",
                "must span at least one audio frame",
            ));
        }

        Ok(Self {
            reference: reference_features,
            score_positions,
            frame_to_beat,
            frame_rate,
            window_frames,
            max_run_count: config.max_run_count.unwrap_or(3),
            distance: config.distance_func,
            accumulated_costs: HashMap::new(),
            input_features: Vec::new(),
            history: vec![(0, 0.0)],
            ref_pointer: 0,
            input_pointer: 0,
            run_count_ref: 0,
            run_count_input: 0,
            best_ref: 0,
            best_input: 0,
            current_index: 0,
            input_index: 0,
            path: AlignmentPath::default(),
        })
    }

    /// Current best reference-frame index.
    pub fn current_frame(&self) -> usize {
        self.best_ref
    }

    fn distance_between(&self, left: &[f32], right: &[f32]) -> f64 {
        match self.distance {
            AudioDistance::Manhattan => left
                .iter()
                .zip(right)
                .map(|(&a, &b)| f64::from((a - b).abs()))
                .sum(),
            AudioDistance::Euclidean => left
                .iter()
                .zip(right)
                .map(|(&a, &b)| {
                    let difference = f64::from(a) - f64::from(b);
                    difference * difference
                })
                .sum::<f64>()
                .sqrt(),
        }
    }

    fn evaluate_path_cost(&mut self, ref_index: usize, input_index: usize) {
        if ref_index >= self.reference.len() {
            return;
        }
        let local = self.distance_between(
            &self.reference[ref_index],
            &self.input_features[input_index],
        );
        if ref_index == 0 && input_index == 0 {
            self.accumulated_costs.insert((0, 0), local);
            return;
        }

        let mut best = f64::INFINITY;
        for (prev_ref, prev_input, weight) in [
            (ref_index.checked_sub(1), input_index.checked_sub(1), 2.0),
            (ref_index.checked_sub(1), Some(input_index), 1.0),
            (Some(ref_index), input_index.checked_sub(1), 1.0),
        ] {
            let previous = prev_ref
                .zip(prev_input)
                .and_then(|cell| self.accumulated_costs.get(&cell).copied())
                .unwrap_or(f64::INFINITY);
            if previous < f64::INFINITY {
                let cost = previous + weight * local;
                if cost <= best {
                    best = cost;
                }
            }
        }
        if best < f64::INFINITY {
            self.accumulated_costs
                .insert((ref_index, input_index), best);
        }
    }

    fn path_cost(&self, ref_index: usize, input_index: usize) -> f64 {
        self.accumulated_costs
            .get(&(ref_index, input_index))
            .copied()
            .map_or(f64::INFINITY, |cost| {
                cost / (1 + ref_index + input_index) as f64
            })
    }

    fn update_best_alignment(&mut self) -> (usize, usize) {
        let current_ref = self.ref_pointer - 1;
        let current_input = self.input_pointer - 1;
        let mut best_cost = self.path_cost(current_ref, current_input);
        self.best_ref = current_ref;
        self.best_input = current_input;

        for ref_index in current_ref
            .saturating_add(1)
            .saturating_sub(self.window_frames)..=current_ref
        {
            let cost = self.path_cost(ref_index, current_input);
            if cost < best_cost {
                best_cost = cost;
                self.best_ref = ref_index;
                self.best_input = current_input;
            }
        }
        for input_index in current_input
            .saturating_add(1)
            .saturating_sub(self.window_frames)..=current_input
        {
            let cost = self.path_cost(current_ref, input_index);
            if cost < best_cost {
                best_cost = cost;
                self.best_ref = current_ref;
                self.best_input = input_index;
            }
        }
        (self.best_ref, self.best_input)
    }

    fn get_inc(&mut self) -> Inc {
        if self.ref_pointer >= self.reference.len() {
            return Inc::Input;
        }
        let (best_ref, best_input) = self.update_best_alignment();
        if self.ref_pointer < self.window_frames || self.input_pointer < self.window_frames {
            return Inc::Both;
        }
        if self.run_count_ref >= self.max_run_count {
            return Inc::Input;
        }
        if self.run_count_input >= self.max_run_count {
            return Inc::Ref;
        }
        let current_ref = self.ref_pointer - 1;
        let current_input = self.input_pointer - 1;
        if best_ref == current_ref && best_input == current_input {
            Inc::Both
        } else if best_ref < current_ref {
            Inc::Input
        } else {
            Inc::Ref
        }
    }

    fn advance_ref(&mut self) {
        self.ref_pointer += 1;
        let ref_index = self.ref_pointer - 1;
        let current_input = self.input_pointer - 1;
        let start = current_input
            .saturating_add(1)
            .saturating_sub(self.window_frames);
        for input_index in start..=current_input {
            self.evaluate_path_cost(ref_index, input_index);
        }
    }

    fn advance_input(&mut self, features: &[f32]) {
        self.input_features.push(features.to_vec());
        self.input_pointer += 1;
        let input_index = self.input_pointer - 1;
        let current_ref = self.ref_pointer - 1;
        let start = current_ref
            .saturating_add(1)
            .saturating_sub(self.window_frames);
        for ref_index in start..=current_ref {
            self.evaluate_path_cost(ref_index, input_index);
        }
        if self.input_pointer.is_multiple_of(self.window_frames) {
            let cutoff = self.ref_pointer.saturating_sub(2 * self.window_frames);
            if cutoff > 0 {
                self.accumulated_costs
                    .retain(|(ref_index, _), _| *ref_index >= cutoff);
            }
        }
    }

    fn save_history(&mut self) {
        let beat = self.frame_to_beat[self.best_ref.min(self.frame_to_beat.len() - 1)];
        self.history.push((self.best_input, beat));
    }

    fn update_path(&mut self) {
        let mut score_beats = vec![None; self.input_pointer];
        for &(input_index, beat) in &self.history {
            if input_index < score_beats.len() {
                score_beats[input_index] = Some(beat);
            }
        }
        let mut previous = self.frame_to_beat[0];
        let mut perf_time = Vec::with_capacity(self.input_pointer);
        let mut score_beat = Vec::with_capacity(self.input_pointer);
        for (index, position) in score_beats.into_iter().enumerate() {
            if let Some(position) = position {
                previous = position;
            }
            perf_time.push(index as f64 / self.frame_rate);
            score_beat.push(previous);
        }
        self.path = AlignmentPath {
            perf_time,
            score_beat,
        };
    }

    fn frame_to_score_index(&self) -> usize {
        let beat = self.frame_to_beat[self.best_ref.min(self.frame_to_beat.len() - 1)];
        self.score_positions
            .partition_point(|position| f64::from(*position) <= beat)
            .saturating_sub(1)
            .min(self.score_positions.len() - 1)
    }

    fn step(&mut self, features: &[f32]) {
        let mut pending = Some(features);
        if self.input_pointer == 0 {
            self.input_features.push(features.to_vec());
            self.input_pointer = 1;
            self.ref_pointer = 1;
            self.evaluate_path_cost(0, 0);
            self.input_index += 1;
            self.current_index = self.frame_to_score_index();
            self.update_path();
            return;
        }

        while self.ref_pointer < self.reference.len() {
            let input_direction = self.get_inc();
            let advance_input = input_direction != Inc::Ref;
            if advance_input && pending.is_none() {
                break;
            }
            if advance_input {
                self.advance_input(pending.take().expect("pending frame"));
            }

            let ref_direction = self.get_inc();
            let advance_ref = ref_direction != Inc::Input;
            if advance_ref {
                self.advance_ref();
            }
            self.update_best_alignment();
            self.update_run_counts(advance_ref, advance_input);
            self.save_history();
        }
        self.current_index = self.frame_to_score_index();
        self.input_index += 1;
        self.update_path();
    }

    fn update_run_counts(&mut self, advance_ref: bool, advance_input: bool) {
        if advance_ref && advance_input {
            self.run_count_ref = 0;
            self.run_count_input = 0;
        } else if advance_ref {
            self.run_count_ref += 1;
            self.run_count_input = 0;
        } else {
            self.run_count_input += 1;
            self.run_count_ref = 0;
        }
    }
}

impl OnlineAlignment for DixonFrame {
    fn call(&mut self, observation: &Observation) -> f64 {
        let _span = tracing::debug_span!("inference", method = "dixon").entered();
        self.step(&observation.features);
        let position = self.frame_to_beat[self.best_ref.min(self.frame_to_beat.len() - 1)];
        tracing::debug!(perf_time = observation.perf_time, position);
        position
    }

    fn is_still_following(&self) -> bool {
        self.ref_pointer < self.reference.len()
    }

    fn alignment_path(&self) -> &AlignmentPath {
        &self.path
    }

    fn score_positions(&self) -> &[f32] {
        &self.score_positions
    }

    fn debug_state(&self) -> StepDebug {
        StepDebug {
            current_index: self.current_index,
            ..StepDebug::default()
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Inc {
    Ref,
    Input,
    Both,
}

fn audio_config(key: &str, reason: &str) -> Error {
    Error::AudioConfig {
        key: key.into(),
        reason: reason.into(),
    }
}
