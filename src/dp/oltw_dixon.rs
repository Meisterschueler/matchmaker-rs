//! Port of `matchmaker.dp.oltw_dixon.OnlineTimeWarpingDixonEvent` (method key `dixon`, MIDI).
//!
//! Upstream: `matchmaker/dp/oltw_dixon.py`.
//!
//! Event-level online time warping after Dixon: a sparse cost matrix over (onset, observation)
//! cells, normalised by path length. Every step decides which axis to advance (`get_inc`) and
//! forces a switch after `max_run_count` advances in one direction.
//!
//! ```
//! use matchmaker_rs::base::OnlineAlignment;
//! use matchmaker_rs::dp::oltw_dixon::{DixonEvent, DixonEventConfig};
//! use matchmaker_rs::score::Score;
//! let score = Score::load("tests/data/simple_mozart_k265_var1.musicxml", true)?;
//! let (features, onsets) = score.onset_pianoroll(true);
//! let follower = DixonEvent::new(features, onsets, DixonEventConfig::default());
//! assert_eq!(follower.score_positions().len(), score.unique_onsets().len());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::collections::HashMap;

use crate::base::{AlignmentPath, OnlineAlignment, StepDebug};
use crate::dp::{Direction, Distance};
use crate::features::Observation;

/// Constructor arguments of `OnlineTimeWarpingDixonEvent` (the class defaults reach it for MIDI).
#[derive(Debug, Clone, PartialEq)]
pub struct DixonEventConfig {
    /// Search width in observations and onsets (default 30).
    pub window_size: usize,
    /// Consecutive advances in one direction before the other is forced (default 3, at least 1).
    pub max_run_count: usize,
    /// Local cost (`distance_func`, default Euclidean).
    pub distance: Distance,
}

impl Default for DixonEventConfig {
    fn default() -> Self {
        DixonEventConfig {
            window_size: 30,
            max_run_count: 3,
            distance: Distance::Euclidean,
        }
    }
}

/// The `dixon` follower for MIDI.
#[derive(Debug, Clone)]
pub struct DixonEvent {
    reference: Vec<Vec<f32>>,
    score_positions: Vec<f32>,
    config: DixonEventConfig,
    /// `_D`: accumulated cost of the finite cells `(onset, observation)`.
    cost: HashMap<(usize, usize), f64>,
    /// `_inputs`: every observation consumed so far.
    inputs: Vec<Vec<f32>>,
    /// `t >= 0` in the reference: the first observation was consumed.
    started: bool,
    /// Last consumed observation.
    t: usize,
    /// Last evaluated onset.
    j: usize,
    run_count_input: usize,
    run_count_ref: usize,
    current_index: usize,
    path: AlignmentPath,
}

impl DixonEvent {
    /// `OnlineTimeWarpingDixonEvent(reference_features, score_positions, ...)`; both come from
    /// [`Score::onset_pianoroll`](crate::score::Score::onset_pianoroll).
    pub fn new(
        reference_features: Vec<Vec<f32>>,
        score_positions: Vec<f32>,
        config: DixonEventConfig,
    ) -> Self {
        DixonEvent {
            reference: reference_features,
            score_positions,
            config,
            cost: HashMap::new(),
            inputs: Vec::new(),
            started: false,
            t: 0,
            j: 0,
            run_count_input: 0,
            run_count_ref: 0,
            current_index: 0,
            path: AlignmentPath::default(),
        }
    }

    /// Index of the current onset.
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// `_evaluate`: accumulated cost of cell `(ref_i, inp_j)` from its three predecessors.
    fn evaluate(&mut self, ref_i: usize, inp_j: usize) {
        if ref_i >= self.reference.len() {
            return;
        }
        let d = self
            .config
            .distance
            .between(&self.reference[ref_i], &self.inputs[inp_j]);
        if ref_i == 0 && inp_j == 0 {
            self.cost.insert((0, 0), d);
            return;
        }
        let mut best_cost = f64::INFINITY;
        for (prev_i, prev_j, weight) in [
            (ref_i.checked_sub(1), Some(inp_j), 1.0),
            (Some(ref_i), inp_j.checked_sub(1), 1.0),
            (ref_i.checked_sub(1), inp_j.checked_sub(1), 2.0),
        ] {
            let prev = prev_i
                .zip(prev_j)
                .and_then(|key| self.cost.get(&key).copied())
                .unwrap_or(f64::INFINITY);
            // `prev < inf` is false for NaN, so NaN predecessors are skipped
            if prev < f64::INFINITY {
                let c = prev + weight * d;
                if c < best_cost {
                    best_cost = c;
                }
            }
        }
        if best_cost < f64::INFINITY {
            self.cost.insert((ref_i, inp_j), best_cost);
        }
    }

    /// `_norm_cost`: cost per unit of path length.
    fn norm_cost(&self, ref_i: usize, inp_j: usize) -> f64 {
        let cost = self
            .cost
            .get(&(ref_i, inp_j))
            .copied()
            .unwrap_or(f64::INFINITY);
        cost / (1 + ref_i + inp_j) as f64
    }

    /// First index of the last `window_size` entries ending at `last` (`max(0, last - w + 1)`).
    fn window_start(&self, last: usize) -> usize {
        (last + 1).saturating_sub(self.config.window_size)
    }

    /// `_argmin_path_cost`: best cell in the last row and the last column, first minimum wins.
    fn argmin_path_cost(&self) -> (usize, usize) {
        let (mut best_cost, mut best_ref, mut best_inp) = (f64::INFINITY, self.j, self.t);
        for k in self.window_start(self.t)..=self.t {
            let nc = self.norm_cost(self.j, k);
            if nc < best_cost {
                (best_cost, best_ref, best_inp) = (nc, self.j, k);
            }
        }
        for k in self.window_start(self.j)..=self.j {
            let nc = self.norm_cost(k, self.t);
            if nc < best_cost {
                (best_cost, best_ref, best_inp) = (nc, k, self.t);
            }
        }
        (best_ref, best_inp)
    }

    fn compute_input_column(&mut self) {
        for k in self.window_start(self.j)..=self.j {
            self.evaluate(k, self.t);
        }
    }

    fn compute_ref_row(&mut self) {
        for k in self.window_start(self.t)..=self.t {
            self.evaluate(self.j, k);
        }
    }

    /// `get_inc`: which axis to advance next.
    fn get_inc(&self) -> Direction {
        let window = self.config.window_size;
        if self.j + 1 >= self.reference.len() {
            return Direction::Input;
        }
        if self.t + 1 < window || self.j + 1 < window {
            return Direction::Both;
        }
        if self.run_count_ref >= self.config.max_run_count {
            return Direction::Input;
        }
        if self.run_count_input >= self.config.max_run_count {
            return Direction::Ref;
        }
        let (best_ref, best_input) = self.argmin_path_cost();
        if best_ref == self.j && best_input == self.t {
            Direction::Both
        } else if best_ref < self.j {
            Direction::Input
        } else {
            Direction::Ref
        }
    }

    fn update_run_counts(&mut self, adv_ref: bool, adv_input: bool) {
        if adv_ref && adv_input {
            self.run_count_ref = 0;
            self.run_count_input = 0;
        } else if adv_ref {
            self.run_count_ref += 1;
            self.run_count_input = 0;
        } else {
            self.run_count_input += 1;
            self.run_count_ref = 0;
        }
    }

    /// `step`: consume the observation, then advance the axes until the next one is needed.
    fn step(&mut self, features: &[f32]) {
        // `_pending` holds at most this one observation and is empty between steps
        let mut pending = Some(features.to_vec());
        if self.started {
            loop {
                let adv_input = self.get_inc() != Direction::Ref;
                if adv_input && pending.is_none() {
                    break;
                }
                if adv_input {
                    self.inputs.extend(pending.take());
                    self.t += 1;
                    self.compute_input_column();
                }
                let adv_ref = self.get_inc() != Direction::Input;
                if adv_ref {
                    self.j += 1;
                    self.compute_ref_row();
                }
                self.update_run_counts(adv_ref, adv_input);
            }
        } else {
            self.inputs.extend(pending.take());
            (self.started, self.t, self.j) = (true, 0, 0);
            self.evaluate(0, 0);
        }
        let (best_ref, _) = self.argmin_path_cost();
        self.current_index = best_ref.min(self.reference.len().saturating_sub(1));
    }
}

impl OnlineAlignment for DixonEvent {
    fn call(&mut self, obs: &Observation) -> f64 {
        let _span = tracing::debug_span!("inference", method = "dixon").entered();
        self.step(&obs.features);
        let position = f64::from(self.score_positions[self.current_index]);
        self.path.push(obs.perf_time, position);
        tracing::debug!(perf_time = obs.perf_time, position);
        position
    }

    fn is_still_following(&self) -> bool {
        self.current_index + 1 < self.score_positions.len()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(key: usize) -> Vec<f32> {
        (0..6).map(|i| f32::from(u8::from(i == key))).collect()
    }

    fn follower(config: DixonEventConfig, onsets: usize) -> DixonEvent {
        let positions = (0..onsets).map(|i| i as f32).collect();
        DixonEvent::new((0..onsets).map(unit).collect(), positions, config)
    }

    fn observation(key: usize, perf_time: f64) -> Observation {
        Observation {
            features: unit(key),
            perf_time,
        }
    }

    #[test]
    fn follows_the_diagonal_and_stops_at_the_last_onset() {
        let mut dixon = follower(DixonEventConfig::default(), 3);
        let positions: Vec<f64> = (0..3)
            .map(|k| dixon.call(&observation(k, k as f64)))
            .collect();
        assert_eq!(positions, [0.0, 1.0, 2.0]);
        assert!(!dixon.is_still_following());
        assert_eq!(dixon.alignment_path().score_beat, [0.0, 1.0, 2.0]);
    }

    #[test]
    fn every_window_and_run_count_terminates_inside_the_score() {
        for window_size in 1..5 {
            for max_run_count in 1..4 {
                let config = DixonEventConfig {
                    window_size,
                    max_run_count,
                    ..DixonEventConfig::default()
                };
                let mut dixon = follower(config, 6);
                for k in 0..30 {
                    let position = dixon.call(&observation(k % 6, k as f64));
                    assert!(
                        (0.0..=5.0).contains(&position),
                        "w={window_size} r={max_run_count}"
                    );
                }
            }
        }
    }

    #[test]
    fn an_all_zero_euclidean_observation_does_not_panic() {
        let mut dixon = follower(DixonEventConfig::default(), 3);
        let zero = Observation {
            features: vec![0.0; 6],
            perf_time: 0.0,
        };
        assert!(dixon.call(&zero).is_finite());
    }
}
