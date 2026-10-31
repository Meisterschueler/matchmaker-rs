//! Port of `matchmaker.dp.oltw_arzt.OnlineTimeWarpingArztEvent` (method key `arzt`, MIDI).
//!
//! Upstream: `matchmaker/dp/oltw_arzt.py`.
//!
//! Event-level online time warping: every observation (a chord) is aligned against the
//! score's onset pianoroll in a window of `window_size` onsets that starts `step_size` onsets
//! before the current position; the position advances by at most `step_size` per observation.
//!
//! ```
//! use matchmaker_rs::base::OnlineAlignment;
//! use matchmaker_rs::dp::oltw_arzt::{ArztEvent, ArztEventConfig};
//! use matchmaker_rs::score::Score;
//! let score = Score::load("tests/data/simple_mozart_k265_var1.musicxml", true)?;
//! let (features, onsets) = score.onset_pianoroll(true);
//! let follower = ArztEvent::new(features, onsets, ArztEventConfig::default());
//! assert_eq!(follower.score_positions().len(), score.unique_onsets().len());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use crate::base::{AlignmentPath, OnlineAlignment, StepDebug};
use crate::dp::Distance;
use crate::features::Observation;

/// Constructor arguments of `OnlineTimeWarpingArztEvent` with the values that actually reach it
/// for MIDI (research R1: the YAML's `window_size: 2` never arrives, the class default does).
#[derive(Debug, Clone, PartialEq)]
pub struct ArztEventConfig {
    /// Search window in onsets (default 30).
    pub window_size: usize,
    /// Window while fewer than this many observations were seen (default 5).
    pub start_window_size: usize,
    /// Largest advance of the position per observation (default 5).
    pub step_size: usize,
    /// Local cost (`distance_func`, default cosine).
    pub distance: Distance,
}

impl Default for ArztEventConfig {
    fn default() -> Self {
        ArztEventConfig {
            window_size: 30,
            start_window_size: 5,
            step_size: 5,
            distance: Distance::Cosine,
        }
    }
}

/// The `arzt` follower for MIDI.
#[derive(Debug, Clone)]
pub struct ArztEvent {
    reference: Vec<Vec<f32>>,
    score_positions: Vec<f32>,
    config: ArztEventConfig,
    /// `global_cost_matrix` of the reference: `float32 (N + 1, 2)`, row 0 stays `inf`.
    cost: Vec<[f32; 2]>,
    current_index: usize,
    input_index: usize,
    path: AlignmentPath,
}

impl ArztEvent {
    /// `OnlineTimeWarpingArztEvent(reference_features, score_positions, ...)`; both come from
    /// [`Score::onset_pianoroll`](crate::score::Score::onset_pianoroll).
    pub fn new(
        reference_features: Vec<Vec<f32>>,
        score_positions: Vec<f32>,
        config: ArztEventConfig,
    ) -> Self {
        ArztEvent {
            cost: vec![[f32::INFINITY; 2]; reference_features.len() + 1],
            reference: reference_features,
            score_positions,
            config,
            current_index: 0,
            input_index: 0,
            path: AlignmentPath::default(),
        }
    }

    /// Index of the current onset.
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// `get_window`: `[start, end)` of the onsets searched for the next observation.
    fn window(&self) -> (usize, usize) {
        let width = if self.input_index < self.config.start_window_size {
            self.config.start_window_size
        } else {
            self.config.window_size
        };
        let start = self.current_index.saturating_sub(self.config.step_size);
        (start, (start + width).min(self.reference.len()))
    }

    /// `step`: one rolling DP column over the window, then the clipped argmin.
    fn step(&mut self, features: &[f32]) {
        let (window_start, window_end) = self.window();
        let mut min_costs = f64::INFINITY;
        let mut min_index = window_start;
        for score_index in window_start..window_end {
            let si = score_index + 1; // 1-indexed in the cost matrix
            let local_dist = self
                .config
                .distance
                .between(&self.reference[score_index], features);
            if score_index == 0 && self.input_index == 0 {
                self.cost[1][1] = local_dist as f32;
                min_costs = local_dist;
                min_index = 0;
                continue;
            }
            // float32 matrix entry + float64 cost = float64; Python `min` keeps the first on NaN
            let d1 = f64::from(self.cost[si - 1][1]) + local_dist;
            let d2 = f64::from(self.cost[si][0]) + local_dist;
            let d3 = f64::from(self.cost[si - 1][0]) + local_dist;
            let mut best = d1;
            if d2 < best {
                best = d2;
            }
            if d3 < best {
                best = d3;
            }
            self.cost[si][1] = best as f32; // stored as float32, `norm_cost` uses the float64 value
            let norm_cost = best / ((self.input_index + score_index) as f64 + 1.0);
            if norm_cost < min_costs {
                min_costs = norm_cost;
                min_index = score_index;
            }
        }
        for cell in &mut self.cost {
            *cell = [cell[1], f32::INFINITY];
        }
        self.current_index = if self.input_index > 0 {
            min_index.clamp(
                self.current_index,
                self.current_index + self.config.step_size,
            )
        } else {
            min_index
        };
        self.input_index += 1;
    }
}

impl OnlineAlignment for ArztEvent {
    fn call(&mut self, obs: &Observation) -> f64 {
        let _span = tracing::debug_span!("inference", method = "arzt").entered();
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
        (0..4).map(|i| f32::from(u8::from(i == key))).collect()
    }

    fn follower(config: ArztEventConfig) -> ArztEvent {
        ArztEvent::new(vec![unit(0), unit(1), unit(2)], vec![0.0, 1.0, 2.0], config)
    }

    fn observation(key: usize, perf_time: f64) -> Observation {
        Observation {
            features: unit(key),
            perf_time,
        }
    }

    #[test]
    fn follows_the_diagonal_and_stops_at_the_last_onset() {
        let mut arzt = follower(ArztEventConfig::default());
        let positions: Vec<f64> = (0..3)
            .map(|k| arzt.call(&observation(k, k as f64)))
            .collect();
        assert_eq!(positions, [0.0, 1.0, 2.0]);
        assert!(!arzt.is_still_following());
        assert_eq!(arzt.alignment_path().perf_time, [0.0, 1.0, 2.0]);
    }

    #[test]
    fn position_advances_by_at_most_step_size() {
        let mut arzt = follower(ArztEventConfig {
            step_size: 1,
            ..ArztEventConfig::default()
        });
        arzt.call(&observation(0, 0.0));
        // the best match is onset 2, but the clip allows one onset per observation
        assert_eq!(arzt.call(&observation(2, 1.0)), 1.0);
    }

    #[test]
    fn an_all_zero_observation_is_nan_cost_and_keeps_the_start() {
        let mut arzt = follower(ArztEventConfig::default());
        assert_eq!(
            arzt.call(&Observation {
                features: vec![0.0; 4],
                perf_time: 0.0
            }),
            0.0
        );
    }
}
