//! Port of `matchmaker.prob.hmm.PitchIOIHMM` (method key `hmm`).
//!
//! Like `pthmm`, but each observation is a pitch list plus the time since the previous
//! observation. The inter-onset-interval term compares it with the score IOI scaled by the beat
//! period of a Kalman tempo model, which is updated whenever the follower advances to a new onset.
//!
//! ```no_run
//! use matchmaker_rs::prob::hmm::{PitchIoiHmm, PitchIoiHmmConfig};
//! use matchmaker_rs::score::Score;
//! let score = Score::load("tests/data/simple_mozart_k265_var1.musicxml", true)?;
//! let follower = PitchIoiHmm::new(&score, PitchIoiHmmConfig::default());
//! assert_eq!(follower.beat_period(), 0.6);
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use super::hidden_markov::{Banded, argmax, forward_algorithm_step};
use super::hmm_models::{
    bernoulli_pitch_probabilities, compute_discrete_pitch_profiles, init_dist,
    ioi_observation_probability, pitch_list_to_vector, score_states, stable_transition_matrix,
};
use crate::base::{AlignmentPath, OnlineAlignment};
use crate::features::Observation;
use crate::score::Score;
use crate::utils::tempo_models::{KalmanParams, KalmanTempoModel};

/// Parameters of [`PitchIoiHmm`]; the defaults are the reference's `hmm` method.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchIoiHmmConfig {
    /// Add a state between consecutive onsets.
    pub has_insertions: bool,
    /// Pitches are indexed in the 88-key piano range.
    pub piano_range: bool,
    /// Precision of the IOI term.
    pub ioi_precision: f64,
    /// Kalman tempo model parameters.
    pub tempo: KalmanParams,
}

impl Default for PitchIoiHmmConfig {
    fn default() -> Self {
        PitchIoiHmmConfig {
            has_insertions: true,
            piano_range: true,
            ioi_precision: 1.0,
            tempo: KalmanParams::default(),
        }
    }
}

/// The `hmm` follower.
#[derive(Debug, Clone)]
pub struct PitchIoiHmm {
    config: PitchIoiHmmConfig,
    score_positions: Vec<f32>,
    pitch_profiles: Vec<Vec<f64>>,
    transition: Banded,
    init: Vec<f64>,
    forward_variable: Option<Vec<f64>>,
    current_index: usize,
    prev_perf_time: Option<f64>,
    tempo: KalmanTempoModel,
    path: AlignmentPath,
}

impl PitchIoiHmm {
    /// Build from a score (`PitchIOIHMM(reference_features, has_insertions, piano_range, ...)`).
    pub fn new(score: &Score, config: PitchIoiHmmConfig) -> Self {
        let onsets = score.unique_onsets();
        let score_positions = score_states(&onsets, config.has_insertions);
        let n = score_positions.len();
        let first_onset = onsets.iter().copied().fold(f32::INFINITY, f32::min);
        PitchIoiHmm {
            pitch_profiles: compute_discrete_pitch_profiles(
                &score.chords(),
                config.piano_range,
                config.has_insertions,
            ),
            transition: stable_transition_matrix(n, 1.0, config.has_insertions),
            init: init_dist(n),
            forward_variable: None,
            current_index: 0,
            prev_perf_time: None,
            tempo: KalmanTempoModel::with_params(config.tempo, first_onset),
            path: AlignmentPath::default(),
            score_positions,
            config,
        }
    }

    /// Follower states in beats.
    pub fn score_positions(&self) -> &[f32] {
        &self.score_positions
    }

    /// Index of the current state.
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// The normalised forward variable after the last step.
    pub fn forward_variable(&self) -> Option<&[f64]> {
        self.forward_variable.as_deref()
    }

    /// Current beat period of the tempo model in seconds per beat.
    pub fn beat_period(&self) -> f64 {
        self.tempo.beat_period()
    }
}

impl OnlineAlignment for PitchIoiHmm {
    fn call(&mut self, obs: &Observation) -> f64 {
        let _span = tracing::debug_span!("inference", method = "hmm").entered();
        let perf_time = obs.perf_time;
        let ioi_obs = self.prev_perf_time.map_or(0.0, |prev| perf_time - prev);
        self.prev_perf_time = Some(perf_time);

        let pitches = pitch_list_to_vector(&obs.features, self.config.piano_range);
        let has_insertions = self.config.has_insertions;
        let (tempo_est, current) = (self.tempo.beat_period(), self.current_index);
        let pitch_probabilities = bernoulli_pitch_probabilities(&pitches, &self.pitch_profiles);
        let observation_prob: Vec<f64> = pitch_probabilities
            .iter()
            .enumerate()
            .map(|(i, &pitch_prob)| {
                pitch_prob
                    * ioi_observation_probability(
                        &self.score_positions,
                        current,
                        i,
                        tempo_est,
                        ioi_obs,
                        self.config.ioi_precision,
                    )
            })
            .collect();
        let fv = forward_algorithm_step(
            &observation_prob,
            &self.transition,
            &self.init,
            self.forward_variable.as_deref(),
        );
        let state = argmax(&fv);
        self.forward_variable = Some(fv);

        // with insertions the tempo only follows onset (even) states; `& 1` keeps MSRV 1.85
        if state > current && (!has_insertions || state & 1 == 0) {
            self.tempo
                .update_beat_period(perf_time, self.score_positions[state]);
        }
        self.current_index = state;

        let position = f64::from(self.score_positions[state]);
        self.path.push(perf_time, position);
        tracing::debug!(perf_time, position);
        position
    }

    fn is_still_following(&self) -> bool {
        self.current_index + 1 < self.score_positions.len()
    }

    fn alignment_path(&self) -> &AlignmentPath {
        &self.path
    }
}
