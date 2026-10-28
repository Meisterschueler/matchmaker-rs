//! Port of `matchmaker.prob.hmm.PitchHMM` (method key `pthmm`, the MIDI default).
//!
//! States are the score onsets plus, with insertions, one state between consecutive onsets.
//! Each observation (binary pitch vector) is scored with a Bernoulli model against pitch
//! profiles; the forward algorithm with a Gumbel-shaped transition matrix picks the state.
//!
//! ```
//! use matchmaker_rs::base::OnlineAlignment;
//! use matchmaker_rs::prob::pthmm::PitchHmm;
//! use matchmaker_rs::score::Score;
//! let score = Score::load("tests/data/simple_mozart_k265_var1.musicxml", true)?;
//! let follower = PitchHmm::new(&score, true, true);
//! assert_eq!(follower.score_positions().len(), 2 * score.unique_onsets().len() - 1);
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use super::hidden_markov::{Banded, argmax, forward_algorithm_step};
use super::hmm_models::{
    bernoulli_pitch_probabilities, compute_discrete_pitch_profiles, init_dist, score_states,
    stable_transition_matrix,
};
use crate::base::{AlignmentPath, OnlineAlignment, StepDebug};
use crate::features::Observation;
use crate::score::Score;

/// The `pthmm` follower.
#[derive(Debug, Clone)]
pub struct PitchHmm {
    score_positions: Vec<f32>,
    pitch_profiles: Vec<Vec<f64>>,
    transition: Banded,
    init: Vec<f64>,
    forward_variable: Option<Vec<f64>>,
    current_index: usize,
    path: AlignmentPath,
}

impl PitchHmm {
    /// Build from a score (`PitchHMM(reference_features, has_insertions, piano_range)`;
    /// the reference defaults are `true`, `true`).
    pub fn new(score: &Score, has_insertions: bool, piano_range: bool) -> Self {
        let score_positions = score_states(&score.unique_onsets(), has_insertions);
        let n = score_positions.len();
        PitchHmm {
            pitch_profiles: compute_discrete_pitch_profiles(
                &score.chords(),
                piano_range,
                has_insertions,
            ),
            transition: stable_transition_matrix(n, 1.0, has_insertions),
            init: init_dist(n),
            forward_variable: None,
            current_index: 0,
            path: AlignmentPath::default(),
            score_positions,
        }
    }

    /// Index of the current state.
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// The normalised forward variable after the last step.
    pub fn forward_variable(&self) -> Option<&[f64]> {
        self.forward_variable.as_deref()
    }
}

impl OnlineAlignment for PitchHmm {
    fn call(&mut self, obs: &Observation) -> f64 {
        let _span = tracing::debug_span!("inference", method = "pthmm").entered();
        let observation_prob = bernoulli_pitch_probabilities(&obs.features, &self.pitch_profiles);
        let fv = forward_algorithm_step(
            &observation_prob,
            &self.transition,
            &self.init,
            self.forward_variable.as_deref(),
        );
        self.current_index = argmax(&fv);
        self.forward_variable = Some(fv);
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
            probabilities: self.forward_variable.clone().unwrap_or_default(),
            ..StepDebug::default()
        }
    }
}
