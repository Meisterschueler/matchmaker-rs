//! Port of `matchmaker.base.OnlineAlignment`: the follower interface and its run loop.
//!
//! Upstream: `matchmaker/base.py`.
//!
//! ```
//! use matchmaker_rs::base::AlignmentPath;
//! let mut path = AlignmentPath::default();
//! path.push(0.5, 1.0);
//! assert_eq!((path.perf_time.len(), path.score_beat[0]), (1, 1.0));
//! ```

use crate::features::Observation;

/// The `(2, T)` alignment path of the reference: performance seconds and score beats.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AlignmentPath {
    /// Row 0: performance time in seconds.
    pub perf_time: Vec<f64>,
    /// Row 1: score position in beats.
    pub score_beat: Vec<f64>,
}

impl AlignmentPath {
    /// Append one `(perf_time, score_beat)` pair.
    pub fn push(&mut self, perf_time: f64, score_beat: f64) {
        self.perf_time.push(perf_time);
        self.score_beat.push(score_beat);
    }
}

/// Inspectable follower state after a call, for the export of a run (FR-018).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct StepDebug {
    /// Index of the current state.
    pub current_index: usize,
    /// Forward variable / state distribution.
    pub probabilities: Vec<f64>,
    /// `hmm`: beat period of the tempo model in seconds.
    pub beat_period: Option<f64>,
    /// `outerhmm`: whether the call folded a note into the previous chord.
    pub is_continuation: Option<bool>,
}

/// A score follower (`OnlineAlignment`).
pub trait OnlineAlignment {
    /// One observation in, current score position (beats) out (`__call__`).
    fn call(&mut self, obs: &Observation) -> f64;

    /// Whether `run` keeps consuming observations.
    fn is_still_following(&self) -> bool;

    /// The alignment path so far.
    fn alignment_path(&self) -> &AlignmentPath;

    /// The follower's states in beats.
    fn score_positions(&self) -> &[f32];

    /// The state after the last call.
    fn debug_state(&self) -> StepDebug;

    /// Port of `run()`: feed observations while following, return the yielded positions.
    fn run(&mut self, observations: &mut dyn Iterator<Item = Observation>) -> Vec<f64> {
        let mut positions = Vec::new();
        while self.is_still_following() {
            let Some(obs) = observations.next() else {
                break;
            };
            positions.push(self.call(&obs));
        }
        positions
    }
}
