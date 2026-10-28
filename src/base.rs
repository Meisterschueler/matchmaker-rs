//! Port of `matchmaker.base.OnlineAlignment`: the follower interface and its run loop.
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

/// A score follower (`OnlineAlignment`).
pub trait OnlineAlignment {
    /// One observation in, current score position (beats) out (`__call__`).
    fn call(&mut self, obs: &Observation) -> f64;

    /// Whether `run` keeps consuming observations.
    fn is_still_following(&self) -> bool;

    /// The alignment path so far.
    fn alignment_path(&self) -> &AlignmentPath;

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
