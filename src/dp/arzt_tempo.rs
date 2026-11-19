//! Port of tempo-aware `OnlineTimeWarpingArztFrame` with `ArztTempoModel`.
//!
//! Upstream: `matchmaker/dp/oltw_arzt.py`, `matchmaker/utils/tempo_models.py`.
//!
//! ```
//! use matchmaker_rs::dp::arzt_tempo::ArztTempoFrame;
//! use matchmaker_rs::matchmaker::{AudioMethod, AudioMethodConfig};
//! let follower = ArztTempoFrame::new(
//!     vec![vec![1.0], vec![0.5]], vec![0.0, 1.0], vec![0.0, 1.0],
//!     AudioMethodConfig::for_method(AudioMethod::ArztTempo),
//! )?;
//! assert_eq!(follower.tempo_state()[2], 1.0);
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use crate::Result;
use crate::base::{AlignmentPath, OnlineAlignment, StepDebug};
use crate::dp::arzt_frame::ArztFrame;
use crate::features::Observation;
use crate::matchmaker::AudioMethodConfig;

/// Seeded tempo-aware Arzt follower; existing plain Arzt remains unchanged.
#[derive(Debug, Clone)]
pub struct ArztTempoFrame {
    follower: ArztFrame,
    initial: ArztFrame,
}

impl ArztTempoFrame {
    /// Construct with the reference tempo-model seed 1984.
    pub fn new(
        reference: Vec<Vec<f32>>,
        score_positions: Vec<f32>,
        frame_to_beat: Vec<f64>,
        config: AudioMethodConfig,
    ) -> Result<Self> {
        let seed = config.random_seed.unwrap_or(1984);
        Self::new_with_seed(reference, score_positions, frame_to_beat, config, seed)
    }

    /// Construct with an explicit NumPy-compatible MT19937 seed.
    pub fn new_with_seed(
        reference: Vec<Vec<f32>>,
        score_positions: Vec<f32>,
        frame_to_beat: Vec<f64>,
        mut config: AudioMethodConfig,
        seed: u32,
    ) -> Result<Self> {
        let sample_rate = config.sample_rate;
        let hop = config
            .hop_length
            .unwrap_or((sample_rate / config.frame_rate.max(1)) as usize);
        config.random_seed = None;
        let mut follower = ArztFrame::new(reference, score_positions, frame_to_beat, config)?;
        follower.enable_tempo(f64::from(sample_rate) / hop as f64, seed)?;
        Ok(Self {
            initial: follower.clone(),
            follower,
        })
    }

    /// Reset the score representation, path, tempo history, and seed state.
    pub fn reset(&mut self) {
        self.follower = self.initial.clone();
    }

    /// Current frame, onset index, tempo, beat period, inserted/deleted/postponed counts, length.
    pub fn tempo_state(&self) -> [f64; 8] {
        self.follower.tempo_state()
    }

    /// Current score feature frames, including seed-dependent insertions and deletions.
    pub fn reference_features(&self) -> &[Vec<f32>] {
        self.follower.reference_features()
    }

    /// Current per-frame beat mapping, including tempo alterations.
    pub fn frame_to_beat(&self) -> &[f64] {
        self.follower.frame_to_beat()
    }
}

impl OnlineAlignment for ArztTempoFrame {
    fn call(&mut self, observation: &Observation) -> f64 {
        self.follower.call(observation)
    }

    fn is_still_following(&self) -> bool {
        self.follower.is_still_following()
    }

    fn alignment_path(&self) -> &AlignmentPath {
        self.follower.alignment_path()
    }

    fn score_positions(&self) -> &[f32] {
        self.follower.score_positions()
    }

    fn debug_state(&self) -> StepDebug {
        self.follower.debug_state()
    }
}
