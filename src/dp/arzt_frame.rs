//! Frame-level `arzt` online time warping for audio input.
//!
//! Upstream: `matchmaker.dp.oltw_arzt.OnlineTimeWarpingArztFrame` and
//! `matchmaker.dp.dtw_loop.oltw_arzt_loop`.

use crate::base::{AlignmentPath, OnlineAlignment, StepDebug};
use crate::error::{Error, Result};
use crate::features::Observation;
use crate::matchmaker::{AudioDistance, AudioMethod, AudioMethodConfig};

#[derive(Debug, Clone)]
/// Frame-level audio Arzt follower with rolling OLTW costs.
pub struct ArztFrame {
    reference: Vec<Vec<f32>>,
    score_positions: Vec<f32>,
    frame_to_beat: Vec<f64>,
    window_frames: usize,
    start_window_frames: usize,
    step_size: usize,
    distance: AudioDistance,
    cost: Vec<[f32; 2]>,
    current_frame: usize,
    current_index: usize,
    input_index: usize,
    path: AlignmentPath,
}

impl ArztFrame {
    /// Create a frame-level follower from score features, onset beats and one beat per frame.
    pub fn new(
        reference_features: Vec<Vec<f32>>,
        score_positions: Vec<f32>,
        frame_to_beat: Vec<f64>,
        config: AudioMethodConfig,
    ) -> Result<Self> {
        config.validate(AudioMethod::Arzt)?;
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
        let window_frames = (config.window_size * frame_rate).round() as usize;
        let start_window_frames =
            (config.start_window_size.unwrap_or(0.1) * frame_rate).round() as usize;

        Ok(Self {
            cost: vec![[f32::INFINITY; 2]; reference_features.len() + 1],
            reference: reference_features,
            score_positions,
            frame_to_beat,
            window_frames,
            start_window_frames,
            step_size: config.step_size.unwrap_or(3),
            distance: config.distance_func,
            current_frame: 0,
            current_index: 0,
            input_index: 0,
            path: AlignmentPath::default(),
        })
    }

    /// Current reference-frame index.
    pub fn current_frame(&self) -> usize {
        self.current_frame
    }

    fn window(&self) -> (usize, usize) {
        let width = if self.current_frame < self.start_window_frames {
            self.start_window_frames
        } else {
            self.window_frames
        };
        (
            self.current_frame.saturating_sub(width),
            self.current_frame
                .saturating_add(width)
                .min(self.reference.len()),
        )
    }

    fn step(&mut self, features: &[f32]) {
        let (window_start, window_end) = self.window();
        let window_cost = self.reference[window_start..window_end]
            .iter()
            .map(|reference| self.distance_between(reference, features))
            .collect::<Vec<_>>();
        let mut min_cost = f32::INFINITY;
        let mut min_index = self.current_frame.saturating_sub(self.step_size);

        for (offset, score_index) in (window_start..window_end).enumerate() {
            let local = window_cost[offset];
            if score_index == 0 && self.input_index == 0 {
                let initial_cost = window_cost.iter().copied().sum();
                self.cost[1][1] = initial_cost;
                min_cost = initial_cost;
                min_index = 0;
                continue;
            }

            let first = self.cost[score_index][1] + local;
            let second = self.cost[score_index + 1][0] + local;
            let diagonal = self.cost[score_index][0] + local;
            let best = first.min(second).min(diagonal);
            self.cost[score_index + 1][1] = best;
            let normalized =
                (f64::from(best) / ((self.input_index + score_index) as f64 + 1.0)) as f32;
            if normalized < min_cost {
                min_cost = normalized;
                min_index = score_index;
            }
        }

        for cell in &mut self.cost {
            *cell = [cell[1], f32::INFINITY];
        }

        if self.input_index == 0 {
            self.current_frame = min_index;
        } else {
            self.current_frame = min_index.clamp(
                self.current_frame,
                self.current_frame.saturating_add(self.step_size),
            );
        }
        let beat = self.frame_to_beat[self.current_frame.min(self.frame_to_beat.len() - 1)];
        self.current_index = self
            .score_positions
            .partition_point(|position| f64::from(*position) <= beat)
            .saturating_sub(1)
            .min(self.score_positions.len() - 1);
        self.input_index += 1;
    }

    fn distance_between(&self, left: &[f32], right: &[f32]) -> f32 {
        match self.distance {
            AudioDistance::Manhattan => left.iter().zip(right).map(|(&a, &b)| (a - b).abs()).sum(),
            AudioDistance::Euclidean => left
                .iter()
                .zip(right)
                .map(|(&a, &b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt(),
        }
    }
}

impl OnlineAlignment for ArztFrame {
    fn call(&mut self, observation: &Observation) -> f64 {
        let _span = tracing::debug_span!("inference", method = "arzt").entered();
        self.step(&observation.features);
        let position = self.frame_to_beat[self.current_frame.min(self.frame_to_beat.len() - 1)];
        self.path.push(observation.perf_time, position);
        tracing::debug!(perf_time = observation.perf_time, position);
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

fn audio_config(key: &str, reason: &str) -> Error {
    Error::AudioConfig {
        key: key.into(),
        reason: reason.into(),
    }
}
