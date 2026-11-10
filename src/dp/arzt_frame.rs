//! Frame-level `arzt` online time warping for audio input.
//!
//! Upstream: `matchmaker.dp.oltw_arzt.OnlineTimeWarpingArztFrame` and
//! `matchmaker.dp.dtw_loop.oltw_arzt_loop`.
//!
//! ```
//! use matchmaker_rs::base::OnlineAlignment;
//! use matchmaker_rs::dp::arzt_frame::ArztFrame;
//! use matchmaker_rs::features::Observation;
//! use matchmaker_rs::matchmaker::{AudioMethod, AudioMethodConfig};
//! let config = AudioMethodConfig::for_method(AudioMethod::Arzt);
//! let mut follower = ArztFrame::new(
//!     vec![vec![1.0, 0.0], vec![0.0, 1.0]],
//!     vec![0.0, 1.0],
//!     vec![0.0, 1.0],
//!     config,
//! )?;
//! let beat = follower.call(&Observation { features: vec![1.0, 0.0], perf_time: 0.0 });
//! assert_eq!(beat, 0.0);
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use crate::base::{AlignmentPath, OnlineAlignment, StepDebug, TempoStepDebug};
use crate::dp::Direction;
use crate::error::{Error, Result};
use crate::features::Observation;
use crate::matchmaker::{AudioDistance, AudioMethod, AudioMethodConfig};
use crate::utils::np_sum::np_sum;
use crate::utils::tempo_models::{ArztTempoModel, ScoreAlteration};
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
struct BackpointerColumn {
    start: usize,
    directions: Vec<Option<Direction>>,
}

#[derive(Debug, Clone)]
struct TempoTracking {
    model: ArztTempoModel,
    onsets: Vec<bool>,
    backpointers: BTreeMap<usize, BackpointerColumn>,
    min_input_needed: usize,
}

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
    tempo: Option<TempoTracking>,
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
            tempo: None,
        })
    }

    /// Current reference-frame index.
    pub fn current_frame(&self) -> usize {
        self.current_frame
    }

    pub(crate) fn current_local_cost(&self, features: &[f32]) -> f32 {
        self.distance_between(
            &self.reference[self.current_frame.min(self.reference.len() - 1)],
            features,
        )
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
        if self.tempo.is_some() && self.input_index > 0 {
            let (relative_tempo, path_end) = self.relative_tempo();
            let tracking = self.tempo.as_mut().unwrap();
            tracking.min_input_needed = if tracking.min_input_needed == 0 {
                path_end
            } else {
                tracking.min_input_needed.min(path_end)
            };
            if let Some(alteration) =
                tracking
                    .model
                    .alteration(relative_tempo, self.current_frame, &tracking.onsets)
            {
                self.alter_score(alteration);
            }
        }
        let (window_start, window_end) = self.window();
        let window_cost = self.reference[window_start..window_end]
            .iter()
            .map(|reference| self.distance_between(reference, features))
            .collect::<Vec<_>>();
        let mut min_cost = f32::INFINITY;
        let mut min_index = self.current_frame.saturating_sub(self.step_size);
        let mut backpointers = self.tempo.as_ref().map(|_| vec![None; window_cost.len()]);

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
            if let Some(directions) = &mut backpointers {
                directions[offset] = Some(if first <= second && first <= diagonal {
                    Direction::Ref
                } else if second <= diagonal {
                    Direction::Input
                } else {
                    Direction::Both
                });
            }
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

        if let Some(directions) = backpointers {
            let tracking = self.tempo.as_mut().unwrap();
            tracking.backpointers.insert(
                self.input_index,
                BackpointerColumn {
                    start: window_start,
                    directions,
                },
            );
            if (self.input_index + 1).is_multiple_of(self.window_frames.max(1)) {
                let keep = (self.input_index as i64 - 2 * self.window_frames as i64)
                    .min(tracking.min_input_needed as i64);
                tracking
                    .backpointers
                    .retain(|&input, _| input as i64 >= keep - 1);
                tracking.min_input_needed = 0;
            }
        }

        if self.input_index == 0 {
            self.current_frame = min_index;
        } else {
            self.current_frame = min_index.clamp(
                self.current_frame
                    .saturating_sub(usize::from(self.tempo.is_some())),
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

    pub(crate) fn enable_tempo(&mut self, frame_rate: f64, seed: u32) -> Result<()> {
        let mut onsets = vec![false; self.reference.len()];
        for &position in &self.score_positions {
            let index = self
                .frame_to_beat
                .partition_point(|&beat| beat < f64::from(position));
            if index < onsets.len() {
                onsets[index] = true;
            }
        }
        self.tempo = Some(TempoTracking {
            model: ArztTempoModel::new(frame_rate, seed)?,
            onsets,
            backpointers: BTreeMap::new(),
            min_input_needed: 0,
        });
        Ok(())
    }

    pub(crate) fn tempo_state(&self) -> [f64; 8] {
        let model = &self.tempo.as_ref().unwrap().model;
        [
            self.current_frame as f64,
            self.current_index as f64,
            model.current_tempo,
            model.beat_period,
            model.n_inserted as f64,
            model.n_deleted as f64,
            model.n_postponed as f64,
            self.reference.len() as f64,
        ]
    }

    pub(crate) fn reference_features(&self) -> &[Vec<f32>] {
        &self.reference
    }

    pub(crate) fn frame_to_beat(&self) -> &[f64] {
        &self.frame_to_beat
    }

    fn backward_path(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        let mut input = self.input_index.saturating_sub(1);
        let mut row = self.current_frame;
        std::iter::from_fn(move || {
            if input == 0 || row == 0 {
                return None;
            }
            let column = self.tempo.as_ref()?.backpointers.get(&input)?;
            let offset = row.checked_sub(column.start)?;
            match column.directions.get(offset).copied().flatten()? {
                Direction::Ref => row -= 1,
                Direction::Input => input -= 1,
                Direction::Both => {
                    input -= 1;
                    row -= 1;
                }
            }
            Some((input, row))
        })
    }

    fn relative_tempo(&self) -> (Option<f64>, usize) {
        let tracking = self.tempo.as_ref().unwrap();
        let half = (tracking.model.window_frames - 1).div_euclid(2) as f64;
        let limit = self.input_index as i64 - 1 - tracking.model.min_past_frames;
        let mut input = self.input_index - 1;
        let mut row = self.current_frame;
        let start_row = row;
        let mut last_row = row;
        let mut path = BTreeMap::from([(row, input)]);
        let mut onsets = Vec::new();
        let mut need_below = None;
        for (next_input, next_row) in self.backward_path() {
            input = next_input;
            row = next_row;
            path.insert(row, input);
            if row != last_row {
                if tracking.onsets[last_row] && path[&last_row] as i64 <= limit {
                    onsets.push(last_row);
                }
                last_row = row;
                if need_below.is_none() && onsets.len() >= 20 {
                    need_below = Some(onsets[19] as f64 - half);
                }
                if need_below.is_some_and(|below| (row as f64) < below) && tracking.onsets[row] {
                    break;
                }
            }
        }
        if tracking.onsets[last_row]
            && path[&last_row] as i64 <= limit
            && !onsets.contains(&last_row)
        {
            onsets.push(last_row);
        }
        if onsets.is_empty() {
            return (None, input);
        }
        let anchors = path
            .iter()
            .filter(|&(&anchor, _)| anchor == row || anchor == start_row || tracking.onsets[anchor])
            .map(|(&anchor, &at)| (anchor as f64, at as f64))
            .collect::<Vec<_>>();
        let interpolate = |at: f64| {
            let first = anchors[0];
            let last = anchors[anchors.len() - 1];
            if at <= first.0 {
                return first.1 - (first.0 - at);
            }
            if at >= last.0 {
                return last.1 + (at - last.0);
            }
            let upper = anchors.partition_point(|&(anchor, _)| anchor <= at);
            let (left, right) = (anchors[upper - 1], anchors[upper]);
            left.1 + (right.1 - left.1) / (right.0 - left.0) * (at - left.0)
        };
        let selected = &onsets[..onsets.len().min(20)];
        let weighted = selected
            .iter()
            .rev()
            .enumerate()
            .map(|(index, &onset)| {
                let lower = onset as f64 - half;
                let upper = onset as f64 + half;
                (upper - lower + 1.0) / (interpolate(upper) - interpolate(lower) + 1.0)
                    * (index + 1) as f64
            })
            .collect::<Vec<_>>();
        let weight_sum = (selected.len() * (selected.len() + 1) / 2) as f64;
        (Some(np_sum(&weighted) / weight_sum), input)
    }

    fn alter_score(&mut self, alteration: ScoreAlteration) {
        let score_frame = self.current_frame;
        let tracking = self.tempo.as_mut().unwrap();
        match alteration {
            ScoreAlteration::Delete => {
                let index = score_frame + 2;
                let previous = index - 1;
                for feature in 0..self.reference[previous].len() {
                    self.reference[previous][feature] =
                        0.5 * (self.reference[previous][feature] + self.reference[index][feature]);
                }
                if tracking.onsets[index] && !tracking.onsets[previous] {
                    self.frame_to_beat[previous] = self.frame_to_beat[index];
                } else if !tracking.onsets[previous] {
                    self.frame_to_beat[previous] =
                        0.5 * (self.frame_to_beat[previous] + self.frame_to_beat[index]);
                }
                tracking.onsets[previous] |= tracking.onsets[index];
                self.reference.remove(index);
                self.frame_to_beat.remove(index);
                tracking.onsets.remove(index);
                self.cost.remove(index + 1);
                for column in tracking.backpointers.values_mut() {
                    if column.start > index {
                        column.start -= 1;
                    } else {
                        let offset = index - column.start;
                        if offset < column.directions.len() {
                            column.directions.remove(offset);
                        }
                    }
                }
            }
            ScoreAlteration::Insert => {
                let index = score_frame + 1;
                let features = self.reference[score_frame]
                    .iter()
                    .zip(&self.reference[index])
                    .map(|(&left, &right)| 0.5 * (left + right))
                    .collect();
                let beat = 0.5 * (self.frame_to_beat[score_frame] + self.frame_to_beat[index]);
                self.reference.insert(index, features);
                self.frame_to_beat.insert(index, beat);
                tracking.onsets.insert(index, false);
                let cost = [
                    0.5 * (self.cost[index][0] + self.cost[index + 1][0]),
                    0.5 * (self.cost[index][1] + self.cost[index + 1][1]),
                ];
                self.cost.insert(index + 1, cost);
                for column in tracking.backpointers.values_mut() {
                    if column.start >= index {
                        column.start += 1;
                    } else {
                        let offset = index - column.start;
                        if offset < column.directions.len() {
                            column.directions.insert(offset, Some(Direction::Ref));
                        }
                    }
                }
            }
        }
    }

    fn current_position(&self) -> f64 {
        let mut frame = self.current_frame;
        if let Some(tracking) = &self.tempo
            && self.input_index > 5
            && let Some((input, row)) = self.backward_path().take(2).last()
        {
            let advance = self.input_index - 1 - input;
            frame = (row as f64 + advance as f64 * tracking.model.current_tempo)
                .clamp(0.0, (self.frame_to_beat.len() - 1) as f64)
                .round_ties_even() as usize;
        }
        self.frame_to_beat[frame.min(self.frame_to_beat.len() - 1)]
    }
}

impl OnlineAlignment for ArztFrame {
    fn call(&mut self, observation: &Observation) -> f64 {
        let method = if self.tempo.is_some() {
            "arzt_tempo"
        } else {
            "arzt"
        };
        let _span = tracing::debug_span!("inference", method).entered();
        self.step(&observation.features);
        let position = self.current_position();
        self.path.push(observation.perf_time, position);
        tracing::debug!(perf_time = observation.perf_time, position);
        position
    }

    fn is_still_following(&self) -> bool {
        self.current_index + 1 < self.score_positions.len()
            && self.input_index < self.reference.len()
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
            beat_period: self
                .tempo
                .as_ref()
                .map(|tracking| tracking.model.beat_period),
            tempo: self.tempo.as_ref().map(|tracking| {
                let column = tracking
                    .backpointers
                    .get(&self.input_index.saturating_sub(1));
                let start = column.map_or(0, |column| column.start);
                let length = column.map_or(0, |column| column.directions.len());
                TempoStepDebug {
                    state: self.tempo_state(),
                    window_start: start,
                    costs: self.cost[start + 1..start + length + 1]
                        .iter()
                        .map(|cell| cell[0])
                        .collect(),
                }
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
