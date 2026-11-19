//! Seeded Korzeniowski audio particle filter.
//!
//! Upstream: `matchmaker.prob.particle_filter_korzeniowski`.

use rand_mt::Mt;

use crate::base::AlignmentPath;
use crate::error::{Error, Result};
use crate::features::KorzeniowskiObservation;
use crate::score::korzeniowski::KorzeniowskiScoreModel;

/// Filter settings. The raw reference constructor uses 1000 particles; the facade overrides this
/// with 500.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PfKorzConfig {
    /// Notated score tempo in quarter-note beats per minute.
    pub notated_tempo: f64,
    /// Duration between observations in seconds.
    pub hop_size: f64,
    /// Number of particles.
    pub num_particles: usize,
    /// Instance-local deterministic seed.
    pub seed: u32,
}

impl Default for PfKorzConfig {
    fn default() -> Self {
        Self {
            notated_tempo: 120.0,
            hop_size: 1.0 / 30.0,
            num_particles: 1_000,
            seed: 1984,
        }
    }
}

/// Inspectable state captured after one PF observation.
#[derive(Debug, Clone, PartialEq)]
pub struct PfKorzStepDebug {
    /// MAP estimate returned for the frame.
    pub current_position: f64,
    /// Particle score positions.
    pub positions: Vec<f64>,
    /// Normalized particle weights.
    pub weights: Vec<f64>,
    /// Sampled particle log-tempi (base 2 BPM).
    pub note_log_tempi: Vec<f64>,
    /// Smoothed particle log-tempi (base 2 BPM).
    pub local_log_tempi: Vec<f64>,
    /// Rao-Blackwellized sounding probabilities.
    pub sound_probabilities: Vec<f64>,
    /// Rao-Blackwellized resting probabilities.
    pub rest_probabilities: Vec<f64>,
    /// ESS measured before any resampling on this call.
    pub effective_sample_size: f64,
    /// Whether this call resampled the particles.
    pub resampled: bool,
}

/// Rao-Blackwellized particle filter for structured Korzeniowski audio observations.
pub struct PfKorz {
    score_model: KorzeniowskiScoreModel,
    config: PfKorzConfig,
    rng: Mt,
    normal_spare: Option<f64>,
    positions: Vec<f64>,
    note_log_tempi: Vec<f64>,
    local_log_tempi: Vec<f64>,
    weights: Vec<f64>,
    sound_probabilities: Vec<f64>,
    rest_probabilities: Vec<f64>,
    last_onset_times: Vec<f64>,
    current_position: f64,
    previous_position: f64,
    previous_time: f64,
    first_input_found: bool,
    path: AlignmentPath,
    last_effective_sample_size: f64,
    last_resampled: bool,
    small_movement: f64,
    large_movement: f64,
    notated_movement: f64,
}

impl PfKorz {
    /// Create a filter with a validated score model and explicit deterministic settings.
    pub fn new(score_model: KorzeniowskiScoreModel, config: PfKorzConfig) -> Result<Self> {
        validate_model(&score_model)?;
        if !config.notated_tempo.is_finite() || config.notated_tempo <= 0.0 {
            return Err(invalid("tempo", "must be a positive number of BPM"));
        }
        if !config.hop_size.is_finite() || config.hop_size <= 0.0 {
            return Err(invalid("hop_size", "must be a positive number of seconds"));
        }
        if config.num_particles == 0 {
            return Err(invalid("num_particles", "must be at least one"));
        }

        let mut rng = Mt::new(config.seed);
        let mut normal_spare = None;
        let initial_log_tempo = config.notated_tempo.log2();
        let note_log_tempi: Vec<f64> = (0..config.num_particles)
            .map(|_| initial_log_tempo + normal(&mut rng, &mut normal_spare, 0.0, 0.1))
            .collect();
        let first_position = f64::from(score_model.onset_positions[0]);
        let uniform_weight = 1.0 / config.num_particles as f64;
        let last_effective_sample_size = config.num_particles as f64;
        let small_movement = (config.notated_tempo / 3.0) * config.hop_size / 60.0;
        let large_movement = (config.notated_tempo * 3.0) * config.hop_size / 60.0;
        let notated_movement = config.notated_tempo * config.hop_size / 60.0;

        Ok(Self {
            score_model,
            config,
            rng,
            normal_spare,
            positions: vec![first_position; config.num_particles],
            local_log_tempi: note_log_tempi.clone(),
            note_log_tempi,
            weights: vec![uniform_weight; config.num_particles],
            sound_probabilities: vec![1.0; config.num_particles],
            rest_probabilities: vec![0.0; config.num_particles],
            last_onset_times: vec![0.0; config.num_particles],
            current_position: first_position,
            previous_position: first_position,
            previous_time: 0.0,
            first_input_found: false,
            path: AlignmentPath::default(),
            last_effective_sample_size,
            last_resampled: false,
            small_movement,
            large_movement,
            notated_movement,
        })
    }

    /// Process one structured audio observation and return the current score beat.
    pub fn call(&mut self, observation: &KorzeniowskiObservation) -> Result<f64> {
        self.validate_observation(observation)?;
        self.last_resampled = false;
        if !self.first_input_found && observation.loudness < -50.0 {
            self.path.push(observation.perf_time, self.current_position);
        } else {
            self.first_input_found = true;
            self.predict(observation.perf_time);
            self.propagate_sound_state();
            self.compute_weights(observation)?;
            let estimate = self.estimate_position();
            let effective_sample_size = effective_sample_size(&self.weights);
            self.last_effective_sample_size = effective_sample_size;
            if effective_sample_size < self.config.num_particles as f64 / 2.0 {
                self.systematic_resample();
                self.last_resampled = true;
            }
            self.previous_position = self.current_position;
            self.current_position = estimate;
        }
        self.path.push(observation.perf_time, self.current_position);
        self.previous_time = observation.perf_time;
        Ok(self.current_position)
    }

    /// Whether the follower should keep consuming observations.
    pub fn is_still_following(&self) -> bool {
        true
    }

    /// Alignment path of performance seconds and score beats.
    pub fn alignment_path(&self) -> &AlignmentPath {
        &self.path
    }

    /// Score-note onset positions.
    pub fn score_positions(&self) -> &[f32] {
        &self.score_model.onset_positions
    }

    /// Current particle and resampling state.
    pub fn debug_state(&self) -> PfKorzStepDebug {
        PfKorzStepDebug {
            current_position: self.current_position,
            positions: self.positions.clone(),
            weights: self.weights.clone(),
            note_log_tempi: self.note_log_tempi.clone(),
            local_log_tempi: self.local_log_tempi.clone(),
            sound_probabilities: self.sound_probabilities.clone(),
            rest_probabilities: self.rest_probabilities.clone(),
            effective_sample_size: self.last_effective_sample_size,
            resampled: self.last_resampled,
        }
    }

    /// The immutable analytical score model used by this filter.
    pub fn score_model(&self) -> &KorzeniowskiScoreModel {
        &self.score_model
    }

    /// Explicit seed and particle-count settings.
    pub fn config(&self) -> PfKorzConfig {
        self.config
    }

    fn validate_observation(&self, observation: &KorzeniowskiObservation) -> Result<()> {
        if observation.spectrum.len() != self.score_model.templates[0].len() {
            return Err(invalid(
                "observation",
                "spectrum width does not match the score templates",
            ));
        }
        if observation.active_notes.is_some() || observation.onset_notes.is_some() {
            return Err(invalid(
                "observation",
                "PF Korzeniowski audio requires audio observations, not MIDI note fields",
            ));
        }
        if !observation.perf_time.is_finite()
            || !observation.onset.is_finite()
            || !observation.loudness.is_finite()
            || observation.spectrum.iter().any(|value| !value.is_finite())
        {
            return Err(invalid("observation", "values must be finite"));
        }
        Ok(())
    }

    fn predict(&mut self, perf_time: f64) {
        let elapsed = perf_time - self.previous_time;
        let previous_positions = self.positions.clone();
        for (index, position) in self.positions.iter_mut().enumerate() {
            let bpm = 2.0f64.powf(self.note_log_tempi[index]);
            let movement = bpm * elapsed / 60.0;
            let movement = if movement > self.large_movement {
                self.notated_movement
            } else if movement < self.small_movement {
                self.small_movement
            } else {
                movement
            };
            *position = (*position + movement).clamp(
                f64::from(self.score_model.onset_positions[0]),
                f64::from(*self.score_model.onset_positions.last().unwrap()),
            );
        }
        self.update_tempi(&previous_positions, perf_time);
    }

    fn update_tempi(&mut self, previous_positions: &[f64], perf_time: f64) {
        let n_onsets = self.score_model.onset_positions.len();
        let mut crossed = Vec::new();
        for (particle, (&previous, &current)) in
            previous_positions.iter().zip(&self.positions).enumerate()
        {
            let previous_index = upper_bound(&self.score_model.onset_positions, previous);
            let current_index = upper_bound(&self.score_model.onset_positions, current);
            if current_index > previous_index {
                crossed.push((particle, current_index - 1, current_index - previous_index));
            }
        }
        if crossed.is_empty() {
            return;
        }

        let mut phases = vec![0.0; crossed.len()];
        for (crossed_index, &(particle, onset_index, delta)) in crossed.iter().enumerate() {
            if onset_index > 0 && onset_index < n_onsets {
                let delta_index = onset_index.saturating_sub(delta).min(n_onsets - 1);
                let delta_beats = f64::from(
                    self.score_model.onset_positions[onset_index]
                        - self.score_model.onset_positions[delta_index],
                );
                let beats_per_second = 2.0f64.powf(self.note_log_tempi[particle]) / 60.0;
                let expected_interval = delta_beats / beats_per_second;
                let actual_interval = perf_time - self.last_onset_times[particle];
                phases[crossed_index] =
                    (actual_interval - expected_interval) / (expected_interval * 60.0);
                self.last_onset_times[particle] = perf_time;
            }
        }
        let high_tempo_noise = normal(&mut self.rng, &mut self.normal_spare, -5.0, 5.0);
        let low_tempo_noise = normal(&mut self.rng, &mut self.normal_spare, 5.0, 5.0);

        let fast_choices: Vec<bool> = (0..crossed.len())
            .map(|_| uniform(&mut self.rng) < 0.2)
            .collect();
        let noise: Vec<f64> = (0..crossed.len())
            .map(|_| normal(&mut self.rng, &mut self.normal_spare, 0.0, 1.0))
            .collect();
        update_tempo_particles(
            &mut self.note_log_tempi,
            &mut self.local_log_tempi,
            &crossed,
            &phases,
            self.config.notated_tempo,
            (high_tempo_noise, low_tempo_noise),
            (&fast_choices, &noise),
        );
    }

    fn propagate_sound_state(&mut self) {
        for particle in 0..self.config.num_particles {
            let rest =
                self.score_model.rest_mask[self.score_model.beat_index(self.positions[particle])];
            if rest {
                let sounding = 0.8 * self.sound_probabilities[particle];
                let resting =
                    0.2 * self.sound_probabilities[particle] + self.rest_probabilities[particle];
                let total = sounding + resting;
                self.sound_probabilities[particle] = sounding / total;
                self.rest_probabilities[particle] = resting / total;
            } else {
                self.sound_probabilities[particle] = 1.0;
                self.rest_probabilities[particle] = 0.0;
            }
        }
    }

    fn compute_weights(&mut self, observation: &KorzeniowskiObservation) -> Result<()> {
        let mut likelihoods = Vec::with_capacity(self.config.num_particles);
        for particle in 0..self.config.num_particles {
            let beat = self.positions[particle];
            let index = self.score_model.beat_index(beat);
            let feature = if self.score_model.rest_mask[index] {
                1.0
            } else {
                observation
                    .spectrum
                    .iter()
                    .zip(&self.score_model.templates[index])
                    .map(|(left, right)| left * f64::from(*right))
                    .sum::<f64>()
                    .max(0.0)
            };
            let nearest_onset = f64::from(self.score_model.nearest_onsets[index]);
            let onset_probability =
                (-(beat - nearest_onset).powi(2) / (2.0 * 0.25f64.powi(2))).exp();
            let onset_observation = onset_observation_probability(observation.onset);
            let onset = onset_observation * onset_probability
                + (1.0 - onset_observation) * (1.0 - onset_probability);
            let sounding = gaussian(observation.loudness, -30.0, 8.0);
            let resting = gaussian(observation.loudness, -70.0, 8.0);
            let loudness = self.sound_probabilities[particle] * sounding
                + self.rest_probabilities[particle] * resting;
            let log_likelihood = 0.7 * feature.max(1e-300).ln()
                + 0.2 * onset.max(1e-300).ln()
                + 0.1 * loudness.max(1e-300).ln();
            likelihoods.push(log_likelihood.exp());
        }
        normalize_likelihood_weights(&mut self.weights, &likelihoods)?;
        Ok(())
    }

    fn estimate_position(&self) -> f64 {
        let count = (0.2 * self.config.num_particles as f64) as usize;
        let count = count.max(1);
        let mut indices: Vec<usize> = (0..self.config.num_particles).collect();
        indices.sort_by(|left, right| {
            self.weights[*left]
                .total_cmp(&self.weights[*right])
                .then_with(|| left.cmp(right))
        });
        indices[self.config.num_particles - count..]
            .iter()
            .map(|&index| self.positions[index])
            .sum::<f64>()
            / count as f64
    }

    fn systematic_resample(&mut self) {
        let indices = systematic_resample_indices(&self.weights, uniform(&mut self.rng));
        self.positions = indices.iter().map(|&index| self.positions[index]).collect();
        self.note_log_tempi = indices
            .iter()
            .map(|&index| self.note_log_tempi[index])
            .collect();
        self.local_log_tempi = indices
            .iter()
            .map(|&index| self.local_log_tempi[index])
            .collect();
        self.sound_probabilities = indices
            .iter()
            .map(|&index| self.sound_probabilities[index])
            .collect();
        self.rest_probabilities = indices
            .iter()
            .map(|&index| self.rest_probabilities[index])
            .collect();
        self.last_onset_times = indices
            .iter()
            .map(|&index| self.last_onset_times[index])
            .collect();
        self.weights.fill(1.0 / self.config.num_particles as f64);
    }
}

fn update_tempo_particles(
    note_log_tempi: &mut [f64],
    local_log_tempi: &mut [f64],
    crossed: &[(usize, usize, usize)],
    phases: &[f64],
    notated_tempo: f64,
    tempo_noise: (f64, f64),
    tempo_draws: (&[bool], &[f64]),
) {
    let (high_tempo_noise, low_tempo_noise) = tempo_noise;
    let (fast_choices, standard_normals) = tempo_draws;
    debug_assert_eq!(crossed.len(), phases.len());
    debug_assert_eq!(crossed.len(), fast_choices.len());
    debug_assert_eq!(crossed.len(), standard_normals.len());

    for (index, &(particle, _, _)) in crossed.iter().enumerate() {
        note_log_tempi[particle] += 0.45 * phases[index];
        let note_bpm = 2.0f64.powf(note_log_tempi[particle]);
        let adjusted_bpm = if note_bpm > 2.0 * notated_tempo {
            notated_tempo + high_tempo_noise
        } else if note_bpm < notated_tempo / 2.0 {
            notated_tempo + low_tempo_noise
        } else {
            note_bpm
        };
        let local_bpm = 2.0f64.powf(local_log_tempi[particle]);
        local_log_tempi[particle] = ((2.0 * local_bpm + adjusted_bpm) / 3.0).log2();
    }

    for (index, &(particle, _, _)) in crossed.iter().enumerate() {
        let sigma = if fast_choices[index] { 0.4 } else { 0.1 };
        note_log_tempi[particle] = local_log_tempi[particle] + standard_normals[index] * sigma;
    }
}

fn validate_model(model: &KorzeniowskiScoreModel) -> Result<()> {
    let rows = model.beat_grid.len();
    let width = model.templates.first().map_or(0, Vec::len);
    if rows == 0
        || model.onset_positions.is_empty()
        || width == 0
        || model.nearest_onsets.len() != rows
        || model.rest_mask.len() != rows
        || model.templates.len() != rows
        || model.active_notes.len() != rows
        || model.templates.iter().any(|row| row.len() != width)
        || model
            .templates
            .iter()
            .flatten()
            .any(|value| !value.is_finite() || *value < 0.0)
        || model.beat_grid.iter().any(|value| !value.is_finite())
        || model.onset_positions.iter().any(|value| !value.is_finite())
    {
        return Err(invalid(
            "score_model",
            "grid, onset, rest, template, and active-note dimensions must agree",
        ));
    }
    if model.beat_grid.windows(2).any(|pair| pair[0] >= pair[1])
        || model
            .onset_positions
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || model.nearest_onsets.iter().any(|nearest| {
            !nearest.is_finite()
                || model
                    .onset_positions
                    .binary_search_by(|onset| onset.total_cmp(nearest))
                    .is_err()
        })
        || model
            .rest_mask
            .iter()
            .zip(&model.active_notes)
            .any(|(&rest, notes)| rest != notes.is_empty())
        || model
            .active_notes
            .iter()
            .flatten()
            .any(|pitch| !(0..=127).contains(pitch))
    {
        return Err(invalid(
            "score_model",
            "model arrays must be finite, ordered, and internally consistent",
        ));
    }
    Ok(())
}

fn normalize_likelihood_weights(weights: &mut [f64], likelihoods: &[f64]) -> Result<()> {
    if weights.is_empty() || weights.len() != likelihoods.len() {
        return Err(invalid(
            "weights",
            "particle weights and likelihoods must have the same nonzero length",
        ));
    }
    for (weight, likelihood) in weights.iter_mut().zip(likelihoods) {
        *weight = *weight * likelihood.max(0.0) + 1e-300;
    }
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err(invalid(
            "weights",
            "normalization total must be positive and finite",
        ));
    }
    for weight in weights {
        *weight /= total;
    }
    Ok(())
}

fn systematic_resample_indices(weights: &[f64], offset: f64) -> Vec<usize> {
    let mut cumulative = Vec::with_capacity(weights.len());
    let mut total = 0.0;
    for &weight in weights {
        total += weight;
        cumulative.push(total);
    }
    (0..weights.len())
        .map(|index| {
            let position = (offset + index as f64) / weights.len() as f64;
            cumulative
                .partition_point(|value| *value < position)
                .min(weights.len() - 1)
        })
        .collect()
}

fn effective_sample_size(weights: &[f64]) -> f64 {
    1.0 / weights.iter().map(|weight| weight * weight).sum::<f64>()
}

fn onset_observation_probability(onset: f64) -> f64 {
    0.5 * (1.0 + (10.0 * (onset - 0.5)).tanh())
}

fn gaussian(value: f64, mean: f64, sigma: f64) -> f64 {
    (-0.5 * ((value - mean) / sigma).powi(2)).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
}

fn upper_bound(onsets: &[f32], beat: f64) -> usize {
    onsets.partition_point(|onset| f64::from(*onset) <= beat)
}

fn uniform(rng: &mut Mt) -> f64 {
    let high = rng.next_u32() >> 5;
    let low = rng.next_u32() >> 6;
    (f64::from(high) * 67_108_864.0 + f64::from(low)) / 9_007_199_254_740_992.0
}

fn normal(rng: &mut Mt, spare: &mut Option<f64>, mean: f64, sigma: f64) -> f64 {
    if let Some(value) = spare.take() {
        return mean + sigma * value;
    }
    loop {
        let x = 2.0 * uniform(rng) - 1.0;
        let y = 2.0 * uniform(rng) - 1.0;
        let radius_squared = x * x + y * y;
        if radius_squared > 0.0 && radius_squared < 1.0 {
            let scale = (-2.0 * radius_squared.ln() / radius_squared).sqrt();
            *spare = Some(x * scale);
            return mean + sigma * y * scale;
        }
    }
}

fn invalid(key: &str, reason: &str) -> Error {
    Error::AudioConfig {
        key: key.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_draws_match_the_reference_polar_pair_and_reuse_its_spare() {
        let mut rng = Mt::new(1984);
        let mut spare = None;
        let first = normal(&mut rng, &mut spare, 0.0, 1.0);
        let second = normal(&mut rng, &mut spare, 0.0, 1.0);
        assert!((first - -0.02690222684723234).abs() < 1e-14);
        assert!((second - -0.6892736860726052).abs() < 1e-14);
        assert!((uniform(&mut rng) - 0.2818432190329727).abs() < 1e-15);
    }

    #[test]
    fn likelihood_collapse_uses_the_reference_floor_and_renormalizes() {
        let mut weights = [0.2, 0.3, 0.5];
        normalize_likelihood_weights(&mut weights, &[0.0, 0.0, 0.0]).unwrap();
        for weight in weights {
            assert!((weight - 1.0 / 3.0).abs() < 1e-15);
        }

        let mut weights = [0.2, 0.3, 0.5];
        normalize_likelihood_weights(&mut weights, &[0.0, 0.5, 1.0]).unwrap();
        assert!(weights[0] > 0.0 && weights[0] < 1e-299);
        assert!((weights[1] - 3.0 / 13.0).abs() < 1e-15);
        assert!((weights[2] - 10.0 / 13.0).abs() < 1e-15);
    }

    #[test]
    fn fixed_draw_fixtures_cover_tempo_updates_and_systematic_resampling() {
        let fast_choices = fixture("fast_choice_uniforms.npy");
        let standard_normals = fixture("standard_normals.npy");
        let resample_offsets = fixture("resample_offsets.npy");
        let mut note_tempi = [300.0f64.log2(), 350.0f64.log2(), 30.0f64.log2()];
        let mut local_tempi = [120.0f64.log2(); 3];
        let high_tempo_noise = -5.0 + 5.0 * standard_normals[4];
        let low_tempo_noise = 5.0 + 5.0 * standard_normals[1];
        let fast_choices = [
            fast_choices[0] < 0.2,
            fast_choices[1] < 0.2,
            fast_choices[2] < 0.2,
        ];
        let standard_normals = [
            standard_normals[4],
            standard_normals[2],
            standard_normals[5],
        ];
        update_tempo_particles(
            &mut note_tempi,
            &mut local_tempi,
            &[(0, 0, 1), (1, 1, 1), (2, 2, 1)],
            &[0.0, 0.0, 0.0],
            120.0,
            (high_tempo_noise, low_tempo_noise),
            (&fast_choices, &standard_normals),
        );
        let smoothed_fast_tempo: f64 = (2.0 * 120.0 + (120.0 - 2.5)) / 3.0;
        assert!((local_tempi[0] - smoothed_fast_tempo.log2()).abs() < 1e-15);
        assert_eq!(local_tempi[0], local_tempi[1]);
        assert!((local_tempi[2] - 120.0f64.log2()).abs() < 1e-15);
        assert!((note_tempi[0] - (smoothed_fast_tempo.log2() + 0.2)).abs() < 1e-15);
        assert!((note_tempi[1] - (smoothed_fast_tempo.log2() - 0.2)).abs() < 1e-15);
        assert!((note_tempi[2] - (120.0f64.log2() + 0.1)).abs() < 1e-15);

        let expected = [[0, 2, 2], [0, 2, 2], [1, 2, 2], [2, 2, 2]];
        for (offset, expected) in resample_offsets.into_iter().zip(expected) {
            assert_eq!(
                systematic_resample_indices(&[0.1, 0.2, 0.7], offset),
                expected
            );
        }
    }

    #[test]
    fn uniform_weights_have_full_effective_sample_size() {
        assert_eq!(effective_sample_size(&[0.25; 4]), 4.0);
    }

    #[test]
    fn score_model_rejects_inconsistent_template_width_and_nearest_onsets() {
        let model = KorzeniowskiScoreModel {
            beat_grid: vec![0.0, 1.0, 2.0],
            onset_positions: vec![0.0, 2.0],
            nearest_onsets: vec![0.0, 0.0, 2.0],
            rest_mask: vec![false, true, false],
            templates: vec![vec![1.0], vec![0.0], vec![1.0]],
            active_notes: vec![vec![60], vec![], vec![62]],
        };
        let config = PfKorzConfig {
            num_particles: 3,
            ..PfKorzConfig::default()
        };

        let mut bad_width = model.clone();
        bad_width.templates[1].clear();
        assert!(PfKorz::new(bad_width, config).is_err());

        let mut bad_nearest_onset = model.clone();
        bad_nearest_onset.nearest_onsets[1] = 0.5;
        assert!(PfKorz::new(bad_nearest_onset, config).is_err());

        let mut non_finite_template = model;
        non_finite_template.templates[1][0] = f32::NAN;
        assert!(PfKorz::new(non_finite_template, config).is_err());
    }

    #[test]
    fn rest_state_propagation_resets_sound_at_notes_and_decays_during_rests() {
        let model = KorzeniowskiScoreModel {
            beat_grid: vec![0.0, 1.0, 2.0],
            onset_positions: vec![0.0, 2.0],
            nearest_onsets: vec![0.0, 0.0, 2.0],
            rest_mask: vec![false, true, false],
            templates: vec![vec![1.0], vec![0.0], vec![1.0]],
            active_notes: vec![vec![60], vec![], vec![62]],
        };
        let mut follower = PfKorz::new(
            model,
            PfKorzConfig {
                num_particles: 3,
                ..PfKorzConfig::default()
            },
        )
        .unwrap();
        follower.positions.copy_from_slice(&[0.0, 1.0, 2.0]);
        follower
            .sound_probabilities
            .copy_from_slice(&[0.3, 0.4, 0.5]);
        follower
            .rest_probabilities
            .copy_from_slice(&[0.7, 0.6, 0.5]);
        follower.propagate_sound_state();
        assert!((follower.sound_probabilities[1] - 0.32).abs() < 1e-15);
        assert!((follower.rest_probabilities[1] - 0.68).abs() < 1e-15);
        assert_eq!(follower.sound_probabilities[0], 1.0);
        assert_eq!(follower.sound_probabilities[2], 1.0);
        assert_eq!(follower.rest_probabilities[0], 0.0);
        assert_eq!(follower.rest_probabilities[2], 0.0);
    }

    fn fixture(name: &str) -> Vec<f64> {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("golden")
            .join("audio")
            .join("pfkorz")
            .join("fixed_draws")
            .join(name);
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
        let header_len = u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize;
        let data = &bytes[10 + header_len..];
        let (chunks, remainder) = data.as_chunks::<8>();
        assert!(remainder.is_empty());
        chunks
            .iter()
            .map(|chunk| f64::from_le_bytes(*chunk))
            .collect()
    }
}
