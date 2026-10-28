//! Port of `matchmaker.prob.outer_product_hmm.OuterProductHMM` (method key `outerhmm`), after
//! Nakamura et al. (2014), "Outer-product hidden Markov model and polyphonic MIDI score
//! following".
//!
//! States are the score chords. Performed notes arrive one at a time; a note less than 35 ms
//! after the previous one is folded into the current chord observation and the Viterbi step is
//! redone from the distribution before the chord (a continuation, no alignment path entry).
//! Transitions combine a band of neighbouring chords with arbitrary skips factorised as
//! `S[j] * r[i]`.
//!
//! ```
//! use matchmaker_rs::base::OnlineAlignment;
//! use matchmaker_rs::prob::outerhmm::{OuterProductHmm, OuterProductHmmConfig};
//! use matchmaker_rs::score::Score;
//! let score = Score::load("tests/data/simple_mozart_k265_var1.musicxml", true)?;
//! let follower = OuterProductHmm::new(&score, OuterProductHmmConfig::default());
//! assert_eq!(follower.score_positions().len(), score.unique_onsets().len());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::collections::BTreeSet;

use super::hidden_markov::argmax;
use crate::base::{AlignmentPath, OnlineAlignment, StepDebug};
use crate::features::Observation;
use crate::score::Score;

/// Notes closer than this (seconds) to the previous note belong to the same chord.
pub const IOI_THRESHOLD: f64 = 0.035;

/// Band of the transition matrix: from state `j` the Viterbi step reads columns `j - D1 ..= j + D2`.
const D1: usize = 3;
const D2: usize = 3;
const BAND: usize = D1 + D2 + 1;
const PIANO_KEYS: usize = 88;

/// Probabilities of correct pitches and pitch-error classes (`DEFAULT_PITCH_ERROR_PROBS`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PitchErrorProbs {
    /// Probability mass of the chord's own pitches.
    pub correct_pitch: f64,
    /// Mass of pitches one semitone away.
    pub semi_tone_error: f64,
    /// Mass of pitches two semitones away.
    pub whole_tone_error: f64,
    /// Mass of pitches an octave away.
    pub octave_error: f64,
    /// Mass of the remaining pitches within an octave.
    pub within_one_octave_error: f64,
}

impl Default for PitchErrorProbs {
    fn default() -> Self {
        PitchErrorProbs {
            correct_pitch: 0.9497,
            semi_tone_error: 0.0145 / 2.0,
            whole_tone_error: 0.0224 / 2.0,
            octave_error: 0.0047 / 2.0,
            within_one_octave_error: 0.0086 / 9.0 / 2.0,
        }
    }
}

/// Parameters of [`OuterProductHmm`]; the defaults are the reference's `outerhmm` method.
#[derive(Debug, Clone, PartialEq)]
pub struct OuterProductHmmConfig {
    /// `(delta, probability)` of moving `delta` chords ahead (`DEFAULT_TRANSITIONS`); deltas
    /// outside `-3..=3` are normalised into the rows but never read.
    pub transitions: Vec<(i64, f64)>,
    /// Pitch-error model.
    pub pitch_error_probs: PitchErrorProbs,
    /// Probability of pitches outside every error class.
    pub other_prob: f64,
    /// Skip-from probability of each chord (`S`); `None` is uniform `1/N`.
    pub skip_from: Option<Vec<f64>>,
    /// Skip-to probability of each chord (`r`); `None` is uniform `1/N`.
    pub skip_to: Option<Vec<f64>>,
    /// Consecutive chords without a state change after which following stops.
    pub patience: usize,
}

impl Default for OuterProductHmmConfig {
    fn default() -> Self {
        OuterProductHmmConfig {
            transitions: vec![
                (-3, 0.00509),
                (-2, 0.00516),
                (-1, 0.00886),
                (0, 0.01342),
                (1, 0.94531),
                (2, 0.00610),
                (3, 0.00073),
            ],
            pitch_error_probs: PitchErrorProbs::default(),
            other_prob: 1e-6,
            skip_from: None,
            skip_to: None,
            patience: 10,
        }
    }
}

/// Port of `compute_OuterProductHMM_pitch_probabilities`: row-major `(n_chords, 128)`.
pub fn pitch_probabilities(
    chords: &[BTreeSet<i32>],
    probs: &PitchErrorProbs,
    other_prob: f64,
) -> Vec<f64> {
    const MAX_PITCH: i32 = 128;
    let mut table = vec![other_prob; chords.len() * MAX_PITCH as usize];
    let shifted = |chord: &BTreeSet<i32>, d: i32| -> BTreeSet<i32> {
        chord
            .iter()
            .flat_map(|&p| [p + d, p - d])
            .filter(|p| (0..MAX_PITCH).contains(p))
            .collect()
    };
    for (i, chord) in chords.iter().enumerate() {
        if chord.is_empty() {
            continue;
        }
        let correct = chord.clone();
        let semi: BTreeSet<i32> = shifted(chord, 1).difference(&correct).copied().collect();
        let whole: BTreeSet<i32> = shifted(chord, 2)
            .into_iter()
            .filter(|p| !correct.contains(p) && !semi.contains(p))
            .collect();
        let octave: BTreeSet<i32> = shifted(chord, 12)
            .into_iter()
            .filter(|p| !correct.contains(p) && !semi.contains(p) && !whole.contains(p))
            .collect();
        let within: BTreeSet<i32> = chord
            .iter()
            .flat_map(|&p| p - 11..p + 12)
            .filter(|x| {
                (0..MAX_PITCH).contains(x)
                    && ![&correct, &semi, &whole, &octave]
                        .iter()
                        .any(|class| class.contains(x))
            })
            .collect();
        let row = &mut table[i * MAX_PITCH as usize..(i + 1) * MAX_PITCH as usize];
        let classes = [
            (&correct, probs.correct_pitch),
            (&semi, probs.semi_tone_error),
            (&whole, probs.whole_tone_error),
            (&octave, probs.octave_error),
            (&within, probs.within_one_octave_error),
        ];
        for (class, mass) in classes {
            let each = mass / class.len().max(1) as f64;
            for &p in class {
                row[p as usize] = each;
            }
        }
    }
    table
}

/// Port of `compute_transition_matrix`, keeping only the band the Viterbi step reads:
/// entry `j * 7 + (i + D1 - j)` is `alpha[j, i]` for `i` in `j - D1 ..= j + D2`.
fn transition_band(n: usize, transitions: &[(i64, f64)]) -> Vec<f64> {
    const BASE: f64 = 1e-6;
    let reach = transitions
        .iter()
        .map(|(d, _)| d.unsigned_abs())
        .max()
        .unwrap_or(0) as i64;
    let raw = |j: usize, i: usize| -> f64 {
        let delta = i as i64 - j as i64;
        let value = if delta.abs() <= reach {
            // later entries overwrite earlier ones, as in the reference's assignment loop
            transitions
                .iter()
                .rev()
                .find(|(d, _)| *d == delta)
                .map_or(BASE, |&(_, p)| p)
        } else {
            BASE
        };
        value + f64::EPSILON
    };
    let mut band = vec![0.0; n * BAND];
    for j in 0..n {
        let row_sum: f64 = (0..n).map(|i| raw(j, i)).sum();
        for i in j.saturating_sub(D1)..(j + D2 + 1).min(n) {
            band[j * BAND + (i + D1 - j)] = raw(j, i) / row_sum;
        }
    }
    band
}

/// The `outerhmm` follower.
#[derive(Debug, Clone)]
pub struct OuterProductHmm {
    patience: usize,
    score_positions: Vec<f32>,
    alpha: Vec<f64>,
    skip_from: Vec<f64>,
    skip_to: Vec<f64>,
    /// `ln(max(b_table[:, 21:109], 1e-300))`, row-major `(N, 88)`.
    log_b: Vec<f64>,
    state_probabilities: Vec<f64>,
    pre_chord_probs: Vec<f64>,
    current_chord: Vec<f32>,
    current_index: usize,
    prev_perf_time: Option<f64>,
    is_continuation: bool,
    same_state_counter: usize,
    finished: bool,
    path: AlignmentPath,
}

impl OuterProductHmm {
    /// Build from a score (`OuterProductHMM(reference_features, ...)`).
    pub fn new(score: &Score, config: OuterProductHmmConfig) -> Self {
        let chords = score.chords();
        let n = chords.len();
        let table = pitch_probabilities(&chords, &config.pitch_error_probs, config.other_prob);
        let log_b = (0..n)
            .flat_map(|i| table[i * 128 + 21..i * 128 + 21 + PIANO_KEYS].iter())
            .map(|&b| b.max(1e-300).ln())
            .collect();
        let uniform = || vec![1.0 / n as f64; n];
        let mut state_probabilities = vec![0.0; n];
        if let Some(first) = state_probabilities.first_mut() {
            *first = 1.0;
        }
        OuterProductHmm {
            patience: config.patience,
            score_positions: score.unique_onsets(),
            alpha: transition_band(n, &config.transitions),
            skip_from: config.skip_from.unwrap_or_else(uniform),
            skip_to: config.skip_to.unwrap_or_else(uniform),
            log_b,
            pre_chord_probs: state_probabilities.clone(),
            state_probabilities,
            current_chord: vec![0.0; PIANO_KEYS],
            current_index: 0,
            prev_perf_time: None,
            is_continuation: false,
            same_state_counter: 0,
            finished: false,
            path: AlignmentPath::default(),
        }
    }

    /// Index of the current state.
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// State distribution after the last call.
    pub fn state_probabilities(&self) -> &[f64] {
        &self.state_probabilities
    }

    /// Whether the last call folded a note into the previous chord.
    pub fn is_continuation(&self) -> bool {
        self.is_continuation
    }

    /// `compute_obs_likelihood`: per-state likelihood of the active pitches, shifted by its
    /// maximum for stability.
    fn obs_likelihood(&self, observation: &[f32]) -> Vec<f64> {
        // inactive pitches add exactly 0 to the reference's matrix-vector product
        let active: Vec<(usize, f64)> = observation
            .iter()
            .take(PIANO_KEYS)
            .enumerate()
            .filter(|&(_, &x)| x != 0.0)
            .map(|(k, &x)| (k, f64::from(x)))
            .collect();
        let mut log_em: Vec<f64> = (0..self.log_b.len() / PIANO_KEYS)
            .map(|i| {
                let row = &self.log_b[i * PIANO_KEYS..(i + 1) * PIANO_KEYS];
                active.iter().map(|&(k, x)| row[k] * x).sum()
            })
            .collect();
        let max = log_em.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        log_em.iter_mut().for_each(|v| *v = (*v - max).exp());
        log_em
    }

    /// `viterbi_step`: one outer-product Viterbi update with the observation `observation`.
    fn viterbi_step(&self, prev: &[f64], observation: &[f32]) -> Vec<f64> {
        let n = prev.len();
        let b = self.obs_likelihood(observation);
        let global_skip_max = prev
            .iter()
            .zip(&self.skip_from)
            .map(|(p, s)| p * s)
            .fold(f64::NEG_INFINITY, f64::max);
        let mut new_probs: Vec<f64> = (0..n)
            .map(|i| {
                let mut local_max = 0.0f64;
                let (start, end) = (i.saturating_sub(D2), (i + D1 + 1).min(n));
                for (j, &p) in prev.iter().enumerate().take(end).skip(start) {
                    let val = p * self.alpha[j * BAND + (i + D1 - j)];
                    if val > local_max {
                        local_max = val;
                    }
                }
                let skip = self.skip_to[i] * global_skip_max;
                b[i] * if skip >= local_max { skip } else { local_max }
            })
            .collect();
        let sum: f64 = new_probs.iter().sum();
        if sum > 0.0 {
            new_probs.iter_mut().for_each(|v| *v /= sum);
            new_probs
        } else {
            vec![1.0 / n as f64; n]
        }
    }

    fn position(&self) -> f64 {
        f64::from(self.score_positions[self.current_index])
    }
}

impl OnlineAlignment for OuterProductHmm {
    fn call(&mut self, obs: &Observation) -> f64 {
        let _span = tracing::debug_span!("inference", method = "outerhmm").entered();
        let prev_state = self.current_index;
        let continuation = self
            .prev_perf_time
            .is_some_and(|prev| obs.perf_time - prev < IOI_THRESHOLD);
        self.prev_perf_time = Some(obs.perf_time);
        self.is_continuation = continuation;

        if continuation {
            // redo the step from the distribution before the chord with all chord pitches
            for (chord, &x) in self.current_chord.iter_mut().zip(&obs.features) {
                *chord = chord.max(x);
            }
            self.state_probabilities =
                self.viterbi_step(&self.pre_chord_probs, &self.current_chord);
            self.current_index = argmax(&self.state_probabilities);
            return self.position();
        }

        self.pre_chord_probs = std::mem::take(&mut self.state_probabilities);
        self.current_chord = obs.features.clone();
        self.state_probabilities = self.viterbi_step(&self.pre_chord_probs, &self.current_chord);
        self.current_index = argmax(&self.state_probabilities);
        let position = self.position();
        self.path.push(obs.perf_time, position);
        tracing::debug!(perf_time = obs.perf_time, position);

        // `run()` stops after `patience` chords without a state change
        if self.current_index == prev_state {
            if self.same_state_counter >= self.patience {
                self.finished = true;
            }
            self.same_state_counter += 1;
        } else {
            self.same_state_counter = 0;
        }
        position
    }

    fn is_still_following(&self) -> bool {
        !self.finished
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
            probabilities: self.state_probabilities.clone(),
            beat_period: None,
            is_continuation: Some(self.is_continuation),
        }
    }
}
