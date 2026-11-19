//! Port of `matchmaker.prob.outer_product_hmm_audio.AudioOuterProductHMM` (audio method `outerhmm`).
//!
//! Upstream: `matchmaker/prob/outer_product_hmm_audio.py`.
//!
//! Unlike MIDI [`super::outerhmm`], this follower consumes an 88-bin CQT vector, builds a
//! weighted chord-template model from score notes, and tracks sounding and pause states.
//!
//! ```
//! use matchmaker_rs::base::OnlineAlignment;
//! use matchmaker_rs::features::Observation;
//! use matchmaker_rs::prob::outerhmm_audio::{AudioOuterHmm, AudioOuterHmmConfig};
//! use matchmaker_rs::score::{Score, ScoreNote};
//! let score = Score::load("tests/data/simple_mozart_k265_var1.musicxml", true)?;
//! let templates = vec![vec![1.0 / 88.0; 88]; 88];
//! let mut follower = AudioOuterHmm::new(&score.notes, templates, AudioOuterHmmConfig::default())?;
//! let position = follower.call(&Observation { features: vec![0.0; 89], perf_time: 0.0 });
//! assert!(position.is_finite());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use crate::base::{AlignmentPath, OnlineAlignment, StepDebug};
use crate::error::{Error, Result};
use crate::features::{Observation, cqt::CqtProcessor};
use crate::score::ScoreNote;
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};
use sha2::{Digest, Sha256};

const PITCHES: usize = 88;
const EMISSION_BETA: f64 = 5.0;
const UNIFORM_FLOOR: f64 = 0.2;
const PITCHEDNESS_VAR0: f64 = 2e-4;
const SUSTAINED_DECAY: f64 = 0.3;
const PAUSE_ENTRY_PROB: f64 = 1e-100;
const PAUSE_SELF_TRANSITION: f64 = 0.999;
const PAUSE_EMISSION_MAX: f64 = 1e-3;
const PROB_FLOOR: f64 = 1e-300;
const LEARNED_TEMPLATE_SAMPLE_RATE: u32 = 16_000;
const LEARNED_TEMPLATE_HOP_LENGTH: usize = 640;
/// SHA-256 of the Apache-2.0 learned-template archive from the reference release.
pub const LEARNED_TEMPLATE_SHA256: &str =
    "7448383AA9C7B4B49B3A4A52B8BED2AFF994B0F3266BA0372A28738A3DDE8A03";
const LEARNED_TEMPLATE_NPZ: &[u8] = include_bytes!("assets/pitch_templates_16000_640.npz");
type PitchTemplateCache = Mutex<HashMap<(PathBuf, u32, usize), (Vec<Vec<f64>>, Vec<bool>)>>;
static PITCH_TEMPLATE_CACHE: OnceLock<PitchTemplateCache> = OnceLock::new();

/// Runtime parameters of [`AudioOuterHmm`].
#[derive(Debug, Clone, PartialEq)]
pub struct AudioOuterHmmConfig {
    /// Score tempo in quarter-note beats per minute.
    pub tempo: f64,
    /// Processor sample rate in hertz; the facade follows the reference constructor's 16 kHz default.
    pub sample_rate: u32,
    /// CQT hop in samples.
    pub hop_length: usize,
    /// Probability of a non-local repeat/skip transition from each chord.
    pub s_j: f64,
}

impl Default for AudioOuterHmmConfig {
    fn default() -> Self {
        Self {
            tempo: 120.0,
            sample_rate: 16_000,
            hop_length: 640,
            s_j: 0.0,
        }
    }
}

/// Score chord templates derived from weighted note onsets and per-pitch CQT templates.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioOuterHmmScoreModel {
    score_positions: Vec<f32>,
    onset_quarters: Vec<f32>,
    weighted_chords: Vec<Vec<(usize, f64)>>,
    pitch_templates: Vec<Vec<f64>>,
    learned_support: Vec<bool>,
    soundfont_sha256: Option<String>,
    chord_templates: Vec<Vec<f64>>,
    log_templates: Vec<Vec<f64>>,
}

impl AudioOuterHmmScoreModel {
    /// Build the model from score notes and an 88-by-88 pitch-template matrix.
    pub fn new(notes: &[ScoreNote], pitch_templates: Vec<Vec<f64>>) -> Result<Self> {
        if notes.is_empty() {
            return Err(config("score", "must contain at least one note"));
        }
        if pitch_templates.len() != PITCHES
            || pitch_templates.iter().any(|row| {
                row.len() != PITCHES || row.iter().any(|value| !value.is_finite() || *value < 0.0)
            })
        {
            return Err(config(
                "pitch_templates",
                "must be a finite, nonnegative 88-by-88 matrix",
            ));
        }

        let mut onsets: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (index, note) in notes.iter().enumerate() {
            if !note.onset_beat.is_finite()
                || !note.onset_quarter.is_finite()
                || !note.duration_beat.is_finite()
                || note.duration_beat < 0.0
            {
                return Err(config("score", "note times and durations must be finite"));
            }
            onsets
                .entry(note.onset_beat.to_bits())
                .or_default()
                .push(index);
        }
        let mut order: Vec<(f32, Vec<usize>)> = onsets
            .into_values()
            .map(|indices| (notes[indices[0]].onset_beat, indices))
            .collect();
        order.sort_by(|left, right| left.0.total_cmp(&right.0));
        let score_positions: Vec<f32> = order.iter().map(|(beat, _)| *beat).collect();
        let mut weighted_chords = Vec::with_capacity(order.len());
        for (onset, _) in &order {
            let mut chord = BTreeMap::new();
            for note in notes.iter().filter(|note| note.onset_beat == *onset) {
                if (21..=108).contains(&note.pitch) {
                    chord.insert((note.pitch - 21) as usize, 1.0);
                }
            }
            for note in notes.iter().filter(|note| {
                note.onset_beat < *onset
                    && note.onset_beat + note.duration_beat > *onset
                    && (21..=108).contains(&note.pitch)
            }) {
                let pitch = (note.pitch - 21) as usize;
                chord.entry(pitch).or_insert_with(|| {
                    (-SUSTAINED_DECAY * f64::from(*onset - note.onset_beat)).exp()
                });
            }
            weighted_chords.push(chord.into_iter().collect::<Vec<_>>());
        }

        let chord_templates: Vec<Vec<f64>> = weighted_chords
            .iter()
            .map(|chord| {
                let mut row = vec![0.0; PITCHES];
                for &(pitch, weight) in chord {
                    for (value, template) in row.iter_mut().zip(&pitch_templates[pitch]) {
                        *value += weight * template;
                    }
                }
                let sum: f64 = row.iter().sum();
                row.iter_mut().for_each(|value| *value /= sum.max(1e-12));
                row.iter_mut().for_each(|value| {
                    *value = (1.0 - UNIFORM_FLOOR) * *value + UNIFORM_FLOOR / 88.0
                });
                row
            })
            .collect();
        let log_templates = chord_templates
            .iter()
            .map(|row| row.iter().map(|value| value.ln()).collect())
            .collect();
        Ok(Self {
            score_positions,
            onset_quarters: order
                .iter()
                .map(|(_, indices)| notes[indices[0]].onset_quarter)
                .collect(),
            weighted_chords,
            pitch_templates,
            learned_support: vec![false; PITCHES],
            soundfont_sha256: None,
            chord_templates,
            log_templates,
        })
    }

    /// Score positions of the chord onsets in beats.
    pub fn score_positions(&self) -> &[f32] {
        &self.score_positions
    }

    /// Chord templates after weighting and the uniform emission floor.
    pub fn chord_templates(&self) -> &[Vec<f64>] {
        &self.chord_templates
    }

    /// Pitch templates used to construct the chord model.
    pub fn pitch_templates(&self) -> &[Vec<f64>] {
        &self.pitch_templates
    }

    /// Pitches whose rows were replaced by the learned upstream overlay.
    pub fn learned_support(&self) -> &[bool] {
        &self.learned_support
    }

    /// SHA-256 of the SoundFont used to synthesize unsupported learned-template rows.
    pub fn soundfont_sha256(&self) -> Option<&str> {
        self.soundfont_sha256.as_deref()
    }

    /// Weighted pitch indices and amplitudes in each score chord.
    pub fn weighted_chords(&self) -> &[Vec<(usize, f64)>] {
        &self.weighted_chords
    }
}

/// Two-layer audio outer-product HMM with a sounding and pause state per score chord.
#[derive(Debug, Clone)]
pub struct AudioOuterHmm {
    model: AudioOuterHmmScoreModel,
    alpha: Vec<f64>,
    skip_from: Vec<f64>,
    skip_to: Vec<f64>,
    a00: Vec<f64>,
    a01: Vec<f64>,
    e0: Vec<f64>,
    state_probabilities: Vec<f64>,
    current_index: usize,
    path: AlignmentPath,
}

impl AudioOuterHmm {
    /// Build the follower using fixed or synthesized pitch templates.
    pub fn new(
        notes: &[ScoreNote],
        pitch_templates: Vec<Vec<f64>>,
        config: AudioOuterHmmConfig,
    ) -> Result<Self> {
        if !config.tempo.is_finite() || config.tempo <= 0.0 {
            return Err(config_error("tempo", "must be a positive number of BPM"));
        }
        if config.sample_rate == 0 || config.hop_length == 0 {
            return Err(config_error("sample_rate/hop_length", "must be at least 1"));
        }
        if !config.s_j.is_finite() || !(0.0..=1.0).contains(&config.s_j) {
            return Err(config_error("s_j", "must be finite and between 0 and 1"));
        }
        let model = AudioOuterHmmScoreModel::new(notes, pitch_templates)?;
        Self::from_model(model, config)
    }

    /// Build synthesized pitch templates and apply the learned rows shipped by pymatchmaker.
    pub fn from_soundfont(
        notes: &[ScoreNote],
        soundfont_path: impl AsRef<Path>,
        config: AudioOuterHmmConfig,
    ) -> Result<Self> {
        let (pitch_templates, support) = build_pitch_templates(
            soundfont_path.as_ref(),
            config.sample_rate,
            config.hop_length,
        )?;
        let mut model = AudioOuterHmmScoreModel::new(notes, pitch_templates)?;
        model.learned_support = support;
        model.soundfont_sha256 = Some(sha256_file(soundfont_path.as_ref())?);
        Self::from_model(model, config)
    }

    fn from_model(model: AudioOuterHmmScoreModel, config: AudioOuterHmmConfig) -> Result<Self> {
        let n = model.score_positions.len();
        let alpha = transition_matrix(n, config.s_j);
        let skip_from = vec![config.s_j; n];
        let skip_to = vec![1.0 / n as f64; n];
        let onset_quarters: Vec<f64> = model
            .onset_quarters
            .iter()
            .map(|&quarter| f64::from(quarter))
            .collect();
        let a00 = chord_self_transitions(
            &onset_quarters,
            config.tempo,
            f64::from(config.sample_rate) / config.hop_length as f64,
        );
        let a01: Vec<f64> = a00
            .iter()
            .map(|stay| (1.0 - stay) * PAUSE_ENTRY_PROB)
            .collect();
        let e0 = a00
            .iter()
            .zip(&a01)
            .map(|(stay, pause)| (1.0 - stay - pause).clamp(1e-10, 1.0))
            .collect();
        let mut state_probabilities = vec![0.0; 2 * n];
        state_probabilities[0] = 1.0;
        Ok(Self {
            model,
            alpha,
            skip_from,
            skip_to,
            a00,
            a01,
            e0,
            state_probabilities,
            current_index: 0,
            path: AlignmentPath::default(),
        })
    }

    /// Score model and pitch/chord templates used by this follower.
    pub fn model(&self) -> &AudioOuterHmmScoreModel {
        &self.model
    }

    /// Current posterior over interleaved sounding/pause states.
    pub fn state_probabilities(&self) -> &[f64] {
        &self.state_probabilities
    }

    /// Current score chord index.
    pub fn current_index(&self) -> usize {
        self.current_index
    }

    /// Sound emission likelihood for an 88-bin CQT observation (an optional 89th flux bin is ignored).
    pub fn compute_obs_likelihood(&self, observation: &[f32]) -> Result<Vec<f64>> {
        let normalized = normalized_cqt(observation)?;
        Ok(sound_likelihood(&self.model.log_templates, &normalized))
    }

    /// Pause emission likelihood for an 88-bin CQT observation (an optional 89th flux bin is ignored).
    pub fn compute_pause_emission(&self, observation: &[f32]) -> Result<f64> {
        let normalized = normalized_cqt(observation)?;
        if normalized.iter().sum::<f64>() <= 0.0 {
            return Ok(PAUSE_EMISSION_MAX);
        }
        Ok((1.0 / (1.0 + 200.0 * variance(&normalized))).clamp(PROB_FLOOR, PAUSE_EMISSION_MAX))
    }

    fn forward_step(&self, observation: &[f32]) -> Result<Vec<f64>> {
        let cqt = normalized_cqt(observation)?;
        let sum: f64 = observation
            .iter()
            .take(PITCHES)
            .map(|&value| f64::from(value.max(0.0)))
            .sum();
        let variance = variance(&cqt);
        let emit_sound = if sum <= 0.0 {
            vec![PROB_FLOOR; self.model.score_positions.len()]
        } else {
            sound_likelihood(&self.model.log_templates, &cqt)
        };
        let emit_pause = if sum <= 0.0 {
            PAUSE_EMISSION_MAX
        } else {
            (1.0 / (1.0 + 200.0 * variance)).clamp(PROB_FLOOR, PAUSE_EMISSION_MAX)
        };
        let n = self.model.score_positions.len();
        let prev_sound: Vec<f64> = (0..n).map(|i| self.state_probabilities[2 * i]).collect();
        let prev_pause: Vec<f64> = (0..n)
            .map(|i| self.state_probabilities[2 * i + 1])
            .collect();
        let exit_mass: Vec<f64> = (0..n)
            .map(|i| prev_sound[i] * self.e0[i] + prev_pause[i] * (1.0 - PAUSE_SELF_TRANSITION))
            .collect();
        let global_skip: f64 = exit_mass
            .iter()
            .zip(&self.skip_from)
            .map(|(mass, skip)| mass * skip)
            .sum();
        let mut next = vec![0.0; 2 * n];
        for i in 0..n {
            let start = i.saturating_sub(2);
            let end = (i + 1).min(n);
            let local = (start..end)
                .map(|j| exit_mass[j] * self.alpha[j * n + i])
                .sum::<f64>();
            let local_skip: f64 = (start..end).map(|j| exit_mass[j] * self.skip_from[j]).sum();
            let incoming = local + self.skip_to[i] * (global_skip - local_skip);
            let within_sound = prev_sound[i] * self.a00[i];
            let within_pause = prev_sound[i] * self.a01[i] + prev_pause[i] * PAUSE_SELF_TRANSITION;
            next[2 * i] = emit_sound[i] * (within_sound + incoming);
            next[2 * i + 1] = emit_pause * within_pause;
        }
        next.iter_mut()
            .for_each(|value| *value = value.max(PROB_FLOOR));
        let normalizer: f64 = next.iter().sum();
        if normalizer > 0.0 {
            next.iter_mut().for_each(|value| *value /= normalizer);
        } else {
            next.fill(0.0);
            next[0] = 1.0;
        }
        Ok(next)
    }
}

impl OnlineAlignment for AudioOuterHmm {
    fn call(&mut self, observation: &Observation) -> f64 {
        let _span = tracing::debug_span!("inference", method = "outerhmm").entered();
        if let Ok(probabilities) = self.forward_step(&observation.features) {
            self.state_probabilities = probabilities;
            self.current_index = top_state_index(&self.state_probabilities);
        }
        let position = f64::from(self.model.score_positions[self.current_index]);
        self.path.push(observation.perf_time, position);
        position
    }

    fn is_still_following(&self) -> bool {
        true
    }

    fn alignment_path(&self) -> &AlignmentPath {
        &self.path
    }

    fn score_positions(&self) -> &[f32] {
        &self.model.score_positions
    }

    fn debug_state(&self) -> StepDebug {
        StepDebug {
            current_index: self.current_index,
            probabilities: self.state_probabilities.clone(),
            ..StepDebug::default()
        }
    }
}

fn transition_matrix(n: usize, s_j: f64) -> Vec<f64> {
    let transitions = [(1isize, 1.0 - 1e-50), (2, 1e-50)];
    let mut alpha = vec![0.0; n * n];
    for j in 0..n {
        for &(delta, probability) in &transitions {
            let i = j as isize + delta;
            if (0..n as isize).contains(&i) {
                alpha[j * n + i as usize] = probability;
            }
        }
        let total: f64 = alpha[j * n..(j + 1) * n].iter().sum();
        if total > 0.0 {
            for value in &mut alpha[j * n..(j + 1) * n] {
                *value /= total;
            }
        }
        alpha[j * n + j] = 0.0;
        let total: f64 = alpha[j * n..(j + 1) * n].iter().sum();
        if total <= 0.0 && n > 1 {
            for i in 0..n {
                if i != j {
                    alpha[j * n + i] = 1.0 / (n - 1) as f64;
                }
            }
        } else if total > 0.0 {
            for value in &mut alpha[j * n..(j + 1) * n] {
                *value /= total;
            }
        }
        for value in &mut alpha[j * n..(j + 1) * n] {
            *value *= 1.0 - s_j;
        }
    }
    alpha
}

fn chord_self_transitions(onset_quarters: &[f64], tempo: f64, frame_rate: f64) -> Vec<f64> {
    let mut durations: Vec<f64> = onset_quarters
        .windows(2)
        .map(|pair| (pair[1] - pair[0]) * (60.0 / tempo))
        .collect();
    durations.push(durations.last().copied().unwrap_or(0.2));
    let frame_time = 1.0 / frame_rate.max(1e-6);
    durations
        .into_iter()
        .map(|duration| {
            let duration = duration.max(0.05);
            let frames = (duration / frame_time).max(1.0);
            (1.0 - 1.0 / frames).clamp(1e-6, 1.0 - 1e-6)
        })
        .collect()
}

fn normalized_cqt(observation: &[f32]) -> Result<Vec<f64>> {
    if observation.len() != PITCHES && observation.len() != PITCHES + 1 {
        return Err(config(
            "features",
            format!(
                "expected 88 CQT bins with optional flux (89 values), got {}",
                observation.len()
            ),
        ));
    }
    if observation.iter().any(|value| !value.is_finite()) {
        return Err(config("features", "CQT values must be finite"));
    }
    let values: Vec<f64> = observation
        .iter()
        .take(PITCHES)
        .map(|&value| f64::from(value.max(0.0)))
        .collect();
    let sum: f64 = values.iter().sum();
    Ok(if sum <= 0.0 {
        values
    } else {
        values.into_iter().map(|value| value / sum).collect()
    })
}

fn sound_likelihood(log_templates: &[Vec<f64>], normalized: &[f64]) -> Vec<f64> {
    if normalized.iter().sum::<f64>() <= 0.0 {
        return vec![PROB_FLOOR; log_templates.len()];
    }
    let scores: Vec<f64> = log_templates
        .iter()
        .map(|row| {
            row.iter()
                .zip(normalized)
                .map(|(log, value)| log * value)
                .sum()
        })
        .collect();
    let best = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pitchedness = variance(normalized);
    let factor = pitchedness / (pitchedness + PITCHEDNESS_VAR0);
    scores
        .into_iter()
        .map(|score| (EMISSION_BETA * factor * (score - best)).exp().max(1e-12))
        .collect()
}

fn variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64
}

fn top_state_index(probabilities: &[f64]) -> usize {
    let mut best = 0;
    for candidate in 1..probabilities.len() / 2 {
        let candidate_score = probabilities[2 * candidate] + probabilities[2 * candidate + 1];
        let best_score = probabilities[2 * best] + probabilities[2 * best + 1];
        if candidate_score > best_score {
            best = candidate;
        }
    }
    best
}

fn config_error(key: &str, reason: impl Into<String>) -> Error {
    Error::AudioConfig {
        key: key.into(),
        reason: reason.into(),
    }
}

fn config(key: &str, reason: impl Into<String>) -> Error {
    config_error(key, reason)
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|error| Error::io(path, error))?;
    let mut digest = Sha256::new();
    let mut buffer = [0; 8192];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| Error::io(path, error))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02X}"))
        .collect())
}

/// Synthesize all pitch templates and overlay learned rows when the reference archive matches.
pub fn build_pitch_templates(
    soundfont_path: &Path,
    sample_rate: u32,
    hop_length: usize,
) -> Result<(Vec<Vec<f64>>, Vec<bool>)> {
    let key = (soundfont_path.to_path_buf(), sample_rate, hop_length);
    let cache = PITCH_TEMPLATE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Ok(cache) = cache.lock()
        && let Some(templates) = cache.get(&key)
    {
        return Ok(templates.clone());
    }
    let archive_sha256 = format!("{:X}", Sha256::digest(LEARNED_TEMPLATE_NPZ));
    if archive_sha256 != LEARNED_TEMPLATE_SHA256 {
        return Err(config_error(
            "pitch_templates",
            format!("embedded archive SHA-256 mismatch: {archive_sha256}"),
        ));
    }
    let mut templates = synthesize_pitch_templates(soundfont_path, sample_rate, hop_length)?;
    let mut support = vec![false; PITCHES];
    if sample_rate == LEARNED_TEMPLATE_SAMPLE_RATE && hop_length == LEARNED_TEMPLATE_HOP_LENGTH {
        let learned = read_learned_templates(LEARNED_TEMPLATE_NPZ)?;
        for (pitch, row) in learned.into_iter().enumerate() {
            if row.iter().sum::<f64>() > 0.5 {
                templates[pitch] = row;
                support[pitch] = true;
            }
        }
    }
    let templates = (templates, support);
    if let Ok(mut cache) = cache.lock() {
        cache.insert(key, templates.clone());
    }
    Ok(templates)
}

/// Synthesize the reference's three-velocity, per-pitch CQT templates with RustySynth.
pub fn synthesize_pitch_templates(
    soundfont_path: impl AsRef<Path>,
    sample_rate: u32,
    hop_length: usize,
) -> Result<Vec<Vec<f64>>> {
    let path = soundfont_path.as_ref();
    if !(16_000..=192_000).contains(&sample_rate) || hop_length == 0 {
        return Err(config_error(
            "sample_rate/hop_length",
            "sample_rate must be 16000..=192000 Hz and hop_length at least 1",
        ));
    }
    let mut file = File::open(path).map_err(|error| Error::io(path, error))?;
    let render_error = |message: String| Error::AudioRender {
        path: path.to_path_buf(),
        message,
    };
    let soundfont =
        Arc::new(SoundFont::new(&mut file).map_err(|error| render_error(error.to_string()))?);
    let settings = SynthesizerSettings::new(sample_rate as i32);

    let rate = f64::from(sample_rate);
    let note_duration = 0.5;
    let step = 0.75;
    let mut by_pitch: Vec<Vec<Vec<f64>>> = (0..PITCHES).map(|_| Vec::new()).collect();
    for velocity in [50, 80, 110] {
        let mut synthesizer = Synthesizer::new(&soundfont, &settings)
            .map_err(|error| render_error(error.to_string()))?;
        synthesizer.process_midi_message(0, 0xB0, 91, 0);
        let mut events = Vec::with_capacity(2 * PITCHES);
        for pitch in 0..PITCHES {
            let start = (pitch as f64 * step * rate) as usize;
            let end = ((pitch as f64 * step + note_duration) * rate) as usize;
            events.push((start, true, pitch as i32 + 21));
            events.push((end, false, pitch as i32 + 21));
        }
        events.sort_by_key(|event| event.0);
        let total = events.last().map_or(0, |event| event.0);
        let mut audio = Vec::with_capacity(total + hop_length);
        let mut left = vec![0.0; settings.block_size];
        let mut right = vec![0.0; settings.block_size];
        for (sample, is_on, pitch) in events {
            while audio.len() < sample {
                let frames = (sample - audio.len()).min(left.len());
                synthesizer.render(&mut left[..frames], &mut right[..frames]);
                audio.extend(
                    left[..frames]
                        .iter()
                        .zip(&right[..frames])
                        .map(|(l, r)| (l + r) * 0.5),
                );
            }
            synthesizer.process_midi_message(
                0,
                if is_on { 0x90 } else { 0x80 },
                pitch,
                if is_on { velocity } else { 0 },
            );
        }
        let mut padded = vec![0.0; hop_length];
        padded.extend(audio);
        let features =
            CqtProcessor::new(sample_rate, hop_length, Some(f64::INFINITY))?.transform(&padded)?;
        let frame_rate = rate / hop_length as f64;
        for (pitch, pitch_frames) in by_pitch.iter_mut().enumerate() {
            let start = (pitch as f64 * step + 0.02).mul_add(frame_rate, 0.0) as usize;
            let end =
                (pitch as f64 * step + note_duration - 0.02).mul_add(frame_rate, 0.0) as usize;
            let frames = features.get(start..end).ok_or_else(|| {
                config_error(
                    "soundfont_path",
                    format!("CQT synthesis has no frames for pitch {}", pitch + 21),
                )
            })?;
            for frame in frames {
                let sum: f64 = frame.iter().map(|&value| f64::from(value)).sum();
                if sum > 0.0 {
                    pitch_frames.push(frame.iter().map(|&value| f64::from(value) / sum).collect());
                }
            }
        }
    }
    by_pitch
        .into_iter()
        .enumerate()
        .map(|(pitch, frames)| {
            if frames.is_empty() {
                return Err(config_error(
                    "soundfont_path",
                    format!("empty CQT template for pitch {}", pitch + 21),
                ));
            }
            let mut row = vec![0.0; PITCHES];
            for frame in &frames {
                for (value, sample) in row.iter_mut().zip(frame) {
                    *value += sample / frames.len() as f64;
                }
            }
            Ok(row)
        })
        .collect()
}

fn read_learned_templates(bytes: &[u8]) -> Result<Vec<Vec<f64>>> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes))
        .map_err(|error| config_error("pitch_templates", error.to_string()))?;
    let mut file = archive
        .by_name("mu.npy")
        .map_err(|error| config_error("pitch_templates", error.to_string()))?;
    let mut npy = Vec::new();
    file.read_to_end(&mut npy)
        .map_err(|error| config_error("pitch_templates", error.to_string()))?;
    if npy.len() < 10 || &npy[..6] != b"\x93NUMPY" || npy[6] != 1 {
        return Err(config_error(
            "pitch_templates",
            "expected a version-1 NumPy array",
        ));
    }
    let header_len = u16::from_le_bytes([npy[8], npy[9]]) as usize;
    let data_start = 10 + header_len;
    let header = std::str::from_utf8(&npy[10..data_start])
        .map_err(|error| config_error("pitch_templates", error.to_string()))?;
    if !header.contains("'<f8'")
        || !header.contains("fortran_order': True")
        || !header.contains("(88, 88)")
        || npy.len() - data_start != PITCHES * PITCHES * 8
    {
        return Err(config_error(
            "pitch_templates",
            "expected an 88-by-88 little-endian f64 matrix",
        ));
    }
    let (chunks, remainder) = npy[data_start..].as_chunks::<8>();
    if !remainder.is_empty() {
        return Err(config_error(
            "pitch_templates",
            "matrix data is not aligned to float64",
        ));
    }
    let values: Vec<f64> = chunks
        .iter()
        .map(|bytes| f64::from_le_bytes(*bytes))
        .collect();
    Ok((0..PITCHES)
        .map(|row| {
            (0..PITCHES)
                .map(|column| values[column * PITCHES + row])
                .collect()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::top_state_index;

    #[test]
    fn tied_top_states_keep_the_first_index_like_numpy_argmax() {
        assert_eq!(top_state_index(&[0.4, 0.1, 0.4, 0.1]), 0);
    }
}
