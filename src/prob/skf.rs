//! Switching Kalman filter score follower, a port of `matchmaker.prob.skf`.
//!
//! Upstream: `matchmaker/prob/skf.py`, after Jiang and Raphael, "Score Following with Hidden Tempo
//! Using a Switching State-Space Model" (ISMIR 2020).
//!
//! The score is a sequence of chords. A hypothesis is a pair `(chord, age)` with a Gaussian belief
//! about the tempo (seconds per whole note); every audio frame stays in the chord or advances to the
//! next one, weighted by the likelihood of the frame's magnitude spectrum under the chord's
//! spectral template. Hypotheses with equal `(chord, age)` are merged by moment matching and only
//! the most probable [`MAX_HYPOTHESES`] survive.
//!
//! # Template modes
//!
//! The templates decide the quality of the follower (the Gaussian fallback is far less accurate),
//! so the mode is explicit and recorded, never a silent fallback:
//!
//! | [`TemplateMode`] | Templates | Reference behaviour |
//! |---|---|---|
//! | `Synthesized` | averaged magnitude spectra of each chord rendered with a SoundFont ([`synthesize_templates`]) | `_build_synthesized_templates` (FluidSynth) |
//! | `Gaussian` | harmonic Gaussian peaks ([`gaussian_templates`]) | `_build_gaussian_templates`, the reference's fallback after any synthesis error |
//!
//! RustySynth renders at 16 kHz or more, whereas the reference default is 8 kHz. The templates are
//! therefore analysed on a render at the smallest integer multiple of the sample rate that is at
//! least 16 kHz, with a proportionally larger FFT, keeping the lowest `n_fft / 2 + 1` bins: the
//! same frequency grid as the reference without its aliasing.
//!
//! ```
//! use matchmaker_rs::base::OnlineAlignment;
//! use matchmaker_rs::features::Observation;
//! use matchmaker_rs::prob::skf::{
//!     SkfConfig, SkfFollower, SkfScoreModel, TemplateMode, gaussian_templates,
//! };
//! let chords = vec![vec![60], vec![64]];
//! let templates = gaussian_templates(&chords, 512, 8_000, 5);
//! let model = SkfScoreModel::new(
//!     chords, vec![0.25, 0.25], vec![0.0, 1.0], templates, TemplateMode::Gaussian, None,
//! )?;
//! let mut follower = SkfFollower::new(model, SkfConfig::default())?;
//! let beat = follower.call(&Observation { features: vec![1.0; 257], perf_time: 0.0 });
//! assert!((0.0..=1.0).contains(&beat));
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::collections::HashMap;
use std::f64::consts::{FRAC_1_SQRT_2, TAU};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use realfft::RealFftPlanner;
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};

use crate::base::{AlignmentPath, OnlineAlignment, SkfStepDebug, StepDebug};
use crate::error::{Error, Result};
use crate::features::Observation;
use crate::score::ScoreNote;

/// Reference default sample rate.
pub const DEFAULT_SAMPLE_RATE: u32 = 8_000;
/// Reference default FFT size (paper: 512 samples, 64 ms at 8 kHz).
pub const DEFAULT_N_FFT: usize = 512;
/// Reference default hop length (paper: 128 samples, 16 ms at 8 kHz).
pub const DEFAULT_HOP_LENGTH: usize = 128;
/// Onset jitter: `sigma_eps = scale * t_k`.
pub const SIGMA_EPS_SCALE: f64 = 0.05;
/// Tempo drift: `sigma_eta = scale * l_k * t_k`.
pub const SIGMA_ETA_SCALE: f64 = 0.01;
/// Beam width: hypotheses kept after every frame.
pub const MAX_HYPOTHESES: usize = 200;

const PROB_FLOOR: f64 = 1e-300;
const MIN_RENDER_RATE: u32 = 16_000;
const MAX_RENDER_RATE: u32 = 192_000;
/// Gaussian templates: harmonics per pitch.
const GAUSSIAN_HARMONICS: usize = 5;
/// Synthesized templates: sounding time, silence between chords, MIDI velocity, skipped attack.
const NOTE_DURATION: f64 = 0.3;
const CHORD_GAP: f64 = 0.1;
const NOTE_VELOCITY: i32 = 100;
const ATTACK_SKIP: f64 = 0.02;
/// The reference analyses templates with this hop whatever the follower's hop is.
const ANALYSIS_HOP: usize = 128;

/// How the spectral templates of a score were produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TemplateMode {
    /// Chords rendered with a SoundFont and averaged into magnitude spectra.
    Synthesized,
    /// Harmonic Gaussian peaks, the reference's fallback.
    Gaussian,
}

impl TemplateMode {
    /// Names accepted by [`FromStr`].
    pub const AVAILABLE: [&'static str; 2] = ["synthesized", "gaussian"];

    /// Name used in configuration and exports.
    pub fn name(self) -> &'static str {
        match self {
            TemplateMode::Synthesized => "synthesized",
            TemplateMode::Gaussian => "gaussian",
        }
    }
}

impl FromStr for TemplateMode {
    type Err = Error;

    fn from_str(name: &str) -> Result<Self> {
        match name {
            "synthesized" => Ok(TemplateMode::Synthesized),
            "gaussian" => Ok(TemplateMode::Gaussian),
            _ => Err(config_error(
                "template_mode",
                format!(
                    "unknown template mode {name:?}; available: {:?}",
                    TemplateMode::AVAILABLE
                ),
            )),
        }
    }
}

/// Chord sequence of a score (`build_chord_sequence`).
#[derive(Debug, Clone, PartialEq)]
pub struct ChordSequence {
    /// MIDI pitches of each chord, in score order.
    pub chords: Vec<Vec<i32>>,
    /// Nominal length of each chord in whole notes.
    pub lengths: Vec<f64>,
    /// Score onset of each chord in beats.
    pub onset_beats: Vec<f64>,
}

/// Group the notes of a score into chords of equal `onset_beat` (`build_chord_sequence`).
///
/// Lengths follow the reference's float32 arithmetic: `(onset_quarter[k + 1] - onset_quarter[k]) / 4`
/// of each chord's first note, the last chord repeats its predecessor's length (0.25 for a single
/// chord), and every length is at least `1e-6`.
pub fn build_chord_sequence(notes: &[ScoreNote]) -> Result<ChordSequence> {
    if notes.is_empty() {
        return Err(config_error("score", "must contain at least one note"));
    }
    let mut order: Vec<usize> = (0..notes.len()).collect();
    // A stable sort keeps the first note of an onset in front, like `np.unique(return_index=True)`.
    order.sort_by(|&a, &b| notes[a].onset_beat.total_cmp(&notes[b].onset_beat));
    let mut chords: Vec<Vec<i32>> = Vec::new();
    let mut onset_beats = Vec::new();
    let mut onset_quarters = Vec::new();
    let mut previous = f32::NAN;
    for &index in &order {
        let note = &notes[index];
        if chords.is_empty() || note.onset_beat != previous {
            chords.push(Vec::new());
            onset_beats.push(f64::from(note.onset_beat));
            onset_quarters.push(note.onset_quarter);
            previous = note.onset_beat;
        }
        chords
            .last_mut()
            .expect("a chord was just pushed")
            .push(note.pitch);
    }

    let mut lengths: Vec<f64> = onset_quarters
        .windows(2)
        .map(|pair| f64::from((pair[1] - pair[0]) / 4.0))
        .collect();
    lengths.push(lengths.last().copied().unwrap_or(0.25));
    for length in &mut lengths {
        *length = length.max(1e-6);
    }
    Ok(ChordSequence {
        chords,
        lengths,
        onset_beats,
    })
}

/// Harmonic Gaussian-peak templates (`_build_gaussian_templates`), one row per chord that sums to 1.
pub fn gaussian_templates(
    chords: &[Vec<i32>],
    n_fft: usize,
    sample_rate: u32,
    n_harmonics: usize,
) -> Vec<Vec<f64>> {
    let n_bins = n_fft / 2 + 1;
    let nyquist = f64::from(sample_rate) / 2.0;
    // `np.linspace(0, nyquist, n_bins)`: the last point is exactly the stop value.
    let step = nyquist / (n_bins as f64 - 1.0);
    let freqs: Vec<f64> = (0..n_bins)
        .map(|bin| {
            if bin + 1 == n_bins && n_bins > 1 {
                nyquist
            } else {
                bin as f64 * step
            }
        })
        .collect();
    let sigma_hz = 1.7 * (f64::from(sample_rate) / n_fft as f64);
    chords
        .iter()
        .map(|chord| {
            let mut row = vec![1e-10; n_bins];
            for &pitch in chord {
                let f0 = 440.0 * 2.0f64.powf((f64::from(pitch) - 69.0) / 12.0);
                for harmonic in 1..=n_harmonics {
                    let fh = f0 * harmonic as f64;
                    if fh >= nyquist {
                        break;
                    }
                    let amplitude = 1.0 / harmonic as f64;
                    for (value, freq) in row.iter_mut().zip(&freqs) {
                        let z = (freq - fh) / sigma_hz;
                        *value += amplitude * (-0.5 * (z * z)).exp();
                    }
                }
            }
            normalize_row(&mut row);
            row
        })
        .collect()
}

/// Chord templates averaged from a SoundFont rendering (`_build_synthesized_templates`).
///
/// Every chord sounds for 0.3 s at velocity 100 and is followed by 0.1 s of silence; the Hann
/// windowed magnitude spectra between 20 ms and the end of the chord are averaged, floored at
/// `1e-10` and normalized to sum 1. See the module docs for the render rate.
///
/// # Errors
///
/// [`Error::Io`] and [`Error::AudioRender`] for an unreadable SoundFont and
/// [`Error::AudioConfig`] for an empty chord list, a pitch outside 0..=127, or a sample rate
/// RustySynth cannot render at a multiple of.
pub fn synthesize_templates(
    chords: &[Vec<i32>],
    soundfont: impl AsRef<Path>,
    n_fft: usize,
    sample_rate: u32,
) -> Result<Vec<Vec<f64>>> {
    let path = soundfont.as_ref();
    if chords.is_empty() || n_fft == 0 || sample_rate == 0 {
        return Err(config_error(
            "templates",
            "need chords, a positive n_fft and a positive sample_rate",
        ));
    }
    if chords
        .iter()
        .flatten()
        .any(|pitch| !(0..=127).contains(pitch))
    {
        return Err(config_error("score", "pitches must be MIDI notes 0..=127"));
    }
    let multiple = MIN_RENDER_RATE.div_ceil(sample_rate).max(1);
    let render_rate = sample_rate.saturating_mul(multiple);
    if render_rate > MAX_RENDER_RATE {
        return Err(config_error(
            "sample_rate",
            "RustySynth supports rates up to 192000 Hz",
        ));
    }
    let _span = tracing::debug_span!("skf_templates", chords = chords.len(), render_rate).entered();

    let mut file = File::open(path).map_err(|error| Error::io(path, error))?;
    let render_error = |message: String| Error::AudioRender {
        path: path.to_path_buf(),
        message,
    };
    let soundfont =
        Arc::new(SoundFont::new(&mut file).map_err(|error| render_error(error.to_string()))?);
    let settings = SynthesizerSettings::new(render_rate as i32);
    let mut synthesizer =
        Synthesizer::new(&soundfont, &settings).map_err(|error| render_error(error.to_string()))?;
    // FluidSynth starts a channel with reverb send 0, RustySynth with 40.
    synthesizer.process_midi_message(0, 0xB0, 91, 0);

    let rate = f64::from(render_rate);
    let chord_interval = NOTE_DURATION + CHORD_GAP;
    let mut events: Vec<(usize, bool, i32)> = Vec::new();
    for (k, chord) in chords.iter().enumerate() {
        let on = (k as f64 * chord_interval * rate).round() as usize;
        let off = ((k as f64 * chord_interval + NOTE_DURATION) * rate).round() as usize;
        for &pitch in chord {
            events.push((on, true, pitch));
            events.push((off, false, pitch));
        }
    }
    events.sort_by_key(|event| event.0);
    let total = events.last().map_or(0, |event| event.0);
    let mut audio: Vec<f32> = Vec::with_capacity(total);
    let mut left = vec![0.0; settings.block_size];
    let mut right = vec![0.0; settings.block_size];
    let mut render_until = |synthesizer: &mut Synthesizer, end: usize, audio: &mut Vec<f32>| {
        while audio.len() < end {
            let frames = (end - audio.len()).min(left.len());
            synthesizer.render(&mut left[..frames], &mut right[..frames]);
            audio.extend_from_slice(&left[..frames]);
        }
    };
    for (sample, is_on, pitch) in events {
        render_until(&mut synthesizer, sample, &mut audio);
        if is_on {
            synthesizer.process_midi_message(0, 0x90, pitch, NOTE_VELOCITY);
        } else {
            synthesizer.process_midi_message(0, 0x80, pitch, 0);
        }
    }

    let fft_size = n_fft * multiple as usize;
    let hop = ANALYSIS_HOP * multiple as usize;
    let keep = n_fft / 2 + 1;
    let window: Vec<f64> = (0..fft_size)
        .map(|index| 0.5 - 0.5 * (TAU * index as f64 / fft_size as f64).cos())
        .collect();
    let fft = RealFftPlanner::<f64>::new().plan_fft_forward(fft_size);
    let mut input = vec![0.0; fft_size];
    let mut spectrum = fft.make_output_vec();
    let mut templates = Vec::with_capacity(chords.len());
    for k in 0..chords.len() {
        let start = ((k as f64 * chord_interval + ATTACK_SKIP) * rate) as usize;
        let end = ((k as f64 * chord_interval + NOTE_DURATION) * rate) as usize;
        let mut sum = vec![0.0; keep];
        let mut count = 0usize;
        let mut position = start;
        while position + fft_size <= end && position + fft_size <= audio.len() {
            for ((slot, sample), weight) in input.iter_mut().zip(&audio[position..]).zip(&window) {
                *slot = f64::from(*sample) * weight;
            }
            fft.process(&mut input, &mut spectrum)
                .map_err(|error| render_error(error.to_string()))?;
            for (total, value) in sum.iter_mut().zip(&spectrum) {
                *total += value.re.hypot(value.im);
            }
            count += 1;
            position += hop;
        }
        let mut row: Vec<f64> = if count > 0 {
            sum.iter().map(|value| value / count as f64).collect()
        } else {
            vec![1.0; keep]
        };
        for value in &mut row {
            *value = value.max(1e-10);
        }
        normalize_row(&mut row);
        templates.push(row);
    }
    Ok(templates)
}

fn normalize_row(row: &mut [f64]) {
    let sum = row.iter().sum::<f64>().max(1e-20);
    for value in row {
        *value /= sum;
    }
}

/// Score side of the follower: chords, their lengths and onsets, and one template per chord.
#[derive(Debug, Clone, PartialEq)]
pub struct SkfScoreModel {
    /// MIDI pitches of each chord.
    pub chords: Vec<Vec<i32>>,
    /// Nominal length of each chord in whole notes.
    pub lengths: Vec<f64>,
    /// Score onset of each chord in beats.
    pub onset_beats: Vec<f64>,
    /// Spectral template of each chord: `n_fft / 2 + 1` bins summing to 1.
    pub templates: Vec<Vec<f64>>,
    /// How the templates were produced.
    pub mode: TemplateMode,
    /// SoundFont of synthesized templates, kept for provenance.
    pub soundfont: Option<PathBuf>,
}

impl SkfScoreModel {
    /// Check that all parts describe the same chords.
    pub fn new(
        chords: Vec<Vec<i32>>,
        lengths: Vec<f64>,
        onset_beats: Vec<f64>,
        templates: Vec<Vec<f64>>,
        mode: TemplateMode,
        soundfont: Option<PathBuf>,
    ) -> Result<Self> {
        let count = chords.len();
        if count == 0 {
            return Err(config_error("chords", "must contain at least one chord"));
        }
        if lengths.len() != count || onset_beats.len() != count || templates.len() != count {
            return Err(config_error(
                "model",
                "chords, lengths, onset_beats and templates must have one entry per chord",
            ));
        }
        if lengths
            .iter()
            .any(|length| !length.is_finite() || *length <= 0.0)
            || onset_beats.iter().any(|beat| !beat.is_finite())
            || onset_beats.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(config_error(
                "model",
                "lengths must be positive and onset_beats finite and strictly ascending",
            ));
        }
        let width = templates[0].len();
        if width == 0
            || templates.iter().any(|row| {
                row.len() != width || row.iter().any(|value| !value.is_finite() || *value < 0.0)
            })
        {
            return Err(config_error(
                "templates",
                "rows must share one width and hold finite non-negative values",
            ));
        }
        Ok(Self {
            chords,
            lengths,
            onset_beats,
            templates,
            mode,
            soundfont,
        })
    }

    /// Build the model of a score: its chord sequence and templates in the requested mode.
    ///
    /// `soundfont` is required for [`TemplateMode::Synthesized`] and ignored for the Gaussian mode.
    pub fn from_notes(
        notes: &[ScoreNote],
        n_fft: usize,
        sample_rate: u32,
        mode: TemplateMode,
        soundfont: Option<&Path>,
    ) -> Result<Self> {
        let sequence = build_chord_sequence(notes)?;
        let (templates, soundfont) = match mode {
            TemplateMode::Gaussian => (
                gaussian_templates(&sequence.chords, n_fft, sample_rate, GAUSSIAN_HARMONICS),
                None,
            ),
            TemplateMode::Synthesized => {
                let path = soundfont.ok_or_else(|| {
                    config_error("soundfont_path", "is required for synthesized templates")
                })?;
                (
                    synthesize_templates(&sequence.chords, path, n_fft, sample_rate)?,
                    Some(path.to_path_buf()),
                )
            }
        };
        Self::new(
            sequence.chords,
            sequence.lengths,
            sequence.onset_beats,
            templates,
            mode,
            soundfont,
        )
    }
}

/// Parameters of the follower (the reference's constructor arguments).
#[derive(Debug, Clone, PartialEq)]
pub struct SkfConfig {
    /// Initial tempo in quarter notes per minute.
    pub tempo: f64,
    /// Sample rate of the observations.
    pub sample_rate: u32,
    /// FFT size of the observations; templates have `n_fft / 2 + 1` bins.
    pub n_fft: usize,
    /// Hop between observations in samples.
    pub hop_length: usize,
    /// Beam width.
    pub max_hypotheses: usize,
    /// Scale of the onset jitter.
    pub sigma_eps_scale: f64,
    /// Scale of the tempo drift.
    pub sigma_eta_scale: f64,
}

impl Default for SkfConfig {
    fn default() -> Self {
        Self {
            tempo: 120.0,
            sample_rate: DEFAULT_SAMPLE_RATE,
            n_fft: DEFAULT_N_FFT,
            hop_length: DEFAULT_HOP_LENGTH,
            max_hypotheses: MAX_HYPOTHESES,
            sigma_eps_scale: SIGMA_EPS_SCALE,
            sigma_eta_scale: SIGMA_ETA_SCALE,
        }
    }
}

/// One `(chord, age)` hypothesis with its Gaussian tempo belief.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SkfHypothesis {
    /// Chord index.
    pub chord: usize,
    /// Number of frames the chord has lasted.
    pub age: usize,
    /// Filtered probability of the hypothesis.
    pub probability: f64,
    /// Mean of the tempo belief in seconds per whole note.
    pub tempo_mean: f64,
    /// Variance of the tempo belief.
    pub tempo_variance: f64,
}

/// Audio score follower with a switching state-space model and hidden tempo.
///
/// Observations are raw-spectrum vectors of `n_fft / 2 + 1` bins, which the audio facade
/// guarantees. The reference also accepts raw audio frames and transforms them itself; this port
/// does not, and a vector of another length is compared on the bins both sides have.
#[derive(Debug, Clone)]
pub struct SkfFollower {
    model: SkfModel,
    config: SkfConfig,
    delta: f64,
    init_tempo: f64,
    init_tempo_variance: f64,
    hypotheses: Vec<SkfHypothesis>,
    candidates: usize,
    current_index: usize,
    path: AlignmentPath,
}

/// The model with the derived values the recurrence needs.
#[derive(Debug, Clone)]
struct SkfModel {
    lengths: Vec<f64>,
    onset_beats: Vec<f64>,
    positions: Vec<f32>,
    log_templates: Vec<Vec<f64>>,
}

impl SkfFollower {
    /// Start in chord 0 with age 1 and the tempo belief of the configuration.
    pub fn new(model: SkfScoreModel, config: SkfConfig) -> Result<Self> {
        if !config.tempo.is_finite() || config.tempo <= 0.0 {
            return Err(config_error("tempo", "must be a positive number of BPM"));
        }
        if config.sample_rate == 0 || config.hop_length == 0 {
            return Err(config_error("sample_rate/hop_length", "must be at least 1"));
        }
        if config.max_hypotheses == 0 {
            return Err(config_error("max_hypotheses", "must be at least 1"));
        }
        if [config.sigma_eps_scale, config.sigma_eta_scale]
            .iter()
            .any(|scale| !scale.is_finite() || *scale < 0.0)
        {
            return Err(config_error(
                "sigma_eps_scale/sigma_eta_scale",
                "must be finite and not negative",
            ));
        }
        let bins = config.n_fft / 2 + 1;
        if model.templates[0].len() != bins {
            return Err(config_error(
                "n_fft",
                format!(
                    "templates have {} bins, n_fft {} needs {bins}",
                    model.templates[0].len(),
                    config.n_fft
                ),
            ));
        }
        let init_tempo = 240.0 / config.tempo;
        let init_tempo_variance = (init_tempo * 0.1) * (init_tempo * 0.1);
        Ok(Self {
            delta: config.hop_length as f64 / f64::from(config.sample_rate),
            init_tempo,
            init_tempo_variance,
            hypotheses: vec![SkfHypothesis {
                chord: 0,
                age: 1,
                probability: 1.0,
                tempo_mean: init_tempo,
                tempo_variance: init_tempo_variance,
            }],
            candidates: 1,
            current_index: 0,
            path: AlignmentPath::default(),
            model: SkfModel {
                positions: model.onset_beats.iter().map(|&beat| beat as f32).collect(),
                log_templates: model
                    .templates
                    .iter()
                    .map(|row| row.iter().map(|value| value.max(1e-300).ln()).collect())
                    .collect(),
                lengths: model.lengths,
                onset_beats: model.onset_beats,
            },
            config,
        })
    }

    /// Active hypotheses in beam order.
    pub fn hypotheses(&self) -> &[SkfHypothesis] {
        &self.hypotheses
    }

    /// Chord the follower currently believes in.
    pub fn current_chord(&self) -> usize {
        self.current_index
    }

    fn sigma_eps(&self, tempo: f64) -> f64 {
        self.config.sigma_eps_scale * tempo.abs()
    }

    /// Probability that a chord of the given age lasts one more frame (Eq. 12).
    fn stay_probability(&self, age: usize, length: f64, mean: f64, variance: f64) -> f64 {
        let mean_length = length * mean;
        let sigma_eps = self.sigma_eps(mean);
        let mut length_variance = sigma_eps * sigma_eps + (length * length) * variance;
        if length_variance <= 0.0 {
            length_variance = 1e-12;
        }
        let sigma = length_variance.sqrt();
        let current = age as f64 * self.delta;
        let next = (age + 1) as f64 * self.delta;
        let denominator = 1.0 - ndtr((current - mean_length) / sigma);
        if denominator < PROB_FLOOR {
            return 0.0;
        }
        let numerator = 1.0 - ndtr((next - mean_length) / sigma);
        (numerator / denominator).clamp(0.0, 1.0)
    }

    /// Kalman update of the tempo belief when chord `chord` ends after `age` frames (Eq. 2-3).
    fn kalman_update(&self, mean: f64, variance: f64, chord: usize, age: usize) -> (f64, f64) {
        let length = self.model.lengths[chord];
        let observed = age as f64 * self.delta;
        let sigma_eps = self.sigma_eps(mean);
        let observation_variance = sigma_eps * sigma_eps;
        let (posterior_mean, posterior_variance) = if length < 1e-10 {
            (mean, variance)
        } else {
            let innovation = observed - length * mean;
            let innovation_variance =
                ((length * length) * variance + observation_variance).max(1e-20);
            let gain = (length * variance) / innovation_variance;
            (
                mean + gain * innovation,
                variance - gain * length * variance,
            )
        };
        let sigma_eta = self.config.sigma_eta_scale * (length * posterior_mean).abs();
        (
            posterior_mean,
            posterior_variance.max(0.0) + sigma_eta * sigma_eta,
        )
    }

    /// Likelihood of the spectrum under every chord template, scaled so the best chord is 1 (Eq. 1).
    fn likelihoods(&self, spectrum: &[f64]) -> Vec<f64> {
        let count = self.model.log_templates.len();
        let sum: f64 = spectrum.iter().sum();
        if sum <= 0.0 {
            return vec![1e-300; count];
        }
        let mut log_likelihoods: Vec<f64> = self
            .model
            .log_templates
            .iter()
            .map(|row| {
                row.iter()
                    .zip(spectrum)
                    .map(|(log, y)| log * (y / sum))
                    .sum()
            })
            .collect();
        // NumPy's max and maximum propagate NaN (a NaN spectrum ends in the reference's reset).
        let best = log_likelihoods
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, |a, b| {
                if a.is_nan() || b.is_nan() {
                    f64::NAN
                } else {
                    a.max(b)
                }
            });
        for value in &mut log_likelihoods {
            let likelihood = (*value - best).clamp(-500.0, 0.0).exp();
            *value = if likelihood < 1e-300 {
                1e-300
            } else {
                likelihood
            };
        }
        log_likelihoods
    }

    fn merge(
        beam: &mut Vec<SkfHypothesis>,
        index: &mut HashMap<(usize, usize), usize>,
        next: SkfHypothesis,
    ) {
        match index.get(&(next.chord, next.age)) {
            Some(&slot) => {
                let old = &mut beam[slot];
                let total = old.probability + next.probability;
                let w_old = old.probability / total;
                let w_new = next.probability / total;
                let mean = w_old * old.tempo_mean + w_new * next.tempo_mean;
                let variance = w_old * (old.tempo_variance + old.tempo_mean * old.tempo_mean)
                    + w_new * (next.tempo_variance + next.tempo_mean * next.tempo_mean)
                    - mean * mean;
                old.probability = total;
                old.tempo_mean = mean;
                old.tempo_variance = variance.max(1e-20);
            }
            None => {
                index.insert((next.chord, next.age), beam.len());
                beam.push(next);
            }
        }
    }

    fn step(&mut self, features: &[f32]) {
        let spectrum: Vec<f64> = features
            .iter()
            .map(|&value| {
                let value = f64::from(value);
                if value < 0.0 { 0.0 } else { value }
            })
            .collect();
        let likelihoods = self.likelihoods(&spectrum);
        let chords = self.model.lengths.len();

        let mut beam: Vec<SkfHypothesis> = Vec::with_capacity(2 * self.hypotheses.len());
        let mut index = HashMap::new();
        for hypothesis in &self.hypotheses {
            let SkfHypothesis {
                chord: k,
                age,
                probability,
                tempo_mean,
                tempo_variance,
            } = *hypothesis;
            let stay =
                self.stay_probability(age, self.model.lengths[k], tempo_mean, tempo_variance);
            let stay_probability = probability * stay * likelihoods[k];
            if stay_probability > PROB_FLOOR {
                Self::merge(
                    &mut beam,
                    &mut index,
                    SkfHypothesis {
                        chord: k,
                        age: age + 1,
                        probability: stay_probability,
                        tempo_mean,
                        tempo_variance,
                    },
                );
            }
            if k + 1 < chords {
                let (mean, variance) = self.kalman_update(tempo_mean, tempo_variance, k, age);
                let advance_probability = probability * (1.0 - stay) * likelihoods[k + 1];
                if advance_probability > PROB_FLOOR {
                    Self::merge(
                        &mut beam,
                        &mut index,
                        SkfHypothesis {
                            chord: k + 1,
                            age: 1,
                            probability: advance_probability,
                            tempo_mean: mean,
                            tempo_variance: variance,
                        },
                    );
                }
            }
        }
        self.candidates = beam.len();
        if beam.len() > self.config.max_hypotheses {
            // The stable descending sort is the reference's `sorted(..., reverse=True)`; the
            // survivors keep this order for the next frame.
            beam.sort_by(|a, b| b.probability.total_cmp(&a.probability));
            beam.truncate(self.config.max_hypotheses);
        }
        let total: f64 = beam.iter().map(|hypothesis| hypothesis.probability).sum();
        if total > 0.0 {
            for hypothesis in &mut beam {
                hypothesis.probability /= total;
            }
            self.hypotheses = beam;
        } else {
            self.hypotheses = vec![SkfHypothesis {
                chord: self.current_index,
                age: 1,
                probability: 1.0,
                tempo_mean: self.init_tempo,
                tempo_variance: self.init_tempo_variance,
            }];
        }
    }

    /// Position of the most probable chord, interpolated by its expected age.
    fn position(&mut self) -> f64 {
        let chords = self.model.lengths.len();
        let mut probability = vec![0.0; chords];
        let mut weighted_age = vec![0.0; chords];
        let mut weighted_mean = vec![0.0; chords];
        for hypothesis in &self.hypotheses {
            probability[hypothesis.chord] += hypothesis.probability;
            weighted_age[hypothesis.chord] += hypothesis.probability * hypothesis.age as f64;
            weighted_mean[hypothesis.chord] += hypothesis.probability * hypothesis.tempo_mean;
        }
        // `np.argmax`: the first maximum wins.
        let mut best = 0;
        for (chord, value) in probability.iter().enumerate() {
            if *value > probability[best] {
                best = chord;
            }
        }
        self.current_index = best;
        let base = self.model.onset_beats[best];
        if probability[best] > 0.0 && best + 1 < chords {
            let average_age = weighted_age[best] / probability[best];
            let average_mean = weighted_mean[best] / probability[best];
            let elapsed = (average_age - 1.0) * self.delta;
            let expected = self.model.lengths[best] * average_mean;
            if expected > 0.0 {
                let beats = self.model.onset_beats[best + 1] - base;
                return base + (elapsed / expected).min(1.0) * beats;
            }
        }
        base
    }
}

impl OnlineAlignment for SkfFollower {
    fn call(&mut self, observation: &Observation) -> f64 {
        let _span = tracing::debug_span!("inference", method = "skf").entered();
        self.step(&observation.features);
        let position = self.position();
        self.path.push(observation.perf_time, position);
        tracing::debug!(perf_time = observation.perf_time, position);
        position
    }

    fn is_still_following(&self) -> bool {
        self.current_index + 1 < self.model.positions.len()
    }

    fn alignment_path(&self) -> &AlignmentPath {
        &self.path
    }

    fn score_positions(&self) -> &[f32] {
        &self.model.positions
    }

    fn debug_state(&self) -> StepDebug {
        StepDebug {
            current_index: self.current_index,
            skf: Some(SkfStepDebug {
                chords: self.hypotheses.iter().map(|h| h.chord).collect(),
                ages: self.hypotheses.iter().map(|h| h.age).collect(),
                probabilities: self.hypotheses.iter().map(|h| h.probability).collect(),
                tempo_means: self.hypotheses.iter().map(|h| h.tempo_mean).collect(),
                tempo_variances: self.hypotheses.iter().map(|h| h.tempo_variance).collect(),
                candidates: self.candidates,
            }),
            ..StepDebug::default()
        }
    }
}

/// `scipy.special.ndtr`: the standard normal CDF, evaluated like Cephes (`erf` near zero, `erfc`
/// in the tails) so that `1 - ndtr(z)` cancels exactly as in the reference.
fn ndtr(value: f64) -> f64 {
    let x = value * FRAC_1_SQRT_2;
    let z = x.abs();
    if z < FRAC_1_SQRT_2 {
        0.5 + 0.5 * libm::erf(x)
    } else {
        let tail = 0.5 * libm::erfc(z);
        if x > 0.0 { 1.0 - tail } else { tail }
    }
}

fn config_error(key: &str, reason: impl Into<String>) -> Error {
    Error::AudioConfig {
        key: key.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn note(pitch: i32, beat: f32, quarter: f32) -> ScoreNote {
        ScoreNote {
            id: format!("n{pitch}-{beat}"),
            pitch,
            onset_beat: beat,
            duration_beat: 1.0,
            onset_quarter: quarter,
            duration_quarter: 1.0,
            onset_div: (quarter * 480.0) as i64,
            duration_div: 480,
            voice: 1,
        }
    }

    #[test]
    fn ndtr_matches_known_values_in_both_branches() {
        assert_eq!(ndtr(0.0), 0.5);
        assert!((ndtr(1.0) - 0.841_344_746_068_542_9).abs() < 1e-15);
        assert!((ndtr(-1.0) - 0.158_655_253_931_457_05).abs() < 1e-15);
        assert!((ndtr(3.0) - 0.998_650_101_968_369_9).abs() < 1e-15);
        assert_eq!(
            ndtr(9.0),
            1.0,
            "the upper tail rounds to exactly one, like Cephes"
        );
    }

    #[test]
    fn chord_sequence_groups_onsets_and_uses_float32_lengths() {
        let notes = [
            note(67, 1.0, 1.0),
            note(60, 0.0, 0.0),
            note(64, 0.0, 0.0),
            note(72, 3.0, 3.0),
        ];
        let sequence = build_chord_sequence(&notes).unwrap();
        assert_eq!(sequence.onset_beats, [0.0, 1.0, 3.0]);
        assert_eq!(sequence.chords, [vec![60, 64], vec![67], vec![72]]);
        assert_eq!(sequence.lengths, [0.25, 0.5, 0.5]);
        let single = build_chord_sequence(&notes[..1]).unwrap();
        assert_eq!(single.lengths, [0.25]);
        assert!(build_chord_sequence(&[]).is_err());
    }

    #[test]
    fn gaussian_templates_are_normalized_rows_with_peaks_at_the_harmonics() {
        let templates = gaussian_templates(&[vec![69]], 512, 8_000, 5);
        let row = &templates[0];
        assert_eq!(row.len(), 257);
        assert!((row.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        let peak = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(peak, 28, "440 Hz at 15.625 Hz per bin");
    }

    #[test]
    fn follower_advances_through_matching_chords_and_stops_at_the_last() {
        let chords = vec![vec![60], vec![64], vec![67]];
        let templates = gaussian_templates(&chords, 512, 8_000, 5);
        let model = SkfScoreModel::new(
            chords,
            vec![0.25; 3],
            vec![0.0, 1.0, 2.0],
            templates.clone(),
            TemplateMode::Gaussian,
            None,
        )
        .unwrap();
        let mut follower = SkfFollower::new(model, SkfConfig::default()).unwrap();
        let mut last = 0.0;
        'outer: for (chord, template) in templates.iter().enumerate() {
            for frame in 0..40 {
                let features: Vec<f32> = template.iter().map(|&v| v as f32).collect();
                last = follower.call(&Observation {
                    features,
                    perf_time: (chord * 40 + frame) as f64 * 0.016,
                });
                if !follower.is_still_following() {
                    break 'outer;
                }
            }
        }
        assert_eq!(follower.current_chord(), 2);
        assert!(last >= 2.0 - 1e-9);
        assert!(!follower.is_still_following());
        let total: f64 = follower.hypotheses().iter().map(|h| h.probability).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn silent_and_degenerate_frames_reset_instead_of_producing_nan() {
        let chords = vec![vec![60], vec![64]];
        let templates = gaussian_templates(&chords, 512, 8_000, 5);
        let model = SkfScoreModel::new(
            chords,
            vec![0.25; 2],
            vec![0.0, 1.0],
            templates,
            TemplateMode::Gaussian,
            None,
        )
        .unwrap();
        let mut follower = SkfFollower::new(model, SkfConfig::default()).unwrap();
        let silent = follower.call(&Observation {
            features: vec![0.0; 257],
            perf_time: 0.0,
        });
        assert!(silent.is_finite());
        let nan = follower.call(&Observation {
            features: vec![f32::NAN; 257],
            perf_time: 0.016,
        });
        assert!(nan.is_finite());
        assert_eq!(follower.hypotheses().len(), 1);
    }

    #[test]
    fn invalid_models_and_configs_are_rejected() {
        assert!(
            SkfScoreModel::new(vec![], vec![], vec![], vec![], TemplateMode::Gaussian, None)
                .is_err()
        );
        let row = vec![0.5; 4];
        let model = SkfScoreModel::new(
            vec![vec![60]],
            vec![0.25],
            vec![0.0],
            vec![row],
            TemplateMode::Gaussian,
            None,
        )
        .unwrap();
        let error = SkfFollower::new(model.clone(), SkfConfig::default()).unwrap_err();
        assert!(error.to_string().contains("n_fft"));
        let config = SkfConfig {
            n_fft: 6,
            tempo: 0.0,
            ..SkfConfig::default()
        };
        assert!(SkfFollower::new(model, config).is_err());
        assert!("synthesized".parse::<TemplateMode>().is_ok());
        assert!("auto".parse::<TemplateMode>().is_err());
    }
}
