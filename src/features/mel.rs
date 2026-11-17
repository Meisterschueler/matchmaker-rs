//! Port of `matchmaker.features.audio.MelSpectrogramProcessor`.
//!
//! Computes a non-centered, Hann-windowed power spectrum and projects it through the Slaney
//! mel bank used by librosa.

use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

use super::{AudioProcessor, Observation};
use crate::error::{Error, Result};
use crate::io::audio::AudioFrame;

/// Power mel-spectrogram observations.
pub struct MelProcessor {
    sample_rate: u32,
    hop_length: usize,
    n_fft: usize,
    mel_bank: Vec<Vec<f32>>,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl MelProcessor {
    /// Create a mel processor; the reference defaults to 128 bands and `norm=inf`.
    pub fn new(
        sample_rate: u32,
        hop_length: usize,
        n_mels: usize,
        norm: Option<f64>,
    ) -> Result<Self> {
        if sample_rate == 0 {
            return Err(config("sample_rate", "must be at least 1 Hz"));
        }
        if hop_length == 0 {
            return Err(config("hop_length", "must be at least 1 sample"));
        }
        if n_mels == 0 {
            return Err(config("n_mels", "must be at least 1 band"));
        }
        if norm.is_some_and(|norm| norm.is_nan() || norm <= 0.0) {
            return Err(config("norm", "must be positive or None"));
        }
        let n_fft = hop_length
            .checked_mul(2)
            .ok_or_else(|| config("hop_length", "twice the hop length overflows"))?;
        n_mels
            .checked_add(2)
            .ok_or_else(|| config("n_mels", "band count overflows"))?;

        let window = (0..n_fft)
            .map(|index| {
                (0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / n_fft as f64).cos()) as f32
            })
            .collect();
        let mut planner = RealFftPlanner::<f32>::new();
        Ok(Self {
            sample_rate,
            hop_length,
            n_fft,
            mel_bank: make_mel_bank(sample_rate, n_fft, n_mels, norm),
            window,
            fft: planner.plan_fft_forward(n_fft),
        })
    }

    /// FFT size (`2 * hop_length` in the reference).
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    /// Sample rate used to configure the reference processor.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Hop length used to configure the reference processor.
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Filter-bank rows, one per mel band.
    pub fn mel_bank(&self) -> &[Vec<f32>] {
        &self.mel_bank
    }

    /// Unnormalized power spectrum for one non-centered analysis window.
    pub fn power_spectrum(&self, samples: &[f32]) -> Option<Vec<f32>> {
        let samples = samples.get(..self.n_fft)?;
        let mut input: Vec<f32> = samples
            .iter()
            .zip(&self.window)
            .map(|(sample, window)| sample * window)
            .collect();
        let mut spectrum = vec![realfft::num_complex::Complex32::default(); self.n_fft / 2 + 1];
        self.fft.process(&mut input, &mut spectrum).ok()?;
        Some(
            spectrum
                .iter()
                .map(|value| value.re * value.re + value.im * value.im)
                .collect(),
        )
    }
}

impl AudioProcessor for MelProcessor {
    fn process(&mut self, frame: &AudioFrame) -> Option<Observation> {
        let _span = tracing::debug_span!("feature", processor = "mel").entered();
        let power = self.power_spectrum(&frame.samples)?;
        let features = self
            .mel_bank
            .iter()
            .map(|filter| {
                filter
                    .iter()
                    .zip(&power)
                    .map(|(weight, value)| weight * value)
                    .sum()
            })
            .collect();
        Some(Observation {
            features,
            perf_time: frame.perf_time,
        })
    }
}

fn make_mel_bank(
    sample_rate: u32,
    n_fft: usize,
    n_mels: usize,
    norm: Option<f64>,
) -> Vec<Vec<f32>> {
    let mel_frequencies = mel_frequencies(sample_rate, n_mels);
    let fft_bin_width = f64::from(sample_rate) / n_fft as f64;
    let fft_frequencies: Vec<f64> = (0..=n_fft / 2)
        .map(|index| index as f64 * fft_bin_width)
        .collect();

    mel_frequencies
        .windows(3)
        .map(|edges| {
            let (lower, center, upper) = (edges[0], edges[1], edges[2]);
            let mut filter: Vec<f32> = fft_frequencies
                .iter()
                .map(|&frequency| {
                    let lower_slope = (frequency - lower) / (center - lower);
                    let upper_slope = (upper - frequency) / (upper - center);
                    lower_slope.min(upper_slope).max(0.0) as f32
                })
                .collect();
            if let Some(norm) = norm {
                let magnitude = if norm.is_infinite() {
                    filter.iter().map(|value| value.abs()).fold(0.0, f32::max) as f64
                } else {
                    filter
                        .iter()
                        .map(|value| f64::from(value.abs()).powf(norm))
                        .sum::<f64>()
                        .powf(1.0 / norm)
                };
                if magnitude > 0.0 {
                    for value in &mut filter {
                        *value = (f64::from(*value) / magnitude) as f32;
                    }
                }
            }
            filter
        })
        .collect()
}

pub(crate) fn make_slaney_mel_bank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Vec<Vec<f32>> {
    let mel_frequencies = mel_frequencies(sample_rate, n_mels);
    let mut bank = make_mel_bank(sample_rate, n_fft, n_mels, None);
    for (filter, edges) in bank.iter_mut().zip(mel_frequencies.windows(3)) {
        let slaney_scale = 2.0 / (edges[2] - edges[0]);
        for value in filter {
            *value = (f64::from(*value) * slaney_scale) as f32;
        }
    }
    bank
}

fn mel_frequencies(sample_rate: u32, n_mels: usize) -> Vec<f64> {
    let max_mel = hz_to_mel(f64::from(sample_rate) / 2.0);
    let step = max_mel / (n_mels + 1) as f64;
    (0..n_mels + 2)
        .map(|index| mel_to_hz(step * index as f64))
        .collect()
}

fn hz_to_mel(frequency: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    const LOGSTEP: f64 = 0.068_751_777_420_949_12;
    if frequency < MIN_LOG_HZ {
        frequency / F_SP
    } else {
        MIN_LOG_MEL + (frequency / MIN_LOG_HZ).ln() / LOGSTEP
    }
}

fn mel_to_hz(mel: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1_000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP;
    const LOGSTEP: f64 = 0.068_751_777_420_949_12;
    if mel < MIN_LOG_MEL {
        mel * F_SP
    } else {
        MIN_LOG_HZ * (LOGSTEP * (mel - MIN_LOG_MEL)).exp()
    }
}

fn config(key: &str, reason: impl Into<String>) -> Error {
    Error::AudioConfig {
        key: key.into(),
        reason: reason.into(),
    }
}
