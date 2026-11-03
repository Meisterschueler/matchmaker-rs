//! Port of `matchmaker.features.audio.LogSpectralEnergyProcessor`.
//!
//! Uses a Hamming-windowed STFT, the reference's linear/log frequency map, positive spectral
//! differences, and optional per-frame normalization.
//!
//! ```
//! use matchmaker_rs::features::{AudioProcessor, lse::LseProcessor};
//! use matchmaker_rs::io::audio::AudioFrame;
//! let mut processor = LseProcessor::new(44_100, 882, Some(2.0));
//! let frame = AudioFrame { index: 0, samples: vec![0.0; 2_048], perf_time: 0.0 };
//! assert_eq!(processor.process(&frame).unwrap().features.len(), 84);
//! ```

use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

use super::{AudioProcessor, Observation};
use crate::io::audio::AudioFrame;

const WINDOW_DURATION: f64 = 2048.0 / 44_100.0;
const REF_FREQ: f64 = 440.0;

/// Log-spectral energy onset features with state across input frames.
pub struct LseProcessor {
    n_fft: usize,
    norm: Option<f64>,
    window: Vec<f32>,
    freq_map: Vec<usize>,
    dim: usize,
    fft: Arc<dyn RealToComplex<f32>>,
    previous_spectrum: Option<Vec<f32>>,
}

impl LseProcessor {
    /// Create an LSE processor; `norm` defaults to L2 in pymatchmaker.
    pub fn new(sample_rate: u32, _hop_length: usize, norm: Option<f64>) -> Self {
        let exponent = (WINDOW_DURATION * f64::from(sample_rate))
            .log2()
            .round_ties_even();
        let n_fft = 2usize.pow(exponent as u32);
        let window: Vec<f32> = (0..n_fft)
            .map(|index| {
                (0.54 - 0.46 * (std::f64::consts::TAU * index as f64 / (n_fft - 1) as f64).cos())
                    as f32
            })
            .collect();
        let df = f64::from(sample_rate) / n_fft as f64;
        let cross = (2.0 / (2.0f64.powf(1.0 / 12.0) - 1.0)) as usize;
        let offset = ((cross as f64 * df / REF_FREQ).log2() * 12.0 + 69.0).round_ties_even();
        let mut freq_map: Vec<usize> = (0..=n_fft / 2).collect();
        for (bin, mapped) in freq_map.iter_mut().enumerate().skip(cross + 1) {
            let midi = ((bin as f64 * df / REF_FREQ).log2() * 12.0 + 69.0).min(127.0);
            *mapped = cross + midi.round_ties_even() as usize - offset as usize;
        }
        let dim = freq_map.last().copied().unwrap_or(0) + 1;
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            n_fft,
            norm,
            window,
            freq_map,
            dim,
            fft: planner.plan_fft_forward(n_fft),
            previous_spectrum: None,
        }
    }

    /// FFT size selected from the reference's 2048-sample-at-44.1-kHz duration.
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }
}

impl AudioProcessor for LseProcessor {
    fn process(&mut self, frame: &AudioFrame) -> Option<Observation> {
        let _span = tracing::debug_span!("feature", processor = "lse").entered();
        let samples = frame.samples.get(..self.n_fft)?;
        let mut input: Vec<f32> = samples
            .iter()
            .zip(&self.window)
            .map(|(sample, window)| sample * window)
            .collect();
        let mut spectrum = vec![realfft::num_complex::Complex32::default(); self.n_fft / 2 + 1];
        self.fft.process(&mut input, &mut spectrum).ok()?;

        let mut current = vec![0.0f32; self.dim];
        for (bin, value) in spectrum.iter().enumerate() {
            current[self.freq_map[bin]] += value.re * value.re + value.im * value.im;
        }
        let mut features: Vec<f32> = current
            .iter()
            .enumerate()
            .map(|(index, &value)| {
                self.previous_spectrum
                    .as_ref()
                    .map_or(value, |previous| (value - previous[index]).max(0.0))
            })
            .collect();
        self.previous_spectrum = Some(current);
        normalize(&mut features, self.norm);
        Some(Observation {
            features,
            perf_time: frame.perf_time,
        })
    }

    fn reset(&mut self) {
        self.previous_spectrum = None;
    }
}

fn normalize(features: &mut [f32], norm: Option<f64>) {
    let Some(norm) = norm else { return };
    let magnitude = if norm.is_infinite() {
        features.iter().map(|value| value.abs()).fold(0.0, f32::max) as f64
    } else {
        features
            .iter()
            .map(|value| f64::from(value.abs()).powf(norm))
            .sum::<f64>()
            .powf(1.0 / norm)
    };
    if magnitude > 0.0 {
        for value in features {
            *value = (f64::from(*value) / magnitude) as f32;
        }
    }
}
