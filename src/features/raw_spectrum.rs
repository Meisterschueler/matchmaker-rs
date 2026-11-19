//! Port of `matchmaker.features.audio.RawSpectrumProcessor`.
//!
//! Applies a non-centered Hann-windowed STFT and emits magnitude bins.
//!
//! ```
//! use matchmaker_rs::features::{AudioProcessor, raw_spectrum::RawSpectrumProcessor};
//! use matchmaker_rs::io::audio::AudioFrame;
//! let mut processor = RawSpectrumProcessor::new(8_000, 128, 512);
//! let frame = AudioFrame { index: 0, samples: vec![0.0; 512], perf_time: 0.0 };
//! assert_eq!(processor.process(&frame).unwrap().features.len(), 257);
//! ```

use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

use super::{AudioProcessor, Observation};
use crate::io::audio::AudioFrame;

/// Magnitude spectrum from a non-centered, Hann-windowed STFT.
pub struct RawSpectrumProcessor {
    sample_rate: u32,
    hop_length: usize,
    n_fft: usize,
    window: Vec<f32>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl RawSpectrumProcessor {
    /// Create a processor with the reference defaults of 8000 Hz, hop 128, and FFT size 512.
    pub fn new(sample_rate: u32, hop_length: usize, n_fft: usize) -> Self {
        let window = (0..n_fft)
            .map(|index| {
                (0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / n_fft as f64).cos()) as f32
            })
            .collect();
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            sample_rate,
            hop_length,
            n_fft,
            window,
            fft: planner.plan_fft_forward(n_fft),
        }
    }

    /// Sample rate used to configure the reference processor.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Hop length used to configure the reference processor.
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// FFT size used to configure the reference processor.
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }
}

impl AudioProcessor for RawSpectrumProcessor {
    fn process(&mut self, frame: &AudioFrame) -> Option<Observation> {
        let _span = tracing::debug_span!("feature", processor = "raw_spectrum").entered();
        let samples = frame.samples.get(..self.n_fft)?;
        let mut input: Vec<f32> = samples
            .iter()
            .zip(&self.window)
            .map(|(sample, window)| sample * window)
            .collect();
        let mut spectrum = vec![realfft::num_complex::Complex32::default(); self.n_fft / 2 + 1];
        self.fft.process(&mut input, &mut spectrum).ok()?;
        Some(Observation {
            features: spectrum
                .iter()
                .map(|value| value.re.hypot(value.im))
                .collect(),
            perf_time: frame.perf_time,
        })
    }
}
