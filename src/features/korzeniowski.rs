//! Structured audio observations from `matchmaker.features.audio.KorzeniowskiAudioProcessor`.
//!
//! ```
//! use matchmaker_rs::features::korzeniowski::KorzeniowskiProcessor;
//! use matchmaker_rs::io::audio::AudioFrame;
//! let mut processor = KorzeniowskiProcessor::new(44_100, 512, 2_048, 4_096)?;
//! let observation = processor.process(&AudioFrame {
//!     index: 0,
//!     samples: vec![0.0; 2_048],
//!     perf_time: 0.0,
//! }).unwrap();
//! assert_eq!(observation.spectrum.len(), 2_049);
//! assert_eq!(observation.onset, 0.0);
//! assert!(observation.active_notes.is_none());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::features::KorzeniowskiObservation;
use crate::io::audio::AudioFrame;
use crate::utils::np_sum::np_sum_squares_f32;
use realfft::{RealFftPlanner, RealToComplex};

/// Korzeniowski's structured spectrum/onset/loudness processor.
pub struct KorzeniowskiProcessor {
    hop_length: usize,
    win_length: usize,
    n_fft: usize,
    window: Vec<f64>,
    fft: Arc<dyn RealToComplex<f64>>,
    previous_spectrum: Option<Vec<f64>>,
}

impl KorzeniowskiProcessor {
    /// Create a processor with an explicit reference-compatible window and FFT size.
    pub fn new(
        sample_rate: u32,
        hop_length: usize,
        win_length: usize,
        n_fft: usize,
    ) -> Result<Self> {
        if sample_rate == 0 {
            return Err(config("sample_rate", "must be at least 1 Hz"));
        }
        if hop_length == 0 {
            return Err(config("hop_length", "must be at least 1 sample"));
        }
        if win_length == 0 || hop_length > win_length {
            return Err(config("win_length", "must be at least hop_length"));
        }
        if n_fft < win_length {
            return Err(config("n_fft", "must be at least win_length"));
        }
        let window = (0..win_length)
            .map(|index| {
                0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / win_length as f64).cos()
            })
            .collect();
        let fft = RealFftPlanner::<f64>::new().plan_fft_forward(n_fft);
        Ok(Self {
            hop_length,
            win_length,
            n_fft,
            window,
            fft,
            previous_spectrum: None,
        })
    }

    /// Number of samples between observations.
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Analysis window length.
    pub fn win_length(&self) -> usize {
        self.win_length
    }

    /// Real FFT size.
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    /// Process one frame, or return `None` when it does not contain a full analysis window.
    pub fn process(&mut self, frame: &AudioFrame) -> Option<KorzeniowskiObservation> {
        let samples = frame.samples.get(..self.win_length)?;
        let mut input = vec![0.0; self.n_fft];
        for ((target, &sample), &window) in input.iter_mut().zip(samples).zip(&self.window) {
            *target = f64::from(sample) * window;
        }
        let mut raw = vec![realfft::num_complex::Complex64::default(); self.n_fft / 2 + 1];
        self.fft
            .process(&mut input, &mut raw)
            .expect("FFT buffers match the planned transform size");
        let raw: Vec<f64> = raw.iter().map(|value| value.norm()).collect();

        let magnitude_sum: f64 = raw.iter().sum();
        let onset = self.previous_spectrum.as_ref().map_or(0.0, |previous| {
            raw.iter()
                .zip(previous)
                .map(|(current, old)| (current - old).max(0.0))
                .sum::<f64>()
                / (magnitude_sum + 1e-8)
        });
        self.previous_spectrum = Some(raw.clone());

        let norm = raw.iter().map(|value| value * value).sum::<f64>().sqrt();
        let spectrum = if norm > 0.0 {
            raw.into_iter().map(|value| value / norm).collect()
        } else {
            raw
        };
        let rms = (np_sum_squares_f32(samples) / self.win_length as f32).sqrt();
        let loudness = f64::from(10.0f32 * (rms * rms).max(1e-10).log10());

        Some(KorzeniowskiObservation {
            spectrum,
            active_notes: None,
            onset,
            onset_notes: None,
            loudness,
            perf_time: frame.perf_time,
        })
    }

    /// Clear the causal onset history.
    pub fn reset(&mut self) {
        self.previous_spectrum = None;
    }
}

fn config(key: &str, reason: impl Into<String>) -> Error {
    Error::AudioConfig {
        key: key.into(),
        reason: reason.into(),
    }
}
