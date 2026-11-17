//! Port of `matchmaker.features.audio.MFCCProcessor`.
//! Upstream: `matchmaker/features/audio.py`.
//!
//! Computes log-power Slaney mel features and the reference DCT-II with 13 coefficients and
//! `norm="backward"` by default.
//!
//! ```
//! use matchmaker_rs::features::{AudioProcessor, mfcc::{MfccNorm, MfccProcessor}};
//! use matchmaker_rs::io::audio::AudioFrame;
//! let mut processor = MfccProcessor::new(44_100, 1_470, 13, Some(MfccNorm::Backward))?;
//! let observation = processor.process(&AudioFrame {
//!     index: 0,
//!     samples: vec![0.0; 2_940],
//!     perf_time: 0.0,
//! }).unwrap();
//! assert_eq!(observation.features[0], -25_600.0);
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::str::FromStr;
use std::sync::Arc;

use super::{AudioProcessor, Observation, mel::make_slaney_mel_bank};
use crate::error::{Error, Result};
use crate::io::audio::AudioFrame;
use realfft::{RealFftPlanner, RealToComplex};

const N_MELS: usize = 128;
const AMIN: f32 = 1e-10;
const TOP_DB: f32 = 80.0;

/// Normalization modes accepted by SciPy's type-II DCT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MfccNorm {
    /// Unscaled forward DCT-II; the released facade default.
    Backward,
    /// Orthonormal DCT-II.
    Ortho,
}

impl FromStr for MfccNorm {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "backward" => Ok(Self::Backward),
            "ortho" => Ok(Self::Ortho),
            _ => Err(config("norm", "must be backward, ortho, or None")),
        }
    }
}

/// MFCC observations using the reference-compatible Slaney mel bank and DCT-II scaling.
pub struct MfccProcessor {
    hop_length: usize,
    n_mfcc: usize,
    norm: Option<MfccNorm>,
    n_fft: usize,
    window: Vec<f64>,
    fft: Arc<dyn RealToComplex<f64>>,
    dct_fft: Arc<dyn RealToComplex<f64>>,
    mel_bank: Vec<Vec<f32>>,
}

impl MfccProcessor {
    /// Create an MFCC processor; the reference defaults to 13 coefficients and backward norm.
    pub fn new(
        sample_rate: u32,
        hop_length: usize,
        n_mfcc: usize,
        norm: Option<MfccNorm>,
    ) -> Result<Self> {
        if n_mfcc == 0 || n_mfcc > N_MELS {
            return Err(config("n_mfcc", format!("must be between 1 and {N_MELS}")));
        }
        if sample_rate == 0 {
            return Err(config("sample_rate", "must be at least 1 Hz"));
        }
        if hop_length == 0 {
            return Err(config("hop_length", "must be at least 1 sample"));
        }
        let n_fft = hop_length
            .checked_mul(2)
            .ok_or_else(|| config("hop_length", "twice the hop length overflows"))?;
        let window: Vec<f64> = (0..n_fft)
            .map(|index| 0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / n_fft as f64).cos())
            .collect();
        let mut planner = RealFftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n_fft);
        let mut dct_planner = RealFftPlanner::<f64>::new();
        let dct_fft = dct_planner.plan_fft_forward(2 * N_MELS);
        let mel_bank = make_slaney_mel_bank(sample_rate, n_fft, N_MELS);
        Ok(Self {
            hop_length,
            n_mfcc,
            norm,
            n_fft,
            window,
            fft,
            dct_fft,
            mel_bank,
        })
    }

    /// FFT size (`2 * hop_length` in the reference).
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    /// Number of returned coefficients.
    pub fn n_mfcc(&self) -> usize {
        self.n_mfcc
    }

    /// DCT normalization; `None` has the same scaling as SciPy's backward default.
    pub fn norm(&self) -> Option<MfccNorm> {
        self.norm
    }

    /// Log-power mel spectrogram before the DCT, with the reference's call-wide `top_db` floor.
    pub fn log_power_spectrogram(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        let n_fft = self.n_fft();
        if samples.len() < n_fft {
            return Ok(Vec::new());
        }
        let mut rows: Vec<Vec<f32>> = samples
            .windows(n_fft)
            .step_by(self.hop_length)
            .map(|window| {
                let power = self.power_spectrum(window)?;
                Ok(self
                    .mel_bank
                    .iter()
                    .map(|filter| {
                        let energy: f64 = filter
                            .iter()
                            .zip(&power)
                            .map(|(weight, value)| f64::from(*weight) * f64::from(*value))
                            .sum();
                        (10.0 * energy.max(f64::from(AMIN)).log10()) as f32
                    })
                    .collect())
            })
            .collect::<Result<_>>()?;
        if let Some(max_db) = rows.iter().flatten().copied().reduce(f32::max) {
            let floor = max_db - TOP_DB;
            for value in rows.iter_mut().flatten() {
                *value = value.max(floor);
            }
        }
        Ok(rows)
    }

    fn power_spectrum(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let samples = samples
            .get(..self.n_fft)
            .ok_or_else(|| config("n_fft", "input is shorter than one analysis window"))?;
        let mut input: Vec<f64> = samples
            .iter()
            .zip(&self.window)
            .map(|(&sample, &window)| f64::from(sample) * window)
            .collect();
        let mut spectrum = vec![realfft::num_complex::Complex64::default(); self.n_fft / 2 + 1];
        self.fft
            .process(&mut input, &mut spectrum)
            .map_err(|error| config("n_fft", format!("FFT failed: {error}")))?;
        Ok(spectrum
            .iter()
            .map(|value| (value.re * value.re + value.im * value.im) as f32)
            .collect())
    }

    /// Transform a whole signal using non-centered, hop-spaced windows.
    pub fn transform(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        self.log_power_spectrogram(samples)
            .map(|rows| rows.iter().map(|row| self.dct(row)).collect())
    }

    fn dct(&self, values: &[f32]) -> Vec<f32> {
        let mut input = vec![0.0f64; 2 * N_MELS];
        for (index, &value) in values.iter().enumerate() {
            input[index] = f64::from(value);
            input[2 * N_MELS - 1 - index] = f64::from(value);
        }
        let mut spectrum = vec![realfft::num_complex::Complex64::default(); N_MELS + 1];
        self.dct_fft
            .process(&mut input, &mut spectrum)
            .expect("fixed-size DCT FFT input");
        (0..self.n_mfcc)
            .map(|coefficient| {
                if coefficient > 0
                    && values
                        .first()
                        .is_some_and(|first| values.iter().all(|value| value == first))
                {
                    return 0.0;
                }
                let angle = std::f64::consts::PI * coefficient as f64 / (2 * N_MELS) as f64;
                let shifted = spectrum[coefficient]
                    * realfft::num_complex::Complex64::from_polar(1.0, -angle);
                let mut value = shifted.re;
                value *= match self.norm {
                    Some(MfccNorm::Ortho) if coefficient == 0 => {
                        1.0 / (2.0 * (N_MELS as f64).sqrt())
                    }
                    Some(MfccNorm::Ortho) => 1.0 / (2.0 * N_MELS as f64).sqrt(),
                    Some(MfccNorm::Backward) | None => 1.0,
                };
                value as f32
            })
            .collect()
    }
}

impl AudioProcessor for MfccProcessor {
    fn process(&mut self, frame: &AudioFrame) -> Option<Observation> {
        let _span = tracing::debug_span!("feature", processor = "mfcc").entered();
        let samples = frame.samples.get(..self.n_fft)?;
        let log_power = self.log_power_spectrogram(samples).ok()?;
        Some(Observation {
            features: self.dct(log_power.first()?),
            perf_time: frame.perf_time,
        })
    }
}

fn config(key: &str, reason: impl Into<String>) -> Error {
    Error::AudioConfig {
        key: key.into(),
        reason: reason.into(),
    }
}
