//! Port of `matchmaker.features.audio.ChromagramProcessor`.
//!
//! Uses Librosa's centered-false STFT, automatically estimated tuning, and normalized chroma
//! filter bank for each audio hop.
//!
//! ```
//! use matchmaker_rs::features::{AudioProcessor, chroma::ChromaProcessor};
//! use matchmaker_rs::io::audio::AudioFrame;
//! let mut processor = ChromaProcessor::new(44_100, 1_470, Some(f64::INFINITY));
//! let frame = AudioFrame { index: 0, samples: vec![0.0; 2_940], perf_time: 0.0 };
//! assert_eq!(processor.process(&frame).unwrap().features.len(), 12);
//! ```

use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

use super::{AudioProcessor, Observation};
use crate::io::audio::AudioFrame;

const N_CHROMA: usize = 12;
const A440: f64 = 440.0;

/// Chroma from the magnitude-squared STFT, normalized per frame.
pub struct ChromaProcessor {
    sample_rate: u32,
    n_fft: usize,
    norm: Option<f64>,
    reference_tuning: Option<f64>,
    fft: Arc<dyn RealToComplex<f32>>,
}

impl ChromaProcessor {
    /// Create a chroma processor; `norm` defaults to positive infinity in pymatchmaker.
    pub fn new(sample_rate: u32, hop_length: usize, norm: Option<f64>) -> Self {
        let n_fft = 2 * hop_length;
        let mut planner = RealFftPlanner::<f32>::new();
        Self {
            sample_rate,
            n_fft,
            norm,
            reference_tuning: None,
            fft: planner.plan_fft_forward(n_fft),
        }
    }

    /// FFT size (`2 * hop_length` by default in the reference).
    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    pub(crate) fn for_reference(
        sample_rate: u32,
        hop_length: usize,
        norm: Option<f64>,
        samples: &[f32],
    ) -> crate::Result<Self> {
        let mut processor = Self::new(sample_rate, hop_length, norm);
        let mut pitches = Vec::new();
        for frame in samples.windows(processor.n_fft).step_by(hop_length) {
            let power =
                processor
                    .power_spectrum(frame)
                    .ok_or_else(|| crate::Error::AudioConfig {
                        key: "score_features".into(),
                        reason: "FFT failed for the score reference".into(),
                    })?;
            pitches.extend(spectral_pitches(&power, sample_rate, processor.n_fft));
        }
        processor.reference_tuning = Some(tuning_from_pitches(pitches));
        Ok(processor)
    }

    fn power_spectrum(&self, samples: &[f32]) -> Option<Vec<f32>> {
        let samples = samples.get(..self.n_fft)?;
        let mut input: Vec<f32> = samples
            .iter()
            .enumerate()
            .map(|(index, sample)| {
                let window =
                    0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / self.n_fft as f64).cos();
                sample * window as f32
            })
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

impl AudioProcessor for ChromaProcessor {
    fn process(&mut self, frame: &AudioFrame) -> Option<Observation> {
        let _span = tracing::debug_span!("feature", processor = "chroma").entered();
        let power = self.power_spectrum(&frame.samples)?;
        let tuning = self
            .reference_tuning
            .unwrap_or_else(|| estimate_tuning(&power, self.sample_rate, self.n_fft));
        let bank = chroma_filterbank(self.sample_rate, self.n_fft, tuning);
        let mut features = vec![0.0f32; N_CHROMA];
        for (row, feature) in features.iter_mut().enumerate() {
            for (weight, power) in bank[row].iter().zip(&power) {
                *feature += weight * power;
            }
        }
        normalize(&mut features, self.norm);
        Some(Observation {
            features,
            perf_time: frame.perf_time,
        })
    }
}

fn estimate_tuning(power: &[f32], sample_rate: u32, n_fft: usize) -> f64 {
    tuning_from_pitches(spectral_pitches(power, sample_rate, n_fft))
}

fn spectral_pitches(power: &[f32], sample_rate: u32, n_fft: usize) -> Vec<(f64, f64)> {
    let maximum = power.iter().copied().fold(0.0f32, f32::max);
    if maximum == 0.0 {
        return Vec::new();
    }
    let threshold = maximum * 0.1;
    let mut pitches = Vec::new();
    for bin in 1..power.len().saturating_sub(1) {
        let current = if power[bin] > threshold {
            power[bin]
        } else {
            0.0
        };
        let previous = if power[bin - 1] > threshold {
            power[bin - 1]
        } else {
            0.0
        };
        let next = if power[bin + 1] > threshold {
            power[bin + 1]
        } else {
            0.0
        };
        let frequency = bin as f64 * f64::from(sample_rate) / n_fft as f64;
        if !(150.0..(f64::from(sample_rate) / 2.0).min(4_000.0)).contains(&frequency)
            || current <= previous
            || current < next
        {
            continue;
        }

        let a = power[bin + 1] + power[bin - 1] - 2.0 * power[bin];
        let b = (power[bin + 1] - power[bin - 1]) / 2.0;
        let shift = if b.abs() >= a.abs() || a == 0.0 {
            0.0
        } else {
            -f64::from(b / a)
        };
        let gradient = if bin == 0 {
            power[1] - power[0]
        } else if bin + 1 == power.len() {
            power[bin] - power[bin - 1]
        } else {
            (power[bin + 1] - power[bin - 1]) / 2.0
        };
        let magnitude = f64::from(power[bin] + 0.5 * gradient * shift as f32);
        let interpolated = (bin as f64 + shift) * f64::from(sample_rate) / n_fft as f64;
        if interpolated > 0.0 {
            pitches.push((interpolated, magnitude));
        }
    }
    pitches
}

fn tuning_from_pitches(pitches: Vec<(f64, f64)>) -> f64 {
    if pitches.is_empty() {
        return 0.0;
    }
    let mut magnitudes: Vec<f64> = pitches.iter().map(|(_, magnitude)| *magnitude).collect();
    magnitudes.sort_by(f64::total_cmp);
    let median = magnitudes[magnitudes.len() / 2];
    let mut histogram = [0usize; 100];
    for (frequency, _) in pitches.into_iter().filter(|(_, mag)| *mag >= median) {
        let mut residual = (12.0 * (frequency / (A440 / 16.0)).log2()).rem_euclid(1.0);
        if residual >= 0.5 {
            residual -= 1.0;
        }
        let index = (((residual + 0.5) * 100.0).floor() as isize).clamp(0, 99) as usize;
        histogram[index] += 1;
    }
    let peak = histogram
        .iter()
        .enumerate()
        .max_by_key(|(index, count)| (**count, std::cmp::Reverse(*index)))
        .map_or(0, |(index, _)| index);
    -0.5 + peak as f64 / 100.0
}

fn chroma_filterbank(sample_rate: u32, n_fft: usize, tuning: f64) -> Vec<Vec<f32>> {
    let reference = A440 * 2.0f64.powf(tuning / N_CHROMA as f64) / 16.0;
    let mut frqbins: Vec<f64> = (1..n_fft)
        .map(|bin| {
            N_CHROMA as f64
                * ((bin as f64 * f64::from(sample_rate) / n_fft as f64) / reference).log2()
        })
        .collect();
    let first = frqbins[0] - 1.5 * N_CHROMA as f64;
    frqbins.insert(0, first);
    let mut bandwidth = Vec::with_capacity(n_fft);
    for pair in frqbins.windows(2) {
        bandwidth.push((pair[1] - pair[0]).max(1.0));
    }
    bandwidth.push(1.0);

    let mut bank = vec![vec![0.0f32; n_fft / 2 + 1]; N_CHROMA];
    for bin in 0..=n_fft / 2 {
        let weights: Vec<f64> = (0..N_CHROMA)
            .map(|chroma| {
                let pitch_class = ((chroma + 3) % N_CHROMA) as f64;
                let distance = (frqbins[bin] - pitch_class + 6.0 + 120.0).rem_euclid(12.0) - 6.0;
                let scaled = 2.0 * distance / bandwidth[bin];
                (-0.5 * scaled * scaled).exp()
            })
            .collect();
        let norm = weights
            .iter()
            .map(|weight| weight * weight)
            .sum::<f64>()
            .sqrt();
        let octave = (-0.5 * (((frqbins[bin] / N_CHROMA as f64 - 5.0) / 2.0).powi(2))).exp();
        if norm > 0.0 {
            for chroma in 0..N_CHROMA {
                bank[chroma][bin] = (weights[chroma] / norm * octave) as f32;
            }
        }
    }
    bank
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::audio::AudioStream;

    #[test]
    fn tuning_histogram_ties_choose_numpy_first_bin() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/audio/simple_mozart_k265_var1_sf2.wav");
        let stream = AudioStream::open(path, 44_100, 1470, 1470).unwrap();
        let frames = stream.take(13).collect::<Vec<_>>();
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(2940);
        for (frame, expected) in [(0, -0.19), (1, -0.18), (2, -0.10), (12, -0.04)] {
            let mut input = frames[frame]
                .samples
                .iter()
                .enumerate()
                .map(|(index, sample)| {
                    sample
                        * (0.5 - 0.5 * (std::f64::consts::TAU * index as f64 / 2940.0).cos()) as f32
                })
                .collect::<Vec<_>>();
            let mut spectrum = fft.make_output_vec();
            fft.process(&mut input, &mut spectrum).unwrap();
            let power = spectrum
                .iter()
                .map(|value| value.re * value.re + value.im * value.im)
                .collect::<Vec<_>>();
            let actual = estimate_tuning(&power, 44_100, 2940);
            assert!(
                (actual - expected).abs() < 1e-12,
                "frame {frame}: tuning={actual}, expected NumPy first bin={expected}"
            );
        }
    }

    #[test]
    fn whole_reference_tuning_matches_librosa() {
        let samples = (0..17_640)
            .map(|index| {
                let time = index as f64 / 44_100.0;
                (0.35 * (std::f64::consts::TAU * 440.0 * time).sin()
                    + 0.2 * (std::f64::consts::TAU * 659.2551138257398 * time).sin()
                    + 0.1 * (std::f64::consts::TAU * 783.9908719634985 * time).sin())
                    as f32
            })
            .collect::<Vec<_>>();
        let processor =
            ChromaProcessor::for_reference(44_100, 1470, Some(f64::INFINITY), &samples).unwrap();
        assert!((processor.reference_tuning.unwrap() + 0.07).abs() < 1e-12);
    }
}
