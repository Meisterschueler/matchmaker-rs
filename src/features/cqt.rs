//! Port of `matchmaker.features.audio.CQTProcessor`: the magnitude of `librosa.cqt` over the 88
//! piano keys (A0 to C8), one bin per semitone.
//!
//! The transform follows librosa's recursive algorithm: one sparse FFT kernel per octave (Hann
//! windowed phasors, normalized by `norm`, sparsified to 99 % of their spectral mass), the signal
//! halved in rate for the lower octaves, frames centered with zero padding, and a response scaled
//! by `1 / sqrt(filter length)`. The first and last frame of every call are dropped, as in the
//! reference. A performance frame carries two hops (the stream caches one), so a stream call
//! yields exactly one row; the score is transformed in one whole-signal call
//! ([`CqtProcessor::transform`]).
//!
//! Numerics (research R13): every internal value is `f64`. The reference works in `float32` and
//! halves the rate with `soxr_hq`, whose rounding noise the low bins amplify about 8000-fold, so
//! no implementation that is not bit-identical can meet a per-element `1e-5` limit. The features
//! agree with the reference to about `1e-6` of the largest bin of a frame instead; the contract's
//! cqt limit adds exactly that frame-maximum term. The rate halving is a Kaiser-windowed sinc
//! fitted to the measured response of `soxr_hq` (delay 0, zero-extended edges, gain `sqrt(2)`).
//!
//! ```
//! use matchmaker_rs::features::{AudioProcessor, cqt::CqtProcessor};
//! use matchmaker_rs::io::audio::AudioFrame;
//! let mut processor = CqtProcessor::new(44_100, 1_470, Some(f64::INFINITY)).unwrap();
//! let samples = (0..2_940).map(|n| (n as f32 * 0.062).sin()).collect();
//! let frame = AudioFrame { index: 0, samples, perf_time: 0.0 };
//! assert_eq!(processor.process(&frame).unwrap().features.len(), 88);
//! ```

use std::f64::consts::PI;
use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

use super::{AudioProcessor, Observation};
use crate::error::{Error, Result};
use crate::io::audio::AudioFrame;

/// Number of constant-Q bins: the 88 piano keys.
pub const N_BINS: usize = 88;
const BINS_PER_OCTAVE: usize = 12;
const N_OCTAVES: usize = N_BINS.div_ceil(BINS_PER_OCTAVE);
/// `librosa.note_to_hz("A0")`.
const FMIN_HZ: f64 = 27.5;
/// Share of each kernel's spectral mass that may be dropped (`librosa.cqt(sparsity=0.01)`).
const SPARSITY: f64 = 0.01;
/// Equivalent noise bandwidth of the Hann window (`librosa.filters.WINDOW_BANDWIDTHS`).
const HANN_ENBW: f64 = 1.500_183_105_468_75;
/// Kaiser design fitted to the measured taps of the `soxr_hq` decimator by 2 (rms error 3e-9,
/// maximum 2e-8): half length, window half-width, `beta` and cutoff in cycles per input sample.
const DECIMATOR_HALF_LENGTH: usize = 194;
const DECIMATOR_WINDOW: f64 = 194.5;
const DECIMATOR_BETA: f64 = 13.040_074;
const DECIMATOR_CUTOFF: f64 = 0.239_217_956_5;

/// One sparse row of an octave's frequency-domain kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRow {
    /// FFT bins with a non-zero coefficient, ascending.
    pub bins: Vec<usize>,
    /// Complex `(real, imaginary)` coefficients of those bins, rounded to `f32` like the reference.
    pub values: Vec<[f32; 2]>,
}

/// The kernel of one octave of the transform and how the signal reaches it.
pub struct OctaveKernel {
    /// Sample rate of the signal this octave analyses.
    pub sample_rate: f64,
    /// FFT size of this octave.
    pub n_fft: usize,
    /// One row per bin of the octave, in ascending frequency.
    pub rows: Vec<KernelRow>,
    hop_length: usize,
    first_bin: usize,
    rate_scale: f64,
    decimate_after: bool,
    fft: Arc<dyn RealToComplex<f64>>,
}

/// Halves the sample rate like `librosa.resample(orig_sr=2, target_sr=1, res_type="soxr_hq")`.
struct Decimator {
    taps: Vec<f64>,
}

/// Constant-Q magnitude features over the 88 piano keys.
pub struct CqtProcessor {
    hop_length: usize,
    early: Option<Decimator>,
    halver: Decimator,
    octaves: Vec<OctaveKernel>,
    inverse_root_lengths: Vec<f64>,
}

impl CqtProcessor {
    /// Create the processor with the reference defaults of 88 bins from A0.
    ///
    /// `norm` is the filter normalization passed to librosa (`Some(f64::INFINITY)` in the
    /// reference, `None` for none). Fails when the rate is too low for the highest filter, or so
    /// high that the reference would downsample the input by more than 2 up front (above about
    /// 70 kHz with a suitable hop), which this port does not reproduce.
    pub fn new(sample_rate: u32, hop_length: usize, norm: Option<f64>) -> Result<Self> {
        if sample_rate == 0 {
            return Err(config("sample_rate", "must be at least 1 Hz".into()));
        }
        if hop_length == 0 {
            return Err(config("hop_length", "must be at least 1 sample".into()));
        }
        if norm.is_some_and(|norm| norm.is_nan() || norm <= 0.0) {
            return Err(config("norm", "must be positive or None".into()));
        }
        let rate = f64::from(sample_rate);
        let frequencies = key_frequencies();
        let alphas = relative_bandwidths(&frequencies);
        let cutoff = frequencies
            .iter()
            .zip(&alphas)
            .map(|(frequency, alpha)| frequency * (1.0 + 0.5 * HANN_ENBW / (1.0 / alpha)))
            .fold(f64::MIN, f64::max);
        let nyquist = rate / 2.0;
        if cutoff > nyquist {
            return Err(config(
                "sample_rate",
                format!(
                    "the constant-Q basis reaches {cutoff:.1} Hz, above the Nyquist frequency {nyquist:.1} Hz"
                ),
            ));
        }

        let early_count = ((nyquist / cutoff).log2().ceil() as i64 - 2).max(0);
        let hop_count = (i64::from(hop_length.trailing_zeros()) - N_OCTAVES as i64 + 1).max(0);
        let early_factor = 1usize << early_count.min(hop_count);
        if early_factor > 2 {
            return Err(config(
                "sample_rate",
                format!("early downsampling by {early_factor} is not supported"),
            ));
        }
        let early = (early_factor == 2).then(Decimator::new);
        let rate = rate / early_factor as f64;
        let inverse_root_lengths = frequencies
            .iter()
            .zip(&alphas)
            .map(|(frequency, alpha)| 1.0 / ((1.0 / alpha) * rate / frequency).sqrt())
            .collect();

        let mut planner = RealFftPlanner::<f64>::new();
        let mut octaves = Vec::with_capacity(N_OCTAVES);
        let (mut octave_rate, mut hop) = (rate, hop_length / early_factor);
        for octave in 0..N_OCTAVES {
            let end = N_BINS - octave * BINS_PER_OCTAVE;
            let start = end.saturating_sub(BINS_PER_OCTAVE);
            let lengths: Vec<f64> = (start..end)
                .map(|bin| (1.0 / alphas[bin]) * octave_rate / frequencies[bin])
                .collect();
            let n_fft = (lengths.iter().copied().fold(0.0, f64::max))
                .log2()
                .ceil()
                .exp2() as usize;
            if n_fft < 2 {
                return Err(config(
                    "sample_rate",
                    "an octave needs an FFT of at least 2 samples".into(),
                ));
            }
            let fft = planner.plan_fft_forward(n_fft);
            let rows = (start..end)
                .zip(&lengths)
                .map(|(bin, &length)| {
                    kernel_row(&*fft, n_fft, length, frequencies[bin], octave_rate, norm)
                })
                .collect();
            let decimate_after = octave + 1 < N_OCTAVES
                && hop % 2 == 0
                && frequencies[start - 1] <= octave_rate / 5.0;
            octaves.push(OctaveKernel {
                sample_rate: octave_rate,
                n_fft,
                rows,
                hop_length: hop,
                first_bin: start,
                rate_scale: (rate / octave_rate).sqrt(),
                decimate_after,
                fft,
            });
            if decimate_after {
                hop /= 2;
                octave_rate /= 2.0;
            }
        }
        Ok(Self {
            hop_length,
            early,
            halver: Decimator::new(),
            octaves,
            inverse_root_lengths,
        })
    }

    /// Hop length in samples at the input rate.
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// Kernels of the octaves, highest first, as the reference builds them.
    pub fn kernels(&self) -> &[OctaveKernel] {
        &self.octaves
    }

    /// Number of frames of the untrimmed transform of `samples` samples: the columns of the matrix
    /// `librosa.cqt` returns, of which [`CqtProcessor::transform`] drops the first and the last.
    pub fn frames(&self, samples: usize) -> usize {
        let mut length = if self.early.is_some() {
            samples.div_ceil(2)
        } else {
            samples
        };
        let mut columns = usize::MAX;
        for octave in &self.octaves {
            columns = columns.min(1 + length / octave.hop_length);
            if octave.decimate_after {
                length = length.div_ceil(2);
            }
        }
        columns
    }

    /// Constant-Q magnitudes of a whole signal: one row of [`N_BINS`] values per hop, without the
    /// first and last frame (`abs(cqt).T[1:-1]`). Signals shorter than two hops give no row.
    pub fn transform(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        if samples.iter().any(|sample| !sample.is_finite()) {
            return Err(config(
                "samples",
                "audio contains NaN or infinite samples".into(),
            ));
        }
        let mut signal: Vec<f64> = samples.iter().map(|&sample| f64::from(sample)).collect();
        if let Some(early) = &self.early {
            if signal.len() < 2 {
                return Err(config(
                    "samples",
                    format!(
                        "input of {} samples is too short for early downsampling",
                        signal.len()
                    ),
                ));
            }
            signal = early.halve(&signal);
        }
        let columns = self.frames(samples.len());
        if columns < 3 {
            return Ok(Vec::new());
        }
        let mut rows = vec![vec![0.0f32; N_BINS]; columns - 2];
        for octave in &self.octaves {
            octave.respond(&signal, columns, &self.inverse_root_lengths, &mut rows)?;
            if octave.decimate_after {
                signal = self.halver.halve(&signal);
            }
        }
        Ok(rows)
    }
}

impl AudioProcessor for CqtProcessor {
    /// The first row of the frame's transform; a stream frame (two hops) has exactly one.
    fn process(&mut self, frame: &AudioFrame) -> Option<Observation> {
        let _span = tracing::debug_span!("feature", processor = "cqt").entered();
        let features = self.transform(&frame.samples).ok()?.into_iter().next()?;
        Some(Observation {
            features,
            perf_time: frame.perf_time,
        })
    }
}

/// Halve the rate of `samples` the way `librosa.resample(orig_sr=2, target_sr=1,
/// res_type="soxr_hq", scale=True)` does: zero delay, zero-extended edges, gain `sqrt(2)`,
/// `ceil(len / 2)` outputs.
pub fn halve_rate(samples: &[f32]) -> Vec<f32> {
    let signal: Vec<f64> = samples.iter().map(|&sample| f64::from(sample)).collect();
    Decimator::new()
        .halve(&signal)
        .into_iter()
        .map(|sample| sample as f32)
        .collect()
}

impl OctaveKernel {
    fn respond(
        &self,
        signal: &[f64],
        columns: usize,
        inverse_root_lengths: &[f64],
        rows: &mut [Vec<f32>],
    ) -> Result<()> {
        let mut frame = vec![0.0f64; self.n_fft];
        let mut spectrum = self.fft.make_output_vec();
        let mut scratch = self.fft.make_scratch_vec();
        let half = (self.n_fft / 2) as isize;
        for (row, column) in rows.iter_mut().zip(1..columns - 1) {
            let first = (column * self.hop_length) as isize - half;
            let lower = first.max(0);
            let upper = (first + self.n_fft as isize).min(signal.len() as isize);
            frame.fill(0.0);
            if lower < upper {
                frame[(lower - first) as usize..(upper - first) as usize]
                    .copy_from_slice(&signal[lower as usize..upper as usize]);
            }
            self.fft
                .process_with_scratch(&mut frame, &mut spectrum, &mut scratch)
                .map_err(|error| config("n_fft", error.to_string()))?;
            for (offset, kernel) in self.rows.iter().enumerate() {
                let (mut real, mut imaginary) = (0.0, 0.0);
                for (&bin, &[kernel_real, kernel_imaginary]) in
                    kernel.bins.iter().zip(&kernel.values)
                {
                    let (kernel_real, kernel_imaginary) =
                        (f64::from(kernel_real), f64::from(kernel_imaginary));
                    real += kernel_real * spectrum[bin].re - kernel_imaginary * spectrum[bin].im;
                    imaginary +=
                        kernel_real * spectrum[bin].im + kernel_imaginary * spectrum[bin].re;
                }
                let bin = self.first_bin + offset;
                row[bin] =
                    (real.hypot(imaginary) * self.rate_scale * inverse_root_lengths[bin]) as f32;
            }
        }
        Ok(())
    }
}

impl Decimator {
    fn new() -> Self {
        let normalizer = bessel_i0(DECIMATOR_BETA);
        let mut taps: Vec<f64> = (0..=2 * DECIMATOR_HALF_LENGTH)
            .map(|index| {
                let offset = index as f64 - DECIMATOR_HALF_LENGTH as f64;
                let window = bessel_i0(
                    DECIMATOR_BETA * (1.0 - (offset / DECIMATOR_WINDOW).powi(2)).max(0.0).sqrt(),
                ) / normalizer;
                2.0 * DECIMATOR_CUTOFF * sinc(2.0 * DECIMATOR_CUTOFF * offset) * window
            })
            .collect();
        let sum: f64 = taps.iter().sum();
        taps.iter_mut().for_each(|tap| *tap /= sum);
        Self { taps }
    }

    fn halve(&self, signal: &[f64]) -> Vec<f64> {
        let half = self.taps.len() / 2;
        let mut padded = vec![0.0; signal.len() + 2 * half];
        padded[half..half + signal.len()].copy_from_slice(signal);
        (0..signal.len().div_ceil(2))
            .map(|output| {
                let center = 2 * output + half;
                let mut sum = self.taps[half] * padded[center];
                for distance in 1..=half {
                    sum += self.taps[half + distance]
                        * (padded[center - distance] + padded[center + distance]);
                }
                sum * std::f64::consts::SQRT_2
            })
            .collect()
    }
}

/// Center frequencies of the 88 piano keys (`librosa.interval_frequencies`, equal temperament).
fn key_frequencies() -> Vec<f64> {
    (0..N_BINS)
        .map(|bin| {
            let ratio = ((bin % BINS_PER_OCTAVE) as f64 / BINS_PER_OCTAVE as f64).exp2();
            f64::from(1u32 << (bin / BINS_PER_OCTAVE)) * ratio * FMIN_HZ
        })
        .collect()
}

/// `librosa.filters._relative_bandwidth`: twice the relative spacing of neighboring frequencies.
fn relative_bandwidths(frequencies: &[f64]) -> Vec<f64> {
    let logs: Vec<f64> = frequencies
        .iter()
        .map(|frequency| frequency.log2())
        .collect();
    let last = logs.len() - 1;
    (0..logs.len())
        .map(|index| {
            let bins_per_octave = match index {
                0 => 1.0 / (logs[1] - logs[0]),
                _ if index == last => 1.0 / (logs[last] - logs[last - 1]),
                _ => 2.0 / (logs[index + 1] - logs[index - 1]),
            };
            let ratio = (2.0 / bins_per_octave).exp2();
            (ratio - 1.0) / (ratio + 1.0)
        })
        .collect()
}

/// One filter of `librosa.filters.wavelet` as a sparse FFT row (`__vqt_filter_fft`).
fn kernel_row(
    fft: &dyn RealToComplex<f64>,
    n_fft: usize,
    length: f64,
    frequency: f64,
    rate: f64,
    norm: Option<f64>,
) -> KernelRow {
    let start = (-length / 2.0).floor();
    let count = ((length / 2.0).floor() - start) as usize;
    let step = 2.0 * PI / count as f64;
    let mut phasors: Vec<(f64, f64)> = (0..count)
        .map(|index| {
            let angle = (start + index as f64) * 2.0 * PI * frequency / rate;
            let window = 0.5 + 0.5 * (index as f64 * step - PI).cos();
            (angle.cos() * window, angle.sin() * window)
        })
        .collect();
    let magnitudes = phasors
        .iter()
        .map(|&(real, imaginary)| real.hypot(imaginary));
    let scale = match norm {
        None => 1.0,
        Some(order) if order.is_infinite() => magnitudes.fold(0.0, f64::max),
        Some(order) => magnitudes
            .map(|value| value.powf(order))
            .sum::<f64>()
            .powf(1.0 / order),
    };
    let scale = if scale < f64::MIN_POSITIVE {
        1.0
    } else {
        scale
    };
    let factor = length / n_fft as f64;
    let to_f32 = |value: f64| f64::from(value as f32);
    let mut real = vec![0.0f64; n_fft];
    let mut imaginary = vec![0.0f64; n_fft];
    let offset = (n_fft - count) / 2;
    for (index, (re, im)) in phasors.drain(..).enumerate() {
        let (re, im) = (to_f32(re / scale), to_f32(im / scale));
        real[offset + index] = to_f32(re * factor);
        imaginary[offset + index] = to_f32(im * factor);
    }
    let mut scratch = fft.make_scratch_vec();
    let mut transformed_real = fft.make_output_vec();
    let mut transformed_imaginary = fft.make_output_vec();
    // Planned for this size, so the buffers cannot be mis-sized.
    let _ = fft.process_with_scratch(&mut real, &mut transformed_real, &mut scratch);
    let _ = fft.process_with_scratch(&mut imaginary, &mut transformed_imaginary, &mut scratch);
    let values: Vec<[f32; 2]> = transformed_real
        .iter()
        .zip(&transformed_imaginary)
        .map(|(a, b)| [(a.re - b.im) as f32, (a.im + b.re) as f32])
        .collect();
    sparsify(&values)
}

/// `librosa.util.sparsify_rows`: drop the smallest entries holding up to `SPARSITY` of the mass.
fn sparsify(values: &[[f32; 2]]) -> KernelRow {
    let magnitudes: Vec<f64> = values
        .iter()
        .map(|&[re, im]| f64::from(re).hypot(f64::from(im)))
        .collect();
    let total: f64 = magnitudes.iter().sum();
    let mut sorted = magnitudes.clone();
    sorted.sort_by(f64::total_cmp);
    let mut cumulative = 0.0;
    let mut threshold = sorted[0];
    for &magnitude in &sorted {
        cumulative += magnitude / total;
        if cumulative >= SPARSITY {
            threshold = magnitude;
            break;
        }
    }
    let kept: Vec<usize> = (0..values.len())
        .filter(|&bin| magnitudes[bin] >= threshold && magnitudes[bin] > 0.0)
        .collect();
    KernelRow {
        values: kept.iter().map(|&bin| values[bin]).collect(),
        bins: kept,
    }
}

fn bessel_i0(x: f64) -> f64 {
    let quarter = x * x / 4.0;
    let (mut term, mut sum) = (1.0, 1.0);
    for k in 1..200 {
        term *= quarter / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

fn config(key: &str, reason: String) -> Error {
    Error::AudioConfig {
        key: key.into(),
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(frequency: f64, rate: f64, length: usize) -> Vec<f32> {
        (0..length)
            .map(|n| (2.0 * PI * frequency * n as f64 / rate).sin() as f32)
            .collect()
    }

    #[test]
    fn halving_keeps_alignment_and_scales_by_the_square_root_of_two() {
        let signal = sine(1_000.0, 44_100.0, 4_001);
        let halved = halve_rate(&signal);
        assert_eq!(halved.len(), 2_001);
        for (index, &value) in halved.iter().enumerate().take(1_800).skip(200) {
            let expected = std::f64::consts::SQRT_2
                * (2.0 * PI * 1_000.0 * (2 * index) as f64 / 44_100.0).sin();
            assert!((f64::from(value) - expected).abs() < 1e-5, "sample {index}");
        }
    }

    #[test]
    fn kernels_cover_the_88_keys_in_eight_octaves_with_sparse_rows() {
        let processor = CqtProcessor::new(44_100, 1_470, Some(f64::INFINITY)).unwrap();
        let rows: Vec<usize> = processor.kernels().iter().map(|k| k.rows.len()).collect();
        assert_eq!(rows, [12, 12, 12, 12, 12, 12, 12, 4]);
        let rates: Vec<f64> = processor.kernels().iter().map(|k| k.sample_rate).collect();
        assert_eq!(
            rates,
            [
                44_100.0, 22_050.0, 22_050.0, 22_050.0, 22_050.0, 22_050.0, 22_050.0, 22_050.0
            ]
        );
        let sizes: Vec<usize> = processor.kernels().iter().map(|k| k.n_fft).collect();
        assert_eq!(
            sizes,
            [512, 512, 1_024, 2_048, 4_096, 8_192, 16_384, 16_384]
        );
        assert!(
            processor
                .kernels()
                .iter()
                .flat_map(|k| &k.rows)
                .all(|row| (7..=16).contains(&row.bins.len()))
        );
    }

    #[test]
    fn a_tone_peaks_in_its_own_bin() {
        let processor = CqtProcessor::new(44_100, 1_470, Some(f64::INFINITY)).unwrap();
        let rows = processor.transform(&sine(440.0, 44_100.0, 14_700)).unwrap();
        assert_eq!(rows.len(), 9);
        let strongest = rows[3]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(strongest, 48); // A4 is 48 semitones above A0
    }

    #[test]
    fn call_lengths_decide_the_number_of_rows_and_silence_stays_silent() {
        let processor = CqtProcessor::new(44_100, 1_470, Some(f64::INFINITY)).unwrap();
        assert_eq!(processor.transform(&vec![0.0; 2 * 1_470]).unwrap().len(), 1);
        assert_eq!(processor.transform(&vec![0.0; 4 * 1_470]).unwrap().len(), 3);
        assert_eq!(
            [100, 2 * 1_470, 4 * 1_470].map(|length| processor.frames(length)),
            [1, 3, 5]
        );
        assert!(processor.transform(&[0.0; 100]).unwrap().is_empty());
        assert!(processor.transform(&[]).unwrap().is_empty());
        assert!(
            processor
                .transform(&vec![0.0; 2 * 1_470])
                .unwrap()
                .iter()
                .flatten()
                .all(|&value| value == 0.0)
        );
        assert!(processor.transform(&[f32::NAN; 2_940]).is_err());
    }

    #[test]
    fn unusable_rates_hops_and_norms_are_rejected_by_name() {
        let message = |result: Result<CqtProcessor>| match result {
            Ok(_) => String::new(),
            Err(error) => error.to_string(),
        };
        assert!(message(CqtProcessor::new(8_000, 128, None)).contains("sample_rate"));
        assert!(message(CqtProcessor::new(0, 128, None)).contains("sample_rate"));
        assert!(message(CqtProcessor::new(44_100, 0, None)).contains("hop_length"));
        assert!(message(CqtProcessor::new(44_100, 1_470, Some(-1.0))).contains("norm"));
        assert!(message(CqtProcessor::new(44_100, 1_470, Some(f64::NAN))).contains("norm"));
        assert!(message(CqtProcessor::new(96_000, 2_048, None)).contains("early downsampling"));
        assert!(message(CqtProcessor::new(1_000_000, 4_096, None)).contains("early downsampling"));
    }
}
