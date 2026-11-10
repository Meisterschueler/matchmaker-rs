//! Port of `matchmaker.features.audio.CQTSpectralFluxProcessor`: the [`cqt`](super::cqt) rows of
//! every call, plus the half-wave rectified spectral flux against the previous call as an 89th
//! value, or the 88 rows alone when the flux is disabled.
//!
//! The reference subtracts the whole matrix of the previous call frame by frame, then drops the
//! first and last frame, and keeps the matrix for the next call. Reproduced here:
//! - the first call has no predecessor, its flux is 0;
//! - a stream call (two hops, one row) is compared with the previous stream call. The zero-primed
//!   first call of a stream sets the history although its row is dropped, so the first emitted
//!   row already has a flux;
//! - the history survives [`AudioProcessor::reset`], which the reference inherits as a no-op: use
//!   one processor per run, never a reset one (research R5);
//! - calls with another frame count cannot be subtracted: NumPy raises its broadcast error there,
//!   this port returns [`Error::AudioConfig`], and the history stays as it was. A one-frame
//!   history (a call shorter than a hop), which NumPy would broadcast, is a shape error here too.
//!   The reference runs the rendered score and the performance through one processor, so its
//!   frame followers fail on the first performance frame (research C07).
//!
//! Not ported: the constructor's `fmin`, `n_bins` and `bins_per_octave` (the facade cannot reach
//! them; the 88 keys are the layout Audio-OuterHMM needs).
//!
//! ```
//! use matchmaker_rs::features::{AudioProcessor, cqt_spectral_flux::CqtSpectralFluxProcessor};
//! use matchmaker_rs::io::audio::AudioFrame;
//! let mut processor =
//!     CqtSpectralFluxProcessor::new(44_100, 1_470, Some(f64::INFINITY), true).unwrap();
//! let frame = |level: f32| AudioFrame {
//!     index: 0,
//!     samples: (0..2_940).map(|n| level * (n as f32 * 0.062).sin()).collect(),
//!     perf_time: 0.0,
//! };
//! let first = processor.process(&frame(0.1)).unwrap().features;
//! assert_eq!((first.len(), first[88]), (89, 0.0));
//! assert!(processor.process(&frame(0.2)).unwrap().features[88] > 0.0);
//! ```

use super::cqt::CqtProcessor;
use super::{AudioProcessor, Observation};
use crate::error::{Error, Result};
use crate::io::audio::AudioFrame;

/// Constant-Q magnitudes over the 88 piano keys with optional spectral flux.
pub struct CqtSpectralFluxProcessor {
    cqt: CqtProcessor,
    include_flux: bool,
    /// Frames of the previous call's untrimmed matrix, and its rows without the edge frames.
    previous: Option<(usize, Vec<Vec<f32>>)>,
}

impl CqtSpectralFluxProcessor {
    /// Create the processor; `norm` is passed on as for [`CqtProcessor::new`], `include_flux`
    /// appends the flux to every row.
    pub fn new(
        sample_rate: u32,
        hop_length: usize,
        norm: Option<f64>,
        include_flux: bool,
    ) -> Result<Self> {
        Ok(Self {
            cqt: CqtProcessor::new(sample_rate, hop_length, norm)?,
            include_flux,
            previous: None,
        })
    }

    /// One call of the reference processor: the rows [`CqtProcessor::transform`] gives for
    /// `samples`, each followed by its flux when enabled. Updates the history.
    pub fn transform(&mut self, samples: &[f32]) -> Result<Vec<Vec<f32>>> {
        let mut rows = self.cqt.transform(samples)?;
        if !self.include_flux {
            return Ok(rows);
        }
        let frames = self.cqt.frames(samples.len());
        let flux: Vec<f32> = match &self.previous {
            None => vec![0.0; rows.len()],
            Some((before, _)) if *before != frames => {
                return Err(Error::AudioConfig {
                    key: "samples".into(),
                    reason: format!(
                        "{frames} frames cannot be subtracted from the {before} frames of the previous call"
                    ),
                });
            }
            Some((_, previous)) => rows
                .iter()
                .zip(previous)
                .map(|(row, before)| positive_change(row, before))
                .collect(),
        };
        self.previous = Some((frames, rows.clone()));
        for (row, flux) in rows.iter_mut().zip(flux) {
            row.push(flux);
        }
        Ok(rows)
    }
}

/// `np.sum(np.maximum(row - before, 0))`.
fn positive_change(row: &[f32], before: &[f32]) -> f32 {
    row.iter()
        .zip(before)
        .map(|(now, then)| f64::from((now - then).max(0.0)))
        .sum::<f64>() as f32
}

impl AudioProcessor for CqtSpectralFluxProcessor {
    /// The first row of the frame's transform; a stream frame (two hops) has exactly one.
    fn process(&mut self, frame: &AudioFrame) -> Option<Observation> {
        let _span = tracing::debug_span!("feature", processor = "cqt_spectral_flux").entered();
        let features = self.transform(&frame.samples).ok()?.into_iter().next()?;
        Some(Observation {
            features,
            perf_time: frame.perf_time,
        })
    }

    // `reset` stays the trait's no-op: the reference inherits it, so the history survives.
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(level: f32, length: usize) -> Vec<f32> {
        (0..length)
            .map(|n| level * (n as f32 * 0.062).sin())
            .collect()
    }

    fn frame(level: f32) -> AudioFrame {
        AudioFrame {
            index: 0,
            samples: tone(level, 2_940),
            perf_time: 0.0,
        }
    }

    fn processor(include_flux: bool) -> CqtSpectralFluxProcessor {
        CqtSpectralFluxProcessor::new(44_100, 1_470, Some(f64::INFINITY), include_flux).unwrap()
    }

    #[test]
    fn flux_is_the_positive_change_against_the_previous_call_and_zero_for_the_first() {
        let cqt = CqtProcessor::new(44_100, 1_470, Some(f64::INFINITY)).unwrap();
        let rows: Vec<Vec<f32>> = [0.1, 0.2, 0.1]
            .map(|level| cqt.transform(&tone(level, 2_940)).unwrap().remove(0))
            .into();
        let mut processor = processor(true);
        let outputs: Vec<Vec<f32>> = [0.1, 0.2, 0.1]
            .map(|level| processor.process(&frame(level)).unwrap().features)
            .into();
        for (output, row) in outputs.iter().zip(&rows) {
            assert_eq!(&output[..88], row.as_slice());
            assert_eq!(output.len(), 89);
        }
        assert_eq!(outputs[0][88], 0.0);
        let rise: f32 = rows[1].iter().zip(&rows[0]).map(|(a, b)| a - b).sum();
        assert!(rise > 0.0 && (outputs[1][88] - rise).abs() <= 1e-5 * rise);
        assert_eq!(outputs[2][88], 0.0, "a fall has no positive change");
    }

    #[test]
    fn reset_keeps_the_history_like_the_reference() {
        let mut continued = processor(true);
        continued.process(&frame(0.1)).unwrap();
        let expected = continued.process(&frame(0.2)).unwrap();
        let mut reset = processor(true);
        reset.process(&frame(0.1)).unwrap();
        reset.reset();
        assert_eq!(reset.process(&frame(0.2)).unwrap(), expected);
        assert!(expected.features[88] > 0.0);
    }

    #[test]
    fn without_flux_the_rows_are_those_of_cqt_and_nothing_is_remembered() {
        let cqt = CqtProcessor::new(44_100, 1_470, Some(f64::INFINITY)).unwrap();
        let mut processor = processor(false);
        for length in [2_940, 8_820, 2_940] {
            let samples = tone(0.1, length);
            assert_eq!(
                processor.transform(&samples).unwrap(),
                cqt.transform(&samples).unwrap()
            );
        }
    }

    #[test]
    fn calls_of_different_length_fail_like_numpy_and_keep_the_history() {
        let mut processor = processor(true);
        let score = processor.transform(&tone(0.1, 14_700)).unwrap();
        assert_eq!(score.len(), 9);
        assert!(score.iter().all(|row| row[88] == 0.0));
        let message = processor
            .transform(&tone(0.1, 2_940))
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("3 frames") && message.contains("11 frames"),
            "{message}"
        );
        assert_eq!(processor.transform(&tone(0.2, 14_700)).unwrap().len(), 9);
    }
}
