//! Port of `hiddenmarkov.forward_algorithm_step` and `ConstantTransitionModel`
//! (python-hiddenmarkov), as used by the MIDI HMMs with `log_probabilities=False`.
//!
//! Upstream: `hiddenmarkov/__init__.py` (package `python-hiddenmarkov`, not pymatchmaker).
//!
//! Transition rows are stored as their band of entries that are not exactly zero (research
//! R2): skipping exact zeros does not change any product or sum.
//!
//! ```
//! use matchmaker_rs::prob::hidden_markov::{argmax, Banded};
//! let t = Banded::from_bands(vec![(1, vec![1.0]), (1, vec![1.0])]);
//! assert_eq!(t.transpose_dot(&[0.5, 0.5]), vec![0.0, 1.0]);
//! assert_eq!(argmax(&[0.2, 0.7, 0.7]), 1);
//! ```

/// A square matrix stored as one `(first column, values)` band per row.
#[derive(Debug, Clone)]
pub struct Banded {
    rows: Vec<(usize, Vec<f64>)>,
}

impl Banded {
    /// Build from bands `(first column, values)`.
    pub fn from_bands(rows: Vec<(usize, Vec<f64>)>) -> Self {
        Banded { rows }
    }

    /// `transition_matrix.T @ v`.
    pub fn transpose_dot(&self, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.rows.len()];
        for ((start, values), &vi) in self.rows.iter().zip(v) {
            for (o, &t) in out[*start..].iter_mut().zip(values) {
                *o += t * vi;
            }
        }
        out
    }
}

/// One forward step: `fv = obs * prior; fv /= max(sum(fv), 1e-6)`, where the prior is the
/// initial distribution on the first step and `T.T @ fv` afterwards.
pub fn forward_algorithm_step(
    observation_prob: &[f64],
    transition: &Banded,
    init: &[f64],
    forward_variable: Option<&[f64]>,
) -> Vec<f64> {
    let prior = match forward_variable {
        None => init.to_vec(),
        Some(fv) => transition.transpose_dot(fv),
    };
    let mut fv: Vec<f64> = observation_prob
        .iter()
        .zip(&prior)
        .map(|(o, p)| o * p)
        .collect();
    let sum = fv.iter().sum::<f64>().max(1e-6);
    fv.iter_mut().for_each(|v| *v /= sum);
    fv
}

/// `np.argmax`: index of the first maximum.
pub fn argmax(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}
