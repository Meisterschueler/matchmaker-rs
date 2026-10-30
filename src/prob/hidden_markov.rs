//! Port of `hiddenmarkov.forward_algorithm_step` and `ConstantTransitionModel`
//! (python-hiddenmarkov), as used by the MIDI HMMs with `log_probabilities=False`.
//!
//! Upstream: `hiddenmarkov/__init__.py` (package `python-hiddenmarkov`, not pymatchmaker).
//!
//! Transition rows are copies of one kernel centred on `loc(i)` and divided by their row sum, so
//! `T.T @ v` is a convolution with that kernel instead of a pass over an N x band matrix, which
//! is limited by memory bandwidth (the reference's dense matrix has the same rows).
//!
//! ```
//! use matchmaker_rs::prob::hidden_markov::argmax;
//! use matchmaker_rs::prob::hmm_models::stable_transition_matrix;
//! let t = stable_transition_matrix(4, 1.0, false);
//! let prior = t.transpose_dot(&[1.0, 0.0, 0.0, 0.0]);
//! assert!((prior.iter().sum::<f64>() - 1.0).abs() < 1e-12); // row 0 sums to one
//! assert_eq!(argmax(&[0.2, 0.7, 0.7]), 1);
//! ```

use ndarray::{Array1, s};

/// Square matrix whose row `i` is `kernel`, centred on column `loc(i)` (`i + 2` for even `i`
/// with inserted states, else `i + 1`), restricted to the columns `0..n` and divided by its sum.
#[derive(Debug, Clone)]
pub struct KernelTransition {
    /// Per residue `j % step`: the kernel at offsets `top, top - step, ...` and `top`. With
    /// inserted states all centres are even, so only every second offset can contribute.
    taps: Vec<(Array1<f64>, i64)>,
    row_sum: Vec<f64>,
    inserted: bool,
    /// Zero padding left of the shifted input, in slots.
    pad: usize,
}

impl KernelTransition {
    /// `kernel[k]` is the unnormalised entry at offset `k - below` from the centre; `row_sum[i]`
    /// the sum of row `i` over its columns in `0..n`.
    pub fn new(kernel: &[f64], below: usize, row_sum: Vec<f64>, inserted: bool) -> Self {
        let step = if inserted { 2 } else { 1 };
        let (below, max_offset) = (below as i64, (kernel.len() - below) as i64 - 1);
        let taps = (0..step)
            .map(|residue| {
                let top = max_offset - (max_offset - residue).rem_euclid(step);
                let entries = std::iter::successors(Some(top), |z| Some(z - step))
                    .take_while(|&z| z >= -below)
                    .map(|z| kernel[(z + below) as usize])
                    .collect();
                (entries, top)
            })
            .collect();
        KernelTransition {
            taps,
            row_sum,
            inserted,
            pad: (max_offset as usize + 1) / step as usize + 2,
        }
    }

    fn loc(&self, i: usize) -> usize {
        // `& 1` keeps the MSRV
        if self.inserted && i & 1 == 0 {
            i + 2
        } else {
            i + 1
        }
    }

    /// `transition_matrix.T @ v`: `out[j] = sum_i kernel[j - loc(i)] * v[i] / row_sum[i]`.
    pub fn transpose_dot(&self, v: &[f64]) -> Vec<f64> {
        let n = self.row_sum.len();
        let step = self.taps.len();
        let mut shifted = Array1::<f64>::zeros(n + self.pad + self.taps[0].0.len() + 12);
        for (i, (&vi, &sum)) in v.iter().zip(&self.row_sum).enumerate() {
            shifted[self.pad + self.loc(i) / step] += vi / sum;
        }
        (0..n)
            .map(|j| {
                let (taps, top) = &self.taps[j % step];
                let first = ((j as i64 - top) / step as i64 + self.pad as i64) as usize;
                taps.dot(&shifted.slice(s![first..first + taps.len()]))
            })
            .collect()
    }
}

/// One forward step: `fv = obs * prior; fv /= max(sum(fv), 1e-6)`, where the prior is the
/// initial distribution on the first step and `T.T @ fv` afterwards.
pub fn forward_algorithm_step(
    observation_prob: &[f64],
    transition: &KernelTransition,
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
