//! Port of `hiddenmarkov.forward_algorithm_step` and `ConstantTransitionModel`
//! (python-hiddenmarkov), as used by the MIDI HMMs with `log_probabilities=False`.
//!
//! Upstream: `hiddenmarkov/__init__.py` (package `python-hiddenmarkov`, not pymatchmaker).
//!
//! Transition rows are stored as their band of entries that are not exactly zero (research
//! R2): skipping exact zeros does not change any product or sum. pymatchmaker 0.5.2 delegates
//! the HMM forward step to python-hiddenmarkov 0.3.0's optimized implementation; this port
//! bounds its product to the reached rows and transition columns while preserving the guarded
//! normalization (`fv /= max(fv.sum(), 1e-6)`).
//!
//! ```
//! use matchmaker_rs::prob::hidden_markov::{argmax, Banded};
//! let t = Banded::from_bands(vec![(1, vec![1.0]), (1, vec![1.0])]);
//! assert_eq!(t.transpose_dot(&[0.5, 0.5]), vec![0.0, 1.0]);
//! assert_eq!(argmax(&[0.2, 0.7, 0.7]), 1);
//! ```

use crate::utils::np_sum::np_sum;

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

    fn transpose_dot_reachable(&self, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.rows.len()];
        let Some(lo) = v.iter().position(|&value| value != 0.0) else {
            return out;
        };
        let hi = v.iter().rposition(|&value| value != 0.0).unwrap() + 1;
        let max_column = self.rows[lo..hi]
            .iter()
            .filter_map(|(start, values)| {
                values
                    .iter()
                    .rposition(|&value| value != 0.0)
                    .map(|offset| start + offset)
            })
            .max();
        let Some(max_column) = max_column else {
            return out;
        };
        let end = max_column + 1;

        for ((start, values), &weight) in self.rows[lo..hi].iter().zip(&v[lo..hi]) {
            if weight == 0.0 || *start >= end {
                continue;
            }
            let count = values.len().min(end - start);
            for (output, &transition) in
                out[*start..*start + count].iter_mut().zip(&values[..count])
            {
                *output += transition * weight;
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
        Some(fv) => transition.transpose_dot_reachable(fv),
    };
    let mut fv: Vec<f64> = observation_prob
        .iter()
        .zip(&prior)
        .map(|(o, p)| o * p)
        .collect();
    let sum = np_sum(&fv).max(1e-6);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn transition() -> Banded {
        Banded::from_bands(vec![
            (0, vec![0.7, 0.3]),
            (0, vec![0.1, 0.7, 0.2]),
            (1, vec![0.2, 0.6, 0.2]),
            (2, vec![0.2, 0.6, 0.2]),
            (3, vec![0.2, 0.6, 0.2]),
            (4, vec![0.2, 0.8]),
        ])
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (&actual, &expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-14, "{actual} != {expected}");
        }
    }

    #[test]
    fn reachable_product_matches_full_product_for_noncontiguous_support() {
        let transition = transition();
        let forward = [0.0, 0.4, 0.0, 0.6, 0.0, 0.0];

        assert_close(
            &transition.transpose_dot_reachable(&forward),
            &transition.transpose_dot(&forward),
        );
    }

    #[test]
    fn reachable_product_keeps_unreachable_columns_zero() {
        let transition = transition();
        let forward = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let prior = transition.transpose_dot_reachable(&forward);

        assert_close(&prior, &transition.transpose_dot(&forward));
        assert_eq!(prior[3..], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn reachable_product_of_zero_forward_variable_is_zero() {
        let transition = transition();
        let prior = transition.transpose_dot_reachable(&[0.0; 6]);

        assert_eq!(prior, [0.0; 6]);
    }

    #[test]
    fn forward_step_uses_initial_distribution_on_first_observation() {
        let transition = transition();
        let observation = [0.5, 1.0, 0.25, 0.0, 0.0, 0.0];
        let initial = [0.4, 0.3, 0.2, 0.1, 0.0, 0.0];

        let actual = forward_algorithm_step(&observation, &transition, &initial, None);
        let expected = [0.4 * 0.5, 0.3, 0.2 * 0.25, 0.0, 0.0, 0.0];
        let sum: f64 = expected.iter().sum();
        let expected: Vec<f64> = expected.iter().map(|value| value / sum).collect();

        assert_close(&actual, &expected);
    }

    #[test]
    fn forward_step_keeps_zero_forward_variable_zero() {
        let transition = transition();

        assert_eq!(
            forward_algorithm_step(&[1.0; 6], &transition, &[1.0; 6], Some(&[0.0; 6])),
            [0.0; 6]
        );
    }
}
