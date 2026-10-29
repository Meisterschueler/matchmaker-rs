//! Building blocks of `matchmaker.prob.hmm` shared by `pthmm` and `hmm`:
//! `stable_transition_matrix`, `init_dist`, `compute_discrete_pitch_profiles`,
//! `compute_bernoulli_pitch_probabilities` and the inserted-state positions.
//!
//! Upstream: `matchmaker/prob/hmm.py`.
//!
//! ```
//! use matchmaker_rs::prob::hmm_models::gumbel_l_pdf;
//! assert!((gumbel_l_pdf(0.0, 0.0, 1.0) - (-1.0f64).exp()).abs() < 1e-15);
//! ```

use std::collections::BTreeSet;

use super::hidden_markov::Banded;

/// `scipy.stats.gumbel_l.pdf(x, loc, scale) = exp(z - exp(z)) / scale`, `z = (x - loc) / scale`.
pub fn gumbel_l_pdf(x: f64, loc: f64, scale: f64) -> f64 {
    let z = (x - loc) / scale;
    (z - z.exp()).exp() / scale
}

/// Port of `stable_transition_matrix(n_states, mp_trans_state=1, dist=gumbel_l, scale,
/// inserted_states)`: row `i` is `gumbel_l.pdf(arange(n), loc, scale)` normalised, with
/// `loc = i + 2` for onset states and `i + 1` otherwise.
pub fn stable_transition_matrix(n: usize, scale: f64, inserted_states: bool) -> Banded {
    // gumbel_l.pdf is exactly 0 for z > ~6.6 and z < ~-745; only that band is evaluated
    let (below, above) = ((760.0 * scale) as usize, (10.0 * scale) as usize + 1);
    let rows = (0..n)
        .map(|i| {
            let loc = if inserted_states && i % 2 == 0 {
                i + 2
            } else {
                i + 1
            };
            let start = loc.saturating_sub(below).min(n);
            let end = (loc + above).min(n);
            let mut values: Vec<f64> = (start..end)
                .map(|j| gumbel_l_pdf(j as f64, loc as f64, scale))
                .collect();
            let sum: f64 = values.iter().sum();
            values.iter_mut().for_each(|v| *v /= sum);
            (start, values)
        })
        .collect();
    Banded::from_bands(rows)
}

/// Port of `init_dist(n_states, dist=gumbel_l, loc=0, scale=10)` (not normalised).
pub fn init_dist(n: usize) -> Vec<f64> {
    let scale = if 10.0 < n as f64 {
        10.0
    } else {
        n as f64 / 10.0
    };
    (0..n).map(|j| gumbel_l_pdf(j as f64, 0.0, scale)).collect()
}

/// Score states: unique onsets, with `has_insertions` a midpoint between consecutive onsets
/// (`np.insert(u, arange(1, len(u)), u[:-1] + 0.5 * np.diff(u))`, in `float32`).
pub fn score_states(unique_onsets: &[f32], has_insertions: bool) -> Vec<f32> {
    if !has_insertions {
        return unique_onsets.to_vec();
    }
    let mut states = Vec::with_capacity(2 * unique_onsets.len());
    for (i, &o) in unique_onsets.iter().enumerate() {
        states.push(o);
        if let Some(&next) = unique_onsets.get(i + 1) {
            states.push(o + 0.5 * (next - o));
        }
    }
    states
}

/// Port of `compute_discrete_pitch_profiles(chord_pitches, piano_range, inserted_states)`:
/// row-major `(n_states, 88 | 128)`.
pub fn compute_discrete_pitch_profiles(
    chords: &[BTreeSet<i32>],
    piano_range: bool,
    inserted_states: bool,
) -> Vec<Vec<f64>> {
    const PROFILE: [f64; 5] = [0.02, 0.02, 1.0, 0.02, 0.02];
    const EPS: f64 = 0.01;
    let mut rows = Vec::new();
    for (c, chord) in chords.iter().enumerate() {
        let mut matrix = [0.0f64; 128];
        for &p in chord {
            if let Some(m) = usize::try_from(p).ok().and_then(|p| matrix.get_mut(p)) {
                *m = 1.0;
            }
        }
        // convolve(chord_matrix, profile[None, :], mode="same")
        let row: Vec<f64> = (0..128usize)
            .map(|j| {
                (0..5)
                    .filter_map(|k| {
                        let s = (j + 2).checked_sub(k).filter(|&s| s < 128)?;
                        Some(matrix[s] * PROFILE[k])
                    })
                    .sum()
            })
            .collect();
        rows.push(row);
        if inserted_states && c + 1 < chords.len() {
            rows.push(vec![0.0; 128]);
        }
    }
    for row in &mut rows {
        let row_eps: Vec<f64> = row.iter().map(|v| v + EPS).collect();
        let row_eps = if piano_range {
            row_eps[21..109].to_vec()
        } else {
            row_eps
        };
        let sum: f64 = row_eps.iter().sum();
        *row = row_eps.into_iter().map(|v| v / sum).collect();
    }
    rows
}

/// Binary pitch vector of `ACCPitchIOIObservationModel.compute_pitch_observation_probability`
/// (`pitch_prof_obs[0, pitch_obs.astype(int)] = 1`). Pitches outside the vector are ignored
/// (the reference wraps negative ones and raises for large ones, research R3).
pub fn pitch_list_to_vector(pitches: &[f32], piano_range: bool) -> Vec<f32> {
    let mut vector = vec![0.0f32; if piano_range { 88 } else { 128 }];
    for &p in pitches {
        let slot = (p as i64)
            .try_into()
            .ok()
            .and_then(|i: usize| vector.get_mut(i));
        if let Some(slot) = slot {
            *slot = 1.0;
        }
    }
    vector
}

/// `ACCPitchIOIObservationModel.compute_ioi_observation_probability` for state `state` when the
/// follower was in `current_state`: `norm * exp(-0.5 * (tempo * |s_cur - s_j| - ioi_obs)^2 * precision)`
/// with `norm = sqrt(0.5 * precision / pi)`. Row `current_state` of the reference's IOI matrix
/// is `|s_cur - s_j|` in `f64` (`pdist`), so it is computed on the fly.
pub fn ioi_observation_probability(
    score_positions: &[f32],
    current_state: usize,
    state: usize,
    tempo_est: f64,
    ioi_obs: f64,
    ioi_precision: f64,
) -> f64 {
    let norm_term = (0.5 * ioi_precision / std::f64::consts::PI).sqrt();
    let ioi_score =
        (f64::from(score_positions[current_state]) - f64::from(score_positions[state])).abs();
    let diff = tempo_est * ioi_score - ioi_obs;
    norm_term * (-0.5 * (diff * diff) * ioi_precision).exp()
}

/// Port of `compute_bernoulli_pitch_probabilities`: `prod(p^x * (1 - p)^(1 - x))` per state
/// for a binary observation `x` (where `p^1 = p` and `p^0 = 1` exactly).
pub fn bernoulli_pitch_probabilities(observation: &[f32], profiles: &[Vec<f64>]) -> Vec<f64> {
    profiles
        .iter()
        .map(|p| {
            p.iter()
                .zip(observation)
                .map(|(&p, &x)| if x > 0.5 { p } else { 1.0 - p })
                .product()
        })
        .collect()
}
