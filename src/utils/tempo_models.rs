//! Port of `KalmanTempoModel` and the audio tracker integration of `ArztTempoModel`.
//!
//! The reference mixes Python floats with NumPy scalars, so single (`f32`) and double (`f64`)
//! precision alternate between operands (research R4). Every mixed-precision line names the
//! Python expression it mirrors.
//!
//! Upstream: `matchmaker/utils/tempo_models.py`.
//!
//! ```
//! use matchmaker_rs::utils::tempo_models::KalmanTempoModel;
//! let mut tempo = KalmanTempoModel::new(0.6, 0.0);
//! tempo.update_beat_period(0.0, 0.0); // the first update only initialises
//! tempo.update_beat_period(0.5, 1.0);
//! assert!(tempo.beat_period() < 0.6);
//! ```

/// Tunable parameters of the filter (`KalmanTempoModel.__init__`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KalmanParams {
    /// Initial beat period in seconds (`60 / 100` in `PitchIOIHMM`).
    pub init_beat_period: f64,
    /// `trans_par`.
    pub trans_par: f64,
    /// `trans_var`.
    pub trans_var: f64,
    /// `obs_var`.
    pub obs_var: f64,
    /// `init_var`.
    pub init_var: f64,
}

impl Default for KalmanParams {
    fn default() -> Self {
        KalmanParams {
            init_beat_period: 0.6,
            trans_par: 1.0,
            trans_var: 0.03,
            obs_var: 0.0213,
            init_var: 1.0,
        }
    }
}

/// Linear Kalman filter tracking the beat period (seconds per beat).
#[derive(Debug, Clone)]
pub struct KalmanTempoModel {
    params: KalmanParams,
    beat_period: f64,
    var_est: f64,
    prev_perf_onset: f64,
    prev_score_onset: f32,
    counter: u32,
}

impl KalmanTempoModel {
    /// Model with the reference defaults except for the initial beat period and score onset.
    pub fn new(init_beat_period: f64, init_score_onset: f32) -> Self {
        Self::with_params(
            KalmanParams {
                init_beat_period,
                ..KalmanParams::default()
            },
            init_score_onset,
        )
    }

    /// Model with explicit parameters.
    pub fn with_params(params: KalmanParams, init_score_onset: f32) -> Self {
        KalmanTempoModel {
            beat_period: params.init_beat_period,
            var_est: params.init_var,
            prev_perf_onset: 0.0,
            prev_score_onset: init_score_onset,
            counter: 0,
            params,
        }
    }

    /// Current beat period in seconds per beat.
    pub fn beat_period(&self) -> f64 {
        self.beat_period
    }

    /// One predict/update cycle for a performed onset (seconds) and its score onset (beats).
    pub fn update_beat_period(&mut self, performed_onset: f64, score_onset: f32) {
        let KalmanParams {
            trans_par,
            trans_var,
            obs_var,
            ..
        } = self.params;
        if self.counter == 0 {
            self.prev_perf_onset = performed_onset;
            self.prev_score_onset = score_onset;
            self.counter += 1;
            return;
        }
        let performed_ioi = (performed_onset - self.prev_perf_onset).abs();
        // abs(score_onset - self.prev_score_onset): both float32 scalars
        let score_ioi: f32 = (score_onset - self.prev_score_onset).abs();
        self.prev_score_onset = score_onset;
        self.prev_perf_onset = performed_onset;

        // `beat_period` and `var_est` are Python numbers until the first update has run;
        // afterwards `beat_period` is float64 and `var_est` float32
        let first = self.counter == 1;
        let period_pred = self.beat_period * trans_par;
        // err = performed_ioi - score_ioi * period_pred
        let predicted_ioi = if first {
            f64::from(score_ioi * period_pred as f32)
        } else {
            f64::from(score_ioi) * period_pred
        };
        let err = performed_ioi - predicted_ioi;

        // var_pred = (trans_par**2) * var_est + trans_var
        let var_pred: f32 = if first {
            ((trans_par * trans_par) * self.var_est + trans_var) as f32
        } else {
            (trans_par * trans_par) as f32 * self.var_est as f32 + trans_var as f32
        };
        // float(var_pred * score_ioi) / ((score_ioi**2) * var_pred + obs_var), all float32
        let kalman_gain: f32 =
            (var_pred * score_ioi) / ((score_ioi * score_ioi) * var_pred + obs_var as f32);
        // beat_period = period_pred + kalman_gain * err
        self.beat_period = period_pred + f64::from(kalman_gain) * err;
        // var_est = (1 - kalman_gain * score_ioi) * var_pred
        self.var_est = f64::from((1.0 - kalman_gain * score_ioi) * var_pred);
        self.counter += 1;
    }
}

/// Port of `ArztTempoModel`: seeded score-frame alterations and relative tempo state.
#[derive(Debug, Clone)]
pub struct ArztTempoModel {
    pub(crate) window_frames: i64,
    pub(crate) min_past_frames: i64,
    pub(crate) current_tempo: f64,
    pub(crate) beat_period: f64,
    pub(crate) n_inserted: usize,
    pub(crate) n_deleted: usize,
    pub(crate) n_postponed: usize,
    last_altered: i64,
    consecutive: usize,
    insert_pending: bool,
    random: Box<rand_mt::Mt>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum ScoreAlteration {
    Insert,
    Delete,
}

impl ArztTempoModel {
    /// Initialize reference defaults at a finite positive frame rate and explicit MT19937 seed.
    pub fn new(frame_rate: f64, seed: u32) -> crate::Result<Self> {
        if !frame_rate.is_finite() || frame_rate <= 0.0 {
            return Err(crate::Error::AudioConfig {
                key: "frame_rate".into(),
                reason: "must be finite and positive".into(),
            });
        }
        Ok(Self {
            window_frames: (3.0 * frame_rate).round_ties_even() as i64,
            min_past_frames: frame_rate.round_ties_even() as i64,
            current_tempo: 1.0,
            beat_period: 0.5,
            n_inserted: 0,
            n_deleted: 0,
            n_postponed: 0,
            last_altered: -1,
            consecutive: 0,
            insert_pending: false,
            random: Box::new(rand_mt::Mt::new(seed)),
        })
    }

    /// Current relative performance/score tempo factor.
    pub fn current_tempo(&self) -> f64 {
        self.current_tempo
    }

    /// Current seconds per beat.
    pub fn beat_period(&self) -> f64 {
        self.beat_period
    }

    pub(crate) fn alteration(
        &mut self,
        relative_tempo: Option<f64>,
        score_frame: usize,
        onsets: &[bool],
    ) -> Option<ScoreAlteration> {
        let Some(tempo) = relative_tempo.filter(|tempo| tempo.is_finite() && *tempo > 0.0) else {
            self.consecutive = 0;
            return None;
        };
        let beat_period = 0.5 / tempo.max(1e-4);
        if !beat_period.is_finite() || beat_period <= 0.0 {
            self.consecutive = 0;
            return None;
        }
        self.current_tempo = tempo;
        self.beat_period = beat_period;
        let tempo = tempo.clamp(0.85, 1.20);
        if tempo >= 1.0 {
            self.insert_pending = false;
        }
        if score_frame as i64 <= self.last_altered + 10 || score_frame + 2 >= onsets.len() {
            return None;
        }
        if self.consecutive >= 3 {
            self.consecutive = 0;
            return None;
        }
        let high = self.random.next_u32() >> 5;
        let low = self.random.next_u32() >> 6;
        let uniform = (f64::from(high) * 67_108_864.0 + f64::from(low)) / 9_007_199_254_740_992.0;
        let alteration = if tempo > 1.0 && uniform > 1.0 / tempo {
            self.n_deleted += 1;
            ScoreAlteration::Delete
        } else if tempo < 1.0 && (self.insert_pending || uniform > tempo) {
            if onsets[score_frame] || onsets[score_frame + 1] {
                self.insert_pending = true;
                self.n_postponed += 1;
                self.consecutive = 0;
                return None;
            }
            self.insert_pending = false;
            self.n_inserted += 1;
            ScoreAlteration::Insert
        } else {
            self.consecutive = 0;
            return None;
        };
        self.consecutive += 1;
        self.last_altered = score_frame as i64;
        Some(alteration)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_update_only_initialises() {
        let mut tempo = KalmanTempoModel::new(0.6, 0.0);
        tempo.update_beat_period(1.0, 2.0);
        assert_eq!(tempo.beat_period(), 0.6);
    }

    #[test]
    fn converges_towards_observed_tempo() {
        let mut tempo = KalmanTempoModel::new(0.6, 0.0);
        for k in 0..40 {
            tempo.update_beat_period(0.4 * f64::from(k), k as f32);
        }
        assert!(
            (tempo.beat_period() - 0.4).abs() < 0.01,
            "{}",
            tempo.beat_period()
        );
    }
}
