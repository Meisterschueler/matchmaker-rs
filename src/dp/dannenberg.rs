//! Port of `matchmaker.dp.dannenberg.DPDannenberg` (method key `dannenberg`, MIDI).
//!
//! Upstream: `matchmaker/dp/dannenberg.py`.
//!
//! Windowed dynamic-programming matcher (Bloch & Dannenberg 1985, Sec. 6). Rows of the DP
//! matrix are score compound events (one pitch set per unique onset beat); columns are
//! individual performance MIDI notes. Each incoming note updates one column by considering
//! three predecessors — same row (extend), diagonal (advance and match), and vertical
//! (skip note) — choosing the one that maximises the rating, which penalises wrong and
//! missing notes. Pitch sets are `u128` bitmasks (bit = MIDI pitch 0–127); the set algebra
//! is unchanged.
//!
//! ```
//! use matchmaker_rs::base::OnlineAlignment;
//! use matchmaker_rs::dp::dannenberg::{DannenbergConfig, DannenbergEvent};
//! use matchmaker_rs::score::Score;
//! let score = Score::load("tests/data/simple_mozart_k265_var1.musicxml", true)?;
//! let (cevts, onsets) = score.score_cevts();
//! let follower = DannenbergEvent::new(cevts, onsets, DannenbergConfig::default(), 120.0);
//! assert_eq!(follower.score_positions().len(), score.unique_onsets().len());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use crate::base::{AlignmentPath, OnlineAlignment, StepDebug};
use crate::features::Observation;
use crate::prob::hidden_markov::argmax;

/// Constructor arguments of `DPDannenberg` (the `methods.yaml` defaults reach it for MIDI).
#[derive(Debug, Clone, PartialEq)]
pub struct DannenbergConfig {
    /// `c_m`: cost per missing score note when advancing past a score cevt (default 0.5).
    pub c_m: f64,
    /// `c_e`: cost per extra performance note not in the current score cevt (default 0.0).
    pub c_e: f64,
    /// `epsilon`: absolute time threshold in seconds for grouping notes into the same
    /// performance cevt (default 0.090 s).
    pub epsilon: f64,
    /// `epsilon_fraction`: fraction of the predicted next-cevt time for the relative
    /// grouping test (default 0.25).
    pub epsilon_fraction: f64,
    /// `window`: number of score cevts ahead of the current best match to consider
    /// (default 50, at least 1).
    pub window: usize,
    /// `smoothing`: exponential smoothing coefficient for tempo tracking (default 0.5).
    pub smoothing: f64,
}

impl Default for DannenbergConfig {
    fn default() -> Self {
        DannenbergConfig {
            c_m: 0.5,
            c_e: 0.0,
            epsilon: 0.090,
            epsilon_fraction: 0.25,
            window: 50,
            smoothing: 0.5,
        }
    }
}

/// The `dannenberg` follower for MIDI.
#[derive(Debug, Clone)]
pub struct DannenbergEvent {
    /// Score compound events from [`Score::score_cevts`](crate::score::Score::score_cevts).
    cevts: Vec<u128>,
    score_positions: Vec<f32>,
    config: DannenbergConfig,
    /// `_values`: current DP rating column.
    values: Vec<f64>,
    /// `_used`: pitches already credited per row in the current column.
    used: Vec<u128>,
    /// `_prev_values` / `_prev_used`: the previous column.
    prev_values: Vec<f64>,
    prev_used: Vec<u128>,
    /// `_best_value` / `_best_row`: the monotone best rating and its row.
    best_value: f64,
    best_row: usize,
    current_index: usize,
    /// `_prev_note_time`: only feeds the cevt-grouping test.
    prev_note_time: Option<f64>,
    /// `_beats_per_second`: smoothed tempo, seeded from the Matchmaker tempo.
    beats_per_second: f64,
    /// `_anchor_beat` / `_anchor_time`: last matched beat and its performance time.
    anchor_beat: f64,
    anchor_time: f64,
    /// `_prev_match_time` / `_prev_match_beat`: previous best-match update.
    prev_match_time: Option<f64>,
    prev_match_beat: Option<f64>,
    path: AlignmentPath,
}

impl DannenbergEvent {
    /// `DPDannenberg(reference_features, score_positions, ..., tempo)`; the features and
    /// positions come from [`Score::score_cevts`](crate::score::Score::score_cevts),
    /// `tempo` (BPM) from the Matchmaker's score tempo or default.
    pub fn new(
        cevts: Vec<u128>,
        score_positions: Vec<f32>,
        config: DannenbergConfig,
        tempo: f64,
    ) -> Self {
        let n_states = cevts.len();
        // Initial column (before any performance note): matching must start at the first
        // score cevt, so row r begins with the cost of missing every note in cevts 0..r
        // (rather than 0 everywhere, which would allow a wrong first note to lock onto
        // any row in the window).
        let mut values = vec![0.0f64; n_states];
        let mut missing = 0.0;
        for r in 1..n_states {
            missing += f64::from(cevts[r - 1].count_ones());
            values[r] = -config.c_m * missing;
        }
        let anchor_beat = score_positions.first().map_or(0.0, |&b| f64::from(b));
        DannenbergEvent {
            cevts,
            score_positions,
            values: values.clone(),
            used: vec![0; n_states],
            prev_values: values,
            prev_used: vec![0; n_states],
            best_value: f64::NEG_INFINITY,
            best_row: 0,
            current_index: 0,
            prev_note_time: None,
            beats_per_second: tempo / 60.0,
            anchor_beat,
            anchor_time: 0.0,
            prev_match_time: None,
            prev_match_beat: None,
            path: AlignmentPath::default(),
            config,
        }
    }

    /// Port of `_is_new_cevt`: epsilon and epsilon-fraction tests (Sec. 5, p. 283-284).
    fn is_new_cevt(&self, note_time: f64) -> bool {
        let Some(prev) = self.prev_note_time else {
            return true;
        };
        let dt = note_time - prev;
        if dt > self.config.epsilon {
            return true;
        }
        if self.beats_per_second > 0.0 && self.best_row + 1 < self.cevts.len() {
            let beat_gap = f64::from(self.score_positions[self.best_row + 1])
                - f64::from(self.score_positions[self.best_row]);
            let predicted_dt = beat_gap / self.beats_per_second;
            if predicted_dt > 0.0 && dt > self.config.epsilon_fraction * predicted_dt {
                return true;
            }
        }
        false
    }

    /// Port of `step`: one DP column update for one performance note.
    fn step(&mut self, pitch: u32, note_time: f64) {
        let bit = 1u128 << pitch;

        // Upstream computes the cevt-grouping classification and ignores it
        // (`if self._is_new_cevt(note_time): pass`); ported as written.
        let _ = self.is_new_cevt(note_time);
        self.prev_note_time = Some(note_time);

        let lo = self.best_row.saturating_sub(10);
        let hi = (self.best_row + self.config.window).min(self.cevts.len());

        // Shift columns: save current as previous (Sec. 6, p. 285).
        self.prev_values[lo..hi].copy_from_slice(&self.values[lo..hi]);
        self.prev_used[lo..hi].copy_from_slice(&self.used[lo..hi]);

        // DP update: one column per performance note (Fig. 6-2, p. 285).
        for r in lo..hi {
            let b = self.cevts[r];

            let y_val = self.prev_values[r];
            let y_used = self.prev_used[r];

            let mut z_val = y_val;
            let mut z_used = y_used;

            if b & bit == 0 {
                z_val -= self.config.c_e;
            } else {
                if y_used & bit == 0 {
                    z_val = y_val + 1.0;
                    z_used = y_used | bit;
                }
                if r > lo {
                    let a = self.cevts[r - 1];
                    let w_val = self.prev_values[r - 1];
                    let w_used = self.prev_used[r - 1];
                    let missing_in_a = f64::from((a & !w_used).count_ones());
                    let d = w_val + 1.0 - self.config.c_m * missing_in_a;
                    if d > z_val {
                        z_val = d;
                        z_used = bit;
                    }
                }
            }

            if r > lo {
                // the vertical predecessor reads the already-updated current column
                let a_skip = self.cevts[r - 1];
                let x_val = self.values[r - 1];
                let x_used = self.used[r - 1];
                let missing_in_a_skip = f64::from((a_skip & !x_used).count_ones());
                let v = x_val - self.config.c_m * missing_in_a_skip;
                if v >= z_val {
                    z_val = v;
                    z_used = 0;
                }
            }

            self.values[r] = z_val;
            self.used[r] = z_used;
        }

        let best_r = lo + argmax(&self.values[lo..hi]);
        let best_v = self.values[best_r];

        if best_v > self.best_value {
            self.best_value = best_v;
            self.best_row = best_r;
            self.current_index = best_r;

            let beat = f64::from(self.score_positions[best_r]);
            if let (Some(prev_time), Some(prev_beat)) = (self.prev_match_time, self.prev_match_beat)
            {
                let elapsed = note_time - prev_time;
                let advanced = beat - prev_beat;
                if elapsed > 0.05 && 0.0 < advanced && advanced < 16.0 {
                    let rate = advanced / elapsed;
                    self.beats_per_second = self.config.smoothing * self.beats_per_second
                        + (1.0 - self.config.smoothing) * rate;
                }
            }

            self.anchor_beat = beat;
            self.anchor_time = note_time;
            self.prev_match_time = Some(note_time);
            self.prev_match_beat = Some(beat);
        }
    }

    /// Port of `get_current_position`: anchor plus smoothed-tempo drift, capped at the
    /// next score event.
    fn current_position(&self, perf_time: f64) -> f64 {
        let drift = self.beats_per_second * (perf_time - self.anchor_time);
        let mut pos = self.anchor_beat + drift;
        if self.current_index + 1 < self.cevts.len() {
            pos = pos.min(f64::from(self.score_positions[self.current_index + 1]));
        }
        pos
    }
}

impl OnlineAlignment for DannenbergEvent {
    fn call(&mut self, obs: &Observation) -> f64 {
        let _span = tracing::debug_span!("inference", method = "dannenberg").entered();
        let perf_time = obs.perf_time;
        // `int(np.atleast_1d(features)[0])`; the stream never yields an empty pitch list,
        // and a pitch outside MIDI range never enters the stream either.
        if let Some(pitch) = obs
            .features
            .first()
            .and_then(|&p| u32::try_from(p as i64).ok())
            .filter(|&p| p < 128)
        {
            self.step(pitch, perf_time);
        }
        let position = self.current_position(perf_time);
        self.path.push(perf_time, position);
        tracing::debug!(perf_time, position);
        position
    }

    fn is_still_following(&self) -> bool {
        self.current_index + 1 < self.cevts.len()
    }

    fn alignment_path(&self) -> &AlignmentPath {
        &self.path
    }

    fn score_positions(&self) -> &[f32] {
        &self.score_positions
    }

    fn debug_state(&self) -> StepDebug {
        StepDebug {
            current_index: self.current_index,
            probabilities: self.values.clone(),
            beat_period: None,
            is_continuation: None,
            tempo: None,
            multi_fold: None,
            skf: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn follower(cevts: Vec<u128>, beats: Vec<f32>) -> DannenbergEvent {
        DannenbergEvent::new(cevts, beats, DannenbergConfig::default(), 120.0)
    }

    fn mask(pitches: &[u32]) -> u128 {
        pitches.iter().map(|&p| 1u128 << p).sum()
    }

    fn note(pitch: u32, time: f64) -> Observation {
        Observation {
            features: vec![pitch as f32],
            perf_time: time,
        }
    }

    #[test]
    fn wrong_first_note_does_not_lock_onto_a_later_row() {
        // The initial column penalises skipped cevts, so a pitch that only exists far
        // ahead must not move the best row there.
        let cevts = vec![mask(&[60]), mask(&[62]), mask(&[64]), mask(&[65])];
        let mut f = follower(cevts, vec![0.0, 1.0, 2.0, 3.0]);
        f.call(&note(65, 0.0));
        assert_eq!(f.debug_state().current_index, 0);
        // The real first notes still align from the start.
        f.call(&note(60, 0.5));
        assert_eq!(f.debug_state().current_index, 0);
        f.call(&note(62, 1.0));
        assert_eq!(f.debug_state().current_index, 1);
    }

    #[test]
    fn follows_matching_notes_and_stops_after_the_last_event() {
        let cevts = vec![mask(&[60, 64]), mask(&[62]), mask(&[64])];
        let mut f = follower(cevts, vec![0.0, 1.0, 2.0]);
        assert!(f.is_still_following());
        f.call(&note(60, 0.0));
        f.call(&note(64, 0.02));
        assert_eq!(f.debug_state().current_index, 0);
        f.call(&note(62, 0.5));
        assert_eq!(f.debug_state().current_index, 1);
        f.call(&note(64, 1.0));
        assert_eq!(f.debug_state().current_index, 2);
        assert!(!f.is_still_following());
        assert_eq!(f.alignment_path().perf_time.len(), 4);
    }

    #[test]
    fn single_cevt_score_never_follows_and_window_clamps() {
        let mut f = follower(vec![mask(&[60])], vec![0.0]);
        assert!(!f.is_still_following());
        // A direct call must still clamp its window to the single row without panicking.
        assert_eq!(f.call(&note(60, 0.0)), 0.0);
        assert_eq!(f.debug_state().current_index, 0);
    }

    #[test]
    fn position_drifts_with_tempo_but_is_capped_at_the_next_event() {
        let cevts = vec![mask(&[60]), mask(&[62]), mask(&[64])];
        let mut f = follower(cevts, vec![0.0, 1.0, 2.0]);
        f.call(&note(60, 0.0));
        // 120 BPM = 2 beats/s: after 0.25 s the drift is 0.5 beats...
        let pos = f.current_position(0.25);
        assert!((pos - 0.5).abs() < 1e-12, "{pos}");
        // ...but it never passes the next score event at beat 1.
        assert_eq!(f.current_position(10.0), 1.0);
    }
}
