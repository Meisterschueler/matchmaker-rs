//! Time maps: port of `partitura.score.Part._time_interpolator` (`beat_map`, `quarter_map`,
//! the musical-beat variant and their inverses).
//!
//! Upstream: `partitura/score.py` (`Part._time_interpolator`).
//!
//! Maps score time in divisions to beats in units of the time-signature denominator.
//! Evaluation follows `numpy.interp`, which `scipy.interpolate.interp1d` delegates to for 1-D
//! `float64` data, so results are bit-identical to the reference.

use super::part::{Part, Time};

/// What a [`TimeMap`] measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unit {
    /// `beat_map`: beats of the time-signature denominator.
    Beat,
    /// `beat_map` after `use_musical_beat()` (e.g. two beats per bar in 6/8).
    MusicalBeat,
    /// `quarter_map`: quarter notes, independent of time signatures.
    Quarter,
}

/// A piecewise-linear map from divisions to beats.
#[derive(Debug, Clone)]
pub(crate) struct TimeMap {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl TimeMap {
    /// Build the map for `unit`.
    ///
    /// Returns `None` when the part has fewer than two timepoints (partitura returns zeros).
    pub fn new(part: &Part, unit: Unit) -> Option<Self> {
        if part.points.len() < 2 {
            return None;
        }
        // keypoints[t] = [quarter duration, beat_type / 4]; None = carry forward.
        let mut keypoints: std::collections::BTreeMap<Time, [Option<f64>; 2]> =
            std::collections::BTreeMap::new();
        keypoints.entry(part.first_point()?).or_default();
        keypoints.entry(part.last_point()?).or_default();
        for (t, q) in part.quarter_times.iter().zip(&part.quarter_durations) {
            keypoints.entry(*t).or_default()[0] = Some(*q as f64);
        }
        for ts in part
            .time_signatures
            .iter()
            .filter(|_| unit != Unit::Quarter)
        {
            let beat_type = ts.beat_type as f64 / 4.0;
            keypoints.entry(ts.start).or_default()[1] = Some(if unit == Unit::MusicalBeat {
                beat_type * (ts.musical_beats() as f64 / ts.beats as f64)
            } else {
                beat_type
            });
        }
        let (mut cur_div, mut cur_bt) = (1.0, 1.0);
        let mut rows: Vec<[f64; 3]> = Vec::with_capacity(keypoints.len());
        for (t, kp) in keypoints {
            let div = kp[0].unwrap_or(cur_div);
            cur_div = div;
            let bt = kp[1].unwrap_or(cur_bt);
            cur_bt = bt;
            rows.push([t as f64, div, bt]);
        }
        let x: Vec<f64> = rows.iter().map(|r| r[0]).collect();
        // y = r_[0, cumsum((bt[:-1] * diff(x)) / div[:-1])]
        let mut y = Vec::with_capacity(rows.len());
        y.push(0.0);
        let mut acc = 0.0;
        for w in rows.windows(2) {
            acc += (w[0][2] * (w[1][0] - w[0][0])) / w[0][1];
            y.push(acc);
        }
        let mut map = TimeMap { x, y };

        // Anacrusis: if the first measure is shorter than its time signature, shift so that
        // the first full measure starts at 0.
        if let Some((m_start, m_end)) = part.first_measure()
            && let Some(ts) = part.time_signatures.iter().find(|ts| ts.start == m_start)
        {
            let actual_dur = map.eval(m_end as f64)? - map.eval(m_start as f64)?;
            let normal_dur = match unit {
                Unit::Beat => ts.beats as f64,
                Unit::MusicalBeat => ts.musical_beats() as f64,
                Unit::Quarter => ts.beats as f64 * 4.0 / ts.beat_type as f64,
            };
            if actual_dur < normal_dur {
                for v in &mut map.y {
                    *v -= actual_dur;
                }
            }
        }
        Some(map)
    }

    /// `interp1d(y, x)` (`inv_beat_map`): the map from beats back to divisions, sorted by beat
    /// with a stable sort like `interp1d`.
    pub fn inverse(&self) -> TimeMap {
        let mut order: Vec<usize> = (0..self.y.len()).collect();
        order.sort_by(|&a, &b| self.y[a].total_cmp(&self.y[b]));
        TimeMap {
            x: order.iter().map(|&i| self.y[i]).collect(),
            y: order.iter().map(|&i| self.x[i]).collect(),
        }
    }

    /// Evaluate at `t` exactly like `numpy.interp`; `None` outside the keypoint range
    /// (`interp1d` raises a bounds error there).
    pub fn eval(&self, t: f64) -> Option<f64> {
        let (x, y) = (&self.x, &self.y);
        let n = x.len();
        if t.is_nan() || t < x[0] || t > x[n - 1] {
            return None;
        }
        // j such that x[j] <= t < x[j + 1] (j = n - 1 if t == x[n - 1])
        let j = x.partition_point(|&v| v <= t) - 1;
        if j == n - 1 || x[j] == t {
            return Some(y[j]);
        }
        let slope = (y[j + 1] - y[j]) / (x[j + 1] - x[j]);
        let mut r = slope * (t - x[j]) + y[j];
        if r.is_nan() {
            r = slope * (t - x[j + 1]) + y[j + 1];
            if r.is_nan() && y[j] == y[j + 1] {
                r = y[j];
            }
        }
        Some(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::part::{Span, TimeSignature};

    fn part(divs: i64, beats: i64, beat_type: i64, first_measure_len: i64, end: i64) -> Part {
        let mut p = Part::new("P1");
        p.set_quarter_duration(0, divs);
        p.time_signatures.push(TimeSignature {
            start: 0,
            beats,
            beat_type,
        });
        p.measures.push(Span {
            start: Some(0),
            end: Some(first_measure_len),
            number: None,
        });
        p.add_point(0);
        p.add_point(first_measure_len);
        p.add_point(end);
        p
    }

    #[test]
    fn four_four() {
        let p = part(2, 4, 4, 8, 16);
        let m = TimeMap::new(&p, Unit::Beat).unwrap();
        assert_eq!(m.eval(0.0), Some(0.0));
        assert_eq!(m.eval(3.0), Some(1.5));
        assert_eq!(m.eval(16.0), Some(8.0));
    }

    #[test]
    fn six_eight_counts_eighths() {
        let p = part(2, 6, 8, 6, 12);
        let m = TimeMap::new(&p, Unit::Beat).unwrap();
        assert_eq!(m.eval(6.0), Some(6.0));
        // quarters ignore the time signature, musical beats count 6/8 as two beats per bar
        assert_eq!(
            TimeMap::new(&p, Unit::Quarter).unwrap().eval(6.0),
            Some(3.0)
        );
        assert_eq!(
            TimeMap::new(&p, Unit::MusicalBeat).unwrap().eval(6.0),
            Some(2.0)
        );
    }

    #[test]
    fn pickup_measure_starts_negative() {
        // 3/4 with a one-quarter pickup (divs = 2)
        let p = part(2, 3, 4, 2, 8);
        let m = TimeMap::new(&p, Unit::Beat).unwrap();
        assert_eq!(m.eval(0.0), Some(-1.0));
        assert_eq!(m.eval(2.0), Some(0.0));
        assert_eq!(m.inverse().eval(-1.0), Some(0.0));
        assert_eq!(m.inverse().eval(0.0), Some(2.0));
    }
}
