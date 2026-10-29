//! Port of `matchmaker.utils.eval`: ground truth, beat-error metrics and SPARC smoothness
//! (research R8).
//!
//! Upstream: `matchmaker/utils/eval.py`.
//!
//! ```
//! use std::path::Path;
//! use matchmaker_rs::matchmaker::{Matchmaker, Method};
//! use matchmaker_rs::utils::eval::{GroundTruth, evaluate_alignment};
//!
//! let perf = Path::new("tests/data/simple_mozart_k265_var1.mid");
//! let mut mm = Matchmaker::new("tests/data/simple_mozart_k265_var1.musicxml", Method::Pthmm, None, true)?;
//! mm.run_file(perf)?;
//! let gt = GroundTruth::load("tests/data/simple_mozart_k265_var1.match", mm.score(), Some(perf))?;
//! let result = evaluate_alignment(mm.alignment_path(), &gt);
//! println!("{}", serde_json::to_string(&result).unwrap());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::cmp::Ordering;
use std::collections::HashMap;
use std::path::Path;

use realfft::RealFftPlanner;
use serde::Serialize;

use crate::base::AlignmentPath;
use crate::error::{Error, Result};
use crate::performance::Performance;
use crate::score::Score;

/// Forward-window duration of the SPARC anchors in seconds (`ONSET_WINDOW_DURATION`).
pub const ONSET_WINDOW_DURATION: f64 = 30.0;
const SPARC_FS: f64 = 50.0;
const SPARC_FC: f64 = 10.0;
const SPARC_PADLEVEL: u32 = 4;
const SPARC_AMP_TH: f64 = 0.05;

/// Ground-truth note onsets: performance seconds and score beats (`resolve_gt`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GroundTruth {
    /// Performance onset times in seconds.
    pub perf_sec: Vec<f64>,
    /// Score positions in beats.
    pub score_beat: Vec<f64>,
}

/// Output of [`evaluate_alignment`]; serialised with the reference's keys.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EvaluationResult {
    /// Mean absolute beat error.
    pub mean_b: f64,
    /// Median absolute beat error.
    pub median_b: f64,
    /// Standard deviation of the absolute beat error.
    pub std_b: f64,
    /// Share of GT onsets within 0.05 beats.
    #[serde(rename = "accuracy_0.05b")]
    pub accuracy_0_05b: f64,
    /// Share of GT onsets within 0.1 beats.
    #[serde(rename = "accuracy_0.1b")]
    pub accuracy_0_1b: f64,
    /// Share of GT onsets within 0.3 beats.
    #[serde(rename = "accuracy_0.3b")]
    pub accuracy_0_3b: f64,
    /// Share of GT onsets within 0.5 beats.
    #[serde(rename = "accuracy_0.5b")]
    pub accuracy_0_5b: f64,
    /// Share of GT onsets within 1 beat.
    pub accuracy_1b: f64,
    /// Share of GT onsets within 2 beats.
    pub accuracy_2b: f64,
    /// Skewness of the signed beat error (biased, SciPy defaults).
    pub skewness: f64,
    /// Excess kurtosis of the signed beat error (biased, Fisher).
    pub kurtosis: f64,
    /// Mean adaptive SPARC over GT-onset windows; `None` when undefined.
    pub smoothness: Option<f64>,
}

impl GroundTruth {
    /// Load ground truth from a `.match` file or a TSV with header columns `perf_sec` and
    /// `score_beat` (`resolve_gt`). For `.match`, `performance` is the MIDI file the follower
    /// received: its onsets replace the match file's when ids and pitches agree.
    pub fn load(
        path: impl AsRef<Path>,
        score: &Score,
        performance: Option<&Path>,
    ) -> Result<GroundTruth> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("match"))
        {
            from_match(path, &text, score, performance)
        } else {
            from_tsv(path, &text)
        }
    }
}

fn from_tsv(path: &Path, text: &str) -> Result<GroundTruth> {
    let err = |reason: String| Error::GroundTruth {
        path: path.into(),
        reason,
    };
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or_default().split('\t').collect();
    let column = |name: &str| {
        header
            .iter()
            .position(|h| h.trim() == name)
            .ok_or_else(|| err(format!("missing column {name:?}")))
    };
    let (perf_col, beat_col) = (column("perf_sec")?, column("score_beat")?);
    let mut gt = GroundTruth::default();
    for (i, line) in lines.enumerate() {
        // np.loadtxt: '#' starts a comment, blank lines are skipped
        let line = line.split('#').next().unwrap_or_default();
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let value = |col: usize| {
            fields
                .get(col)
                .and_then(|f| f.trim().parse::<f64>().ok())
                .ok_or_else(|| err(format!("line {}: no number in column {}", i + 2, col + 1)))
        };
        gt.perf_sec.push(value(perf_col)?);
        gt.score_beat.push(value(beat_col)?);
    }
    Ok(gt)
}

/// `format_pnote_id`.
fn pnote_id(id: &str) -> String {
    if id.starts_with('n') {
        id.to_string()
    } else {
        format!("n{id}")
    }
}

/// Port of `gt_from_match` for match files of version 1.x and the legacy versions 3.0-5.0
/// ((n)ASAP). Score notes are found by shared note
/// id, else (e.g. Vienna 4x22) by MIDI pitch and onset beat rounded to 4 decimals; the match
/// score's onset beat is the snote's `OnsetInBeats`. Other files are converted with the
/// reference by `tools/golden/match_to_tsv.py`.
fn from_match(
    path: &Path,
    text: &str,
    score: &Score,
    performance: Option<&Path>,
) -> Result<GroundTruth> {
    let err = |reason: &str| Error::GroundTruth {
        path: path.into(),
        reason: format!("{reason}; convert it to TSV with tools/golden/match_to_tsv.py"),
    };
    let info = |key: &str| {
        text.lines().find_map(|l| {
            l.trim()
                .strip_prefix("info(")?
                .strip_suffix(").")?
                .strip_prefix(key)?
                .strip_prefix(',')
        })
    };
    let version = info("matchFileVersion").unwrap_or_default();
    // partitura reads "5.0" as legacy 0.5.0 and "1.0.0" as 1.0.0
    let legacy = matches!(version, "3.0" | "4.0" | "5.0");
    if !legacy && !(version.starts_with("1.") && version.matches('.').count() == 2) {
        return Err(err(
            "only match file versions 1.x.y and 3.0-5.0 are supported",
        ));
    }

    // snote(Anchor,[NoteName,Modifier],Octave,Measure:Beat,Offset,Duration,OnsetInBeats,...)
    let mut snotes = Vec::new();
    // snote-note lines: (score id, performance id, MIDI pitch, onset in ticks)
    let mut matches = Vec::new();
    for line in text.lines().map(str::trim) {
        let Some(rest) = line.strip_prefix("snote(") else {
            continue;
        };
        let snote = rest.split(')').next().unwrap_or_default();
        let score_id = snote.split(',').next().unwrap_or_default();
        snotes.push((score_id, snote));
        if let Some((_, note)) = rest.split_once(")-note(") {
            // 1.x: note(Id,MidiPitch,Onset,Offset,Velocity,Channel,Track)
            // legacy: note(Id,[NoteName,Modifier],Octave,Onset,Offset,AdjOffset,Velocity)
            let fields: Vec<&str> = note.trim_end_matches(").").split(',').collect();
            let (pitch, onset) = if legacy {
                let pitch = fields
                    .get(1..4)
                    .and_then(|f| spelled_pitch(f[0], f[1], f[2]));
                (pitch, fields.get(4))
            } else {
                (fields.get(1).and_then(|p| p.parse().ok()), fields.get(2))
            };
            let onset = onset.and_then(|o| o.parse::<f64>().ok());
            let (Some(pitch), Some(onset)) = (pitch, onset) else {
                return Err(err(&format!("cannot read the note in {line:?}")));
            };
            matches.push((score_id, pnote_id(fields[0]), pitch, onset));
        }
    }

    let mut beat_of = HashMap::new();
    let by_id: HashMap<&str, f64> = score
        .notes
        .iter()
        .map(|n| (n.id.as_str(), f64::from(n.onset_beat)))
        .collect();
    if snotes.iter().any(|(id, _)| by_id.contains_key(id)) {
        for (id, _) in &snotes {
            if let Some(&beat) = by_id.get(id) {
                beat_of.insert(*id, beat);
            }
        }
    } else {
        // `+ 0.0` turns -0.0 into 0.0, which Python's dict treats as equal
        let key = |pitch: i64, beat: f64| (pitch, (round4(beat) + 0.0).to_bits());
        let by_pitch_beat: HashMap<_, f64> = score
            .notes
            .iter()
            .map(|n| {
                let beat = f64::from(n.onset_beat);
                (key(i64::from(n.pitch), beat), beat)
            })
            .collect();
        for (id, snote) in &snotes {
            let (pitch, beat) = snote_pitch_beat(snote)
                .ok_or_else(|| err(&format!("cannot read snote({snote})")))?;
            if let Some(&b) = by_pitch_beat.get(&key(pitch, beat)) {
                beat_of.insert(*id, b);
            }
        }
    }
    if beat_of.is_empty() {
        return Err(err("no score note of the match file is found in the score"));
    }

    // _streamed_onsets: partitura numbers the notes of each MIDI track n0, n1, ...
    let streamed = match performance {
        Some(p) => {
            let mut notes = HashMap::new();
            for part in Performance::load(p)?.parts {
                for (k, n) in part.notes.iter().enumerate() {
                    notes.insert(format!("n{k}"), (i64::from(n.pitch), n.note_on));
                }
            }
            matches
                .iter()
                .all(|(_, pid, pitch, _)| notes.get(pid).is_some_and(|(p, _)| p == pitch))
                .then_some(notes)
        }
        None => None,
    };
    let num = |key| info(key).and_then(|v| v.parse::<f64>().ok());
    let clock = num("midiClockRate").zip(num("midiClockUnits"));

    let mut pairs = Vec::new();
    for (score_id, pid, _, onset) in &matches {
        let Some(&beat) = beat_of.get(score_id) else {
            continue;
        };
        let sec = match &streamed {
            Some(notes) => notes[pid].1,
            None => {
                let (mpq, ppq) =
                    clock.ok_or_else(|| err("missing midiClockRate or midiClockUnits"))?;
                // midi_ticks_to_seconds
                mpq * onset / (1e6 * ppq)
            }
        };
        pairs.push((sec, beat));
    }
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let mut gt = GroundTruth::default();
    for (i, &(sec, beat)) in pairs.iter().enumerate() {
        if i == 0 || beat != pairs[i - 1].1 {
            gt.perf_sec.push(sec);
            gt.score_beat.push(beat);
        }
    }
    Ok(gt)
}

/// MIDI pitch and `OnsetInBeats` (through f32, like partitura's note array) of an snote.
fn snote_pitch_beat(snote: &str) -> Option<(i64, f64)> {
    let f: Vec<&str> = snote.split(',').collect();
    let pitch = spelled_pitch(f.get(1)?, f.get(2)?, f.get(3)?)?;
    let beat: f32 = f.get(7)?.parse().ok()?;
    Some((pitch, f64::from(beat)))
}

/// MIDI pitch of the match fields `[NoteName`, `Modifier]`, `Octave`.
fn spelled_pitch(name: &str, modifier: &str, octave: &str) -> Option<i64> {
    let step = match name.trim_start_matches('[').to_ascii_uppercase().as_str() {
        "C" => 0,
        "D" => 2,
        "E" => 4,
        "F" => 5,
        "G" => 7,
        "A" => 9,
        "B" => 11,
        _ => return None,
    };
    let alter = match modifier.trim_end_matches(']') {
        "n" | "" => 0,
        "#" => 1,
        "##" | "x" => 2,
        "b" => -1,
        "bb" => -2,
        _ => return None,
    };
    let octave: i64 = octave.parse().ok()?;
    Some(12 * (octave + 1) + step + alter)
}

/// `transfer_positions(wp, queries, frame_rate=1, domain="score")`: the last path entry per
/// performance time, held until the next one; NaN before the first entry.
pub fn transfer_positions(path: &AlignmentPath, queries: &[f64]) -> Vec<f64> {
    let mut points: Vec<(f64, f64)> = path
        .perf_time
        .iter()
        .copied()
        .zip(path.score_beat.iter().copied())
        .collect();
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let (mut frames, mut beats): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for (t, b) in points {
        if frames.last() == Some(&t) {
            *beats.last_mut().expect("same length") = b;
        } else {
            frames.push(t);
            beats.push(b);
        }
    }
    queries
        .iter()
        .map(|&q| {
            // np.searchsorted sorts NaN after every number
            let i = if q.is_nan() {
                frames.len()
            } else {
                frames.partition_point(|&f| f <= q)
            };
            if i == 0 { f64::NAN } else { beats[i - 1] }
        })
        .collect()
}

/// Python's `float(f"{x:.4f}")`: both round the exact binary value half to even.
fn round4(x: f64) -> f64 {
    format!("{x:.4}").parse().unwrap_or(x)
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    match s.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => s[n / 2],
        n => (s[n / 2 - 1] + s[n / 2]) / 2.0,
    }
}

/// `scipy.stats.skew` and `kurtosis` with `bias=True`, Fisher; NaN when all values are equal.
fn skew_kurtosis(v: &[f64]) -> (f64, f64) {
    let m = mean(v);
    let moment = |k: i32| v.iter().map(|x| (x - m).powi(k)).sum::<f64>() / v.len() as f64;
    let m2 = moment(2);
    if m2 <= (f64::EPSILON * m).powi(2) {
        return (f64::NAN, f64::NAN);
    }
    (moment(3) / m2.powf(1.5), moment(4) / m2.powi(2) - 3.0)
}

/// Port of `evaluate_alignment` with the default tolerances and SPARC windows.
pub fn evaluate_alignment(path: &AlignmentPath, gt: &GroundTruth) -> EvaluationResult {
    let predicted = transfer_positions(path, &gt.perf_sec);
    let errors: Vec<f64> = gt
        .score_beat
        .iter()
        .zip(&predicted)
        .map(|(g, p)| g - p)
        .collect();
    let known: Vec<f64> = errors.iter().copied().filter(|e| !e.is_nan()).collect();
    let abs: Vec<f64> = known.iter().map(|e| e.abs()).collect();
    let abs_mean = mean(&abs);
    let std = mean(
        &abs.iter()
            .map(|a| (a - abs_mean).powi(2))
            .collect::<Vec<_>>(),
    )
    .sqrt();
    let accuracy = |tau: f64| {
        round4(errors.iter().filter(|e| e.abs() <= tau).count() as f64 / gt.score_beat.len() as f64)
    };
    let (skewness, kurtosis) = skew_kurtosis(&known);
    let onsets: Vec<f64> = gt
        .perf_sec
        .iter()
        .zip(&gt.score_beat)
        .filter(|(t, b)| t.is_finite() && b.is_finite())
        .map(|(t, _)| *t)
        .collect();
    EvaluationResult {
        mean_b: round4(abs_mean),
        median_b: round4(median(&abs)),
        std_b: round4(std),
        accuracy_0_05b: accuracy(0.05),
        accuracy_0_1b: accuracy(0.1),
        accuracy_0_3b: accuracy(0.3),
        accuracy_0_5b: accuracy(0.5),
        accuracy_1b: accuracy(1.0),
        accuracy_2b: accuracy(2.0),
        skewness: round4(skewness),
        kurtosis: round4(kurtosis),
        smoothness: compute_sparc(
            &path.score_beat,
            &path.perf_time,
            &onsets,
            ONSET_WINDOW_DURATION,
        )
        .map(round4),
    }
}

/// `onset_wise_windows`: `(t0, t1)` of forward windows anchored at the unique finite onsets,
/// up to the first window that reaches the final onset.
fn onset_wise_windows(onsets: &[f64], duration: f64) -> Vec<(f64, f64)> {
    let mut times: Vec<f64> = onsets.iter().copied().filter(|t| t.is_finite()).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let Some(&last) = times.last() else {
        return Vec::new();
    };
    let mut windows = Vec::new();
    for left in 0..times.len() {
        if left > 0 && times[left] == times[left - 1] {
            continue;
        }
        let stop = times[left] + duration;
        windows.push((times[left], stop.min(last)));
        if times.partition_point(|&t| t <= stop) == times.len() {
            break;
        }
    }
    windows
}

/// `spectral_arc_length` (adaptive SPARC, Balasubramanian 2015) with the reference defaults.
fn spectral_arc_length(speed: &[f64], planner: &mut RealFftPlanner<f64>) -> Option<f64> {
    if speed.len() < 2 || !speed.iter().all(|s| s.is_finite()) || speed.iter().all(|&s| s == 0.0) {
        return None;
    }
    let n_fft = speed.len().next_power_of_two() << SPARC_PADLEVEL;
    let fft = planner.plan_fft_forward(n_fft);
    let mut input = fft.make_input_vec();
    input[..speed.len()].copy_from_slice(speed);
    let mut spectrum = fft.make_output_vec();
    fft.process(&mut input, &mut spectrum).ok()?;
    let mut magnitude: Vec<f64> = spectrum.iter().map(|c| c.norm()).collect();
    let max = magnitude.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    magnitude.iter_mut().for_each(|m| *m /= max);
    // np.fft.rfftfreq(n_fft, d=1/fs)
    let step = 1.0 / (n_fft as f64 * (1.0 / SPARC_FS));
    let freq = |k: usize| k as f64 * step;
    let support: Vec<usize> = (0..magnitude.len())
        .filter(|&k| freq(k) <= SPARC_FC && magnitude[k] >= SPARC_AMP_TH)
        .collect();
    if support.len() < 2 {
        return None;
    }
    let (first, last) = (support[0], support[support.len() - 1]);
    let span = freq(last) - freq(first);
    let length: f64 = (first..last)
        .map(|k| ((freq(k + 1) - freq(k)) / span).hypot(magnitude[k + 1] - magnitude[k]))
        .sum();
    Some(-length)
}

/// `compute_sparc`: mean adaptive SPARC over onset-wise windows of the position trajectory
/// resampled at 50 Hz (holding the last position); `None` if no window is defined.
pub fn compute_sparc(
    score_beat: &[f64],
    perf_sec: &[f64],
    onset_times: &[f64],
    window_sec: f64,
) -> Option<f64> {
    let mut points: Vec<(f64, f64)> = perf_sec
        .iter()
        .copied()
        .zip(score_beat.iter().copied())
        .filter(|(t, b)| t.is_finite() && b.is_finite())
        .collect();
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    // keep the final decision at duplicate timestamps, as in transfer_positions
    let points: Vec<(f64, f64)> = (0..points.len())
        .filter(|&i| points.get(i + 1).is_none_or(|n| n.0 != points[i].0))
        .map(|i| points[i])
        .collect();
    let windows = onset_wise_windows(onset_times, window_sec);
    let (Some(&(start, _)), Some(&(_, stop))) = (windows.first(), windows.last()) else {
        return None;
    };
    if points.len() < 2 {
        return None;
    }
    let dt = 1.0 / SPARC_FS;
    // np.arange(start, stop, dt) fills start + i * ((start + dt) - start)
    let delta = (start + dt) - start;
    let n = ((stop - start) / dt).ceil().max(0.0) as usize;
    let times: Vec<f64> = (0..n)
        .map(|i| {
            if i == 1 {
                start + dt
            } else {
                start + i as f64 * delta
            }
        })
        .collect();
    let positions: Vec<f64> = times
        .iter()
        .map(|&t| match points.partition_point(|p| p.0 <= t) {
            0 => f64::NAN,
            i => points[i - 1].1,
        })
        .collect();
    let (covered_from, covered_to) = (points[0].0, points[points.len() - 1].0);
    let mut planner = RealFftPlanner::new();
    let values: Vec<f64> = windows
        .iter()
        .filter(|&&(t0, t1)| t0 >= covered_from && t1 <= covered_to)
        .filter_map(|&(t0, t1)| {
            let left = times.partition_point(|&t| t < t0);
            let right = times.partition_point(|&t| t < t1);
            let p = &positions[left..right];
            if p.len() < 2 {
                return None;
            }
            // np.abs(np.gradient(p, dt)), edge_order 1
            let n = p.len();
            let speed: Vec<f64> = (0..n)
                .map(|i| match i {
                    0 => (p[1] - p[0]) / dt,
                    i if i == n - 1 => (p[n - 1] - p[n - 2]) / dt,
                    i => (p[i + 1] - p[i - 1]) / (2.0 * dt),
                })
                .map(f64::abs)
                .collect();
            spectral_arc_length(&speed, &mut planner)
        })
        .collect();
    (!values.is_empty()).then(|| mean(&values))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Expected values from pymatchmaker v0.4.0 / SciPy 1.18.1 (tools/golden environment).

    #[test]
    fn rounding_matches_python_format() {
        let rounded: Vec<f64> = [0.03125, 0.09375, -0.03125, 1.00005, 2.5e-5]
            .into_iter()
            .map(round4)
            .collect();
        assert_eq!(rounded, [0.0312, 0.0938, -0.0312, 1.0001, 0.0]);
        assert!(round4(f64::NAN).is_nan());
    }

    #[test]
    fn snote_pitch_and_onset_beat() {
        // Vienna 4x22 Chopin_op10_no3_p01.match
        let snote = "n1,[B,n],3,0:1,0,1/8,-0.5000,0.0000,[v1,staff1]";
        assert_eq!(snote_pitch_beat(snote), Some((59, -0.5)));
        let snote = "n3,[G,#],3,1:1,0,1/16,0.0000,0.2500,[v3,staff1]";
        assert_eq!(snote_pitch_beat(snote), Some((56, 0.0)));
        assert_eq!(snote_pitch_beat("n3,[H,#],3"), None);
        // legacy 5.0 note of (n)ASAP Chopin/Barcarolle/Verbaite03.match
        assert_eq!(spelled_pitch("[C", "#]", "1"), Some(25));
    }

    #[test]
    fn skew_and_kurtosis_match_scipy() {
        let (s, k) = skew_kurtosis(&[2.0, 8.0, 0.0, 4.0, 1.0, 9.0, 9.0, 0.0]);
        assert!((s - 0.2650554122698573).abs() < 1e-12);
        assert!((k - -1.6660010752838508).abs() < 1e-12);
        let (s, k) = skew_kurtosis(&[0.5; 4]);
        assert!(s.is_nan() && k.is_nan());
    }

    #[test]
    fn transfer_holds_last_entry_per_time() {
        let path = AlignmentPath {
            perf_time: vec![0.0, 1.0, 1.0, 2.0],
            score_beat: vec![0.0, 1.0, 1.5, 3.0],
        };
        let got = transfer_positions(&path, &[-0.5, 0.0, 0.5, 1.0, 1.7, 2.0, 9.0, f64::NAN]);
        assert!(got[0].is_nan());
        assert_eq!(got[1..], [0.0, 0.0, 1.5, 1.5, 3.0, 3.0, 3.0]);
    }

    fn trajectory() -> (Vec<f64>, Vec<f64>) {
        let perf = (0..40).map(|i| f64::from(i) * 0.37).collect();
        let beats = (0..40)
            .map(|i| f64::from(i) * 0.5 + 0.2 * f64::from(i).sin())
            .collect();
        (perf, beats)
    }

    #[test]
    fn sparc_matches_reference() {
        let (perf, beats) = trajectory();
        let sparc = compute_sparc(&beats, &perf, &perf, 5.0).unwrap();
        assert!((sparc - -14.124139255637598).abs() < 1e-9, "{sparc}");
    }

    #[test]
    fn evaluation_matches_reference() {
        let (perf, beats) = trajectory();
        let path = AlignmentPath {
            perf_time: perf.clone(),
            score_beat: beats.clone(),
        };
        let gt = GroundTruth {
            perf_sec: perf.iter().step_by(3).map(|t| t + 0.01).collect(),
            score_beat: (0..40)
                .step_by(3)
                .map(|i| beats[i] + 0.07 * f64::from(i as u32).cos())
                .collect(),
        };
        let r = evaluate_alignment(&path, &gt);
        let expected = [
            0.0398, 0.0423, 0.0238, 0.5714, 1.0, 1.0, 1.0, 1.0, 1.0, -0.0888, -1.2516,
        ];
        let got = [
            r.mean_b,
            r.median_b,
            r.std_b,
            r.accuracy_0_05b,
            r.accuracy_0_1b,
            r.accuracy_0_3b,
            r.accuracy_0_5b,
            r.accuracy_1b,
            r.accuracy_2b,
            r.skewness,
            r.kurtosis,
        ];
        assert_eq!(got, expected);
        assert_eq!(r.smoothness, None);
    }
}
