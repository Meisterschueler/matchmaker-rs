//! Port of `partitura.score.merge_parts(parts, reassign="voice")`.
//!
//! All parts are rescaled to the least common multiple of their divisions. Measures, time
//! signatures, tempos, endings and `DaCapo`/`Fine` markers are taken from the first part only;
//! voices of later parts are shifted by the maximum voice numbers of the preceding parts.

use super::part::{Marker, Part, Span, Time};

fn gcd(a: Time, b: Time) -> Time {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

/// Merge `parts` into one part (returns the single part unchanged if there is only one).
pub(crate) fn merge_parts(mut parts: Vec<Part>) -> Result<Part, String> {
    if parts.is_empty() {
        return Err("IndexError: score has no parts".into());
    }
    if parts.len() == 1 {
        return Ok(parts.remove(0));
    }
    if parts.iter().any(|p| p.quarter_durations.len() != 1) {
        return Err("Merging parts with multiple divisions is not supported".into());
    }
    let divs: Vec<Time> = parts.iter().map(|p| p.quarter_durations[0]).collect();
    let lcm = divs.iter().fold(1, |acc, &d| acc / gcd(acc, d) * d);
    let multipliers: Vec<Time> = divs.iter().map(|&d| lcm / d).collect();
    // maximum voice per part (from its note array, default 1)
    let maximum_voices: Vec<i32> = parts
        .iter()
        .map(|p| p.notes_tied().map(|(_, n)| n.voice).max().unwrap_or(1))
        .collect();

    let mut new = Part::new(parts[0].id.clone());
    new.quarter_times = vec![0];
    new.quarter_durations = vec![lcm];
    for (p_ind, p) in parts.iter().enumerate() {
        let m = multipliers[p_ind];
        let voice_offset: i32 = maximum_voices[..p_ind].iter().sum();
        let base = new.notes.len();
        for n in &p.notes {
            let mut copy = n.clone();
            copy.start *= m;
            copy.end *= m;
            copy.voice += voice_offset;
            copy.tie_next = n.tie_next.map(|j| j + base);
            copy.tie_prev = n.tie_prev.map(|j| j + base);
            new.add_point(copy.start);
            new.add_point(copy.end);
            new.notes.push(copy);
        }
        let scale = |s: &Span| Span {
            start: s.start.map(|t| t * m),
            end: s.end.map(|t| t * m),
            number: s.number.clone(),
        };
        for r in &p.repeats {
            new.repeats.push(scale(r));
        }
        for &(t, marker) in &p.markers {
            if p_ind == 0 || !matches!(marker, Marker::DaCapo | Marker::Fine) {
                new.markers.push((t * m, marker));
            }
        }
        if p_ind == 0 {
            for &t in &p.points {
                new.add_point(t * m);
            }
            new.measures.extend(p.measures.iter().map(scale));
            new.endings.extend(p.endings.iter().map(scale));
            for ts in &p.time_signatures {
                let mut c = *ts;
                c.start *= m;
                new.time_signatures.push(c);
            }
            for t in &p.tempos {
                let mut c = t.clone();
                c.start *= m;
                new.tempos.push(c);
            }
        }
    }
    new.sort_notes_by_start();
    Ok(new)
}
