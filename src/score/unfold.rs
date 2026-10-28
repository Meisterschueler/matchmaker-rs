//! Port of `partitura.score.unfold_part_maximal(part, update_ids=True, ignore_leaps=False)`
//! and the functions it uses: `add_segments`, `Path`, `unfold_paths`, `get_paths`,
//! `new_part_from_path` (`ScoreVariant.create_variant_part`) and
//! `update_note_ids_after_unfolding`.
//!
//! Segment ids, destination lists and their string sorting follow the Python code exactly,
//! because they decide the order in which repeats, voltas and jumps are taken.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::part::{Marker, Note, Part, Span, Time};

/// Why unfolding failed; pymatchmaker then falls back to the folded part.
#[derive(Debug)]
pub(crate) struct UnfoldError(pub String);

/// Insertion-ordered boundary types at one time (a Python `dict` keyed by type).
#[derive(Debug, Default, Clone)]
struct Boundaries(Vec<(&'static str, Option<usize>)>);

impl Boundaries {
    fn set(&mut self, key: &'static str, value: Option<usize>) {
        match self.0.iter_mut().find(|(k, _)| *k == key) {
            Some(entry) => entry.1 = value,
            None => self.0.push((key, value)),
        }
    }
    fn get(&self, key: &str) -> Option<Option<usize>> {
        self.0.iter().find(|(k, _)| *k == key).map(|(_, v)| *v)
    }
    fn contains(&self, key: &str) -> bool {
        self.get(key).is_some()
    }
}

/// `segment_info[t]` while building segments.
#[derive(Debug, Clone)]
struct SegmentInfo {
    id: String,
    end: Time,
    to: Vec<String>,
    kind: SegmentType,
    volta_numbers: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SegmentType {
    Default,
    LeapStart,
    LeapEnd,
}

/// `partitura.score.Segment`.
#[derive(Debug, Clone)]
struct Segment {
    id: String,
    start: Time,
    end: Time,
    to: Vec<String>,
    await_to: Vec<String>,
    kind: SegmentType,
}

fn segment_char(i: usize) -> Result<String, UnfoldError> {
    match i {
        0..=25 => Ok(char::from(b'A' + i as u8).to_string()),
        26..=51 => Ok(char::from(b'a' + (i - 26) as u8).to_string()),
        _ => Err(UnfoldError(format!(
            "KeyError: {i} (more than 52 segments)"
        ))),
    }
}

/// Port of `add_segments(part)`.
fn add_segments(part: &Part) -> Result<BTreeMap<String, Segment>, UnfoldError> {
    let mut boundaries: BTreeMap<Time, Boundaries> = BTreeMap::new();
    let mut coda_destinations: Vec<Time> = Vec::new();
    let mut segno_destinations: Vec<Time> = Vec::new();

    // valid repeats / endings in iteration (start) order
    let mut repeats: Vec<usize> = (0..part.repeats.len())
        .filter(|&i| part.repeats[i].start.is_some() && part.repeats[i].end.is_some())
        .collect();
    repeats.sort_by_key(|&i| part.repeats[i].start);
    let mut endings: Vec<usize> = (0..part.endings.len())
        .filter(|&i| part.endings[i].start.is_some() && part.endings[i].end.is_some())
        .collect();
    endings.sort_by_key(|&i| part.endings[i].start);

    for &i in &repeats {
        let (s, e) = (
            part.repeats[i].start.unwrap_or(0),
            part.repeats[i].end.unwrap_or(0),
        );
        boundaries
            .entry(s)
            .or_default()
            .set("repeat_start", Some(i));
        boundaries.entry(e).or_default().set("repeat_end", Some(i));
    }
    for &i in &endings {
        let (s, e) = (
            part.endings[i].start.unwrap_or(0),
            part.endings[i].end.unwrap_or(0),
        );
        boundaries.entry(s).or_default().set("volta_start", Some(i));
        boundaries.entry(e).or_default().set("volta_end", Some(i));
    }
    let markers_of = |m: Marker| {
        let mut v: Vec<Time> = part
            .markers
            .iter()
            .filter(|(_, k)| *k == m)
            .map(|(t, _)| *t)
            .collect();
        v.sort();
        v
    };
    for t in markers_of(Marker::Coda) {
        boundaries.entry(t).or_default().set("coda", None);
        coda_destinations.push(t);
    }
    for t in markers_of(Marker::ToCoda) {
        boundaries.entry(t).or_default().set("tocoda", None);
    }
    for t in markers_of(Marker::DaCapo) {
        boundaries.entry(t).or_default().set("dacapo", None);
    }
    for t in markers_of(Marker::Fine) {
        boundaries.entry(t).or_default().set("fine", None);
    }
    for t in markers_of(Marker::Segno) {
        boundaries.entry(t).or_default().set("segno", None);
        segno_destinations.push(t);
    }
    for t in markers_of(Marker::DalSegno) {
        boundaries.entry(t).or_default().set("dalsegno", None);
    }
    let first = part
        .first_point()
        .ok_or_else(|| UnfoldError("empty part".into()))?;
    let last = part
        .last_point()
        .ok_or_else(|| UnfoldError("empty part".into()))?;
    boundaries.entry(last).or_default().set("end", None);
    boundaries.entry(first).or_default().set("start", None);

    let boundary_times: Vec<Time> = boundaries.keys().copied().collect();
    let mut info: BTreeMap<Time, SegmentInfo> = BTreeMap::new();
    for (i, w) in boundary_times.windows(2).enumerate() {
        info.insert(
            w[0],
            SegmentInfo {
                id: segment_char(i)?,
                end: w[1],
                to: Vec::new(),
                kind: SegmentType::Default,
                volta_numbers: Vec::new(),
            },
        );
    }
    let last_time = *boundary_times.last().unwrap_or(&last);
    info.insert(
        last_time,
        SegmentInfo {
            id: "END".into(),
            end: last_time,
            to: Vec::new(),
            kind: SegmentType::Default,
            volta_numbers: Vec::new(),
        },
    );
    let id_at = |info: &BTreeMap<Time, SegmentInfo>, t: Time| -> Result<String, UnfoldError> {
        info.get(&t)
            .map(|s| s.id.clone())
            .ok_or_else(|| UnfoldError(format!("KeyError: {t}")))
    };

    let mut current_volta_repeat_start: Time = 0;
    let mut current_volta_end: Time = 0;
    let mut current_volta_total_number: usize = 0;

    for &ss in &boundary_times[..boundary_times.len().saturating_sub(1)] {
        let se = info[&ss].end;
        let b_se = boundaries.get(&se).cloned().unwrap_or_default();
        for (boundary_type, value) in b_se.0.iter() {
            match *boundary_type {
                "repeat_start" => {
                    let id = id_at(&info, se)?;
                    if let Some(s) = info.get_mut(&ss) {
                        s.to.push(id);
                    }
                }
                "repeat_end" => {
                    if !b_se.contains("volta_end") {
                        let id = id_at(&info, se)?;
                        let repeat_start = part.repeats[value.unwrap_or(0)].start.unwrap_or(0);
                        let start_id = id_at(&info, repeat_start)?;
                        let s = info
                            .get_mut(&ss)
                            .ok_or_else(|| UnfoldError("segment".into()))?;
                        s.to.push(id);
                        s.to.push(start_id);
                    }
                }
                "volta_start" => {
                    if !b_se.contains("volta_end") {
                        current_volta_total_number = 0;
                        current_volta_end = se;
                        for _ in 0..10 {
                            let Some(Some(vi)) = boundaries
                                .get(&current_volta_end)
                                .and_then(|b| b.get("volta_start"))
                            else {
                                continue;
                            };
                            let number = part.endings[vi]
                                .number
                                .clone()
                                .ok_or_else(|| UnfoldError("ending without number".into()))?;
                            let numbers: Vec<String> = number
                                .split(',')
                                .map(|n| {
                                    n.trim().parse::<i64>().map(|v| v.to_string()).map_err(|_| {
                                        UnfoldError(format!("invalid volta number {n:?}"))
                                    })
                                })
                                .collect::<Result<_, _>>()?;
                            current_volta_total_number += numbers.len();
                            let target_id = id_at(&info, current_volta_end)?;
                            for no in &numbers {
                                info.get_mut(&ss)
                                    .ok_or_else(|| UnfoldError("segment".into()))?
                                    .to
                                    .push(format!("{no}_Volta_{target_id}"));
                            }
                            info.get_mut(&current_volta_end)
                                .ok_or_else(|| UnfoldError("segment".into()))?
                                .volta_numbers
                                .extend(numbers);
                            current_volta_end = part.endings[vi].end.unwrap_or(0);
                        }
                    }
                }
                "volta_end" => {
                    let numbers = info[&ss].volta_numbers.clone();
                    let total = current_volta_total_number.to_string();
                    for vn in &numbers {
                        if *vn != total {
                            if let Some(Some(ri)) = b_se.get("repeat_end") {
                                current_volta_repeat_start = current_volta_repeat_start
                                    .max(part.repeats[ri].start.unwrap_or(0));
                            }
                            let target =
                                format!("Z_Volta_{}", id_at(&info, current_volta_repeat_start)?);
                            info.get_mut(&ss)
                                .ok_or_else(|| UnfoldError("segment".into()))?
                                .to
                                .push(target);
                        }
                    }
                    if numbers.contains(&total) {
                        let id = id_at(&info, current_volta_end)?;
                        info.get_mut(&ss)
                            .ok_or_else(|| UnfoldError("segment".into()))?
                            .to
                            .push(id);
                    }
                }
                "coda" => {
                    let id = id_at(&info, se)?;
                    info.get_mut(&ss)
                        .ok_or_else(|| UnfoldError("segment".into()))?
                        .to
                        .push(id);
                    info.get_mut(&se)
                        .ok_or_else(|| UnfoldError("segment".into()))?
                        .kind = SegmentType::LeapEnd;
                }
                "tocoda" => {
                    let id = id_at(&info, se)?;
                    let coda = *coda_destinations
                        .first()
                        .ok_or_else(|| UnfoldError("IndexError: no coda".into()))?;
                    let coda_id = id_at(&info, coda)?;
                    let s = info
                        .get_mut(&ss)
                        .ok_or_else(|| UnfoldError("segment".into()))?;
                    s.to.push(id);
                    s.to.push(format!("Navigation2_{coda_id}"));
                    s.kind = SegmentType::LeapStart;
                }
                "segno" => {
                    let id = id_at(&info, se)?;
                    info.get_mut(&ss)
                        .ok_or_else(|| UnfoldError("segment".into()))?
                        .to
                        .push(id);
                    info.get_mut(&se)
                        .ok_or_else(|| UnfoldError("segment".into()))?
                        .kind = SegmentType::LeapEnd;
                }
                "dalsegno" => {
                    let id = id_at(&info, se)?;
                    let segno = *segno_destinations
                        .first()
                        .ok_or_else(|| UnfoldError("IndexError: no segno".into()))?;
                    let segno_id = id_at(&info, segno)?;
                    let s = info
                        .get_mut(&ss)
                        .ok_or_else(|| UnfoldError("segment".into()))?;
                    s.to.push(id.clone());
                    s.to.push(format!("Navigation1_{segno_id}"));
                    s.to.push(format!("Navigation2_{id}"));
                    s.kind = SegmentType::LeapStart;
                }
                "dacapo" => {
                    let id = id_at(&info, se)?;
                    let first_id = id_at(&info, first)?;
                    let s = info
                        .get_mut(&ss)
                        .ok_or_else(|| UnfoldError("segment".into()))?;
                    s.to.push(id.clone());
                    s.to.push(format!("Navigation1_{first_id}"));
                    s.to.push(format!("Navigation2_{id}"));
                    s.kind = SegmentType::LeapStart;
                }
                "fine" => {
                    let id = id_at(&info, se)?;
                    let last_id = id_at(&info, last)?;
                    let s = info
                        .get_mut(&ss)
                        .ok_or_else(|| UnfoldError("segment".into()))?;
                    s.to.push(id);
                    s.to.push(format!("Navigation2_{last_id}"));
                }
                "end" => {
                    let id = id_at(&info, se)?;
                    info.get_mut(&ss)
                        .ok_or_else(|| UnfoldError("segment".into()))?
                        .to
                        .push(id);
                }
                _ => {}
            }
            // first segment is always a leap destination (da capo)
            if ss == 0 {
                info.get_mut(&ss)
                    .ok_or_else(|| UnfoldError("segment".into()))?
                    .kind = SegmentType::LeapEnd;
            }
        }
    }

    // clean up and order all jump destinations
    let mut segments = BTreeMap::new();
    for &start in &boundary_times[..boundary_times.len().saturating_sub(1)] {
        let s = &info[&start];
        let destinations = &s.to;
        let mut no_volta: Vec<String> = destinations
            .iter()
            .filter(|d| !d.contains("Volta_") && !d.contains("Navigation"))
            .cloned()
            .collect();
        let mut volta: Vec<String> = destinations
            .iter()
            .filter(|d| d.contains("Volta_"))
            .cloned()
            .collect();
        let mut navigation1: Vec<String> = destinations
            .iter()
            .filter(|d| d.contains("Navigation1_"))
            .map(|d| d.chars().skip(12).collect())
            .collect();
        let navigation2: Vec<String> = destinations
            .iter()
            .filter(|d| d.contains("Navigation2_"))
            .map(|d| d.chars().skip(12).collect())
            .collect();
        // list(set(...))
        no_volta = no_volta
            .into_iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let except_await_has_end = volta
            .iter()
            .chain(&no_volta)
            .chain(&navigation1)
            .any(|d| d == "END");
        if except_await_has_end {
            no_volta.retain(|d| d != "END");
            navigation1.push("END".into());
        }
        no_volta.sort();
        volta.sort();
        let volta: Vec<String> = volta.iter().map(|d| d.chars().skip(8).collect()).collect();
        no_volta.retain(|d| !volta.contains(d));
        let mut to = volta;
        to.extend(no_volta);
        to.extend(navigation1);
        segments.insert(
            s.id.clone(),
            Segment {
                id: s.id.clone(),
                start,
                end: s.end,
                to,
                await_to: navigation2,
                kind: s.kind,
            },
        );
    }
    Ok(segments)
}

/// The single path followed with `all_repeats=True` (port of `Path` + `unfold_paths`).
struct PathState {
    path: Vec<String>,
    used: HashMap<String, Vec<String>>,
    no_repeats: bool,
    jumped: bool,
}

impl PathState {
    /// Port of `Path.list_of_destinations_from_last_segment` for `all_repeats=True`.
    fn next_destination(
        &self,
        segments: &BTreeMap<String, Segment>,
    ) -> Result<String, UnfoldError> {
        let last = self
            .path
            .last()
            .ok_or_else(|| UnfoldError("empty path".into()))?;
        let seg = segments
            .get(last)
            .ok_or_else(|| UnfoldError(format!("KeyError: {last}")))?;
        let destinations = &seg.to;
        if destinations.is_empty() {
            return Err(UnfoldError("IndexError: no destination".into()));
        }
        if self.no_repeats {
            return Ok(destinations[destinations.len() - 1].clone());
        }
        let used = self.used.get(last).map(Vec::as_slice).unwrap_or(&[]);
        let Some(last_destination) = used.last() else {
            return Ok(destinations[0].clone());
        };
        let count = used.iter().filter(|d| *d == last_destination).count();
        let repeated = destinations.len() * 100;
        let positions: Vec<usize> = (0..repeated)
            .filter(|i| destinations[i % destinations.len()] == *last_destination)
            .collect();
        let index = positions
            .get(count - 1)
            .ok_or_else(|| UnfoldError("IndexError: destination history".into()))?
            % destinations.len();
        if index < destinations.len() - 1 {
            Ok(destinations[index + 1].clone())
        } else {
            Ok(destinations[0].clone())
        }
    }

    /// Port of `Path.make_copy_with_jump_to` (in place, `ignore_leap_info=False`).
    fn jump_to(
        &mut self,
        destination: String,
        segments: &mut BTreeMap<String, Segment>,
    ) -> Result<(), UnfoldError> {
        let last = self
            .path
            .last()
            .cloned()
            .ok_or_else(|| UnfoldError("empty path".into()))?;
        self.used
            .entry(last.clone())
            .or_default()
            .push(destination.clone());
        self.path.push(destination.clone());
        let dest_kind = segments
            .get(&destination)
            .map(|s| s.kind)
            .ok_or_else(|| UnfoldError(format!("KeyError: {destination}")))?;
        let prev_kind = segments
            .get(&last)
            .map(|s| s.kind)
            .ok_or_else(|| UnfoldError(format!("KeyError: {last}")))?;
        if dest_kind == SegmentType::LeapEnd && prev_kind == SegmentType::LeapStart {
            if !self.jumped {
                self.jumped = true;
                let ids: Vec<String> = segments.keys().cloned().collect();
                for id in ids {
                    if let Some(seg) = segments.get_mut(&id) {
                        if !seg.await_to.is_empty() {
                            let mut to: Vec<String> =
                                seg.to.iter().filter(|d| **d <= seg.id).cloned().collect();
                            to.extend(seg.await_to.iter().cloned());
                            seg.to = to;
                        }
                    }
                    self.used.insert(id, Vec::new());
                }
                self.used.entry(last).or_default().push(destination);
            }
            self.no_repeats = true;
        }
        Ok(())
    }
}

/// Maximum path length; Python hits its recursion limit (10 000) and pymatchmaker falls back.
const MAX_PATH_LEN: usize = 9_000;

/// Port of `get_paths(part, all_repeats=True, ignore_leap_info=False)[0]`: segment
/// `(start, end)` pairs of the maximal unfolding.
fn maximal_path(part: &Part) -> Result<Vec<(Time, Time)>, UnfoldError> {
    let mut segments = add_segments(part)?;
    let mut state = PathState {
        path: vec!["A".to_string()],
        used: HashMap::new(),
        no_repeats: false,
        jumped: false,
    };
    loop {
        let destination = state.next_destination(&segments)?;
        if destination == "END" {
            break;
        }
        state.jump_to(destination, &mut segments)?;
        if state.path.len() > MAX_PATH_LEN {
            return Err(UnfoldError("RecursionError".into()));
        }
    }
    state
        .path
        .iter()
        .map(|id| {
            segments
                .get(id)
                .map(|s| (s.start, s.end))
                .ok_or_else(|| UnfoldError(format!("KeyError: {id}")))
        })
        .collect()
}

/// Port of `unfold_part_maximal(part, update_ids=True, ignore_leaps=False)` for one part.
pub(crate) fn unfold_part_maximal(part: &Part) -> Result<Part, UnfoldError> {
    let path = maximal_path(part)?;
    let mut new = create_variant_part(part, &path)?;
    update_note_ids_after_unfolding(&mut new);
    Ok(new)
}

/// Port of `ScoreVariant.create_variant_part`.
fn create_variant_part(part: &Part, segments: &[(Time, Time)]) -> Result<Part, UnfoldError> {
    let mut new = Part::new(part.id.clone());
    let mut t_unfold: Time = 0;
    for &(start, end) in segments {
        let offset = t_unfold;
        t_unfold += end - start;
        let delta = offset - start;
        for (t, q) in part.quarter_durations_in(start, end) {
            new.set_quarter_duration(t + delta, q);
        }
        if !part.points.contains(&start) || !part.points.contains(&end) || end < start {
            return Err(UnfoldError(
                "segment end not a successor of segment start".into(),
            ));
        }
        let in_segment = |t: Time| t >= start && t < end;

        // timepoints: every point in [start, end) gets a new point (get_or_add_point)
        for &t in part.points.range(start..end) {
            new.add_point(t + delta);
        }
        // measures
        for m in &part.measures {
            if let Some(s) = m.start.filter(|&s| in_segment(s)) {
                let e = m.end.map(|e| e + delta);
                if let Some(e) = e {
                    new.add_point(e);
                }
                new.measures.push(Span {
                    start: Some(s + delta),
                    end: e,
                    number: m.number.clone(),
                });
            }
        }
        // time signatures: skip if equal to the previous one in the new part
        let mut ts_sorted: Vec<_> = part
            .time_signatures
            .iter()
            .filter(|ts| in_segment(ts.start))
            .collect();
        ts_sorted.sort_by_key(|ts| ts.start);
        for ts in ts_sorted {
            let new_t = ts.start + delta;
            let prev = new
                .time_signatures
                .iter()
                .filter(|p| p.start < new_t)
                .max_by_key(|p| p.start);
            if prev.is_some_and(|p| (p.beats, p.beat_type) == (ts.beats, ts.beat_type)) {
                continue;
            }
            let mut copy = *ts;
            copy.start = new_t;
            new.time_signatures.push(copy);
        }
        // tempos, Fine/Segno/Coda markers (Repeat, Ending, ToCoda, DaCapo, DalSegno dropped)
        for tempo in part.tempos.iter().filter(|t| in_segment(t.start)) {
            let mut copy = tempo.clone();
            copy.start += delta;
            new.tempos.push(copy);
        }
        for &(t, m) in part.markers.iter().filter(|(t, _)| in_segment(*t)) {
            if matches!(m, Marker::Fine | Marker::Segno | Marker::Coda) {
                new.markers.push((t + delta, m));
            }
        }
        // notes, with tie references mapped inside this segment only
        let base = new.notes.len();
        let mut map: HashMap<usize, usize> = HashMap::new();
        let mut copied: Vec<usize> = Vec::new();
        for (i, n) in part.notes.iter().enumerate() {
            if in_segment(n.start) {
                map.insert(i, base + copied.len());
                copied.push(i);
            }
        }
        for &i in &copied {
            let n = &part.notes[i];
            new.add_point(n.end + delta);
            new.notes.push(Note {
                start: n.start + delta,
                end: n.end + delta,
                tie_next: n.tie_next.and_then(|j| map.get(&j).copied()),
                tie_prev: n.tie_prev.and_then(|j| map.get(&j).copied()),
                ..n.clone()
            });
        }
    }
    // keep timeline iteration order of notes (by start time, stable)
    let mut order: Vec<usize> = (0..new.notes.len()).collect();
    order.sort_by_key(|&i| new.notes[i].start);
    let mut position = vec![0; order.len()];
    for (new_i, &old_i) in order.iter().enumerate() {
        position[old_i] = new_i;
    }
    let notes: Vec<Note> = order
        .iter()
        .map(|&i| {
            let mut n = new.notes[i].clone();
            n.tie_next = n.tie_next.map(|j| position[j]);
            n.tie_prev = n.tie_prev.map(|j| position[j]);
            n
        })
        .collect();
    new.notes = notes;
    Ok(new)
}

/// Port of `update_note_ids_after_unfolding`: every id gets `-<k>` in time order.
fn update_note_ids_after_unfolding(part: &mut Part) {
    let mut by_id: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, n) in part.notes.iter().enumerate() {
        if let Some(id) = &n.id {
            by_id.entry(id.clone()).or_default().push(i);
        }
    }
    for (id, mut idx) in by_id {
        idx.sort_by_key(|&i| part.notes[i].start);
        for (k, i) in idx.into_iter().enumerate() {
            part.notes[i].id = Some(format!("{id}-{}", k + 1));
        }
    }
}
