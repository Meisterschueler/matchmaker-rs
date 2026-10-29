//! Port of `partitura.io.importmusicxml.load_musicxml` (the parts that matter for score
//! following), called by pymatchmaker as
//! `load_musicxml(path, ignore_invisible_objects=True, force_note_ids="keep")`.
//!
//! Upstream: `partitura/io/importmusicxml.py`.
//!
//! The structure follows the Python functions (`_parse_partlist`, `_parse_parts`,
//! `_handle_measure`, `_handle_note`, `_handle_direction`, `_handle_sound`, …) so both can be
//! read side by side. Objects partitura creates but score following never reads (beams,
//! slurs, clefs, dynamics, …) are not kept; their timepoints only lie inside measures.

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::Path;

use roxmltree::{Document, Node};

use super::part::{Marker, Note, Part, Span, Tempo, Time, TimeSignature};
use crate::error::{Error, Result};

/// `ACCIDENTAL_MAP` of partitura.
fn accidental_alter(text: &str) -> Option<i32> {
    match text {
        "sharp" => Some(1),
        "natural" => Some(0),
        "flat" => Some(-1),
        "double-sharp" => Some(2),
        "double-flat" => Some(-2),
        _ => None,
    }
}

/// `pitch_spelling_to_midi_pitch(step, alter, octave)`.
fn midi_pitch(step: &str, alter: Option<i32>, octave: i32) -> Option<i32> {
    let base = match step.to_ascii_lowercase().as_str() {
        "c" => 0,
        "d" => 2,
        "e" => 4,
        "f" => 5,
        "g" => 7,
        "a" => 9,
        "b" => 11,
        _ => return None,
    };
    Some((octave + 1) * 12 + base + alter.unwrap_or(0))
}

/// Read the score-partwise document, unpacking `.mxl` containers like partitura.
pub(crate) fn read_document_text(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
    let xml_err = |message: String| Error::MusicXml {
        path: path.to_path_buf(),
        message,
    };
    if bytes.starts_with(b"PK\x03\x04") {
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes))
            .map_err(|e| xml_err(format!("invalid .mxl container: {e}")))?;
        let mut container = String::new();
        archive
            .by_name("META-INF/container.xml")
            .map_err(|e| xml_err(format!("missing META-INF/container.xml: {e}")))?
            .read_to_string(&mut container)
            .map_err(|e| xml_err(e.to_string()))?;
        let doc = Document::parse(&container).map_err(|e| xml_err(e.to_string()))?;
        let full_path = doc
            .descendants()
            .find(|n| n.has_tag_name("rootfile"))
            .and_then(|n| n.attribute("full-path"))
            .ok_or_else(|| xml_err("container.xml has no rootfile".into()))?
            .to_string();
        let mut text = String::new();
        archive
            .by_name(&full_path)
            .map_err(|e| xml_err(format!("missing {full_path}: {e}")))?
            .read_to_string(&mut text)
            .map_err(|e| xml_err(e.to_string()))?;
        return Ok(text);
    }
    String::from_utf8(bytes).map_err(|e| xml_err(format!("not UTF-8: {e}")))
}

/// Parse a MusicXML file into its parts (in `part-list` order), with note ids assigned like
/// `force_note_ids="keep"`.
pub(crate) fn load_musicxml(path: &Path) -> Result<Vec<Part>> {
    let text = read_document_text(path)?;
    parse_musicxml(&text).map_err(|message| Error::MusicXml {
        path: path.to_path_buf(),
        message,
    })
}

/// Parse MusicXML text (see [`load_musicxml`]).
pub(crate) fn parse_musicxml(text: &str) -> std::result::Result<Vec<Part>, String> {
    let options = roxmltree::ParsingOptions {
        allow_dtd: true,
        ..Default::default()
    };
    let doc = Document::parse_with_options(text, options).map_err(|e| e.to_string())?;
    let root = doc.root_element();
    if root.tag_name().name() != "score-partwise" {
        return Err("Currently only score-partwise structure is supported".into());
    }
    let Some(partlist) = child(root, "part-list") else {
        return Ok(Vec::new());
    };
    // _parse_partlist: score-part elements define the parts (order of the part list).
    let mut parts: Vec<Part> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for e in partlist.children().filter(Node::is_element) {
        if e.tag_name().name() == "score-part" {
            let id = e.attribute("id").unwrap_or_default().to_string();
            index.insert(id.clone(), parts.len());
            parts.push(Part::new(id));
        }
    }
    // _parse_parts: parts that are not listed in part-list are parsed but discarded.
    let mut generic_notes: Vec<Vec<GenericNote>> = vec![Vec::new(); parts.len()];
    for part_el in root.children().filter(|n| n.has_tag_name("part")) {
        let id = part_el.attribute("id").unwrap_or("P1");
        let Some(&i) = index.get(id) else { continue };
        parse_part(part_el, &mut parts[i], &mut generic_notes[i])?;
    }
    // assign_note_ids(partlist, keep=True)
    for (p, (part, generic)) in parts.iter_mut().zip(&generic_notes).enumerate() {
        assign_note_ids_keep(p, part, generic);
    }
    Ok(parts)
}

/// Kind of a `GenericNote` (the class key under which partitura stores it at a timepoint).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NoteClass {
    Note,
    GraceNote,
    Rest,
    Unpitched,
}

/// Every `GenericNote` in document order, for id assignment.
#[derive(Debug, Clone)]
struct GenericNote {
    start: Time,
    class: NoteClass,
    /// Index into `Part::notes` for pitched notes.
    note: Option<usize>,
    has_id: bool,
}

/// Port of `assign_note_ids(parts, keep=True)`: notes without id get `p{part}n{index}`, where
/// `index` counts all generic notes (incl. rests) in timeline iteration order.
fn assign_note_ids_keep(part_index: usize, part: &mut Part, generic: &[GenericNote]) {
    for (ni, g) in timeline_order(generic).into_iter().enumerate() {
        if let (Some(n), false) = (g.note, g.has_id) {
            part.notes[n].id = Some(format!("p{part_index}n{ni}"));
        }
    }
}

/// Timeline iteration order of `iter_all(GenericNote, include_subclasses=True)`: by start
/// time; at one timepoint grouped by class in order of first insertion, then insertion order.
fn timeline_order(generic: &[GenericNote]) -> Vec<&GenericNote> {
    let mut by_time: BTreeMap<Time, Vec<&GenericNote>> = BTreeMap::new();
    for g in generic {
        by_time.entry(g.start).or_default().push(g);
    }
    let mut out = Vec::with_capacity(generic.len());
    for (_, items) in by_time {
        let mut classes: Vec<NoteClass> = Vec::new();
        for g in &items {
            if !classes.contains(&g.class) {
                classes.push(g.class);
            }
        }
        for c in classes {
            out.extend(items.iter().filter(|g| g.class == c));
        }
    }
    out
}

/// Ties collected per pitch: `ongoing["tie_notes"]`.
#[derive(Default)]
struct TieNotes {
    pitches: Vec<i32>,
    start: HashMap<(i32, Time), usize>,
    stop: HashMap<(i32, Time), usize>,
}

/// Port of `_parse_parts` for one `<part>`.
fn parse_part(
    part_el: Node,
    part: &mut Part,
    generic: &mut Vec<GenericNote>,
) -> std::result::Result<(), String> {
    let mut position: Time = 0;
    let mut ongoing = Ongoing::default();
    // _handle_new_page / _handle_new_system at position 0
    part.add_point(0);
    for measure_el in part_el.children().filter(|n| n.has_tag_name("measure")) {
        position = handle_measure(measure_el, position, part, &mut ongoing, generic)?;
    }

    // add collected ties (pitches in first-seen order, start timepoints in insertion order)
    let ties = std::mem::take(&mut ongoing.ties);
    for pitch in &ties.pitches {
        let mut starts: Vec<(&(i32, Time), &usize)> =
            ties.start.iter().filter(|((p, _), _)| p == pitch).collect();
        starts.sort_by_key(|((_, t), _)| *t);
        for ((_, t), &start_note) in starts {
            if let Some(&stop_note) = ties.stop.get(&(*pitch, *t)) {
                part.notes[stop_note].tie_prev = Some(start_note);
                part.notes[start_note].tie_next = Some(stop_note);
            }
        }
    }

    complete_endings_and_repeats(part);
    Ok(())
}

/// State that partitura keeps in `ongoing` and needs here.
#[derive(Default)]
struct Ongoing {
    ties: TieNotes,
    repeat: Option<usize>,
    /// partitura keys endings by `("ending", getattr(e, "number", "0"))`, which is always
    /// `("ending", "0")` for lxml elements, so there is at most one open ending.
    ending: Option<usize>,
}

/// Port of `_handle_measure`. Returns the end time of the measure.
fn handle_measure(
    measure_el: Node,
    mut position: Time,
    part: &mut Part,
    ongoing: &mut Ongoing,
    generic: &mut Vec<GenericNote>,
) -> std::result::Result<Time, String> {
    let measure_index = part.measures.len();
    part.measures.push(Span {
        start: Some(position),
        end: None,
        number: measure_el.attribute("number").map(str::to_string),
    });
    part.add_point(position);
    let measure_start = position;
    let mut prev_note: Option<PrevNote> = None;
    let mut measure_maxtime = measure_start;

    for e in measure_el.children().filter(Node::is_element) {
        // ignore_invisible_objects=True
        let invisible = e.attribute("print-object") == Some("no")
            || child(e, "notehead").is_some_and(|n| n.text() == Some("none"))
            || child(e, "cue").is_some();
        if invisible {
            if e.has_tag_name("note") && child(e, "chord").is_none() {
                position += value_from_tag::<Time>(e, "duration").unwrap_or(0);
            }
            continue;
        }
        match e.tag_name().name() {
            "backup" => {
                position -= value_from_tag::<Time>(e, "duration").unwrap_or(0);
                if position < measure_start {
                    position = measure_start;
                }
                measure_maxtime = measure_maxtime.max(position);
            }
            "forward" => {
                position += value_from_tag::<Time>(e, "duration").unwrap_or(0);
                measure_maxtime = measure_maxtime.max(position);
            }
            "attributes" => handle_attributes(e, position, part),
            "direction" => handle_direction(e, position, part)?,
            "sound" => handle_sound(e, position, part)?,
            "note" => {
                let (new_position, note) =
                    handle_note(e, position, part, ongoing, prev_note, generic);
                position = new_position;
                prev_note = Some(note);
                measure_maxtime = measure_maxtime.max(position);
            }
            "harmony" => part.add_point(position),
            "barline" => {
                let position_barline = match e.attribute("location") {
                    None | Some("right") => measure_maxtime,
                    Some("left") => measure_start,
                    Some(_) => position,
                };
                if let Some(repeat) = child(e, "repeat") {
                    handle_repeat(repeat, position_barline, part, ongoing);
                }
                if let Some(ending) = child(e, "ending") {
                    handle_ending(ending, position_barline, part, ongoing);
                }
                if child(e, "bar-style").is_some() || child(e, "fermata").is_some() {
                    part.add_point(position);
                }
            }
            _ => {}
        }
    }
    part.measures[measure_index].end = Some(measure_maxtime);
    part.add_point(measure_maxtime);
    Ok(measure_maxtime)
}

/// Port of `_handle_attributes` (time signature and divisions; other attributes only add a
/// timepoint).
fn handle_attributes(e: Node, position: Time, part: &mut Part) {
    let beats = value_from_tag::<i64>(e, "time/beats");
    let beat_type = value_from_tag::<i64>(e, "time/beat-type");
    if let (Some(beats), Some(beat_type)) = (beats, beat_type)
        && beats != 0
        && beat_type != 0
    {
        part.time_signatures.push(TimeSignature {
            start: position,
            beats,
            beat_type,
        });
        part.add_point(position);
    }
    let has_key = find(e, "key/fifths").is_some() || find(e, "key/mode").is_some();
    let has_transpose =
        find(e, "transpose/diatonic").is_some() || find(e, "transpose/chromatic").is_some();
    if has_key || has_transpose || child(e, "clef").is_some() {
        part.add_point(position);
    }
    if let Some(divs) = value_from_tag::<i64>(e, "divisions")
        && divs != 0
    {
        part.set_quarter_duration(position, divs);
    }
}

/// Port of `_handle_direction`: navigation markers and tempos.
fn handle_direction(e: Node, position: Time, part: &mut Part) -> std::result::Result<(), String> {
    for sd in e.children().filter(|n| n.has_tag_name("sound")) {
        let attr = |name: &str| sd.attribute(name);
        let mut markers = Vec::new();
        if matches!(attr("fine"), Some("yes" | "1")) {
            markers.push(Marker::Fine);
        }
        if attr("dacapo") == Some("yes") {
            markers.push(Marker::DaCapo);
        }
        if attr("tocoda") == Some("coda") {
            markers.push(Marker::ToCoda);
        }
        if attr("coda") == Some("coda") {
            markers.push(Marker::Coda);
        }
        if attr("dalsegno") == Some("segno") {
            markers.push(Marker::DalSegno);
        }
        if attr("segno") == Some("segno") {
            markers.push(Marker::Segno);
        }
        for m in markers {
            part.markers.push((position, m));
            part.add_point(position);
        }
        handle_sound(sd, position, part)?;
    }
    // starting_directions in partitura's order; ending directions only add a timepoint.
    enum Item {
        Other,
        Tempo(Tempo),
    }
    let mut starting: Vec<Item> = Vec::new();
    let mut ending = false;
    for direction_type in e.children().filter(|n| n.has_tag_name("direction-type")) {
        let Some(dt) = direction_type.children().find(Node::is_element) else {
            continue;
        };
        match dt.tag_name().name() {
            "dynamics" => {
                for c in direction_type.children().filter(Node::is_element) {
                    if c.children().any(|n| n.is_element()) {
                        starting.push(Item::Other);
                    }
                }
            }
            "words" => {
                for words in direction_type.children().filter(Node::is_element) {
                    let Some(text) = words.text() else { continue };
                    match parse_tempo_indication(text) {
                        Some((prefixed, unit, bpm)) => {
                            if prefixed {
                                starting.push(Item::Other);
                            }
                            starting.push(Item::Tempo(Tempo {
                                start: position,
                                bpm,
                                unit: Some(unit),
                            }));
                        }
                        None => starting.push(Item::Other),
                    }
                }
            }
            "wedge" => match dt.attribute("type") {
                Some("crescendo" | "diminuendo") => starting.push(Item::Other),
                Some("stop") => ending = true,
                _ => {}
            },
            "octave-shift" => match dt.attribute("type") {
                Some("up" | "down") => starting.push(Item::Other),
                Some("stop") => ending = true,
                _ => {}
            },
            "pedal" => match dt.attribute("type") {
                Some("start") => starting.push(Item::Other),
                Some("stop") => ending = true,
                _ => {}
            },
            _ => {}
        }
    }
    for item in starting {
        match item {
            Item::Other => part.add_point(position),
            Item::Tempo(tempo) => add_tempo_if_unique(position, part, tempo),
        }
    }
    if ending {
        part.add_point(position);
    }
    Ok(())
}

/// Port of `_handle_sound`: `<sound tempo="…">` adds a quarter-note tempo.
fn handle_sound(e: Node, position: Time, part: &mut Part) -> std::result::Result<(), String> {
    if let Some(tempo) = e.attribute("tempo") {
        let bpm = tempo
            .trim()
            .parse::<f64>()
            .map_err(|_| format!("could not convert string to float: {tempo:?}"))?;
        add_tempo_if_unique(
            position,
            part,
            Tempo {
                start: position,
                bpm,
                unit: None,
            },
        );
    }
    Ok(())
}

/// Port of `_add_tempo_if_unique`: only if a timepoint exists at `position` and no tempo starts
/// there yet.
fn add_tempo_if_unique(position: Time, part: &mut Part, tempo: Tempo) {
    if part.points.contains(&position) && !part.tempos.iter().any(|t| t.start == position) {
        part.tempos.push(tempo);
    }
}

/// The `tempo_indication` rule of partitura's direction grammar: `[qhe]\.* = [0-9]+`, alone
/// or in parentheses after other words (see research R6 for the simplification).
///
/// Returns `(preceded_by_other_words, unit, bpm)`.
fn parse_tempo_indication(text: &str) -> Option<(bool, String, f64)> {
    let trimmed = text.trim().trim_end_matches(')').trim_end();
    let eq = trimmed.rfind('=')?;
    let bpm_text = trimmed[eq + 1..].trim();
    if bpm_text.is_empty() || !bpm_text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let before = trimmed[..eq].trim_end();
    let unit_start = before
        .rfind(|c: char| {
            !(c == '.'
                || c.eq_ignore_ascii_case(&'q')
                || c.eq_ignore_ascii_case(&'h')
                || c.eq_ignore_ascii_case(&'e'))
        })
        .map_or(0, |i| i + 1);
    let unit = &before[unit_start..];
    let mut chars = unit.chars();
    let first = chars.next()?;
    if !matches!(first.to_ascii_lowercase(), 'q' | 'h' | 'e') || !chars.all(|c| c == '.') {
        return None;
    }
    // what precedes the unit must be empty, "(" or separated by whitespace / "("
    let prefix = before[..unit_start].trim_end();
    if !(prefix.is_empty()
        || prefix.ends_with('(')
        || before[..unit_start].ends_with(char::is_whitespace))
    {
        return None;
    }
    let preceded = !prefix.trim_end_matches('(').trim().is_empty();
    Some((preceded, unit.to_string(), bpm_text.parse::<f64>().ok()?))
}

/// The previous `<note>` (`prev_note` in `_handle_note`).
#[derive(Debug, Clone, Copy)]
struct PrevNote {
    start: Time,
    duration: Time,
}

/// Port of `_handle_note`. Returns the new position and the note's timing.
fn handle_note(
    e: Node,
    mut position: Time,
    part: &mut Part,
    ongoing: &mut Ongoing,
    prev_note: Option<PrevNote>,
    generic: &mut Vec<GenericNote>,
) -> (Time, PrevNote) {
    let mut duration = value_from_tag::<Time>(e, "duration").unwrap_or(0);
    let voice = value_from_tag::<i32>(e, "voice")
        .filter(|v| *v != 0)
        .unwrap_or(1);
    let note_id = e
        .attribute("id")
        .filter(|s| !s.is_empty())
        .or_else(|| e.attribute("ID"))
        .map(str::to_string);

    if child(e, "chord").is_some()
        && let Some(prev) = prev_note
    {
        position = prev.start;
        duration = prev.duration;
    }

    let mut class = NoteClass::Rest;
    let mut note_index = None;
    let mut tie_pitch: Option<i32> = None; // None = "rest"
    if let Some(pitch) = child(e, "pitch") {
        let step = value_from_tag::<String>(pitch, "step");
        let mut alter = value_from_tag::<i32>(pitch, "alter");
        let octave = value_from_tag::<i32>(pitch, "octave");
        if alter.is_none() {
            alter = value_from_tag::<String>(e, "accidental").and_then(|a| accidental_alter(&a));
        }
        let is_grace = child(e, "grace").is_some();
        class = if is_grace {
            NoteClass::GraceNote
        } else {
            NoteClass::Note
        };
        if let (Some(step), Some(octave)) = (step, octave)
            && let Some(mp) = midi_pitch(&step, alter, octave)
        {
            tie_pitch = Some(mp);
            note_index = Some(part.notes.len());
            part.notes.push(Note {
                id: note_id.clone(),
                midi_pitch: mp,
                voice,
                start: position,
                end: position + duration,
                tie_next: None,
                tie_prev: None,
            });
        }
    } else if child(e, "unpitched").is_some() {
        class = NoteClass::Unpitched;
    }
    generic.push(GenericNote {
        start: position,
        class,
        note: note_index,
        has_id: note_id.is_some(),
    });
    part.add_point(position);
    part.add_point(position + duration);

    let tie_types: Vec<&str> = e
        .children()
        .filter(|n| n.has_tag_name("tie"))
        .filter_map(|n| n.attribute("type"))
        .collect();
    if !tie_types.is_empty()
        && let (Some(pitch), Some(n)) = (tie_pitch, note_index)
    {
        let ties = &mut ongoing.ties;
        if !ties.pitches.contains(&pitch) {
            ties.pitches.push(pitch);
        }
        if tie_types.contains(&"stop") {
            ties.stop.insert((pitch, position), n);
        }
        if tie_types.contains(&"start") {
            ties.start.insert((pitch, position + duration), n);
        }
    }
    (
        position + duration,
        PrevNote {
            start: position,
            duration,
        },
    )
}

/// Port of `_handle_repeat`.
fn handle_repeat(e: Node, position: Time, part: &mut Part, ongoing: &mut Ongoing) {
    match e.attribute("direction") {
        Some("forward") => {
            ongoing.repeat = Some(part.repeats.len());
            part.repeats.push(Span {
                start: Some(position),
                end: None,
                number: None,
            });
            part.add_point(position);
        }
        Some("backward") => {
            match ongoing.repeat.take() {
                Some(i) => part.repeats[i].end = Some(position),
                None => part.repeats.push(Span {
                    start: None,
                    end: Some(position),
                    number: None,
                }),
            }
            part.add_point(position);
        }
        _ => {}
    }
}

/// Port of `_handle_ending`.
fn handle_ending(e: Node, position: Time, part: &mut Part, ongoing: &mut Ongoing) {
    let number = e.attribute("number").map(str::to_string);
    match e.attribute("type") {
        Some("start") => {
            ongoing.ending = Some(part.endings.len());
            part.endings.push(Span {
                start: Some(position),
                end: None,
                number,
            });
            part.add_point(position);
        }
        Some("stop" | "discontinue") => {
            match ongoing.ending.take() {
                Some(i) => part.endings[i].end = Some(position),
                None => part.endings.push(Span {
                    start: None,
                    end: Some(position),
                    number,
                }),
            }
            part.add_point(position);
        }
        _ => {}
    }
}

/// `measure_map(t)` without the anacrusis correction: start and end of the measure that
/// contains `t` (`kind="previous"`, extrapolating).
fn measure_containing(part: &Part, t: Time) -> Option<(Time, Time)> {
    let mut measures: Vec<(Time, Time)> = part
        .measures
        .iter()
        .filter_map(|m| Some((m.start?, m.end?)))
        .collect();
    measures.sort_by_key(|m| m.0);
    measures
        .iter()
        .rev()
        .find(|m| m.0 <= t)
        .or(measures.first())
        .copied()
}

/// The completion steps at the end of `_parse_parts` for endings and repeats.
fn complete_endings_and_repeats(part: &mut Part) {
    for i in 0..part.endings.len() {
        let e = &part.endings[i];
        if e.start.is_none() {
            if let Some(end) = e.end
                && let Some((s, _)) = measure_containing(part, end - 1)
            {
                part.endings[i].start = Some(s);
            }
        } else if e.end.is_none()
            && let Some((_, end)) = measure_containing(part, e.start.unwrap_or(0))
        {
            part.endings[i].end = Some(end);
            part.add_point(end);
        }
    }

    // Complete repeats without end (iterated in start order).
    let by_start = |part: &Part| -> Vec<usize> {
        let mut order: Vec<usize> = (0..part.repeats.len())
            .filter(|&i| part.repeats[i].start.is_some())
            .collect();
        order.sort_by_key(|&i| part.repeats[i].start); // stable: insertion order on ties
        order
    };
    for i in by_start(part) {
        if part.repeats[i].end.is_none() {
            let start = part.repeats[i].start.unwrap_or(0);
            let mut end_times: Vec<Time> = by_start(part)
                .iter()
                .filter_map(|&j| part.repeats[j].start)
                .collect();
            end_times.push(part.last_point().unwrap_or(0));
            let idx = searchsorted_left(&end_times, start + 1);
            if let Some(&end) = end_times.get(idx) {
                part.repeats[i].end = Some(end);
            }
        }
    }

    // Complete repeats without start (iterated in end order).
    let mut order: Vec<usize> = (0..part.repeats.len())
        .filter(|&i| part.repeats[i].start.is_none())
        .collect();
    order.sort_by_key(|&i| part.repeats[i].end);
    let mut volta_repeats = Vec::new();
    for i in order {
        let end = part.repeats[i].end.unwrap_or(0);
        // an ending that ends here and starts where another repeat ends -> volta repeat
        let ending_here = part.endings.iter().find(|e| e.end == Some(end));
        if let Some(ending) = ending_here {
            let ending_start = ending.start;
            if part
                .repeats
                .iter()
                .any(|r| r.end.is_some() && r.end == ending_start)
            {
                volta_repeats.push(i);
                continue;
            }
        }
        // start_times = [0] + [r.end.t for r in iter_all(Repeat)] (start order, not sorted)
        let mut start_times: Vec<Time> = vec![0];
        start_times.extend(by_start(part).iter().filter_map(|&j| part.repeats[j].end));
        let idx = searchsorted_left(&start_times, end).saturating_sub(1);
        part.repeats[i].start = Some(start_times[idx]);
    }
    for i in volta_repeats {
        let end = part.repeats[i].end.unwrap_or(0);
        let mut start_times: Vec<Time> = vec![0];
        start_times.extend(by_start(part).iter().filter_map(|&j| part.repeats[j].start));
        let idx = searchsorted_left(&start_times, end).saturating_sub(1);
        part.repeats[i].start = Some(start_times[idx]);
    }
}

/// `numpy.searchsorted(a, v, side="left")`: binary search, also on unsorted input.
pub(crate) fn searchsorted_left(a: &[Time], v: Time) -> usize {
    let (mut lo, mut hi) = (0, a.len());
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if a[mid] < v {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// First child element with the given tag (`e.find(tag)` for a simple tag).
fn child<'a, 'i>(e: Node<'a, 'i>, tag: &str) -> Option<Node<'a, 'i>> {
    e.children().find(|n| n.has_tag_name(tag))
}

/// `e.find("a/b")` for a slash-separated path of child tags.
fn find<'a, 'i>(e: Node<'a, 'i>, path: &str) -> Option<Node<'a, 'i>> {
    let mut parts = path.split('/');
    let first = parts.next()?;
    // lxml's find returns the first match in document order, searching all candidates.
    fn rec<'a, 'i>(node: Node<'a, 'i>, tag: &str, rest: &[&str]) -> Option<Node<'a, 'i>> {
        for c in node.children().filter(|n| n.has_tag_name(tag)) {
            match rest.split_first() {
                None => return Some(c),
                Some((next, tail)) => {
                    if let Some(found) = rec(c, next, tail) {
                        return Some(found);
                    }
                }
            }
        }
        None
    }
    let rest: Vec<&str> = parts.collect();
    rec(e, first, &rest)
}

/// Port of `get_value_from_tag(e, tag, as_type)` (None if missing or not convertible).
fn value_from_tag<T: std::str::FromStr>(e: Node, path: &str) -> Option<T> {
    find(e, path)?.text()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tempo_indication_rule() {
        assert_eq!(
            parse_tempo_indication("q = 100"),
            Some((false, "q".into(), 100.0))
        );
        assert_eq!(
            parse_tempo_indication("Allegro (h.=60)"),
            Some((true, "h.".into(), 60.0))
        );
        assert_eq!(parse_tempo_indication("Allegro"), None);
        assert_eq!(parse_tempo_indication("a=3"), None);
    }

    #[test]
    fn minimal_score() {
        let xml = r#"<?xml version="1.0"?>
<score-partwise><part-list><score-part id="P1"/></part-list>
<part id="P1"><measure number="1">
<attributes><divisions>2</divisions><time><beats>2</beats><beat-type>4</beat-type></time></attributes>
<note id="a"><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration></note>
<note><chord/><pitch><step>E</step><alter>-1</alter><octave>4</octave></pitch><duration>2</duration></note>
<note><rest/><duration>2</duration></note>
</measure></part></score-partwise>"#;
        let parts = parse_musicxml(xml).unwrap();
        let p = &parts[0];
        assert_eq!(p.notes.len(), 2);
        assert_eq!(p.notes[0].midi_pitch, 60);
        assert_eq!(p.notes[1].midi_pitch, 63);
        assert_eq!(p.notes[1].start, 0);
        assert_eq!(p.notes[1].id.as_deref(), Some("p0n1"));
        assert_eq!(p.measures[0].end, Some(4));
    }
}
