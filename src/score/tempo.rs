//! Port of `matchmaker.utils.misc.get_tempo_from_score` (with
//! `extract_tempo_marking_from_musicxml`) and `partitura.utils.music.to_quarter_tempo`.
//!
//! Upstream: `matchmaker/utils/misc.py`, `partitura/utils/music.py`.
//!
//! Sources in order: the first partitura `Tempo` object, the first MusicXML `<sound tempo>`,
//! a text tempo marking such as "Allegro". `None` means the caller's default (120 BPM).

use roxmltree::{Document, Node};

use super::part::Part;

/// `TEMPO_MARKING_TO_BPM` in its dict order (the first matching marking wins).
const TEMPO_MARKING_TO_BPM: [(&str, f64); 16] = [
    ("larghissimo", 24.0),
    ("grave", 30.0),
    ("largo", 40.0),
    ("larghetto", 50.0),
    ("lento", 60.0),
    ("adagio", 60.0),
    ("adagietto", 70.0),
    ("andante", 80.0),
    ("andantino", 90.0),
    ("moderato", 110.0),
    ("allegretto", 120.0),
    ("allegro", 130.0),
    ("vivace", 150.0),
    ("vivacissimo", 170.0),
    ("presto", 180.0),
    ("prestissimo", 200.0),
];

/// Port of `to_quarter_tempo(unit, tempo)`; `None` where Python raises (unknown unit, > 3 dots).
fn to_quarter_tempo(unit: &str, tempo: f64) -> Option<f64> {
    const DOT_MULTIPLIERS: [f64; 4] = [1.0, 1.0 + 1.0 / 2.0, 1.0 + 3.0 / 4.0, 1.0 + 7.0 / 8.0];
    let dots = unit.matches('.').count();
    let label = unit.trim().trim_end_matches('.');
    let dur = match label {
        "long" => 16.0,
        "breve" => 8.0,
        "whole" => 4.0,
        "half" | "h" => 2.0,
        "quarter" | "q" => 1.0,
        "eighth" | "e" => 1.0 / 2.0,
        "16th" => 1.0 / 4.0,
        "32nd" => 1.0 / 8.0,
        "64th" => 1.0 / 16.0,
        "128th" => 1.0 / 32.0,
        "256th" => 1.0 / 64.0,
        _ => return None,
    };
    Some(tempo * DOT_MULTIPLIERS.get(dots)? * dur)
}

/// Port of `get_tempo_from_score(score_part, score_file)`; `xml` is the MusicXML text if the
/// score file is plain MusicXML (`ElementTree.parse` fails on `.mxl` and MIDI files).
pub(crate) fn get_tempo_from_score(part: &Part, xml: Option<&str>) -> Option<f64> {
    // 1. partitura Tempo objects (first in timeline order); any error skips this source
    let mut tempos: Vec<_> = part.tempos.iter().collect();
    tempos.sort_by_key(|t| t.start);
    if let Some(first) = tempos.first()
        && let Some(bpm) = to_quarter_tempo(first.unit.as_deref().unwrap_or("q"), first.bpm)
    {
        return Some(bpm);
    }
    let doc = xml.and_then(|text| {
        let options = roxmltree::ParsingOptions {
            allow_dtd: true,
            ..Default::default()
        };
        Document::parse_with_options(text, options).ok()
    })?;
    // 2. first <sound tempo="..."> anywhere in the document
    if let Some(tempo) = doc
        .descendants()
        .filter(|n| n.has_tag_name("sound"))
        .find_map(|n| n.attribute("tempo"))
        && let Ok(bpm) = tempo.trim().parse::<f64>()
    {
        return Some(bpm);
    }
    // 3. text tempo marking
    extract_tempo_marking_from_musicxml(&doc)
}

/// Port of `extract_tempo_marking_from_musicxml`. Note: the reference breaks out of the
/// measure loop after the first measure of each part, so only first measures are searched.
fn extract_tempo_marking_from_musicxml(doc: &Document) -> Option<f64> {
    let mut current_beats: i64 = 4;
    let mut current_beat_type: i64 = 4;
    for part in doc.descendants().filter(|n| n.has_tag_name("part")) {
        let Some(measure) = part.descendants().find(|n| n.has_tag_name("measure")) else {
            continue;
        };
        for attributes in measure
            .descendants()
            .filter(|n| n.has_tag_name("attributes"))
        {
            for time in attributes.descendants().filter(|n| n.has_tag_name("time")) {
                if let Some(text) = child_text(time, "beats").filter(|t| !t.is_empty()) {
                    current_beats = text.trim().parse().ok()?; // ValueError -> None
                }
                if let Some(text) = child_text(time, "beat-type").filter(|t| !t.is_empty()) {
                    current_beat_type = text.trim().parse().ok()?;
                }
            }
        }
        for direction in measure
            .descendants()
            .filter(|n| n.has_tag_name("direction"))
        {
            for direction_type in direction
                .descendants()
                .filter(|n| n.has_tag_name("direction-type"))
            {
                for words in direction_type
                    .descendants()
                    .filter(|n| n.has_tag_name("words"))
                {
                    let Some(text) = words.text().filter(|t| !t.is_empty()) else {
                        continue;
                    };
                    let text = text.to_lowercase();
                    let text = text.trim();
                    for (marking, bpm) in TEMPO_MARKING_TO_BPM {
                        if starts_with_word(text, marking) {
                            let compound = current_beats >= 6 && current_beats % 3 == 0;
                            return Some(if compound {
                                bpm / 1.5
                            } else {
                                bpm * (4.0 / current_beat_type as f64)
                            });
                        }
                    }
                }
            }
        }
    }
    None
}

fn child_text<'a>(node: Node<'a, 'a>, tag: &str) -> Option<&'a str> {
    node.children().find(|n| n.has_tag_name(tag))?.text()
}

/// `re.match(rf"^{marking}\b", text)`.
fn starts_with_word(text: &str, marking: &str) -> bool {
    text.strip_prefix(marking).is_some_and(|rest| {
        !rest
            .chars()
            .next()
            .is_some_and(|c| c.is_alphanumeric() || c == '_')
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_tempo() {
        assert_eq!(to_quarter_tempo("q", 100.0), Some(100.0));
        assert_eq!(to_quarter_tempo("h", 100.0), Some(200.0));
        assert_eq!(to_quarter_tempo("Q", 100.0), None);
    }

    #[test]
    fn marking_word_boundary() {
        assert!(starts_with_word("allegro molto", "allegro"));
        assert!(!starts_with_word("allegretto", "allegro"));
    }
}
