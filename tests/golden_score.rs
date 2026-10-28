//! Golden test: score loading equals partitura's note array (FR-001, FR-002, FR-003).
//!
//! Tolerance: exact. Integer fields and ids must be equal; `float32` fields bit-equal (same
//! `float64` formula, then cast). Rows are compared after sorting by (onset_beat, pitch, id),
//! because partitura's pitch sort is not stable for equal (onset, pitch) pairs.

mod common;

use common::{Ctx, golden, golden_pieces, json, npy};
use matchmaker_rs::score::Score;

#[derive(Debug, Clone, PartialEq)]
struct Row {
    onset_beat: f32,
    pitch: i64,
    id: String,
    duration_beat: f32,
    onset_quarter: f32,
    duration_quarter: f32,
    onset_div: i64,
    duration_div: i64,
    voice: i64,
}

fn sort_rows(rows: &mut [Row]) {
    rows.sort_by(|a, b| {
        a.onset_beat
            .total_cmp(&b.onset_beat)
            .then(a.pitch.cmp(&b.pitch))
            .then(a.id.cmp(&b.id))
    });
}

fn score_file(piece: &str) -> Option<std::path::PathBuf> {
    if piece.ends_with("_wrongnote") {
        return None;
    }
    if let Some(base) = piece.strip_suffix("_midiscore") {
        return Some(common::data(&format!("{base}.mid")));
    }
    Some(common::data(&format!("{piece}.musicxml")))
}

#[test]
fn note_arrays_equal_reference() {
    let mut checked = 0;
    for piece in golden_pieces() {
        let Some(path) = score_file(&piece) else {
            continue;
        };
        let dir = golden(&piece).join("score");
        let ctx = Ctx::new("score", &piece, "score/note_array.npy");
        let na = npy(dir.join("note_array.npy"));
        let ids: Vec<String> = serde_json::from_value(json(dir.join("note_ids.json"))).unwrap();
        let f = |name: &str| na.field(name);
        let (ob, db) = (f("onset_beat"), f("duration_beat"));
        let (oq, dq) = (f("onset_quarter"), f("duration_quarter"));
        let (od, dd, pitch, voice) = (f("onset_div"), f("duration_div"), f("pitch"), f("voice"));
        let mut expected: Vec<Row> = (0..na.len())
            .map(|i| Row {
                onset_beat: ob[i] as f32,
                pitch: pitch[i] as i64,
                id: ids[i].clone(),
                duration_beat: db[i] as f32,
                onset_quarter: oq[i] as f32,
                duration_quarter: dq[i] as f32,
                onset_div: od[i] as i64,
                duration_div: dd[i] as i64,
                voice: voice[i] as i64,
            })
            .collect();

        let score =
            Score::load(&path, true).unwrap_or_else(|e| panic!("[{piece}] load failed: {e}"));
        let mut actual: Vec<Row> = score
            .notes
            .iter()
            .map(|n| Row {
                onset_beat: n.onset_beat,
                pitch: i64::from(n.pitch),
                id: n.id.clone(),
                duration_beat: n.duration_beat,
                onset_quarter: n.onset_quarter,
                duration_quarter: n.duration_quarter,
                onset_div: n.onset_div,
                duration_div: n.duration_div,
                voice: i64::from(n.voice),
            })
            .collect();
        sort_rows(&mut expected);
        sort_rows(&mut actual);
        assert_eq!(
            expected.len(),
            actual.len(),
            "[{} / {}] number of notes differs",
            ctx.piece,
            ctx.file
        );
        for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
            assert!(
                e == a,
                "[{} / {} / {}] first difference at row {i}:\n expected {e:?}\n actual   {a:?}",
                ctx.method,
                ctx.piece,
                ctx.file
            );
        }

        let meta = json(dir.join("meta.json"));
        assert_eq!(
            meta["unfolded"].as_bool(),
            Some(score.unfolded),
            "[{piece}] unfolded flag differs"
        );
        let expected_tempo = meta["score_tempo"].as_f64();
        assert_eq!(expected_tempo, score.tempo, "[{piece}] score tempo differs");
        checked += 1;
    }
    assert!(checked >= 8, "only {checked} golden scores found");
}
