//! Golden test: performance loading, stream framing (FR-004) and the `pitch` / `chord_onset`
//! processors (FR-006, SC-003).
//!
//! Tolerance: exact. Message times and frame times use the reference's float formulas.
//! Messages are compared as a multiset per timestamp, because the reference sorts them with
//! NumPy's unstable `argsort` (research R5).

mod common;

use common::{Ctx, assert_exact, data, golden, npy};
use matchmaker_rs::features::{Processor, chord_onset::ChordOnsetProcessor, pitch::PitchProcessor};
use matchmaker_rs::io::stream::{Frame, MidiMessage, MidiMessageKind, StreamMode, frames};
use matchmaker_rs::performance::Performance;

const PIECES: [&str; 4] = [
    "simple_mozart_k265_var1",
    "mozart_k265_var1",
    "Bach-fugue_bwv_858",
    "simple_mozart_k265_var1_wrongnote",
];

/// `(kind, channel, note, velocity, control, value, program, time)` as written by generate.py.
type Record = [f64; 8];

fn record(m: &MidiMessage) -> Record {
    let (kind, note, velocity, control, value, program) = match m.kind {
        MidiMessageKind::NoteOn { note, velocity } => (0, note.into(), velocity.into(), -1, -1, -1),
        MidiMessageKind::NoteOff { note } => (1, note.into(), 0, -1, -1, -1),
        MidiMessageKind::ControlChange { control, value } => {
            (2, -1, -1, control.into(), value.into(), -1)
        }
        MidiMessageKind::ProgramChange { program } => (3, -1, -1, -1, -1, program.into()),
    };
    let i = |v: i32| f64::from(v);
    [
        i(kind),
        f64::from(m.channel),
        i(note),
        i(velocity),
        i(control),
        i(value),
        i(program),
        m.time,
    ]
}

fn golden_records(path: std::path::PathBuf) -> Vec<Record> {
    let a = npy(path);
    let cols: Vec<Vec<f64>> = [
        "kind", "channel", "note", "velocity", "control", "value", "program", "time",
    ]
    .iter()
    .map(|f| a.field(f))
    .collect();
    (0..a.len())
        .map(|r| std::array::from_fn(|c| cols[c][r]))
        .collect()
}

/// Sort by (time, rest) so equal-time groups compare as multisets.
fn sorted(mut v: Vec<Record>) -> Vec<Record> {
    v.sort_by(|a, b| a[7].total_cmp(&b[7]).then(a.partial_cmp(b).unwrap()));
    v
}

fn assert_records(ctx: &Ctx, expected: Vec<Record>, actual: Vec<Record>) {
    let (e, a) = (sorted(expected), sorted(actual));
    assert_eq!(
        e.len(),
        a.len(),
        "[{} / {} / {}] message count differs",
        ctx.method,
        ctx.piece,
        ctx.file
    );
    for (i, (e, a)) in e.iter().zip(&a).enumerate() {
        assert!(
            e == a,
            "[{} / {} / {}] first difference at message {i}: expected {e:?}, actual {a:?}",
            ctx.method,
            ctx.piece,
            ctx.file
        );
    }
}

fn messages(piece: &str) -> Vec<MidiMessage> {
    Performance::load(data(&format!("{piece}.mid")))
        .unwrap_or_else(|e| panic!("[{piece}] {e}"))
        .messages()
}

#[test]
fn stream_equals_reference() {
    for piece in PIECES {
        let msgs = messages(piece);
        let dir = golden(piece).join("stream");
        let ctx = Ctx::new("stream", piece, "event/messages.npy");
        assert_records(
            &ctx,
            golden_records(dir.join("event/messages.npy")),
            msgs.iter().map(record).collect(),
        );

        for (label, pp) in [("windowed_0.01", 0.01), ("windowed_0.001", 0.001)] {
            let framed = frames(&msgs, StreamMode::Windowed { polling_period: pp });
            // frame_times.npy holds the function's (k + 0.5) * pp; processors see
            // Buffer.time = k * pp + 0.5 * pp (Frame::time), which the chord_onset golden checks
            let n_frames = npy(dir.join(label).join("frame_times.npy")).len();
            assert_eq!(
                n_frames,
                framed.len(),
                "[stream / {piece} / {label}] frame count differs"
            );

            let offsets = npy(dir.join(label).join("frame_offsets.npy")).values();
            let expected = golden_records(dir.join(label).join("messages.npy"));
            for (k, f) in framed.iter().enumerate() {
                let (from, to) = (offsets[k] as usize, offsets[k + 1] as usize);
                let ctx = Ctx::new("stream", piece, &format!("{label}/messages.npy frame {k}"));
                assert_records(
                    &ctx,
                    expected[from..to].to_vec(),
                    f.messages.iter().map(record).collect(),
                );
            }
        }
    }
}

fn run(processor: &mut dyn Processor, frames: &[Frame]) -> (Vec<Vec<f32>>, Vec<f64>) {
    let mut outputs: Vec<_> = frames.iter().filter_map(|f| processor.process(f)).collect();
    outputs.extend(processor.flush_remaining());
    outputs
        .into_iter()
        .map(|o| (o.features, o.perf_time))
        .unzip()
}

#[test]
fn processors_equal_reference() {
    for piece in PIECES.iter().take(3) {
        let msgs = messages(piece);
        let w10 = frames(
            &msgs,
            StreamMode::Windowed {
                polling_period: 0.01,
            },
        );
        let w1 = frames(
            &msgs,
            StreamMode::Windowed {
                polling_period: 0.001,
            },
        );
        let event = frames(&msgs, StreamMode::Event);
        let cases: [(&str, Box<dyn Processor>, &[Frame]); 5] = [
            (
                "pitch_piano_vector",
                Box::new(PitchProcessor::new(true, false)),
                &w10,
            ),
            (
                "pitch_piano_list",
                Box::new(PitchProcessor::new(true, true)),
                &w10,
            ),
            (
                "pitch_full_vector",
                Box::new(PitchProcessor::new(false, false)),
                &w10,
            ),
            (
                "pitch_piano_vector_event",
                Box::new(PitchProcessor::new(true, false)),
                &event,
            ),
            (
                "chord_onset_piano",
                Box::new(ChordOnsetProcessor::new(true)),
                &w1,
            ),
        ];
        for (name, mut processor, input) in cases {
            let dir = golden(piece).join("processor").join(name);
            let (features, times) = run(processor.as_mut(), input);
            let expected = npy(dir.join("features.npy"));
            let width = expected.shape.get(1).copied().unwrap_or(0);
            // Tie order of the reference's unstable message sort (R5) shows up as pitch order
            // inside a list and, in event mode, as order of same-time observations: compare
            // list rows sorted and event-mode observations as a multiset per timestamp.
            let normalize = |mut rows: Vec<(f64, Vec<f64>)>| {
                if name.ends_with("_list") {
                    rows.iter_mut()
                        .for_each(|(_, r)| r.sort_by(|a, b| b.total_cmp(a)));
                }
                if name.ends_with("_event") {
                    rows.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.partial_cmp(&b.1).unwrap()));
                }
                rows.into_iter().unzip::<_, _, Vec<f64>, Vec<Vec<f64>>>()
            };
            let expected_rows = npy(dir.join("times.npy"))
                .values()
                .into_iter()
                .zip(expected.values().chunks(width.max(1)).map(<[f64]>::to_vec))
                .collect();
            let actual_rows = times
                .into_iter()
                .zip(features.iter().map(|r| {
                    (0..width)
                        .map(|c| r.get(c).map_or(-1.0, |&v| f64::from(v)))
                        .collect()
                }))
                .collect();
            let (e_times, e_rows) = normalize(expected_rows);
            let (a_times, a_rows) = normalize(actual_rows);
            assert_exact(&Ctx::new(name, piece, "times.npy"), &e_times, &a_times);
            assert_exact(
                &Ctx::new(name, piece, "features.npy"),
                &e_rows.concat(),
                &a_rows.concat(),
            );
        }
    }
}
