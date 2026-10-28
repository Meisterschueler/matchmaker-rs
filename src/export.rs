//! Export of a recorded run in the layout of `contracts/golden-data.md`, so that
//! `tools/golden/compare.py` can diff it against the reference stage by stage.
//! `evaluation.json` needs ground truth and is written by the CLI (`--ground-truth`).
//!
//! ```no_run
//! use matchmaker_rs::matchmaker::{Matchmaker, Method};
//! let mut mm = Matchmaker::new("tests/data/simple_mozart_k265_var1.musicxml", Method::Pthmm, None, true)?
//!     .record();
//! mm.run_file("tests/data/simple_mozart_k265_var1.mid")?;
//! matchmaker_rs::export::write_run(&mm, "out/simple_mozart_k265_var1")?;
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::path::Path;

use serde_json::json;

use crate::error::{Error, Result};
use crate::features::Observation;
use crate::io::stream::{MidiMessage, MidiMessageKind};
use crate::matchmaker::{Matchmaker, Method, ProcessorKind};

/// Reference skips `probabilities.npy` above this many values (`MAX_PROBABILITY_VALUES`).
const MAX_PROBABILITY_VALUES: usize = 5_000_000;

const MESSAGE_FIELDS: &str = "[('kind', '<i4'), ('channel', '<i4'), ('note', '<i4'), \
     ('velocity', '<i4'), ('control', '<i4'), ('value', '<i4'), ('program', '<i4'), ('time', '<f8')]";
const NOTE_FIELDS: &str = "[('onset_beat', '<f4'), ('duration_beat', '<f4'), ('onset_quarter', '<f4'), \
     ('duration_quarter', '<f4'), ('onset_div', '<i4'), ('duration_div', '<i4'), ('pitch', '<i4'), \
     ('voice', '<i4')]";

fn write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
    }
    std::fs::write(path, bytes).map_err(|e| Error::io(path, e))
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    write(path, value.to_string().as_bytes())
}

/// NumPy format 1.0 file; `descr` is the Python repr of the dtype.
fn npy(path: &Path, descr: &str, shape: &[usize], data: &[u8]) -> Result<()> {
    let dims = match shape {
        [n] => format!("({n},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut header = format!("{{'descr': {descr}, 'fortran_order': False, 'shape': {dims}, }}");
    // 10 bytes of magic and length, then the header ends in '\n' on a 64-byte boundary
    header.push_str(&" ".repeat((64 - (10 + header.len() + 1) % 64) % 64));
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend((header.len() as u16).to_le_bytes());
    bytes.extend(header.as_bytes());
    bytes.extend(data);
    write(path, &bytes)
}

fn f4(path: &Path, shape: &[usize], values: impl Iterator<Item = f32>) -> Result<()> {
    npy(
        path,
        "'<f4'",
        shape,
        &values.flat_map(f32::to_le_bytes).collect::<Vec<_>>(),
    )
}

fn f8(path: &Path, shape: &[usize], values: impl Iterator<Item = f64>) -> Result<()> {
    npy(
        path,
        "'<f8'",
        shape,
        &values.flat_map(f64::to_le_bytes).collect::<Vec<_>>(),
    )
}

fn i8s(path: &Path, shape: &[usize], values: impl Iterator<Item = usize>) -> Result<()> {
    let bytes: Vec<u8> = values.flat_map(|v| (v as i64).to_le_bytes()).collect();
    npy(path, "'<i8'", shape, &bytes)
}

fn flags(path: &Path, values: &[bool]) -> Result<()> {
    let bytes: Vec<u8> = values.iter().map(|&b| u8::from(b)).collect();
    npy(path, "'|b1'", &[values.len()], &bytes)
}

/// `(kind, channel, note, velocity, control, value, program)` like generate.py's `message_record`.
fn message_fields(m: &MidiMessage) -> [i32; 7] {
    let channel = i32::from(m.channel);
    match m.kind {
        MidiMessageKind::NoteOn { note, velocity } => {
            [0, channel, note.into(), velocity.into(), -1, -1, -1]
        }
        MidiMessageKind::NoteOff { note } => [1, channel, note.into(), 0, -1, -1, -1],
        MidiMessageKind::ControlChange { control, value } => {
            [2, channel, -1, -1, control.into(), value.into(), -1]
        }
        MidiMessageKind::ProgramChange { program } => [3, channel, -1, -1, -1, -1, program.into()],
    }
}

/// `features.npy` / `times.npy` of observations; rows are padded with -1 to the longest one.
fn observations(dir: &Path, prefix: &str, obs: &[&Observation]) -> Result<()> {
    let width = obs.iter().map(|o| o.features.len()).max().unwrap_or(0);
    let rows = obs.iter().flat_map(|o| {
        let padding = width - o.features.len();
        o.features
            .iter()
            .copied()
            .chain(std::iter::repeat_n(-1.0, padding))
    });
    f4(
        &dir.join(format!("{prefix}features.npy")),
        &[obs.len(), width],
        rows,
    )?;
    f8(
        &dir.join(format!("{prefix}times.npy")),
        &[obs.len()],
        obs.iter().map(|o| o.perf_time),
    )
}

/// Write `score/`, `stream/`, `processor/` and `<method>/` of a recorded run into `dir`.
pub fn write_run(mm: &Matchmaker, dir: impl AsRef<Path>) -> Result<()> {
    let dir = dir.as_ref();
    let recording = mm.recording().ok_or_else(|| Error::InvalidParameter {
        key: "export".into(),
        reason: "the run was not recorded; call Matchmaker::record before running".into(),
    })?;
    let config = mm.config();
    let version = env!("CARGO_PKG_VERSION");

    let notes = &mm.score().notes;
    let rows: Vec<u8> = notes
        .iter()
        .flat_map(|n| {
            let mut row = Vec::with_capacity(32);
            for v in [
                n.onset_beat,
                n.duration_beat,
                n.onset_quarter,
                n.duration_quarter,
            ] {
                row.extend(v.to_le_bytes());
            }
            for v in [n.onset_div as i32, n.duration_div as i32, n.pitch, n.voice] {
                row.extend(v.to_le_bytes());
            }
            row
        })
        .collect();
    npy(
        &dir.join("score/note_array.npy"),
        NOTE_FIELDS,
        &[notes.len()],
        &rows,
    )?;
    write_json(
        &dir.join("score/note_ids.json"),
        &json!(notes.iter().map(|n| &n.id).collect::<Vec<_>>()),
    )?;
    write_json(
        &dir.join("score/meta.json"),
        &json!({
            "score_tempo": mm.score().tempo,
            "tempo": mm.tempo(),
            "unfolded": mm.score().unfolded,
            "features": {},
        }),
    )?;

    let stream = dir.join("stream").join(match config.polling_period {
        Some(pp) => format!("windowed_{pp}"),
        None => "event".into(),
    });
    let messages: Vec<&MidiMessage> = recording.frames.iter().flat_map(|f| &f.messages).collect();
    let rows: Vec<u8> = messages
        .iter()
        .flat_map(|m| {
            let mut row: Vec<u8> = message_fields(m)
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            row.extend(m.time.to_le_bytes());
            row
        })
        .collect();
    npy(
        &stream.join("messages.npy"),
        MESSAGE_FIELDS,
        &[messages.len()],
        &rows,
    )?;
    if let Some(pp) = config.polling_period {
        let ends = recording.frames.iter().scan(0, |end, f| {
            *end += f.messages.len();
            Some(*end)
        });
        let n = recording.frames.len();
        i8s(
            &stream.join("frame_offsets.npy"),
            &[n + 1],
            std::iter::once(0).chain(ends),
        )?;
        f8(
            &stream.join("frame_times.npy"),
            &[n],
            (0..n).map(|k| (k as f64 + 0.5) * pp),
        )?;
    }

    let (kind, list) = (config.processor, mm.method() == Method::Hmm);
    let range = if config.piano_range { "piano" } else { "full" };
    let mut name = match kind {
        ProcessorKind::Pitch => format!("pitch_{range}_{}", if list { "list" } else { "vector" }),
        ProcessorKind::ChordOnset => format!("chord_onset_{range}"),
    };
    // golden names imply the window: 10 ms for `pitch`, 1 ms for `chord_onset`
    let golden_window = if kind == ProcessorKind::Pitch {
        0.01
    } else {
        0.001
    };
    match config.polling_period {
        None => name.push_str("_event"),
        Some(pp) if pp != golden_window => name.push_str(&format!("_{pp}")),
        Some(_) => {}
    }
    let outputs: Vec<&Observation> = recording.processor_outputs.iter().collect();
    observations(&dir.join("processor").join(name), "", &outputs)?;

    let method_dir = dir.join(mm.method().name());
    let steps = &recording.steps;
    let fed: Vec<&Observation> = steps.iter().map(|s| &s.observation).collect();
    observations(&method_dir, "observations_", &fed)?;
    f4(
        &method_dir.join("score_positions.npy"),
        &[mm.follower_score_positions().len()],
        mm.follower_score_positions().iter().copied(),
    )?;
    f8(
        &method_dir.join("positions.npy"),
        &[steps.len()],
        steps.iter().map(|s| s.position),
    )?;
    i8s(
        &method_dir.join("current_index.npy"),
        &[steps.len()],
        steps.iter().map(|s| s.debug.current_index),
    )?;
    let states = mm.follower_score_positions().len();
    if !steps.is_empty() && steps.len() * states <= MAX_PROBABILITY_VALUES {
        f8(
            &method_dir.join("probabilities.npy"),
            &[steps.len(), states],
            steps
                .iter()
                .flat_map(|s| s.debug.probabilities.iter().copied()),
        )?;
    }
    match mm.method() {
        Method::Hmm => f8(
            &method_dir.join("beat_period.npy"),
            &[steps.len()],
            steps.iter().filter_map(|s| s.debug.beat_period),
        )?,
        Method::OuterHmm => flags(
            &method_dir.join("is_continuation.npy"),
            &steps
                .iter()
                .filter_map(|s| s.debug.is_continuation)
                .collect::<Vec<_>>(),
        )?,
        Method::Pthmm => {}
    }
    let path = mm.alignment_path();
    f8(
        &method_dir.join("alignment_path.npy"),
        &[2, path.perf_time.len()],
        path.perf_time.iter().chain(&path.score_beat).copied(),
    )?;
    write_json(
        &method_dir.join("meta.json"),
        &json!({
            "reference": {"implementation": "matchmaker-rs", "version": version},
            "generator": "matchmaker-rs export",
            "config": {
                "method": mm.method().name(),
                "processor": match kind {
                    ProcessorKind::Pitch => "pitch",
                    ProcessorKind::ChordOnset => "chord_onset",
                },
                "piano_range": config.piano_range,
                "polling_period": config.polling_period,
                "tempo": mm.tempo(),
            },
        }),
    )
}
