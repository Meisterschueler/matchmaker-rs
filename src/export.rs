//! Export of a recorded run in the layout of `contracts/golden-data.md`, so that
//! `tools/golden/compare.py` can diff it against the reference stage by stage.
//! `evaluation.json` needs ground truth and is written by the CLI (`--ground-truth`).
//!
//! ```
//! use matchmaker_rs::matchmaker::{Matchmaker, Method};
//! let mut mm = Matchmaker::new("tests/data/simple_mozart_k265_var1.musicxml", Method::Pthmm, None, true)?
//!     .record();
//! mm.run_file("tests/data/simple_mozart_k265_var1.mid")?;
//! matchmaker_rs::export::write_run(&mm, std::env::temp_dir().join("matchmaker_doc_export"))?;
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::path::Path;

use serde_json::json;

use crate::error::{Error, Result};
use crate::features::{AudioProcessor, Observation, raw_spectrum::RawSpectrumProcessor};
use crate::io::audio::AudioStream;
use crate::io::stream::{MidiMessage, MidiMessageKind};
use crate::matchmaker::{AudioMatchmaker, AudioMethod, Matchmaker, Method, ProcessorKind};
use crate::utils::eval::EvaluationResult;

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
    // followers without a state distribution (arzt, dixon) have nothing to write
    let has_probabilities = steps
        .first()
        .is_some_and(|s| !s.debug.probabilities.is_empty());
    if has_probabilities && steps.len() * states <= MAX_PROBABILITY_VALUES {
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
        Method::Pthmm | Method::Arzt | Method::Dixon => {}
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
                "processor": kind.name(),
                "piano_range": config.piano_range,
                "polling_period": config.polling_period,
                "tempo": mm.tempo(),
            },
        }),
    )
}

/// Write recorded audio stages in a NumPy-readable layout.
pub fn write_audio_run(mm: &AudioMatchmaker, dir: impl AsRef<Path>) -> Result<()> {
    let dir = dir.as_ref();
    let recording = mm.recording().ok_or_else(|| Error::InvalidParameter {
        key: "export".into(),
        reason: "the audio run was not recorded; call AudioMatchmaker::record before running"
            .into(),
    })?;
    let config = mm.config();
    let method = mm.method();
    let processor = config.processor().name();
    let reference = mm.reference();
    let steps = &recording.steps;

    f4(
        &dir.join("score/audio/features.npy"),
        &[
            reference.features.len(),
            reference.features.first().map_or(0, Vec::len),
        ],
        reference
            .features
            .iter()
            .flat_map(|frame| frame.iter().copied()),
    )?;
    f8(
        &dir.join("score/audio/frame_to_beat.npy"),
        &[reference.frame_to_beat.len()],
        reference.frame_to_beat.iter().copied(),
    )?;
    f4(
        &dir.join("score/audio/score_positions.npy"),
        &[reference.score_positions.len()],
        reference.score_positions.iter().copied(),
    )?;
    write_json(
        &dir.join("score/meta.json"),
        &json!({
            "score_tempo": mm.score().tempo,
            "tempo": mm.tempo(),
            "unfolded": mm.score().unfolded,
            "audio_method": method.name(),
            "processor": processor,
        }),
    )?;

    write_json(
        &dir.join("audio/meta.json"),
        &json!({
            "performance": recording.performance_path,
            "source_sample_rate": recording.source_sample_rate,
            "source_channels": recording.source_channels,
            "sample_rate": config.sample_rate,
            "frame_rate": reference.frame_rate,
            "hop_length": mm.config().hop_length.unwrap_or(mm.config().sample_rate as usize / mm.config().frame_rate as usize),
            "frames": recording.frame_times.len(),
        }),
    )?;
    f8(
        &dir.join("audio/frame_times.npy"),
        &[recording.frame_times.len()],
        recording.frame_times.iter().copied(),
    )?;

    let processor_outputs: Vec<&Observation> = recording.processor_outputs.iter().collect();
    observations(
        &dir.join("processor").join(processor),
        "",
        &processor_outputs,
    )?;

    let method_dir = dir.join(method.name());
    let fed: Vec<&Observation> = steps.iter().map(|step| &step.observation).collect();
    observations(&method_dir, "observations_", &fed)?;
    f4(
        &method_dir.join("score_positions.npy"),
        &[mm.score_positions().len()],
        mm.score_positions().iter().copied(),
    )?;
    f8(
        &method_dir.join("positions.npy"),
        &[steps.len()],
        steps.iter().map(|step| step.position),
    )?;
    i8s(
        &method_dir.join("current_index.npy"),
        &[steps.len()],
        steps.iter().map(|step| step.debug.current_index),
    )?;
    let path = mm.alignment_path();
    f8(
        &method_dir.join("alignment_path.npy"),
        &[2, path.perf_time.len()],
        path.perf_time.iter().chain(&path.score_beat).copied(),
    )?;
    if steps.first().is_some_and(|step| step.debug.tempo.is_some()) {
        let tempo = steps
            .iter()
            .filter_map(|step| step.debug.tempo.as_ref())
            .collect::<Vec<_>>();
        f8(
            &method_dir.join("tempo_states.npy"),
            &[tempo.len(), 8],
            tempo.iter().flat_map(|state| state.state),
        )?;
        f8(
            &method_dir.join("beat_period.npy"),
            &[tempo.len()],
            tempo.iter().map(|state| state.state[3]),
        )?;
        let mut offsets = vec![0];
        for state in &tempo {
            offsets.push(offsets.last().copied().unwrap() + state.costs.len());
        }
        i8s(
            &method_dir.join("cost_window_offsets.npy"),
            &[offsets.len()],
            offsets.into_iter(),
        )?;
        i8s(
            &method_dir.join("cost_window_start.npy"),
            &[tempo.len()],
            tempo.iter().map(|state| state.window_start),
        )?;
        let count = tempo.iter().map(|state| state.costs.len()).sum::<usize>();
        f4(
            &method_dir.join("cost_window_values.npy"),
            &[count],
            tempo.iter().flat_map(|state| state.costs.iter().copied()),
        )?;
    }
    let candidates = mm.multi_fold_candidates();
    if !candidates.is_empty() {
        let candidate_dir = method_dir.join("candidates");
        let mut metadata = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let directory = candidate_dir.join(candidate.index.to_string());
            f4(
                &directory.join("score_features.npy"),
                &[
                    candidate.features.len(),
                    candidate.features.first().map_or(0, Vec::len),
                ],
                candidate.features.iter().flatten().copied(),
            )?;
            f4(
                &directory.join("score_positions.npy"),
                &[candidate.score_positions.len()],
                candidate.score_positions.iter().copied(),
            )?;
            f8(
                &directory.join("frame_to_unfolded_beat.npy"),
                &[candidate.unfolded_frame_to_beat.len()],
                candidate.unfolded_frame_to_beat.iter().copied(),
            )?;
            f8(
                &directory.join("frame_to_notated_beat.npy"),
                &[candidate.notated_frame_to_beat.len()],
                candidate.notated_frame_to_beat.iter().copied(),
            )?;
            metadata.push(json!({
                "index": candidate.index,
                "note_count": candidate.note_count,
                "max_onset_beat": candidate.max_onset_beat,
                "reference_frames": candidate.features.len(),
            }));
        }
        write_json(
            &method_dir.join("hypothesis_metadata.json"),
            &json!(metadata),
        )?;

        let states: Vec<_> = steps
            .iter()
            .filter_map(|step| step.debug.multi_fold.as_ref())
            .collect();
        i8s(
            &method_dir.join("selected_hypothesis.npy"),
            &[states.len()],
            states.iter().map(|state| state.selected_hypothesis),
        )?;
        let shape = [states.len(), candidates.len()];
        i8s(
            &method_dir.join("candidate_current_frames.npy"),
            &shape,
            states
                .iter()
                .flat_map(|state| state.current_frames.iter().copied()),
        )?;
        f8(
            &method_dir.join("candidate_local_costs.npy"),
            &shape,
            states
                .iter()
                .flat_map(|state| state.local_costs.iter().copied()),
        )?;
        f8(
            &method_dir.join("candidate_mean_costs.npy"),
            &shape,
            states
                .iter()
                .flat_map(|state| state.mean_costs.iter().copied()),
        )?;
    }
    write_json(
        &method_dir.join("meta.json"),
        &json!({
            "reference": {"implementation": "matchmaker-rs", "version": env!("CARGO_PKG_VERSION")},
            "generator": "matchmaker-rs audio export",
            "config": {
                "method": method.name(),
                "processor": processor,
                "sample_rate": config.sample_rate,
                "frame_rate": reference.frame_rate,
                "hop_length": config.hop_length.unwrap_or(config.sample_rate as usize / config.frame_rate as usize),
                "window_size": config.window_size,
                "distance_func": format!("{:?}", config.distance_func).to_lowercase(),
                "tempo": mm.tempo(),
                "random_seed": config.random_seed,
                "selection_window": config.selection_window,
                "switch_margin": config.switch_margin,
            },
        }),
    )
}

/// Decode and export standalone raw-spectrum features using the existing audio stream.
pub fn write_audio_features(
    performance: impl AsRef<Path>,
    dir: impl AsRef<Path>,
    sample_rate: u32,
    hop_length: usize,
    n_fft: usize,
) -> Result<()> {
    let performance = performance.as_ref();
    let dir = dir.as_ref();
    if sample_rate == 0 || hop_length == 0 || n_fft == 0 || n_fft < hop_length {
        return Err(Error::AudioConfig {
            key: "n_fft".into(),
            reason:
                "sample_rate and hop_length must be positive, and n_fft must be at least hop_length"
                    .into(),
        });
    }
    let mut stream = AudioStream::open(performance, sample_rate, hop_length, n_fft - hop_length)?;
    let source_sample_rate = stream.source_sample_rate();
    let source_channels = stream.source_channels();
    let mut processor = RawSpectrumProcessor::new(sample_rate, hop_length, n_fft);
    stream.prime(&mut processor);
    let mut outputs = Vec::new();
    for frame in stream {
        outputs.push(
            processor
                .process(&frame)
                .ok_or_else(|| Error::AudioConfig {
                    key: "raw_spectrum".into(),
                    reason: "failed to process a complete audio frame".into(),
                })?,
        );
    }
    let output_refs: Vec<&Observation> = outputs.iter().collect();
    observations(&dir.join("processor/raw_spectrum"), "", &output_refs)?;
    let frame_times: Vec<f64> = outputs.iter().map(|output| output.perf_time).collect();
    f8(
        &dir.join("audio/frame_times.npy"),
        &[frame_times.len()],
        frame_times.iter().copied(),
    )?;
    write_json(
        &dir.join("audio/meta.json"),
        &json!({
            "generator": "matchmaker-rs feature export",
            "performance": performance.to_string_lossy(),
            "processor": "raw_spectrum",
            "source_sample_rate": source_sample_rate,
            "source_channels": source_channels,
            "sample_rate": sample_rate,
            "hop_length": hop_length,
            "n_fft": n_fft,
            "frames": frame_times.len(),
        }),
    )
}

/// Write audio ground-truth metrics and identify the direct TSV used to compute them.
pub fn write_audio_evaluation(
    method: AudioMethod,
    dir: impl AsRef<Path>,
    ground_truth_path: &Path,
    evaluation: &EvaluationResult,
) -> Result<()> {
    let method_dir = dir.as_ref().join(method.name());
    write_json(&method_dir.join("evaluation.json"), &json!(evaluation))?;
    write_json(
        &method_dir.join("evaluation_meta.json"),
        &json!({
            "ground_truth_path": ground_truth_path.to_string_lossy(),
            "ground_truth_format": "perf_sec/score_beat TSV",
            "evaluation_file": "evaluation.json",
        }),
    )
}
