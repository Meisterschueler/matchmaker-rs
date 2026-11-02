//! Score-to-audio rendering and frame-to-beat reference construction.
//!
//! Upstream: `matchmaker.utils.misc.generate_score_audio` and
//! `matchmaker.registry._reference_score_audio`.
//!
//! ```no_run
//! use matchmaker_rs::matchmaker::{AudioMethod, AudioMethodConfig};
//! use matchmaker_rs::score::{Score, audio::build_reference};
//! let score = Score::load("tests/data/simple_mozart_k265_var1.musicxml", true)?;
//! let config = AudioMethodConfig::for_method(AudioMethod::Arzt);
//! let reference = build_reference(
//!     &score,
//!     "target/audio-assets/UprightPianoKW-small-SF2-20190703/UprightPianoKW-small-20190703.sf2",
//!     &config,
//!     AudioMethod::Arzt,
//! )?;
//! assert_eq!(reference.features.len(), reference.frame_to_beat.len());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};
use tracing::debug_span;

use crate::error::{Error, Result};
use crate::features::{AudioProcessor, Observation, chroma::ChromaProcessor, lse::LseProcessor};
use crate::io::audio::AudioStream;
use crate::matchmaker::{AudioMethod, AudioMethodConfig, AudioProcessorKind};
use crate::score::Score;
use crate::score::timeline::{TimeMap, Unit};

/// Synthesized score waveform and frame-level alignment reference.
#[derive(Debug, Clone)]
pub struct ScoreAudioReference {
    /// Mono waveform after onset padding and reference-compatible cropping.
    pub samples: Vec<f32>,
    /// One score feature vector per emitted frame.
    pub features: Vec<Vec<f32>>,
    /// Score beat associated with each feature frame.
    pub frame_to_beat: Vec<f64>,
    /// Unique score onset beats.
    pub score_positions: Vec<f32>,
    /// Output sample rate.
    pub sample_rate: u32,
    /// Effective feature frame rate.
    pub frame_rate: f64,
}

#[derive(Debug, Clone, Copy)]
struct MidiEvent {
    sample: usize,
    pitch: i32,
    velocity: i32,
    is_note_on: bool,
}

/// Render a symbolic score with the explicitly selected SF2 SoundFont.
pub fn render_score_audio(
    score: &Score,
    soundfont_path: impl AsRef<Path>,
    sample_rate: u32,
    tempo: f64,
) -> Result<Vec<f32>> {
    let path = soundfont_path.as_ref();
    if !(16_000..=192_000).contains(&sample_rate) {
        return Err(audio_config(
            "sample_rate",
            "RustySynth supports rates from 16000 through 192000 Hz",
        ));
    }
    if !tempo.is_finite() || tempo <= 0.0 {
        return Err(audio_config("tempo", "must be a positive number of BPM"));
    }
    if score.notes.is_empty() {
        return Err(audio_config("score", "must contain at least one note"));
    }

    let _span = debug_span!("audio_render", path = %path.display(), sample_rate).entered();
    let mut file = File::open(path).map_err(|error| Error::io(path, error))?;
    let soundfont = Arc::new(
        SoundFont::new(&mut file).map_err(|error| Error::AudioRender {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?,
    );
    let settings = SynthesizerSettings::new(sample_rate as i32);
    let mut synthesizer =
        Synthesizer::new(&soundfont, &settings).map_err(|error| Error::AudioRender {
            path: path.to_path_buf(),
            message: error.to_string(),
        })?;

    let seconds_per_quarter = 60.0 / tempo;
    let first_onset = score
        .notes
        .iter()
        .map(|note| f64::from(note.onset_quarter) * seconds_per_quarter)
        .fold(f64::INFINITY, f64::min)
        .max(0.0);
    let padding = (first_onset * f64::from(sample_rate)) as usize;
    let last_onset_div = score
        .notes
        .iter()
        .map(|note| note.onset_div)
        .max()
        .unwrap_or(0);
    let last_onset_div = (last_onset_div as f64).floor();
    let last_onset_time =
        last_onset_div / score.part.quarter_duration_at(last_onset_div) * seconds_per_quarter + 0.1;
    let target_samples = (last_onset_time * f64::from(sample_rate)) as usize;
    let render_samples = target_samples.saturating_sub(padding);

    let mut events = Vec::with_capacity(score.notes.len() * 2);
    for note in &score.notes {
        let start = (f64::from(note.onset_quarter) * seconds_per_quarter * f64::from(sample_rate))
            .round()
            .max(0.0) as usize;
        let end = (f64::from(note.onset_quarter + note.duration_quarter)
            * seconds_per_quarter
            * f64::from(sample_rate))
        .round()
        .max(0.0) as usize;
        events.push(MidiEvent {
            sample: start.saturating_sub(padding),
            pitch: note.pitch,
            velocity: 64,
            is_note_on: true,
        });
        events.push(MidiEvent {
            sample: end.saturating_sub(padding),
            pitch: note.pitch,
            velocity: 0,
            is_note_on: false,
        });
    }
    events.sort_by(|left, right| {
        left.sample
            .cmp(&right.sample)
            .then(left.is_note_on.cmp(&right.is_note_on))
            .then(left.pitch.cmp(&right.pitch))
    });

    let mut rendered = Vec::with_capacity(render_samples);
    let mut cursor = 0usize;
    let mut left = vec![0.0; settings.block_size];
    let mut right = vec![0.0; settings.block_size];
    for event in events {
        let event_sample = event.sample.min(render_samples);
        render_until(
            &mut synthesizer,
            &mut rendered,
            &mut cursor,
            event_sample,
            &mut left,
            &mut right,
        );
        if event.sample <= render_samples {
            let (command, velocity) = if event.is_note_on {
                (0x90, event.velocity)
            } else {
                (0x80, event.velocity)
            };
            synthesizer.process_midi_message(0, command, event.pitch, velocity);
        }
    }
    render_until(
        &mut synthesizer,
        &mut rendered,
        &mut cursor,
        render_samples,
        &mut left,
        &mut right,
    );

    let mut output = vec![0.0; padding];
    output.extend(rendered);
    output.resize(target_samples, 0.0);
    output.truncate(target_samples);
    Ok(output)
}

/// Build synthesized score features and the reference-frame-to-beat mapping.
pub fn build_reference(
    score: &Score,
    soundfont_path: impl AsRef<Path>,
    config: &AudioMethodConfig,
    method: AudioMethod,
) -> Result<ScoreAudioReference> {
    config.validate(method)?;
    let tempo = score.tempo.unwrap_or(120.0);
    let samples = render_score_audio(score, soundfont_path, config.sample_rate, tempo)?;
    let hop_length = config
        .hop_length
        .unwrap_or((config.sample_rate / config.frame_rate) as usize);
    let frame_rate = f64::from(config.sample_rate) / hop_length as f64;
    let mut processor: Box<dyn AudioProcessor> = match config.processor() {
        AudioProcessorKind::Chroma => Box::new(ChromaProcessor::new(
            config.sample_rate,
            hop_length,
            config.norm,
        )),
        AudioProcessorKind::Lse => Box::new(LseProcessor::new(
            config.sample_rate,
            hop_length,
            config.norm,
        )),
    };
    let cache_size = match config.processor() {
        AudioProcessorKind::Chroma => hop_length,
        AudioProcessorKind::Lse => {
            LseProcessor::new(config.sample_rate, hop_length, config.norm).n_fft() - hop_length
        }
    };
    let mut stream = AudioStream::from_mono_samples(
        samples.clone(),
        config.sample_rate,
        hop_length,
        cache_size,
    )?;
    stream.prime(processor.as_mut());
    let observations: Vec<Observation> = stream
        .filter_map(|frame| processor.process(&frame))
        .collect();
    let features: Vec<Vec<f32>> = observations
        .iter()
        .map(|observation| observation.features.clone())
        .collect();

    let beat_map = TimeMap::new(&score.part, Unit::Beat);
    let quarter_to_div = TimeMap::new(&score.part, Unit::Quarter).map(|map| map.inverse());
    let frame_to_beat = (0..features.len())
        .map(|index| {
            let quarter = index as f64 / frame_rate * tempo / 60.0;
            let division = quarter_to_div
                .as_ref()
                .and_then(|map| map.eval(quarter))
                .unwrap_or(0.0);
            beat_map
                .as_ref()
                .and_then(|map| map.eval(division))
                .unwrap_or(0.0)
        })
        .collect();

    Ok(ScoreAudioReference {
        samples,
        features,
        frame_to_beat,
        score_positions: score.unique_onsets(),
        sample_rate: config.sample_rate,
        frame_rate,
    })
}

fn render_until(
    synthesizer: &mut Synthesizer,
    output: &mut Vec<f32>,
    cursor: &mut usize,
    end: usize,
    left: &mut [f32],
    right: &mut [f32],
) {
    while *cursor < end {
        let frames = (end - *cursor).min(left.len());
        synthesizer.render(&mut left[..frames], &mut right[..frames]);
        output.extend(
            left[..frames]
                .iter()
                .zip(&right[..frames])
                .map(|(&left, &right)| (left + right) * 0.5),
        );
        *cursor += frames;
    }
}

fn audio_config(key: &str, reason: &str) -> Error {
    Error::AudioConfig {
        key: key.into(),
        reason: reason.into(),
    }
}
