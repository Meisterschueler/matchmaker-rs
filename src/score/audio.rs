//! Score-to-audio rendering and frame-to-beat reference construction.
//!
//! Upstream: `matchmaker.utils.misc.generate_score_audio`,
//! `matchmaker.registry._reference_score_audio` and the peak normalization at the end of
//! `partitura.utils.fluidsynth.synthesize_fluidsynth`.
//!
//! # Reference conventions (the two swap points)
//!
//! When the reference turns a score into alignment features it makes two implicit choices. Both
//! change the *absolute* feature values. Processors that normalize every frame (`chroma`, `lse`)
//! hide this, unnormalized ones (`raw_spectrum`, later `mel`, `cqt`, ...) and the alignment costs
//! on top of them do not. Each choice is one private function with a single call site, so a
//! different convention is a local replacement:
//!
//! | Convention | Function | Reference behaviour |
//! |---|---|---|
//! | Level | `normalize_level` | the rendered piece is peak-normalized to ±1.0 before it is padded and cropped |
//! | Framing | `extract_score_features` | one whole-signal `center=False` call: frame `j` covers samples `j·hop .. j·hop + n_fft`; chroma tuning is estimated once for the whole score |
//!
//! Both apply to every audio method; neither function sees the method or the follower. To try a
//! better convention, replace the function body (keep its contract), then re-run
//! `tests/score_audio.rs`, the audio golden tests and the audio benchmark
//! (`tools/bench/audio.toml`); the benchmark report's parity against the pinned reference decides.
//! The rationale and the alternatives that were weighed are in
//! `specs/004-audio-methods/research.md` (R11).
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
use crate::features::{
    AudioProcessor, chroma::ChromaProcessor, cqt::CqtProcessor, lse::LseProcessor,
    raw_spectrum::RawSpectrumProcessor,
};
use crate::io::audio::AudioFrame;
use crate::matchmaker::{AudioMethod, AudioMethodConfig, AudioProcessorKind};
use crate::score::Score;
use crate::score::timeline::{TimeMap, Unit};

/// Synthesized score waveform and frame-level alignment reference.
#[derive(Debug, Clone)]
pub struct ScoreAudioReference {
    /// Mono waveform, peak-normalized to ±1.0, after onset padding and reference-compatible
    /// cropping.
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
///
/// The waveform is peak-normalized like the reference's (swap point `normalize_level`, see the
/// module docs), so absolute feature values are comparable with the pinned pymatchmaker.
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
    synthesizer.process_midi_message(0, 0xB0, 91, 0);

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

    // The level reference is the peak of the uncropped piece, so render to the last note-off.
    let render_samples = events
        .last()
        .map_or(0, |event| event.sample)
        .max(target_samples.saturating_sub(padding));
    let mut rendered = Vec::with_capacity(render_samples);
    let mut cursor = 0usize;
    let mut left = vec![0.0; settings.block_size];
    let mut right = vec![0.0; settings.block_size];
    for event in events {
        render_until(
            &mut synthesizer,
            &mut rendered,
            &mut cursor,
            event.sample,
            &mut left,
            &mut right,
        );
        let (command, velocity) = if event.is_note_on {
            (0x90, event.velocity)
        } else {
            (0x80, event.velocity)
        };
        synthesizer.process_midi_message(0, command, event.pitch, velocity);
    }
    render_until(
        &mut synthesizer,
        &mut rendered,
        &mut cursor,
        render_samples,
        &mut left,
        &mut right,
    );
    normalize_level(&mut rendered);

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
    let features = extract_score_features(&samples, config, hop_length, frame_rate)?;

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

/// Swap point **level**: scale a rendered piece so that its absolute peak is exactly 1.0.
///
/// Reference: Partitura's `synthesize_fluidsynth` divides the piece by
/// `max(signal.max(), abs(signal.min()))` before `generate_score_audio` pads and crops it. The
/// divisor is therefore the peak of the *uncropped* piece, which is why [`render_score_audio`]
/// renders to the last note-off before calling this. RustySynth alone returns its raw level
/// (peak about 0.14), the reference's FluidSynth signal is divided down from integer sample
/// values; without this step the two scores differ by about 17 dB in every frequency band.
///
/// Known deviations from the reference:
/// - Partitura divides by the peak of the *last* (track, channel) signal, not of their sum. This
///   port renders one mix, so both agree only for single-track scores (all fixtures); multi-track
///   scores are unverified.
/// - A silent piece is left unchanged; the reference would divide by zero and produce NaN.
///
/// This fixes the *score* side only. A performance keeps whatever level it was recorded with, so
/// level-sensitive features (`raw_spectrum`) are as fragile against level changes as in the
/// reference. A better convention (e.g. matching the performance's level at run time) replaces
/// this function; callers only rely on "peak is 1.0 unless the piece is silent".
fn normalize_level(samples: &mut [f32]) {
    let peak = samples
        .iter()
        .fold(0.0f32, |peak, sample| peak.max(sample.abs()));
    if peak > 0.0 && peak.is_finite() {
        for sample in samples {
            *sample /= peak;
        }
    }
}

/// Swap point **framing**: the alignment features of a rendered score, one vector per frame.
///
/// Reference (`registry._reference_score_audio`): a single `processor((score_audio, 0.0))` call
/// over the whole waveform with `center=False`, followed by `processor.reset()`. Reproduced here:
/// - frame `j` covers samples `j * hop .. j * hop + n_fft` without padding and a trailing partial
///   window is dropped, i.e. `1 + (len - n_fft) / hop` frames;
/// - chroma estimates its tuning once from the whole score ([`ChromaProcessor::for_reference`]),
///   whereas performance frames estimate it per frame;
/// - stateful processors start without a predecessor (the first LSE frame is its full spectrum);
/// - `cqt` is centered: the reference transforms the whole waveform in one call and drops the
///   first and last frame, so row `j` is centered on sample `(j + 1) * hop`.
///
/// The performance stream frames differently: a cache of `n_fft - hop` zeros and a primed first
/// block put frame `i` at `i * hop - (n_fft - 2 * hop)`. Both coincide only for `n_fft == 2 * hop`
/// (chroma), so the stream framing must not be reused for the score; for `raw_spectrum` (512/128)
/// it shifts the score by two frames and adds two, which the reference does not have.
///
/// Errors with [`Error::AudioConfig`] when the waveform is shorter than one window (the
/// reference's `librosa.stft` raises there, too).
fn extract_score_features(
    samples: &[f32],
    config: &AudioMethodConfig,
    hop_length: usize,
    frame_rate: f64,
) -> Result<Vec<Vec<f32>>> {
    let (mut processor, n_fft): (Box<dyn AudioProcessor>, usize) = match config.processor() {
        AudioProcessorKind::Chroma => (
            Box::new(ChromaProcessor::for_reference(
                config.sample_rate,
                hop_length,
                config.norm,
                samples,
            )?),
            2 * hop_length,
        ),
        AudioProcessorKind::Lse => {
            let processor = LseProcessor::new(config.sample_rate, hop_length, config.norm);
            let n_fft = processor.n_fft();
            (Box::new(processor), n_fft)
        }
        AudioProcessorKind::RawSpectrum => {
            let n_fft = config.n_fft.unwrap_or(512);
            (
                Box::new(RawSpectrumProcessor::new(
                    config.sample_rate,
                    hop_length,
                    n_fft,
                )),
                n_fft,
            )
        }
        AudioProcessorKind::Cqt => {
            let rows = CqtProcessor::new(config.sample_rate, hop_length, Some(f64::INFINITY))?
                .transform(samples)?;
            if rows.is_empty() {
                return Err(audio_config(
                    "hop_length",
                    "the rendered score is shorter than two hops",
                ));
            }
            return Ok(rows);
        }
        AudioProcessorKind::CqtSpectralFlux => {
            return Err(audio_config(
                "processor",
                "cqt_spectral_flux uses the score note array, not score-audio features",
            ));
        }
    };
    if samples.len() < n_fft {
        return Err(audio_config(
            "n_fft",
            "the rendered score is shorter than one analysis window",
        ));
    }
    Ok(samples
        .windows(n_fft)
        .step_by(hop_length)
        .enumerate()
        .filter_map(|(index, window)| {
            processor
                .process(&AudioFrame {
                    index,
                    samples: window.to_vec(),
                    perf_time: index as f64 / frame_rate,
                })
                .map(|observation| observation.features)
        })
        .collect())
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

#[cfg(test)]
mod tests {
    use super::normalize_level;

    #[test]
    fn normalize_level_scales_the_absolute_peak_to_one_and_keeps_the_sign() {
        let mut samples = [0.25, -0.5, 0.125];
        normalize_level(&mut samples);
        assert_eq!(samples, [0.5, -1.0, 0.25]);
    }

    #[test]
    fn normalize_level_leaves_silence_unchanged_instead_of_producing_nan() {
        let mut samples = [0.0f32; 4];
        normalize_level(&mut samples);
        assert_eq!(samples, [0.0; 4]);
        normalize_level(&mut []);
    }
}
