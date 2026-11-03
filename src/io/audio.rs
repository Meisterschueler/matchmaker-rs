//! Prerecorded audio decoding and reference-compatible hop framing.
//!
//! Upstream: `matchmaker/io/audio.py` (`AudioStream`).
//!
//! ```
//! use matchmaker_rs::io::audio::AudioStream;
//! let stream = AudioStream::open(
//!     "tests/data/audio/codec_samples/decoder_tone.wav",
//!     44_100,
//!     4_410,
//!     4_410,
//! )?;
//! println!("{} audio frames", stream.len());
//! # Ok::<(), matchmaker_rs::Error>(())
//! ```

use std::fs::File;
use std::path::Path;

use rubato::audioadapter::Adapter;
use rubato::{Fft, FixedSync, Resampler};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use tracing::debug_span;

use crate::error::{Error, Result};
use crate::features::AudioProcessor;

/// One processor input block, including its overlap context.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioFrame {
    /// Zero-based emitted frame number.
    pub index: usize,
    /// Context samples followed by one hop of mono `f32` audio.
    pub samples: Vec<f32>,
    /// Reference-compatible performance time in seconds.
    pub perf_time: f64,
}

/// A decoded, mono, resampled prerecorded audio file split into fixed hops.
#[derive(Debug)]
pub struct AudioStream {
    source_sample_rate: u32,
    source_channels: usize,
    sample_rate: u32,
    priming_frame: Option<AudioFrame>,
    frames: std::vec::IntoIter<AudioFrame>,
}

impl AudioStream {
    pub(crate) fn from_mono_samples(
        samples: Vec<f32>,
        sample_rate: u32,
        hop_length: usize,
        cache_size: usize,
    ) -> Result<Self> {
        if sample_rate == 0 {
            return Err(audio_config("sample_rate", "must be at least 1 Hz"));
        }
        if hop_length == 0 {
            return Err(audio_config("hop_length", "must be at least 1 sample"));
        }
        let (priming_frame, frames) = frame_samples(samples, sample_rate, hop_length, cache_size);
        Ok(Self {
            source_sample_rate: sample_rate,
            source_channels: 1,
            sample_rate,
            priming_frame,
            frames: frames.into_iter(),
        })
    }

    /// Decode an audio file and prepare its processor blocks.
    ///
    /// `target_sample_rate` and `hop_length` are positive values. `cache_size` is the number of
    /// preceding samples retained for processor context. The first hop primes this cache and
    /// emits no observation; a partial final hop is zero-padded before processing.
    pub fn open(
        path: impl AsRef<Path>,
        target_sample_rate: u32,
        hop_length: usize,
        cache_size: usize,
    ) -> Result<Self> {
        let path = path.as_ref();
        if target_sample_rate == 0 {
            return Err(audio_config("sample_rate", "must be at least 1 Hz"));
        }
        if hop_length == 0 {
            return Err(audio_config("hop_length", "must be at least 1 sample"));
        }

        let _span = debug_span!("audio_decode", path = %path.display()).entered();
        let (samples, source_sample_rate, source_channels) = decode(path)?;
        let samples = if source_sample_rate == target_sample_rate || samples.is_empty() {
            samples
        } else {
            resample(path, samples, source_sample_rate, target_sample_rate)?
        };
        let (priming_frame, frames) =
            frame_samples(samples, target_sample_rate, hop_length, cache_size);

        Ok(Self {
            source_sample_rate,
            source_channels,
            sample_rate: target_sample_rate,
            priming_frame,
            frames: frames.into_iter(),
        })
    }

    /// Process the first cache-priming block without yielding it as an alignment observation.
    pub fn prime(&mut self, processor: &mut dyn AudioProcessor) {
        if let Some(frame) = self.priming_frame.take() {
            let _ = processor.process(&frame);
        }
    }

    /// Original sample rate reported by the audio decoder.
    pub fn source_sample_rate(&self) -> u32 {
        self.source_sample_rate
    }

    /// Original channel count reported by the decoded audio.
    pub fn source_channels(&self) -> usize {
        self.source_channels
    }

    /// Sample rate after mono conversion and resampling.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Number of emitted frames after first-hop cache priming.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Whether this stream emits no processor frames.
    pub fn is_empty(&self) -> bool {
        self.frames.len() == 0
    }
}

impl Iterator for AudioStream {
    type Item = AudioFrame;

    fn next(&mut self) -> Option<Self::Item> {
        self.frames.next()
    }
}

fn audio_config(key: &str, reason: &str) -> Error {
    Error::AudioConfig {
        key: key.into(),
        reason: reason.into(),
    }
}

fn audio_decode(path: &Path, message: impl Into<String>) -> Error {
    Error::AudioDecode {
        path: path.to_path_buf(),
        message: message.into(),
    }
}

fn decode(path: &Path) -> Result<(Vec<f32>, u32, usize)> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if !matches!(extension.as_str(), "wav" | "flac" | "mp3" | "ogg") {
        return Err(Error::UnsupportedFormat {
            path: path.to_path_buf(),
            extension,
        });
    }

    let file = File::open(path).map_err(|error| Error::io(path, error))?;
    let mut hint = Hint::new();
    hint.with_extension(&extension);
    let stream = MediaSourceStream::new(Box::new(file), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            stream,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|error| audio_decode(path, error.to_string()))?;
    let (track_id, codec_params) = {
        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| audio_decode(path, "no audio track"))?;
        let codec_params = track
            .codec_params
            .clone()
            .ok_or_else(|| audio_decode(path, "audio track has no codec parameters"))?;
        (track.id, codec_params)
    };
    let audio_params = codec_params
        .audio()
        .ok_or_else(|| audio_decode(path, "audio track has no audio parameters"))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|error| audio_decode(path, error.to_string()))?;

    let mut source_sample_rate = None;
    let mut source_channels = None;
    let mut samples = Vec::new();
    while let Some(packet) = format
        .next_packet()
        .map_err(|error| audio_decode(path, error.to_string()))?
    {
        if packet.track_id != track_id {
            continue;
        }
        let decoded = decoder
            .decode(&packet)
            .map_err(|error| audio_decode(path, error.to_string()))?;
        let rate = decoded.spec().rate();
        let channels = decoded.spec().channels().count();
        if rate == 0 || channels == 0 {
            return Err(audio_decode(
                path,
                "invalid decoded sample rate or channel count",
            ));
        }
        if source_sample_rate.is_some_and(|previous| previous != rate)
            || source_channels.is_some_and(|previous| previous != channels)
        {
            return Err(audio_decode(path, "audio format changed within the stream"));
        }
        source_sample_rate = Some(rate);
        source_channels = Some(channels);

        let mut interleaved = vec![0.0f32; decoded.samples_interleaved()];
        decoded.copy_to_slice_interleaved(&mut interleaved);
        samples.extend(
            interleaved
                .chunks_exact(channels)
                .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32),
        );
    }

    let rate = source_sample_rate
        .or_else(|| codec_params.audio().and_then(|params| params.sample_rate))
        .ok_or_else(|| audio_decode(path, "audio sample rate is unknown"))?;
    let channels = source_channels
        .or_else(|| {
            codec_params
                .audio()
                .and_then(|params| params.channels.as_ref().map(|channels| channels.count()))
        })
        .ok_or_else(|| audio_decode(path, "audio channel count is unknown"))?;
    Ok((samples, rate, channels))
}

fn resample(
    path: &Path,
    samples: Vec<f32>,
    source_rate: u32,
    target_rate: u32,
) -> Result<Vec<f32>> {
    let _span = debug_span!(
        "audio_resample",
        source_rate,
        target_rate,
        samples = samples.len()
    )
    .entered();
    let input_len = samples.len();
    let input =
        rubato::audioadapter_buffers::owned::InterleavedOwned::new_from(samples, 1, input_len)
            .map_err(|error| audio_decode(path, format!("invalid resampler input: {error}")))?;
    let mut resampler = Fft::<f32>::new(
        source_rate as usize,
        target_rate as usize,
        1_024,
        1,
        FixedSync::Both,
    )
    .map_err(|error| audio_decode(path, format!("cannot initialize resampler: {error}")))?;
    let output = resampler
        .process_all(&input, input.frames(), None)
        .map_err(|error| audio_decode(path, format!("resampling failed: {error}")))?;
    Ok(output.take_data())
}

fn frame_samples(
    samples: Vec<f32>,
    sample_rate: u32,
    hop_length: usize,
    cache_size: usize,
) -> (Option<AudioFrame>, Vec<AudioFrame>) {
    let _span = debug_span!(
        "audio_frame",
        sample_rate,
        hop_length,
        cache_size,
        samples = samples.len()
    )
    .entered();
    if samples.is_empty() {
        return (None, Vec::new());
    }

    let padded_len = samples.len().div_ceil(hop_length) * hop_length;
    let mut padded = samples;
    padded.resize(padded_len, 0.0);
    let mut cache = vec![0.0; cache_size];
    let mut priming_frame = None;
    let mut frames = Vec::with_capacity(padded_len / hop_length - 1);
    let mut emitted = 0usize;

    for (chunk_index, hop) in padded.chunks_exact(hop_length).enumerate() {
        let mut block = Vec::with_capacity(cache_size + hop_length);
        block.extend_from_slice(&cache);
        block.extend_from_slice(hop);
        if chunk_index == 0 {
            priming_frame = Some(AudioFrame {
                index: 0,
                samples: block.clone(),
                perf_time: 0.0,
            });
        } else {
            frames.push(AudioFrame {
                index: emitted,
                samples: block.clone(),
                perf_time: emitted as f64 * hop_length as f64 / f64::from(sample_rate),
            });
            emitted += 1;
        }
        if cache_size > 0 {
            cache.copy_from_slice(&block[block.len() - cache_size..]);
        }
    }
    (priming_frame, frames)
}
