//! Audio decoding, downmixing, resampling, and reference-compatible hop framing.

use std::path::PathBuf;

use common::data;
use matchmaker_rs::Error;
use matchmaker_rs::io::audio::AudioStream;

mod common;

struct TestFile(PathBuf);

impl Drop for TestFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn temp_file(name: &str, bytes: &[u8]) -> TestFile {
    let path = std::env::temp_dir().join(format!(
        "matchmaker_audio_{name}_{}.wav",
        std::process::id()
    ));
    std::fs::write(&path, bytes).unwrap();
    TestFile(path)
}

fn pcm_wav(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
    let data: Vec<u8> = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect();
    let block_align = channels * 2;
    let byte_rate = sample_rate * u32::from(block_align);
    let mut wav = Vec::with_capacity(44 + data.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data.len() as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
    wav.extend_from_slice(&data);
    wav
}

#[test]
fn mono_wav_primes_cache_pads_final_hop_and_keeps_reference_times() {
    let file = temp_file("framing", &pcm_wav(8, 1, &[1000, 2000, 3000, 4000, 5000]));
    let mut stream = AudioStream::open(&file.0, 8, 2, 2).unwrap();
    assert_eq!(stream.source_sample_rate(), 8);
    assert_eq!(stream.source_channels(), 1);
    assert_eq!(stream.sample_rate(), 8);

    let frames: Vec<_> = stream.by_ref().collect();
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].perf_time, 0.0);
    assert_eq!(frames[1].perf_time, 0.25);
    assert_eq!(frames[0].samples.len(), 4);
    let scale = 1.0 / 32768.0;
    for (actual, expected) in frames[0]
        .samples
        .iter()
        .zip([1000.0, 2000.0, 3000.0, 4000.0].map(|sample| sample * scale))
    {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_eq!(frames[1].samples.last(), Some(&0.0));
}

#[test]
fn stereo_audio_is_averaged_to_mono() {
    let file = temp_file(
        "downmix",
        &pcm_wav(8_000, 2, &[16_000, -16_000, 0, 20_000, 0, 0]),
    );
    let frames: Vec<_> = AudioStream::open(&file.0, 8_000, 1, 1).unwrap().collect();
    assert_eq!(frames.len(), 2);
    assert!(frames[0].samples[0].abs() < 1e-7);
    assert!((frames[0].samples[1] - 10_000.0 / 32768.0).abs() < 1e-6);
    assert!(frames[1].samples[1].abs() < 1e-7);
}

#[test]
fn resampling_preserves_duration_and_uses_target_rate_for_timestamps() {
    let samples = vec![8_000i16; 1_600];
    let file = temp_file("resample", &pcm_wav(8_000, 1, &samples));
    let frames: Vec<_> = AudioStream::open(&file.0, 16_000, 320, 320)
        .unwrap()
        .collect();
    assert_eq!(frames.len(), 9);
    assert_eq!(frames[0].perf_time, 0.0);
    assert!((frames[1].perf_time - 0.02).abs() < 1e-12);
}

#[test]
fn empty_and_corrupt_audio_have_defined_results() {
    let empty = temp_file("empty", &pcm_wav(44_100, 1, &[]));
    assert!(
        AudioStream::open(&empty.0, 44_100, 1_470, 1_470)
            .unwrap()
            .next()
            .is_none()
    );

    let corrupt = temp_file("corrupt", b"not a wave file");
    let error = AudioStream::open(&corrupt.0, 44_100, 1_470, 1_470).unwrap_err();
    assert!(matches!(error, Error::AudioDecode { .. }), "{error}");
    assert!(error.to_string().contains("matchmaker_audio_corrupt"));
}

#[test]
fn unsupported_audio_extension_is_typed() {
    let wav = pcm_wav(8_000, 1, &[0, 0]);
    let file = temp_file("unsupported", &wav);
    let path = file.0.with_extension("aac");
    std::fs::rename(&file.0, &path).unwrap();
    let renamed = TestFile(path);
    let error = AudioStream::open(&renamed.0, 8_000, 1, 1).unwrap_err();
    assert!(matches!(error, Error::UnsupportedFormat { .. }), "{error}");
}

#[test]
fn decodes_wav_flac_mp3_and_ogg_fixtures() {
    for extension in ["wav", "flac", "mp3", "ogg"] {
        let path = data(&format!("audio/codec_samples/decoder_tone.{extension}"));
        let mut stream = AudioStream::open(&path, 44_100, 4_410, 4_410)
            .unwrap_or_else(|error| panic!("{extension}: {error}"));
        assert_eq!(stream.source_sample_rate(), 44_100, "{extension}");
        assert_eq!(stream.source_channels(), 1, "{extension}");
        let frames: Vec<_> = stream.by_ref().collect();
        assert!(frames.len() >= 3, "{extension}: {} frames", frames.len());
        assert_eq!(frames[0].perf_time, 0.0, "{extension}");
        assert_eq!(frames[1].perf_time, 0.1, "{extension}");
        assert!(frames.iter().all(|frame| frame.samples.len() == 8_820));
    }
}
