//! Golden tests for audio `chroma` and `lse` processors at both OLTW frame rates.

mod common;

use common::{Ctx, assert_close, golden, npy};
use matchmaker_rs::features::{AudioProcessor, chroma::ChromaProcessor, lse::LseProcessor};
use matchmaker_rs::io::audio::AudioFrame;

const SAMPLE_RATE: u32 = 44_100;

fn process_samples(
    processor: &mut dyn AudioProcessor,
    samples: &[f32],
    hop_length: usize,
    cache_size: usize,
) -> (Vec<Vec<f32>>, Vec<f64>) {
    let padded_len = samples.len().div_ceil(hop_length) * hop_length;
    let mut padded = samples.to_vec();
    padded.resize(padded_len, 0.0);
    let mut cache = vec![0.0; cache_size];
    let mut features = Vec::new();
    let mut times = Vec::new();

    for (chunk_index, hop) in padded.chunks_exact(hop_length).enumerate() {
        let mut block = cache.clone();
        block.extend_from_slice(hop);
        let frame = AudioFrame {
            index: chunk_index.saturating_sub(1),
            samples: block.clone(),
            perf_time: chunk_index.saturating_sub(1) as f64 * hop_length as f64
                / f64::from(SAMPLE_RATE),
        };
        let output = processor
            .process(&frame)
            .expect("audio frame has a feature");
        if chunk_index > 0 {
            features.push(output.features);
            times.push(output.perf_time);
        }
        if cache_size > 0 {
            cache.copy_from_slice(&block[block.len() - cache_size..]);
        }
    }
    (features, times)
}

#[test]
fn all_method_processor_combinations_match_reference_goldens() {
    let base = golden("audio/processors");
    let samples: Vec<f32> = npy(base.join("input_samples.npy"))
        .values()
        .into_iter()
        .map(|value| value as f32)
        .collect();

    for (method, processor_name, frame_rate, hop_length) in [
        ("arzt", "chroma", 30, 1_470),
        ("arzt", "lse", 30, 1_470),
        ("dixon", "chroma", 50, 882),
        ("dixon", "lse", 50, 882),
    ] {
        let mut processor: Box<dyn AudioProcessor> = match processor_name {
            "chroma" => Box::new(ChromaProcessor::new(
                SAMPLE_RATE,
                hop_length,
                Some(f64::INFINITY),
            )),
            "lse" => Box::new(LseProcessor::new(SAMPLE_RATE, hop_length, Some(2.0))),
            _ => unreachable!(),
        };
        let cache_size = if processor_name == "chroma" {
            hop_length
        } else {
            2_048 - hop_length
        };
        let (actual_features, actual_times) =
            process_samples(processor.as_mut(), &samples, hop_length, cache_size);
        let reference = base.join(format!("{method}_{processor_name}"));
        let expected_features = npy(reference.join("features.npy"));
        let expected_times = npy(reference.join("times.npy"));
        let actual_flat: Vec<f64> = actual_features
            .iter()
            .flat_map(|frame| frame.iter().map(|&value| f64::from(value)))
            .collect();
        // macOS LSE differs from the pinned reference by up to 1.68e-5 due to FFT rounding.
        let feature_atol = if processor_name == "lse" && cfg!(target_os = "macos") {
            2e-5
        } else {
            1e-6
        };
        assert_close(
            &Ctx::new(method, "synthetic_audio", "features.npy"),
            &expected_features.values(),
            &actual_flat,
            2e-4,
            feature_atol,
        );
        assert_close(
            &Ctx::new(method, "synthetic_audio", "times.npy"),
            &expected_times.values(),
            &actual_times,
            0.0,
            0.0,
        );
        assert_eq!(expected_features.shape[1], actual_features[0].len());
        assert_eq!(frame_rate, SAMPLE_RATE as usize / hop_length);
    }
}

#[test]
fn lse_reset_and_silence_follow_reference_state_rules() {
    let mut processor = LseProcessor::new(SAMPLE_RATE, 882, Some(2.0));
    let silent = AudioFrame {
        index: 0,
        samples: vec![0.0; 2_048],
        perf_time: 0.0,
    };
    let first = processor.process(&silent).unwrap();
    assert!(first.features.iter().all(|&value| value == 0.0));
    processor.reset();
    let after_reset = processor.process(&silent).unwrap();
    assert_eq!(first, after_reset);
}
