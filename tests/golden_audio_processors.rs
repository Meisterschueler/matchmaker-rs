//! Golden tests for audio `chroma`, `lse`, `raw_spectrum`, `cqt` and `cqt_spectral_flux` processors.

mod common;

use common::{Ctx, assert_close, assert_exact, data, golden, json, npy};
use matchmaker_rs::export::{write_cqt_features, write_cqt_spectral_flux_features};
use matchmaker_rs::features::{
    AudioProcessor,
    chroma::ChromaProcessor,
    cqt::{CqtProcessor, N_BINS, halve_rate},
    cqt_spectral_flux::CqtSpectralFluxProcessor,
    lse::LseProcessor,
    raw_spectrum::RawSpectrumProcessor,
};
use matchmaker_rs::io::audio::AudioFrame;
use std::path::PathBuf;

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

fn process_raw_spectrum_samples(
    processor: &mut dyn AudioProcessor,
    samples: &[f32],
    sample_rate: u32,
    hop_length: usize,
    n_fft: usize,
) -> (Vec<Vec<f32>>, Vec<f64>) {
    let padded_len = samples.len().div_ceil(hop_length) * hop_length;
    let mut padded = samples.to_vec();
    padded.resize(padded_len, 0.0);
    let cache_size = n_fft - hop_length;
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
                / f64::from(sample_rate),
        };
        let output = processor
            .process(&frame)
            .expect("raw spectrum frame has a feature");
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
fn raw_spectrum_matches_default_override_silent_and_short_reference_cases() {
    let base = golden("audio/processors/raw_spectrum");
    for (name, sample_rate, hop_length, n_fft) in [
        ("default", 8_000, 128, 512),
        ("override", 16_000, 256, 1_024),
        ("silent", 8_000, 128, 512),
    ] {
        let case = base.join(name);
        let samples: Vec<f32> = npy(case.join("input_samples.npy"))
            .values()
            .into_iter()
            .map(|value| value as f32)
            .collect();
        let mut processor = RawSpectrumProcessor::new(sample_rate, hop_length, n_fft);
        let (actual_features, actual_times) =
            process_raw_spectrum_samples(&mut processor, &samples, sample_rate, hop_length, n_fft);
        let expected_features = npy(case.join("features.npy"));
        let expected_times = npy(case.join("times.npy"));
        let actual_flat: Vec<f64> = actual_features
            .iter()
            .flat_map(|frame| frame.iter().map(|&value| f64::from(value)))
            .collect();
        assert_close(
            &Ctx::new("raw_spectrum", name, "features.npy"),
            &expected_features.values(),
            &actual_flat,
            1e-5,
            1e-5,
        );
        assert_close(
            &Ctx::new("raw_spectrum", name, "times.npy"),
            &expected_times.values(),
            &actual_times,
            0.0,
            0.0,
        );
        assert_eq!(
            expected_features.shape,
            [actual_features.len(), n_fft / 2 + 1]
        );
    }

    let mut processor = RawSpectrumProcessor::new(8_000, 128, 512);
    let short = AudioFrame {
        index: 0,
        samples: vec![0.0; 511],
        perf_time: 0.0,
    };
    assert!(processor.process(&short).is_none());
    assert_eq!(npy(base.join("short/features.npy")).shape, [0, 257]);
}

#[test]
fn raw_spectrum_reset_preserves_its_stateless_output() {
    let frame = AudioFrame {
        index: 0,
        samples: (0..512).map(|index| (index as f32 * 0.1).sin()).collect(),
        perf_time: 0.25,
    };
    let mut processor = RawSpectrumProcessor::new(8_000, 128, 512);
    let first = processor.process(&frame).unwrap();
    processor.reset();
    assert_eq!(processor.process(&frame).unwrap(), first);
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

const CQT_CASES: [&str; 7] = [
    "default", "override", "tones", "early", "norm2", "silent", "short",
];

struct CqtCase {
    directory: PathBuf,
    meta: serde_json::Value,
    samples: Vec<f32>,
    sample_rate: u32,
    hop_length: usize,
    whole_samples: usize,
}

impl CqtCase {
    fn load(name: &str) -> Self {
        Self::open("cqt", name)
    }

    fn open(processor: &str, name: &str) -> Self {
        let directory = golden(&format!("audio/processors/{processor}")).join(name);
        let meta = json(directory.join("meta.json"));
        Self {
            samples: npy(directory.join("input_samples.npy"))
                .values()
                .into_iter()
                .map(|value| value as f32)
                .collect(),
            sample_rate: meta["sample_rate"].as_u64().unwrap() as u32,
            hop_length: meta["hop_length"].as_u64().unwrap() as usize,
            whole_samples: meta["whole_samples"].as_u64().unwrap() as usize,
            directory,
            meta,
        }
    }

    fn processor(&self) -> CqtProcessor {
        let norm = match &self.meta["norm"] {
            serde_json::Value::String(text) if text == "inf" => Some(f64::INFINITY),
            value => value.as_f64(),
        };
        CqtProcessor::new(self.sample_rate, self.hop_length, norm).unwrap()
    }

    fn array(&self, file: &str) -> Vec<f64> {
        npy(self.directory.join(file)).values()
    }
}

fn widen(values: impl IntoIterator<Item = f32>) -> Vec<f64> {
    values.into_iter().map(f64::from).collect()
}

/// The cqt limit (contract "Validation and comparison limits", research R13): the per-element
/// float32 limit plus a term relative to the largest reference value of the same group (a frame
/// of features or a kernel row), because the reference's float32 and `soxr` rounding noise scales
/// with the group, not with the element.
fn assert_group_scaled_close(
    ctx: &Ctx,
    expected: &[f64],
    actual: &[f64],
    group_sizes: &[usize],
    (rtol, atol, group_rtol): (f64, f64, f64),
) {
    assert_eq!(
        expected.len(),
        actual.len(),
        "[{} / {} / {}] length differs",
        ctx.method,
        ctx.piece,
        ctx.file
    );
    assert_eq!(group_sizes.iter().sum::<usize>(), expected.len());
    let (mut max_abs, mut max_rel, mut worst_share) = (0.0f64, 0.0f64, 0.0f64);
    let mut first = None;
    let mut start = 0;
    for (group, &size) in group_sizes.iter().enumerate() {
        let expected_group = &expected[start..start + size];
        let group_max = expected_group
            .iter()
            .fold(0.0f64, |max, value| max.max(value.abs()));
        for (index, (e, a)) in expected_group.iter().zip(&actual[start..]).enumerate() {
            let difference = (a - e).abs();
            let allowed = atol + rtol * e.abs() + group_rtol * group_max;
            max_abs = max_abs.max(difference);
            if e.abs() >= 1e-200 {
                max_rel = max_rel.max(difference / e.abs());
            }
            worst_share = worst_share.max(difference / allowed);
            if difference > allowed && first.is_none() {
                first = Some((group, index, *e, *a));
            }
        }
        start += size;
    }
    let class = match (first.is_none(), max_abs == 0.0) {
        (true, true) => "EXACT",
        (true, false) => "WITHIN_TOL",
        _ => "EXCEEDS",
    };
    println!(
        "DEVIATION\t{}\t{}\t{}\t{class}\t{max_abs:e}\t{max_rel:e}\t{}",
        ctx.method,
        ctx.piece,
        ctx.file,
        expected.len()
    );
    println!(
        "CQT_LIMIT_SHARE\t{}\t{}\t{worst_share:.4}",
        ctx.piece, ctx.file
    );
    if let Some((group, index, e, a)) = first {
        panic!(
            "[{} / {} / {}] first difference in group {group} at index {index}: expected {e:?}, actual {a:?}",
            ctx.method, ctx.piece, ctx.file
        );
    }
}

fn assert_cqt_close(ctx: &Ctx, expected: &[f64], actual: &[f64], tolerances: &serde_json::Value) {
    let limit = |key: &str| tolerances[key].as_f64().unwrap();
    assert_group_scaled_close(
        ctx,
        expected,
        actual,
        &vec![N_BINS; expected.len() / N_BINS],
        (
            limit("features_rtol"),
            limit("features_atol"),
            limit("features_frame_max_rtol"),
        ),
    );
}

#[test]
fn cqt_kernels_match_the_reference_octave_by_octave() {
    for name in CQT_CASES {
        let case = CqtCase::load(name);
        let processor = case.processor();
        let kernels = processor.kernels();
        let ctx = |file: &str| Ctx::new("cqt", name, file);
        let rows: Vec<f64> = kernels
            .iter()
            .map(|kernel| kernel.rows.len() as f64)
            .collect();
        assert_exact(
            &ctx("kernel_octave_rows.npy"),
            &case.array("kernel_octave_rows.npy"),
            &rows,
        );
        let sizes: Vec<f64> = kernels.iter().map(|kernel| kernel.n_fft as f64).collect();
        assert_exact(
            &ctx("kernel_octave_n_fft.npy"),
            &case.array("kernel_octave_n_fft.npy"),
            &sizes,
        );
        let rates: Vec<f64> = kernels.iter().map(|kernel| kernel.sample_rate).collect();
        assert_exact(
            &ctx("kernel_octave_rate.npy"),
            &case.array("kernel_octave_rate.npy"),
            &rates,
        );
        let all_rows = || kernels.iter().flat_map(|kernel| &kernel.rows);
        let support: Vec<f64> = all_rows().map(|row| row.bins.len() as f64).collect();
        assert_exact(
            &ctx("kernel_row_nnz.npy"),
            &case.array("kernel_row_nnz.npy"),
            &support,
        );
        let bins: Vec<f64> = all_rows()
            .flat_map(|row| row.bins.iter().map(|&bin| bin as f64))
            .collect();
        assert_exact(
            &ctx("kernel_indices.npy"),
            &case.array("kernel_indices.npy"),
            &bins,
        );
        let values = widen(all_rows().flat_map(|row| row.values.iter().flatten().copied()));
        let tolerances = &case.meta["tolerances"];
        let limit = |key: &str| tolerances[key].as_f64().unwrap();
        let row_sizes: Vec<usize> = all_rows().map(|row| 2 * row.bins.len()).collect();
        assert_group_scaled_close(
            &ctx("kernel_values.npy"),
            &case.array("kernel_values.npy"),
            &values,
            &row_sizes,
            (
                limit("kernel_rtol"),
                limit("kernel_atol"),
                limit("kernel_row_max_rtol"),
            ),
        );
    }
}

#[test]
fn cqt_rate_halving_matches_soxr_hq_stage_by_stage() {
    for name in CQT_CASES {
        let case = CqtCase::load(name);
        let tolerances = &case.meta["tolerances"];
        let mut input = case.samples[..case.whole_samples].to_vec();
        for (index, resample) in case.meta["resamples"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let factor = resample["factor"].as_u64().unwrap();
            assert_eq!(factor, 2, "{name}: only halvings are generated");
            assert_eq!(
                input.len() as u64,
                resample["input_length"].as_u64().unwrap()
            );
            let file = format!("decimated_{index}.npy");
            let expected = case.array(&file);
            assert_close(
                &Ctx::new("cqt", name, &file),
                &expected,
                &widen(halve_rate(&input)),
                tolerances["decimated_rtol"].as_f64().unwrap(),
                tolerances["decimated_atol"].as_f64().unwrap(),
            );
            // The reference's own output feeds the next stage, so errors do not accumulate.
            input = expected.into_iter().map(|value| value as f32).collect();
        }
    }
}

#[test]
fn cqt_matches_reference_stream_and_whole_signal_features() {
    for name in CQT_CASES {
        let case = CqtCase::load(name);
        let tolerances = &case.meta["tolerances"];
        let mut processor = case.processor();
        let (rows, times) = process_raw_spectrum_samples(
            &mut processor,
            &case.samples,
            case.sample_rate,
            case.hop_length,
            2 * case.hop_length,
        );
        let expected = npy(case.directory.join("stream_features.npy"));
        assert_eq!(expected.shape, [rows.len(), N_BINS], "{name} stream shape");
        assert_cqt_close(
            &Ctx::new("cqt", name, "stream_features.npy"),
            &expected.values(),
            &widen(rows.into_iter().flatten()),
            tolerances,
        );
        assert_close(
            &Ctx::new("cqt", name, "stream_times.npy"),
            &case.array("stream_times.npy"),
            &times,
            0.0,
            0.0,
        );

        let whole = processor
            .transform(&case.samples[..case.whole_samples])
            .unwrap();
        let expected = npy(case.directory.join("whole_features.npy"));
        assert_eq!(
            expected.shape,
            [whole.len(), N_BINS],
            "{name} whole-signal shape"
        );
        assert_cqt_close(
            &Ctx::new("cqt", name, "whole_features.npy"),
            &expected.values(),
            &widen(whole.into_iter().flatten()),
            tolerances,
        );
    }
}

#[test]
fn cqt_export_of_the_bundled_performance_matches_the_reference() {
    let directory = golden("audio/processors/cqt/performance");
    let meta = json(directory.join("meta.json"));
    let output = std::env::temp_dir().join("matchmaker_golden_cqt_performance");
    let _ = std::fs::remove_dir_all(&output);
    write_cqt_features(
        data("audio/simple_mozart_k265_var1_sf2.wav"),
        &output,
        meta["sample_rate"].as_u64().unwrap() as u32,
        meta["hop_length"].as_u64().unwrap() as usize,
    )
    .unwrap();
    let expected = npy(directory.join("stream_features.npy"));
    let actual = npy(output.join("processor/cqt/features.npy"));
    assert_eq!(actual.shape, expected.shape);
    assert_eq!(
        expected.shape[0] as u64,
        meta["stream_frames"].as_u64().unwrap()
    );
    assert_cqt_close(
        &Ctx::new("cqt", "performance", "stream_features.npy"),
        &expected.values(),
        &actual.values(),
        &meta["tolerances"],
    );
    assert_close(
        &Ctx::new("cqt", "performance", "stream_times.npy"),
        &npy(directory.join("stream_times.npy")).values(),
        &npy(output.join("processor/cqt/times.npy")).values(),
        0.0,
        0.0,
    );
}

#[test]
fn cqt_reset_preserves_its_stateless_output() {
    let frame = AudioFrame {
        index: 0,
        samples: (0..2_940).map(|index| (index as f32 * 0.1).sin()).collect(),
        perf_time: 0.25,
    };
    let mut processor = CqtProcessor::new(44_100, 1_470, Some(f64::INFINITY)).unwrap();
    let first = processor.process(&frame).unwrap();
    processor.reset();
    assert_eq!(processor.process(&frame).unwrap(), first);
}

const FLUX_CASES: [&str; 5] = ["default", "override", "melody", "silent", "short"];

fn flux_processor(case: &CqtCase, include_flux: bool) -> CqtSpectralFluxProcessor {
    CqtSpectralFluxProcessor::new(
        case.sample_rate,
        case.hop_length,
        Some(f64::INFINITY),
        include_flux,
    )
    .unwrap()
}

/// The cqt limit for rows of 88 bins plus the flux. The frame maximum is taken over the bins
/// alone (the flux can exceed them), so the flux is judged as strictly as the bins.
fn assert_flux_close(ctx: &Ctx, expected: &[f64], actual: &[f64], tolerances: &serde_json::Value) {
    const WIDTH: usize = N_BINS + 1;
    assert_eq!(
        expected.len(),
        actual.len(),
        "[{} / {} / {}] length differs",
        ctx.method,
        ctx.piece,
        ctx.file
    );
    let limit = |key: &str| tolerances[key].as_f64().unwrap();
    let (mut worst_bins, mut worst_flux) = (0.0f64, 0.0f64);
    for (row, (expected_row, actual_row)) in
        expected.chunks(WIDTH).zip(actual.chunks(WIDTH)).enumerate()
    {
        let scale = expected_row[..N_BINS]
            .iter()
            .fold(0.0f64, |max, value| max.max(value.abs()));
        for (column, (e, a)) in expected_row.iter().zip(actual_row).enumerate() {
            let allowed = limit("features_atol")
                + limit("features_rtol") * e.abs()
                + limit("features_frame_max_rtol") * scale;
            let share = (a - e).abs() / allowed;
            if column < N_BINS {
                worst_bins = worst_bins.max(share);
            } else {
                worst_flux = worst_flux.max(share);
            }
            assert!(
                share <= 1.0,
                "[{} / {} / {}] row {row}, column {column}: expected {e:?}, actual {a:?}, allowed {allowed:e}",
                ctx.method,
                ctx.piece,
                ctx.file
            );
        }
    }
    println!(
        "CQT_LIMIT_SHARE\t{}\t{}\tbins {worst_bins:.4}\tflux {worst_flux:.4}",
        ctx.piece, ctx.file
    );
}

#[test]
fn cqt_spectral_flux_matches_reference_stream_first_call_and_whole_signal_rows() {
    for name in FLUX_CASES {
        let case = CqtCase::open("cqt_spectral_flux", name);
        let tolerances = &case.meta["tolerances"];
        let ctx = |file: &str| Ctx::new("cqt_spectral_flux", name, file);
        let hop = case.hop_length;

        let mut processor = flux_processor(&case, true);
        let (rows, times) = process_raw_spectrum_samples(
            &mut processor,
            &case.samples,
            case.sample_rate,
            hop,
            2 * hop,
        );
        let expected = npy(case.directory.join("stream_features.npy"));
        assert_eq!(
            expected.shape,
            [rows.len(), N_BINS + 1],
            "{name} stream shape"
        );
        assert_flux_close(
            &ctx("stream_features.npy"),
            &expected.values(),
            &widen(rows.into_iter().flatten()),
            tolerances,
        );
        assert_close(
            &ctx("stream_times.npy"),
            &case.array("stream_times.npy"),
            &times,
            0.0,
            0.0,
        );

        // The first call of a stream has no predecessor: its row is dropped and its flux is 0.
        let mut block = vec![0.0f32; hop];
        block.extend(
            case.samples
                .iter()
                .copied()
                .chain(std::iter::repeat(0.0))
                .take(hop),
        );
        let first = flux_processor(&case, true)
            .process(&AudioFrame {
                index: 0,
                samples: block,
                perf_time: 0.0,
            })
            .unwrap();
        assert_eq!(first.features[N_BINS], 0.0);
        assert_flux_close(
            &ctx("first_features.npy"),
            &case.array("first_features.npy"),
            &widen(first.features),
            tolerances,
        );

        let whole = flux_processor(&case, true)
            .transform(&case.samples[..case.whole_samples])
            .unwrap();
        let expected = npy(case.directory.join("whole_features.npy"));
        assert_eq!(
            expected.shape,
            [whole.len(), N_BINS + 1],
            "{name} whole-signal shape"
        );
        assert_flux_close(
            &ctx("whole_features.npy"),
            &expected.values(),
            &widen(whole.into_iter().flatten()),
            tolerances,
        );
    }
}

#[test]
fn cqt_spectral_flux_without_flux_matches_the_reference_rows() {
    for name in ["default", "override", "melody"] {
        let case = CqtCase::open("cqt_spectral_flux", name);
        let mut processor = flux_processor(&case, false);
        let (rows, _) = process_raw_spectrum_samples(
            &mut processor,
            &case.samples,
            case.sample_rate,
            case.hop_length,
            2 * case.hop_length,
        );
        let expected = npy(case.directory.join("stream_features_no_flux.npy"));
        assert_eq!(expected.shape, [rows.len(), N_BINS], "{name} stream shape");
        assert_cqt_close(
            &Ctx::new("cqt_spectral_flux", name, "stream_features_no_flux.npy"),
            &expected.values(),
            &widen(rows.into_iter().flatten()),
            &case.meta["tolerances"],
        );
    }
}

#[test]
fn cqt_spectral_flux_history_survives_reset_and_the_score_call_breaks_stream_calls() {
    let case = CqtCase::open("cqt_spectral_flux", "default");
    let lifecycle = &case.meta["lifecycle"];
    assert_eq!(lifecycle["reset_keeps_history"], true);
    let hop = case.hop_length;

    // reset() after the second call changes nothing: the reference's rows come out unchanged.
    let mut processor = flux_processor(&case, true);
    let mut padded = case.samples.clone();
    padded.resize(padded.len().div_ceil(hop) * hop, 0.0);
    let mut cache = vec![0.0f32; hop];
    let mut rows = Vec::new();
    for (index, chunk) in padded.chunks_exact(hop).enumerate() {
        let mut block = cache.clone();
        block.extend_from_slice(chunk);
        let observation = processor
            .process(&AudioFrame {
                index,
                samples: block.clone(),
                perf_time: 0.0,
            })
            .unwrap();
        if index > 0 {
            rows.push(observation.features);
        }
        if index == 1 {
            processor.reset();
        }
        cache.copy_from_slice(&block[block.len() - hop..]);
    }
    assert_flux_close(
        &Ctx::new("cqt_spectral_flux", "default", "stream_features.npy"),
        &case.array("stream_features.npy"),
        &widen(rows.into_iter().flatten()),
        &case.meta["tolerances"],
    );

    // The score is one whole-signal call; its matrix stays as history and no stream call fits it.
    let mut processor = flux_processor(&case, true);
    let score = processor
        .transform(&case.samples[..case.whole_samples])
        .unwrap();
    let history = lifecycle["history_frames"].as_u64().unwrap() as usize;
    assert_eq!(score.len() + 2, history);
    processor.reset();
    let call = [vec![0.0; hop], case.samples[..hop].to_vec()].concat();
    let message = processor.transform(&call).unwrap_err().to_string();
    assert!(
        message.contains("3 frames") && message.contains(&format!("{history} frames")),
        "{message}"
    );
    let reference = lifecycle["after_score_call"]["message"].as_str().unwrap();
    assert!(
        reference.contains(&format!("(3,88) ({history},88)")),
        "{reference}"
    );
}

#[test]
fn cqt_spectral_flux_export_of_the_bundled_performance_matches_the_reference() {
    let directory = golden("audio/processors/cqt_spectral_flux/performance");
    let meta = json(directory.join("meta.json"));
    let sample_rate = meta["sample_rate"].as_u64().unwrap() as u32;
    let hop_length = meta["hop_length"].as_u64().unwrap() as usize;
    let performance = data("audio/simple_mozart_k265_var1_sf2.wav");
    let ctx = |file: &str| Ctx::new("cqt_spectral_flux", "performance", file);
    let output = std::env::temp_dir().join("matchmaker_golden_cqt_spectral_flux_performance");
    let _ = std::fs::remove_dir_all(&output);
    write_cqt_spectral_flux_features(&performance, &output, sample_rate, hop_length, true).unwrap();
    let expected = npy(directory.join("stream_features.npy"));
    let actual = npy(output.join("processor/cqt_spectral_flux/features.npy"));
    assert_eq!(actual.shape, expected.shape);
    assert_eq!(
        expected.shape,
        [meta["stream_frames"].as_u64().unwrap() as usize, N_BINS + 1]
    );
    assert_flux_close(
        &ctx("stream_features.npy"),
        &expected.values(),
        &actual.values(),
        &meta["tolerances"],
    );
    assert_close(
        &ctx("stream_times.npy"),
        &npy(directory.join("stream_times.npy")).values(),
        &npy(output.join("processor/cqt_spectral_flux/times.npy")).values(),
        0.0,
        0.0,
    );

    // Without the flux the export holds the cqt rows: compare them with the cqt goldens.
    let cqt = golden("audio/processors/cqt/performance");
    let without = std::env::temp_dir().join("matchmaker_golden_cqt_spectral_flux_off");
    let _ = std::fs::remove_dir_all(&without);
    write_cqt_spectral_flux_features(&performance, &without, sample_rate, hop_length, false)
        .unwrap();
    assert_cqt_close(
        &ctx("stream_features_no_flux"),
        &npy(cqt.join("stream_features.npy")).values(),
        &npy(without.join("processor/cqt_spectral_flux/features.npy")).values(),
        &json(cqt.join("meta.json"))["tolerances"],
    );
}
