//! Golden tests for `skf`: fixed-template replay against the pinned reference (pymatchmaker 0.4.0).
//!
//! The reference builds its templates inside the constructor; the generator injects the stored
//! ones, so these tests isolate the recurrence (beam, merging, Kalman update, interpolation) from
//! template synthesis. The synthesized templates themselves are compared separately and need the
//! pinned SoundFont.

mod common;

use common::{Ctx, assert_close, assert_exact, data, golden, json, npy};
use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::features::Observation;
use matchmaker_rs::prob::skf::{
    SkfConfig, SkfFollower, SkfScoreModel, TemplateMode, build_chord_sequence, gaussian_templates,
    synthesize_templates,
};
use matchmaker_rs::score::{Score, ScoreNote};
use std::path::{Path, PathBuf};

const CASES: [&str; 4] = ["synthetic", "override", "one_chord", "bundled"];

fn case_dir(case: &str) -> PathBuf {
    golden("audio/skf").join(case)
}

fn ints(path: impl AsRef<Path>) -> Vec<usize> {
    npy(path).values().into_iter().map(|v| v as usize).collect()
}

fn reference_chords(dir: &Path) -> Vec<Vec<i32>> {
    let flat = npy(dir.join("chords_flat.npy")).values();
    let offsets = ints(dir.join("chord_offsets.npy"));
    offsets
        .windows(2)
        .map(|pair| flat[pair[0]..pair[1]].iter().map(|&v| v as i32).collect())
        .collect()
}

fn templates(dir: &Path) -> Vec<Vec<f64>> {
    let file = npy(dir.join("templates.npy"));
    file.values()
        .chunks(file.shape[1])
        .map(<[f64]>::to_vec)
        .collect()
}

fn golden_notes(dir: &Path) -> Vec<ScoreNote> {
    let beats = npy(dir.join("note_onset_beat.npy")).values();
    let quarters = npy(dir.join("note_onset_quarter.npy")).values();
    let pitches = npy(dir.join("note_pitch.npy")).values();
    (0..beats.len())
        .map(|i| ScoreNote {
            id: format!("n{i}"),
            pitch: pitches[i] as i32,
            onset_beat: beats[i] as f32,
            duration_beat: 1.0,
            onset_quarter: quarters[i] as f32,
            duration_quarter: 1.0,
            onset_div: 0,
            duration_div: 1,
            voice: 1,
        })
        .collect()
}

fn notes_of(case: &str, dir: &Path) -> Vec<ScoreNote> {
    if case == "bundled" {
        Score::load(data("simple_mozart_k265_var1.musicxml"), true)
            .unwrap()
            .notes
    } else {
        golden_notes(dir)
    }
}

fn config(meta: &serde_json::Value) -> SkfConfig {
    let follower = &meta["follower"];
    let mut config = SkfConfig {
        tempo: meta["tempo"].as_f64().unwrap(),
        sample_rate: follower["sample_rate"].as_u64().unwrap() as u32,
        n_fft: follower["n_fft"].as_u64().unwrap() as usize,
        hop_length: follower["hop_length"].as_u64().unwrap() as usize,
        ..SkfConfig::default()
    };
    if let Some(width) = follower["max_hypotheses"].as_u64() {
        config.max_hypotheses = width as usize;
    }
    config
}

#[test]
fn chord_sequences_match_the_reference_for_tables_and_the_bundled_score() {
    for case in CASES {
        let dir = case_dir(case);
        let sequence = build_chord_sequence(&notes_of(case, &dir)).unwrap();
        assert_eq!(sequence.chords, reference_chords(&dir), "{case}: chords");
        for (file, actual) in [
            ("lengths.npy", &sequence.lengths),
            ("onset_beats.npy", &sequence.onset_beats),
        ] {
            assert_exact(
                &Ctx::new("skf", case, file),
                &npy(dir.join(file)).values(),
                actual,
            );
        }
    }
}

#[test]
fn gaussian_templates_match_the_reference_fallback() {
    for case in ["synthetic", "override", "one_chord"] {
        let dir = case_dir(case);
        let meta = json(dir.join("meta.json"));
        assert_eq!(meta["template_mode"], "gaussian");
        let follower = config(&meta);
        let actual = gaussian_templates(
            &reference_chords(&dir),
            follower.n_fft,
            follower.sample_rate,
            5,
        );
        assert_close(
            &Ctx::new("skf", case, "templates.npy"),
            &npy(dir.join("templates.npy")).values(),
            &actual.concat(),
            meta["tolerances"]["model_rtol"].as_f64().unwrap(),
            meta["tolerances"]["model_atol"].as_f64().unwrap(),
        );
    }
}

#[test]
fn fixed_template_replay_matches_positions_and_hypothesis_state() {
    for case in CASES {
        let dir = case_dir(case);
        let meta = json(dir.join("meta.json"));
        let tolerances = &meta["tolerances"];
        let (rtol, atol) = (
            tolerances["model_rtol"].as_f64().unwrap(),
            tolerances["model_atol"].as_f64().unwrap(),
        );
        let mode: TemplateMode = meta["template_mode"].as_str().unwrap().parse().unwrap();
        let model = SkfScoreModel::new(
            reference_chords(&dir),
            npy(dir.join("lengths.npy")).values(),
            npy(dir.join("onset_beats.npy")).values(),
            templates(&dir),
            mode,
            None,
        )
        .unwrap();
        let mut follower = SkfFollower::new(model, config(&meta)).unwrap();

        let observations = npy(dir.join("observations.npy"));
        let times = npy(dir.join("perf_times.npy")).values();
        let state_frames = ints(dir.join("state_frames.npy"));
        let offsets = ints(dir.join("state_offsets.npy"));
        let (mut positions, mut current) = (Vec::new(), Vec::new());
        for (frame, (row, perf_time)) in observations
            .values()
            .chunks(observations.shape[1])
            .zip(&times)
            .enumerate()
        {
            positions.push(follower.call(&Observation {
                features: row.iter().map(|&v| v as f32).collect(),
                perf_time: *perf_time,
            }));
            let debug = follower.debug_state();
            current.push(debug.current_index as f64);
            let Some(slot) = state_frames.iter().position(|&stored| stored == frame) else {
                continue;
            };
            let state = debug.skf.expect("skf exposes its hypotheses");
            let range = offsets[slot]..offsets[slot + 1];
            let label = |name: &str| Ctx::new("skf", case, &format!("state[{frame}].{name}"));
            let stored = |file: &str| npy(dir.join(file)).values()[range.clone()].to_vec();
            let to_f64 = |values: &[usize]| values.iter().map(|&v| v as f64).collect::<Vec<_>>();
            assert_exact(
                &label("chord"),
                &stored("state_chord.npy"),
                &to_f64(&state.chords),
            );
            assert_exact(
                &label("age"),
                &stored("state_age.npy"),
                &to_f64(&state.ages),
            );
            for (name, file, actual) in [
                ("probability", "state_probability.npy", &state.probabilities),
                ("tempo_mean", "state_tempo_mean.npy", &state.tempo_means),
                (
                    "tempo_variance",
                    "state_tempo_variance.npy",
                    &state.tempo_variances,
                ),
            ] {
                assert_close(&label(name), &stored(file), actual, rtol, atol);
            }
        }
        assert_close(
            &Ctx::new("skf", case, "positions.npy"),
            &npy(dir.join("positions.npy")).values(),
            &positions,
            0.0,
            tolerances["position_atol"].as_f64().unwrap(),
        );
        assert_exact(
            &Ctx::new("skf", case, "current_index.npy"),
            &npy(dir.join("current_index.npy")).values(),
            &current,
        );
        let path = follower.alignment_path();
        assert_exact(
            &Ctx::new("skf", case, "alignment_path.perf_time"),
            &times,
            &path.perf_time,
        );
    }
}

#[test]
fn synthesized_templates_agree_with_the_reference_synthesizer() {
    let soundfont = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target/audio-assets/UprightPianoKW-small-SF2-20190703")
        .join("UprightPianoKW-small-20190703.sf2");
    let dir = case_dir("bundled");
    let meta = json(dir.join("meta.json"));
    assert_eq!(meta["template_mode"], "synthesized");
    let chords = reference_chords(&dir);
    let actual = synthesize_templates(&chords, &soundfont, 512, 8_000).unwrap();
    let expected = templates(&dir);
    assert_eq!(actual.len(), expected.len());
    // RustySynth against FluidSynth at 8 kHz (aliasing included): measured minimum cosine
    // 0.9959 and maximum total variation 0.0391; the limits keep a margin. The full-pipeline
    // limits of the audio contract decide whether the difference matters (the audio benchmark).
    for (chord, (rust, reference)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(rust.len(), reference.len());
        assert!(
            (rust.iter().sum::<f64>() - 1.0).abs() < 1e-12,
            "chord {chord} sums to 1"
        );
        let dot: f64 = rust.iter().zip(reference).map(|(a, b)| a * b).sum();
        let norm = |row: &[f64]| row.iter().map(|v| v * v).sum::<f64>().sqrt();
        let cosine = dot / (norm(rust) * norm(reference));
        let variation = 0.5
            * rust
                .iter()
                .zip(reference)
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
        println!("TEMPLATE\tchord {chord}\tcosine {cosine:.5}\ttotal variation {variation:.4}");
        assert!(cosine >= 0.99, "chord {chord}: cosine {cosine}");
        assert!(
            variation <= 0.06,
            "chord {chord}: total variation {variation}"
        );
    }
}
