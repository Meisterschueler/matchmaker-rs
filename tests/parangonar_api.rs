//! Golden/API coverage for the remaining Parangonar MIDI matchers.

mod common;

use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::dp::parangonar::{
    OptmConfig, OptmMatcher, SlOltwConfig, SlOltwMatcher, SltOltwConfig, SltOltwMatcher,
};
use matchmaker_rs::features::Observation;
use matchmaker_rs::matchmaker::{Matchmaker, Method};
use matchmaker_rs::score::{Score, ScoreNote};

fn test_notes() -> Vec<ScoreNote> {
    (0..5)
        .map(|index| ScoreNote {
            id: index.to_string(),
            is_grace: false,
            pitch: 60 + index,
            onset_beat: index as f32,
            duration_beat: 1.0,
            onset_quarter: index as f32,
            duration_quarter: 1.0,
            onset_div: i64::from(index),
            duration_div: 1,
            voice: 1,
        })
        .collect()
}

#[test]
fn oltw_checked_calls_reject_invalid_observations_without_mutating_the_path() {
    let notes = test_notes();
    let mut followers: Vec<Box<dyn OnlineAlignment>> = vec![
        Box::new(SlOltwMatcher::new(&notes, SlOltwConfig::default()).unwrap()),
        Box::new(SltOltwMatcher::new(&notes, SltOltwConfig::default()).unwrap()),
    ];
    for follower in &mut followers {
        for (features, perf_time) in [
            (vec![], 0.0),
            (vec![60.0, 61.0], 0.0),
            (vec![60.5], 0.0),
            (vec![f32::NAN], 0.0),
            (vec![128.0], 0.0),
            (vec![60.0], f64::NAN),
        ] {
            assert!(
                follower
                    .try_call(&Observation {
                        features,
                        perf_time
                    })
                    .is_err()
            );
            assert!(follower.alignment_path().perf_time.is_empty());
        }
        assert!(
            follower
                .try_call(&Observation {
                    features: vec![0.0],
                    perf_time: 0.0
                })
                .is_ok()
        );
    }
}

#[test]
fn oltw_extreme_windows_match_a_full_score_window() {
    let notes = test_notes();
    let mut pairs: Vec<(Box<dyn OnlineAlignment>, Box<dyn OnlineAlignment>)> = vec![
        (
            Box::new(
                SlOltwMatcher::new(
                    &notes,
                    SlOltwConfig {
                        window_size: usize::MAX,
                        ..SlOltwConfig::default()
                    },
                )
                .unwrap(),
            ),
            Box::new(
                SlOltwMatcher::new(
                    &notes,
                    SlOltwConfig {
                        window_size: notes.len(),
                        ..SlOltwConfig::default()
                    },
                )
                .unwrap(),
            ),
        ),
        (
            Box::new(
                SltOltwMatcher::new(
                    &notes,
                    SltOltwConfig {
                        window_size: usize::MAX,
                        ..SltOltwConfig::default()
                    },
                )
                .unwrap(),
            ),
            Box::new(
                SltOltwMatcher::new(
                    &notes,
                    SltOltwConfig {
                        window_size: notes.len(),
                        ..SltOltwConfig::default()
                    },
                )
                .unwrap(),
            ),
        ),
    ];
    for (large, full) in &mut pairs {
        for pitch in [60.0, 61.0, 62.0, 63.0] {
            let obs = Observation {
                features: vec![pitch],
                perf_time: f64::from(pitch),
            };
            assert_eq!(large.try_call(&obs).unwrap(), full.try_call(&obs).unwrap());
        }
        assert!(large.debug_state().current_index > 0);
    }
}

#[test]
fn slt_matches_reference_float32_onsets_without_rounding_path_times() {
    let mut notes = test_notes();
    notes.truncate(2);
    notes[0].pitch = 62;
    notes[1].pitch = 61;
    notes[1].onset_beat = 0.5;
    let mut follower = SltOltwMatcher::new(&notes, SltOltwConfig::default()).unwrap();

    assert!(
        follower
            .try_call(&Observation {
                features: vec![60.0],
                perf_time: f64::MAX,
            })
            .is_err()
    );
    assert!(follower.alignment_path().perf_time.is_empty());
    for (time, pitch, expected) in [(99.98, 60.0, 0.0), (100.32, 62.0, 0.5)] {
        assert_eq!(
            follower
                .try_call(&Observation {
                    features: vec![pitch],
                    perf_time: time,
                })
                .unwrap(),
            expected
        );
    }
    assert_eq!(follower.alignment_path().perf_time, [99.98, 100.32]);
}

#[test]
fn oltw_rejects_invalid_scores_and_tempo_parameters() {
    let mut notes = test_notes();
    notes[1].onset_beat = f32::NAN;
    assert!(SlOltwMatcher::new(&notes, SlOltwConfig::default()).is_err());
    assert!(SltOltwMatcher::new(&notes, SltOltwConfig::default()).is_err());
    notes[1].onset_beat = -1.0;
    assert!(SlOltwMatcher::new(&notes, SlOltwConfig::default()).is_err());
    assert!(SltOltwMatcher::new(&notes, SltOltwConfig::default()).is_err());
    for config in [
        SltOltwConfig {
            init_tempo: f64::MIN_POSITIVE,
            ..SltOltwConfig::default()
        },
        SltOltwConfig {
            init_tempo: f64::MAX,
            ..SltOltwConfig::default()
        },
        SltOltwConfig {
            tempo_factor: f64::NAN,
            ..SltOltwConfig::default()
        },
        SltOltwConfig {
            tempo_factor: f64::MAX,
            ..SltOltwConfig::default()
        },
        SltOltwConfig {
            time_weight: f64::INFINITY,
            ..SltOltwConfig::default()
        },
        SltOltwConfig {
            time_weight: f64::MAX,
            ..SltOltwConfig::default()
        },
    ] {
        assert!(SltOltwMatcher::new(&test_notes(), config).is_err());
    }
}

fn golden_path(method: &str) -> Vec<(f64, f64)> {
    let file = common::data(&format!(
        "../golden/simple_mozart_k265_var1/{method}/alignment_path.npy"
    ));
    let bytes = std::fs::read(file).unwrap();
    let header = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let values: Vec<f64> = bytes[10 + header..]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|chunk| f64::from_le_bytes(*chunk))
        .collect();
    let (time, beat) = values.split_at(values.len() / 2);
    time.iter().copied().zip(beat.iter().copied()).collect()
}

#[test]
fn remaining_parangonar_methods_match_goldens() {
    let score = common::data("simple_mozart_k265_var1.musicxml");
    let performance = common::data("simple_mozart_k265_var1.mid");
    for name in ["OPTM", "SLT_OLTW", "SL_OLTW"] {
        let method = name.parse::<Method>().unwrap();
        let mut mm = Matchmaker::new(&score, method, None, true).unwrap();
        let positions = mm.run_file(&performance).unwrap();
        assert!(!positions.is_empty());
        let expected = golden_path(name);
        assert_eq!(mm.alignment_path().perf_time.len(), expected.len());
        for ((actual_time, actual_beat), (expected_time, expected_beat)) in mm
            .alignment_path()
            .perf_time
            .iter()
            .zip(mm.alignment_path().score_beat.iter())
            .zip(expected)
        {
            assert!((*actual_time - expected_time).abs() < 1e-6, "{name}");
            assert!((*actual_beat - expected_beat).abs() < 1e-6, "{name}");
        }
    }
}

#[test]
fn optm_replays_reference_observations() {
    for (piece, score) in [
        (
            "simple_mozart_k265_var1",
            "simple_mozart_k265_var1.musicxml",
        ),
        (
            "simple_mozart_k265_var1_midiscore",
            "simple_mozart_k265_var1.mid",
        ),
        (
            "simple_mozart_k265_var1_wrongnote",
            "simple_mozart_k265_var1.musicxml",
        ),
        ("mozart_k265_var1", "mozart_k265_var1.musicxml"),
        ("Bach-fugue_bwv_858", "Bach-fugue_bwv_858.musicxml"),
    ] {
        let dir = common::golden(piece).join("OPTM");
        let notes = Score::load(common::data(score), true).unwrap().otm_notes();
        let mut matcher = OptmMatcher::new(&notes, OptmConfig::default()).unwrap();
        let observations = common::golden_observations(&dir);
        let expected = common::npy(dir.join("positions.npy")).values();
        let actual: Vec<_> = observations
            .iter()
            .map(|obs| matcher.try_call(obs).unwrap())
            .collect();
        common::assert_close(
            &common::Ctx::new("OPTM", piece, "positions"),
            &expected,
            &actual,
            0.0,
            1e-6,
        );
    }
}

#[test]
fn optm_jump_configuration_and_grace_notes_match_reference() {
    let notes = Score::load(common::data("otm/grace.musicxml"), true)
        .unwrap()
        .otm_notes();
    let pitches = [
        60, 64, 67, 72, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72, 74, 60, 67, 48, 52, 55,
        59,
    ];
    // Recorded from Parangonar 3.4.1 on CPU with backup_steps=4 and jump_trigger=2.
    for (allow_jump, expected) in [
        (
            false,
            [
                0, 0, 0, 3, 1, 1, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
            ],
        ),
        (
            true,
            [
                0, 0, 3, 3, 0, 1, 0, 3, 0, 3, 0, 3, 1, 3, 0, 3, 0, 3, 0, 3, 3, 0, 3, 0,
            ],
        ),
    ] {
        let mut matcher = OptmMatcher::new(
            &notes,
            OptmConfig {
                allow_jump,
                backup_steps: 4,
                jump_trigger: 2,
            },
        )
        .unwrap();
        for (index, pitch) in pitches.into_iter().enumerate() {
            matcher
                .try_call(&Observation {
                    features: vec![pitch as f32],
                    perf_time: index as f64 * 0.25,
                })
                .unwrap();
            assert_eq!(
                matcher.debug_state().current_index,
                expected[index],
                "allow_jump={allow_jump}, event={index}"
            );
        }
    }
}

#[test]
fn oltw_methods_replay_reference_observations() {
    for (piece, score) in [
        (
            "simple_mozart_k265_var1",
            "simple_mozart_k265_var1.musicxml",
        ),
        (
            "simple_mozart_k265_var1_midiscore",
            "simple_mozart_k265_var1.mid",
        ),
        (
            "simple_mozart_k265_var1_wrongnote",
            "simple_mozart_k265_var1.musicxml",
        ),
        ("mozart_k265_var1", "mozart_k265_var1.musicxml"),
        ("Bach-fugue_bwv_858", "Bach-fugue_bwv_858.musicxml"),
    ] {
        let notes = Score::load(common::data(score), true).unwrap().otm_notes();
        let timed = notes
            .iter()
            .find(|note| note.duration_quarter > 0.0)
            .unwrap();
        let tempo =
            f64::from((60.0_f32 / 90.0_f32) / (timed.duration_beat / timed.duration_quarter));
        for name in ["SL_OLTW", "SLT_OLTW"] {
            let mut matcher: Box<dyn OnlineAlignment> = if name == "SL_OLTW" {
                Box::new(SlOltwMatcher::new(&notes, SlOltwConfig::default()).unwrap())
            } else {
                Box::new(
                    SltOltwMatcher::new(
                        &notes,
                        SltOltwConfig {
                            init_tempo: tempo,
                            ..SltOltwConfig::default()
                        },
                    )
                    .unwrap(),
                )
            };
            let dir = common::golden(piece).join(name);
            let expected = common::npy(dir.join("positions.npy")).values();
            let actual: Vec<_> = common::golden_observations(&dir)
                .iter()
                .map(|obs| matcher.try_call(obs).unwrap())
                .collect();
            common::assert_close(
                &common::Ctx::new(name, piece, "positions"),
                &expected,
                &actual,
                0.0,
                1e-6,
            );
        }
    }
}
