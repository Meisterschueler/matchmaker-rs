//! Golden/API coverage for the remaining Parangonar MIDI matchers.

mod common;

use matchmaker_rs::base::OnlineAlignment;
use matchmaker_rs::dp::parangonar::{
    OptmConfig, OptmMatcher, SlOltwConfig, SlOltwMatcher, SltOltwConfig, SltOltwMatcher,
};
use matchmaker_rs::features::Observation;
use matchmaker_rs::matchmaker::{Matchmaker, Method};
use matchmaker_rs::score::Score;

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
        let tempo = 60.0 / 90.0 / f64::from(timed.duration_beat / timed.duration_quarter);
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
