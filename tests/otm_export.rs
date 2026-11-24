//! NumPy-compatible event-aligned OTM diagnostics and skipped-inference masks.
mod common;

use matchmaker_rs::matchmaker::{Matchmaker, Method};

#[test]
fn diagnostics_include_every_event_and_explicit_model_presence() {
    let mut follower = Matchmaker::new(
        common::data("simple_mozart_k265_var1.musicxml"),
        Method::Otm,
        None,
        true,
    )
    .unwrap()
    .record();
    follower
        .run_file(common::data("simple_mozart_k265_var1.mid"))
        .unwrap();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("otm-export-test");
    matchmaker_rs::export::write_run(&follower, &dir).unwrap();
    let method_dir = dir.join("OTM");
    let count = follower.alignment_path().perf_time.len();
    for (name, shape) in [
        ("observations_features", vec![count]),
        ("observations_times", vec![count]),
        ("token_windows", vec![count, 26, 7]),
        ("token_windows_present", vec![count]),
        ("model_logits", vec![count, 26, 2]),
        ("model_logits_present", vec![count]),
        ("tempo_state", vec![count, 2]),
        ("current_index", vec![count]),
        ("alignment_path", vec![2, count]),
        ("internal_path", vec![2, count]),
    ] {
        assert_eq!(
            common::npy(method_dir.join(format!("{name}.npy"))).shape,
            shape,
            "{name}"
        );
    }
    let token_mask = common::npy(method_dir.join("token_windows_present.npy")).values();
    let logit_mask = common::npy(method_dir.join("model_logits_present.npy")).values();
    assert_eq!(token_mask, logit_mask);
    assert!(token_mask.contains(&0.0));
    assert!(token_mask.contains(&1.0));
    let tokens = common::npy(method_dir.join("token_windows.npy")).values();
    let logits = common::npy(method_dir.join("model_logits.npy")).values();
    for (index, present) in token_mask.iter().enumerate() {
        if *present == 0.0 {
            assert!(
                tokens[index * 182..(index + 1) * 182]
                    .iter()
                    .all(|value| *value == 0.0)
            );
            assert!(
                logits[index * 52..(index + 1) * 52]
                    .iter()
                    .all(|value| *value == 0.0)
            );
        }
    }
    let metadata = common::json(method_dir.join("meta.json"));
    assert_eq!(metadata["config"]["method"], "OTM");
    let notes = common::npy(method_dir.join("score_notes.npy"));
    assert_eq!(notes.len(), follower.score().notes.len());
    assert_eq!(
        notes.string_field("id"),
        follower
            .score()
            .otm_notes()
            .iter()
            .map(|note| note.id.clone())
            .collect::<Vec<_>>()
    );
}
