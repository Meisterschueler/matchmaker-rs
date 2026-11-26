//! Grace identity and OTM score-ID normalization.
mod common;

use matchmaker_rs::score::Score;

#[test]
fn grace_identity_survives_score_loading_and_unfolding() {
    let score = Score::load(common::data("otm/grace.musicxml"), true).unwrap();
    assert!(score.notes.iter().any(|note| note.is_grace));
    let notes = score.otm_notes();
    assert_eq!(notes.len(), score.notes.len());
    assert!(notes.iter().all(|note| !note.id.is_empty()));
    assert!(
        notes
            .windows(2)
            .all(|pair| pair[0].onset_beat <= pair[1].onset_beat)
    );
    let mut zero_duration = score;
    zero_duration
        .notes
        .iter_mut()
        .filter(|note| !note.is_grace)
        .for_each(|note| note.duration_beat = 0.0);
    assert!(
        zero_duration
            .otm_notes()
            .iter()
            .any(|note| note.duration_beat == 0.0 && !note.is_grace)
    );
}

#[test]
fn invalid_ids_are_replaced_as_a_group() {
    let mut score = Score::load(common::data("simple_mozart_k265_var1.mid"), false).unwrap();
    score.notes[0].id.clear();
    for (i, note) in score.otm_notes().iter().enumerate() {
        assert_eq!(note.id, format!("s{i}"));
    }
}
