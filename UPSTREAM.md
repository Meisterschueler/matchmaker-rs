# Following upstream

matchmaker-rs is a port. This file says what it ports, from which version, and how to pull in upstream changes.

## Pinned upstream

| Package | Version | Source in this repo |
|---|---|---|
| pymatchmaker | 0.4.0 | `tools/golden/uv.lock` |
| partitura | 1.9.0 | `tools/golden/uv.lock` |
| python-hiddenmarkov | 0.1.4 | `tools/golden/uv.lock` |

The reference sources are installed by `uv sync --locked` into `tools/golden/.venv/Lib/site-packages/` (`matchmaker/`, `partitura/`, `hiddenmarkov/`).

## Convention

- Every module under `src/` starts with `Port of <python symbol>` and an `Upstream: <file>` line. The path is relative to `site-packages/`.
- Functions and structs that port a Python function name it in their doc comment ("Port of `_handle_measure`"). Deviations are explained next to the code (`research R*`, `shortcut:`).
- Numeric code that mirrors a mixed `float32`/`float64` expression names the Python expression (see `src/utils/tempo_models.rs`).

## Module map

| Rust | Upstream file | Upstream symbol |
|---|---|---|
| `src/base.rs` | `matchmaker/base.py` | `OnlineAlignment` |
| `src/matchmaker.rs` | `matchmaker/matchmaker.py`, `matchmaker/methods.yaml` | `Matchmaker`, per-method defaults |
| `src/features/pitch.rs` | `matchmaker/features/midi.py` | `PitchProcessor` |
| `src/features/chord_onset.rs` | `matchmaker/features/midi.py` | `ChordOnsetProcessor` |
| `src/io/stream.rs` | `matchmaker/io/midi.py`, `matchmaker/utils/symbolic.py` | `MidiStream`, `Buffer`, `midi_messages_to_framed_midi` |
| `src/performance.rs` | `partitura/io/importmidi.py`, `matchmaker/utils/symbolic.py` | `load_performance_midi`, `midi_messages_from_performance` |
| `src/prob/pthmm.rs` | `matchmaker/prob/hmm.py` | `PitchHMM` |
| `src/prob/hmm.rs` | `matchmaker/prob/hmm.py` | `PitchIOIHMM` |
| `src/prob/hmm_models.rs` | `matchmaker/prob/hmm.py` | `stable_transition_matrix`, `compute_discrete_pitch_profiles`, `compute_bernoulli_pitch_probabilities` |
| `src/prob/outerhmm.rs` | `matchmaker/prob/outer_product_hmm.py` | `OuterProductHMM` |
| `src/prob/hidden_markov.rs` | `hiddenmarkov/__init__.py` | `forward_algorithm_step`, `ConstantTransitionModel` |
| `src/utils/tempo_models.rs` | `matchmaker/utils/tempo_models.py` | `KalmanTempoModel` |
| `src/utils/eval.rs` | `matchmaker/utils/eval.py` | `evaluate_alignment`, `evaluate_beat_errors`, `compute_sparc` |
| `src/score/mod.rs` | `matchmaker/matchmaker.py` | `Matchmaker.__init__` score loading |
| `src/score/musicxml.rs` | `partitura/io/importmusicxml.py` | `load_musicxml` |
| `src/score/midi.rs` | `partitura/io/importmidi.py` | `load_score_midi` |
| `src/score/part.rs` | `partitura/score.py`, `partitura/utils/music.py` | `Part`, `Part.note_array` |
| `src/score/timeline.rs` | `partitura/score.py` | `Part._time_interpolator` |
| `src/score/unfold.rs` | `partitura/score.py` | `unfold_part_maximal`, `add_segments`, `Path`, `get_paths`, `new_part_from_path` |
| `src/score/merge.rs` | `partitura/score.py` | `merge_parts` |
| `src/score/tempo.rs` | `matchmaker/utils/misc.py`, `partitura/utils/music.py` | `get_tempo_from_score`, `to_quarter_tempo` |
| `crates/matchmaker-py/src/lib.rs` | `matchmaker/matchmaker.py` | public `Matchmaker` API (MIDI methods only) |
| `crates/matchmaker-cli/src/main.rs` | `matchmaker/matchmaker.py` | CLI surface |

Not ported: audio features and followers (`dp/`, `features/audio.py`, `io/audio.py`, `prob/outer_product_hmm_audio.py`, `skf.py`, `particle_filter_korzeniowski.py`), live MIDI devices and the thread/queue plumbing.

## Procedure: a new upstream release

1. Bump the pins in `tools/golden/pyproject.toml`, then `cd tools/golden; uv lock; uv sync --locked` (setup and regeneration: `specs/001-midi-score-following/quickstart.md`).
2. Diff the old and new upstream files of the rows above (`pip download pymatchmaker==<old>` and `<new>`, or `git diff v<old> v<new>` in the upstream repo). Only the files in the table matter.
3. For each changed Python function, change the Rust function named in its doc comment. Keep the Python structure so both stay readable side by side.
4. Regenerate the golden data with `tools/golden/generate.py` (layout: `specs/001-midi-score-following/contracts/golden-data.md`) and review the diff of `tests/golden/`: every changed number must trace back to a changed upstream line.
5. `cargo test --workspace --exclude matchmaker-py` (golden tests compare stage by stage), then `cargo clippy --all-targets --all-features -- -D warnings` and `cargo fmt`.
6. Update the version table above and the defaults in `src/matchmaker.rs` if `methods.yaml` changed.
7. If the Python API changed, mirror it in `crates/matchmaker-py/src/lib.rs` and `crates/matchmaker-py/tests/`.

New upstream code that is not in the table: add a Rust module with the `Port of` / `Upstream:` header and a row here.
