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
- Functions and structs that port a Python function name it in their doc comment ("Port of `_handle_measure`"). Deviations are explained next to the code (`research R*`, `shortcut:`); `research R*` means `specs/001-midi-score-following/research.md`, the research of later features is written `002 research R*` (`specs/002-midi-oltw/research.md`), and so are their `FR-*` and `SC-*`.
- Numeric code that mirrors a mixed `float32`/`float64` expression names the Python expression (see `src/utils/tempo_models.rs`).

## Module map

| Rust | Upstream file | Upstream symbol |
|---|---|---|
| `src/base.rs` | `matchmaker/base.py` | `OnlineAlignment` |
| `src/matchmaker.rs` | `matchmaker/matchmaker.py`, `matchmaker/methods.yaml` | `Matchmaker`, per-method defaults |
| `src/io/audio.rs` | `matchmaker/io/audio.py` | mono decode, resample, frame-hop stream |
| `src/features/chroma.rs` | `matchmaker/features/audio.py` | `ChromagramProcessor` |
| `src/features/lse.rs` | `matchmaker/features/audio.py` | `LogSpectralEnergyProcessor` |
| `src/features/raw_spectrum.rs` | `matchmaker/features/audio.py` | `RawSpectrumProcessor` |
| `src/features/cqt.rs` | `matchmaker/features/audio.py`, `librosa/core/constantq.py`, `librosa/core/audio.py` | `CQTProcessor`; librosa's recursive `cqt` (octave kernels, early downsampling, response scaling) with the `soxr_hq` 2:1 decimator as a fitted Kaiser sinc (research R13) |
| `src/features/cqt_spectral_flux.rs` | `matchmaker/features/audio.py` | `CQTSpectralFluxProcessor` over `src/features/cqt.rs`: previous call's matrix as flux history, inherited no-op `reset` (research R5, C07) |
| `src/score/audio.rs` | `matchmaker/utils/misc.py`, `matchmaker/registry.py`, `partitura/utils/fluidsynth.py` | peak-normalized synthesized score, whole-signal score features, frame-to-beat map (swap points `normalize_level`, `extract_score_features`; research R11) |
| `src/features/pitch.rs` | `matchmaker/features/midi.py` | `PitchProcessor` |
| `src/features/chord_onset.rs` | `matchmaker/features/midi.py` | `ChordOnsetProcessor` |
| `src/io/stream.rs` | `matchmaker/io/midi.py`, `matchmaker/utils/symbolic.py` | `MidiStream`, `Buffer`, `midi_messages_to_framed_midi` |
| `src/performance.rs` | `partitura/io/importmidi.py`, `matchmaker/utils/symbolic.py` | `load_performance_midi`, `midi_messages_from_performance` |
| `src/prob/pthmm.rs` | `matchmaker/prob/hmm.py` | `PitchHMM` |
| `src/prob/hmm.rs` | `matchmaker/prob/hmm.py` | `PitchIOIHMM` |
| `src/prob/hmm_models.rs` | `matchmaker/prob/hmm.py` | `stable_transition_matrix`, `compute_discrete_pitch_profiles`, `compute_bernoulli_pitch_probabilities` |
| `src/prob/outerhmm.rs` | `matchmaker/prob/outer_product_hmm.py` | `OuterProductHMM` |
| `src/prob/skf.rs` | `matchmaker/prob/skf.py` | `SwitchingKalmanFilterFollower`, `build_chord_sequence`, `build_spectral_templates` (synthesized with RustySynth, Gaussian fallback as an explicit mode) |
| `src/prob/hidden_markov.rs` | `hiddenmarkov/__init__.py` | `forward_algorithm_step`, `ConstantTransitionModel` |
| `src/dp/mod.rs` | `matchmaker/dp/direction.py`, SciPy `cdist` | `Direction`; metrics `cosine`, `euclidean` |
| `src/dp/arzt.rs` | `matchmaker/dp/oltw_arzt.py` | `OnlineTimeWarpingArztEvent` |
| `src/dp/arzt_frame.rs` | `matchmaker/dp/oltw_arzt.py`, `matchmaker/dp/dtw_loop.pyx` | `OnlineTimeWarpingArztFrame` |
| `src/dp/arzt_tempo.rs` | `matchmaker/dp/oltw_arzt.py`, `matchmaker/utils/tempo_models.py` | tempo-aware `OnlineTimeWarpingArztFrame` |
| `src/dp/arzt_multi_fold.rs` | `matchmaker/dp/oltw_arzt_multi_fold.py` | parallel repeat-hypothesis Arzt follower |
| `src/dp/dixon.rs` | `matchmaker/dp/oltw_dixon.py` | `OnlineTimeWarpingDixonEvent` |
| `src/dp/dixon_frame.rs` | `matchmaker/dp/oltw_dixon.py` | `OnlineTimeWarpingDixonFrame` |
| `src/export.rs` | `matchmaker-rs audio pipeline` | recorded MIDI/audio stage exports |
| `src/utils/tempo_models.rs` | `matchmaker/utils/tempo_models.py` | `KalmanTempoModel`, tracker integration of `ArztTempoModel` |
| `src/utils/eval.rs` | `matchmaker/utils/eval.py` | `evaluate_alignment`, `evaluate_beat_errors`, `compute_sparc` |
| `src/score/mod.rs` | `matchmaker/matchmaker.py` | `Matchmaker.__init__` score loading |
| `src/score/mod.rs` (`onset_pianoroll`) | `matchmaker/features/midi.py` | `onset_pianoroll` |
| `src/score/musicxml.rs` | `partitura/io/importmusicxml.py` | `load_musicxml` |
| `src/score/midi.rs` | `partitura/io/importmidi.py` | `load_score_midi` |
| `src/score/part.rs` | `partitura/score.py`, `partitura/utils/music.py` | `Part`, `Part.note_array` |
| `src/score/timeline.rs` | `partitura/score.py` | `Part._time_interpolator` |
| `src/score/unfold.rs` | `partitura/score.py` | `unfold_part_maximal`, `add_segments`, `Path`, `get_paths`, `new_part_from_path` |
| `src/score/merge.rs` | `partitura/score.py` | `merge_parts` |
| `src/score/tempo.rs` | `matchmaker/utils/misc.py`, `partitura/utils/music.py` | `get_tempo_from_score`, `to_quarter_tempo` |
| `crates/matchmaker-py/src/lib.rs` | `matchmaker/matchmaker.py`, `matchmaker/features/audio.py` | public `Matchmaker` API and `extract_audio_features` |
| `crates/matchmaker-cli/src/main.rs` | `matchmaker/matchmaker.py`, `matchmaker/features/audio.py` | MIDI/audio CLI, standalone feature extraction |
| `tools/bench/run_reference.py` | `matchmaker/matchmaker.py`, `matchmaker/utils/eval.py` | pinned MIDI/audio benchmark runner |

Not ported: audio `outerhmm`, `prob/outer_product_hmm_audio.py`, `particle_filter_korzeniowski.py`, standalone onset updates of `ArztTempoModel`, and live audio/MIDI devices and thread/queue plumbing. `skf.py` is ported except for its raw-audio input branch (`_extract_features` computing an FFT when a frame is not a spectrum) and its console prints. The reference chooses `skf` templates by trying FluidSynth with Partitura's bundled MuseScore SoundFont and falling back to Gaussian peaks on any error; this port makes the mode explicit and uses the pinned SF2 for the benchmark reference. Audio ground-truth CLI evaluation accepts direct `perf_sec`/`score_beat` TSV; note-ID `.match` evaluation still requires an explicit MIDI performance. `cqt` does not reproduce `soxr_hq` bit for bit (libsoxr is LGPL C with SIMD paths, so there is no pure-Rust port to reuse): a Kaiser-windowed sinc fitted to its measured response halves the rate, everything is computed in `f64`, and the features meet the reference to the amended per-frame limit of the contract instead of the `float32` per-element one; early downsampling by more than 2 (above about 70 kHz) is rejected instead of ported (research R13). `CQTSpectralFluxProcessor` keeps `fmin`, `n_bins` and `bins_per_octave` at the 88 keys, reports a shape error where NumPy would broadcast a one-frame history, and is not offered to the frame followers, because the reference fails there itself: its score and performance share one processor whose no-op `reset` leaves the score's matrix as history (research C07). The CI workflows download the pinned FreePats archive from a fixed HTTPS URL, verify its archive and extracted SF2 hashes, and provide the SoundFont to audio tests as a temporary artifact.

## Procedure: a new upstream release

1. Bump the pins in `tools/golden/pyproject.toml`, then `cd tools/golden; uv lock; uv sync --locked` (setup and regeneration: `specs/001-midi-score-following/quickstart.md`).
2. Diff the old and new upstream files of the rows above (`pip download pymatchmaker==<old>` and `<new>`, or `git diff v<old> v<new>` in the upstream repo). Only the files in the table matter.
3. For each changed Python function, change the Rust function named in its doc comment. Keep the Python structure so both stay readable side by side.
4. Regenerate the golden data with `tools/golden/generate.py` (layout: `specs/001-midi-score-following/contracts/golden-data.md`) and review the diff of `tests/golden/`: every changed number must trace back to a changed upstream line.
5. `cargo test --workspace --exclude matchmaker-py` (golden tests compare stage by stage), then `cargo clippy --all-targets --all-features -- -D warnings` and `cargo fmt`.
6. Update the version table above and the defaults in `src/matchmaker.rs` and `default_kwargs` in `crates/matchmaker-py/src/lib.rs` if `methods.yaml` changed.
7. If the Python API changed, mirror it in `crates/matchmaker-py/src/lib.rs` and `crates/matchmaker-py/tests/`.

New upstream code that is not in the table: add a Rust module with the `Port of` / `Upstream:` header and a row here.
