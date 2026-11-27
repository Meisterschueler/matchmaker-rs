# Following upstream

matchmaker-rs is a port. This file says what it ports, from which version, and how to pull in upstream changes.

## Pinned upstream

| Package | Version | Source in this repo |
|---|---|---|
| pymatchmaker | 0.5.2 | `tools/golden/uv.lock` |
| partitura | 1.9.0 | `tools/golden/uv.lock` |
| python-hiddenmarkov | 0.3.0 | `tools/golden/uv.lock` |
| Parangonar | 3.4.1 | `tools/golden/uv.lock`; OTM source and converted checkpoint |

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
| `src/features/mel.rs` | `matchmaker/features/audio.py`, `librosa/feature/spectral.py`, `librosa/filters.py` | `MelSpectrogramProcessor`; non-centered Hann power spectrum and Slaney mel bank |
| `src/features/cqt.rs` | `matchmaker/features/audio.py`, `librosa/core/constantq.py`, `librosa/core/audio.py` | `CQTProcessor`; librosa's recursive `cqt` (octave kernels, early downsampling, response scaling) with the `soxr_hq` 2:1 decimator as a fitted Kaiser sinc (research R13) |
| `src/features/cqt_spectral_flux.rs` | `matchmaker/features/audio.py` | `CQTSpectralFluxProcessor` over `src/features/cqt.rs`: previous call's matrix as flux history, inherited no-op `reset` (research R5, C07) |
| `src/features/mfcc.rs` | `matchmaker/features/audio.py`, `librosa/feature/spectral.py`, `scipy/fft` | `MFCCProcessor`; Slaney log-power mel features and DCT-II coefficients |
| `src/features/korzeniowski.rs` | `matchmaker/features/audio.py` | `KorzeniowskiAudioProcessor`; structured normalized spectrum, causal onset flux and RMS loudness |
| `src/score/audio.rs` | `matchmaker/utils/misc.py`, `matchmaker/registry.py`, `partitura/utils/fluidsynth.py` | peak-normalized synthesized score, whole-signal score features, frame-to-beat map (swap points `normalize_level`, `extract_score_features`; research R11) |
| `src/features/pitch.rs` | `matchmaker/features/midi.py` | `PitchProcessor` |
| `src/features/chord_onset.rs` | `matchmaker/features/midi.py` | `ChordOnsetProcessor` |
| `src/features/parangonar.rs` | `matchmaker/external.py` | event-based `ParangonarProcessor` |
| `src/dp/otm.rs` | `parangonar/match/online_matchers.py` | `OnlineTransformerMatcher.__call__`, `TempoModel`, CPython integer-set token ordering |
| `src/dp/otm_model.rs` | `parangonar/match/pretrained_models.py`, `parangonar/match/online_matchers.py` | fixed CPU `AlignmentTransformer` and `tokenize` |
| `src/dp/parangonar.rs` | `parangonar/match/online_matchers.py` | `OnlinePureTransformerMatcher`, `OLTWMatcher`, `TOLTWMatcher` |
| `src/io/stream.rs` | `matchmaker/io/midi.py`, `matchmaker/utils/symbolic.py` | `MidiStream`, `Buffer`, `midi_messages_to_framed_midi` |
| `src/performance.rs` | `partitura/io/importmidi.py`, `matchmaker/utils/symbolic.py` | `load_performance_midi`, `midi_messages_from_performance` |
| `src/prob/pthmm.rs` | `matchmaker/prob/hmm.py` | `PitchHMM` |
| `src/prob/hmm.rs` | `matchmaker/prob/hmm.py` | `PitchIOIHMM` |
| `src/prob/hmm_models.rs` | `matchmaker/prob/hmm.py` | `stable_transition_matrix`, `compute_discrete_pitch_profiles`, `compute_bernoulli_pitch_probabilities` |
| `src/prob/outerhmm.rs` | `matchmaker/prob/outer_product_hmm.py` | `OuterProductHMM` |
| `src/prob/outerhmm_audio.rs` | `matchmaker/prob/outer_product_hmm_audio.py` | `AudioOuterProductHMM`, weighted chord/pitch templates, sound/pause posterior |
| `src/prob/skf.rs` | `matchmaker/prob/skf.py` | `SwitchingKalmanFilterFollower`, `build_chord_sequence`, `build_spectral_templates` (synthesized with RustySynth, Gaussian fallback as an explicit mode) |
| `src/prob/pfkorz.rs` | `matchmaker/prob/particle_filter_korzeniowski.py` | seeded Korzeniowski particle filter and inspectable particle state |
| `src/prob/hidden_markov.rs` | `hiddenmarkov/__init__.py` | `forward_algorithm_step_optimized`, `ConstantTransitionModel` |
| `src/dp/mod.rs` | `matchmaker/dp/direction.py`, SciPy `cdist` | `Direction`; metrics `cosine`, `euclidean` |
| `src/dp/arzt.rs` | `matchmaker/dp/oltw_arzt.py` | `OnlineTimeWarpingArztEvent` |
| `src/dp/arzt_frame.rs` | `matchmaker/dp/oltw_arzt.py`, `matchmaker/dp/dtw_loop.pyx` | `OnlineTimeWarpingArztFrame` |
| `src/dp/arzt_tempo.rs` | `matchmaker/dp/oltw_arzt.py`, `matchmaker/utils/tempo_models.py` | tempo-aware `OnlineTimeWarpingArztFrame` |
| `src/dp/arzt_multi_fold.rs` | `matchmaker/dp/oltw_arzt_multi_fold.py` | parallel repeat-hypothesis Arzt follower |
| `src/dp/dixon.rs` | `matchmaker/dp/oltw_dixon.py` | `OnlineTimeWarpingDixonEvent` |
| `src/dp/dixon_frame.rs` | `matchmaker/dp/oltw_dixon.py` | `OnlineTimeWarpingDixonFrame` |
| `src/dp/dannenberg.rs` | `matchmaker/dp/dannenberg.py` | `DPDannenberg` |
| `src/score/mod.rs` (`score_cevts`) | `matchmaker/registry.py` | `score_cevts` reference builder |
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
| `src/score/korzeniowski.rs` | `matchmaker/features/processor.py` | analytical `KorzeniowskiScoreModel` |
| `crates/matchmaker-py/src/lib.rs` | `matchmaker/matchmaker.py`, `matchmaker/features/audio.py` | public `Matchmaker` API and `extract_audio_features` |
| `crates/matchmaker-cli/src/main.rs` | `matchmaker/matchmaker.py`, `matchmaker/features/audio.py` | MIDI/audio CLI, standalone feature extraction |
| `tools/bench/run_reference.py` | `matchmaker/matchmaker.py`, `matchmaker/utils/eval.py` | pinned MIDI/audio benchmark runner |

Not ported: standalone onset updates of `ArztTempoModel`, and live audio/MIDI devices and thread/queue plumbing — including pymatchmaker's `wait=True` feeder thread of `AudioStream.run_offline`, which paces file blocks at playback deadlines (offline block iteration is unchanged and stays ported). `dannenberg` ports `dp/dannenberg.py` as written: the compound-event grouping test (`_is_new_cevt`) is computed and its result discarded, as upstream does; pitch sets are `u128` bitmasks with unchanged set algebra. pymatchmaker 0.5.2's `PitchIOIHMM.step` fast path delegates to python-hiddenmarkov 0.3.0's optimized forward step; this port limits work to reached rows and transition columns in `src/prob/hidden_markov.rs`, including the `max(sum, 1e-6)` guard. Its inserted-states observation shortcut and `_iter_first_unfolded_parts(part, 4)` are value-preserving and match the existing banded observation handling and lazy `iter_unfolded_parts` in `src/score/unfold.rs`. The SPARC smoothness metric follows 0.5.2: the held position path is resampled at 3000 Hz and per-window values come from the reference's sliding-spectrum computation (`windowed_spectral_arc_lengths`), falling back to plain per-window FFTs by the same cost heuristic; the port returns the mean only (no per-window diagnostics list). `skf.py` is ported except for its raw-audio input branch (`_extract_features` computing an FFT when a frame is not a spectrum) and its console prints. `particle_filter_korzeniowski.py` is ported with a seeded MT19937 stream, default seed 1984 and facade count 500; NumPy stream identity is not promised, while same-runtime/config/seed Rust runs are exact. Its score model is analytical and particle-state exports are bounded. Audio `outerhmm` ports `prob/outer_product_hmm_audio.py` with the upstream Apache-2.0 pitch-template archive (SHA-256 recorded in C08 evidence); the upstream distribution does not identify its training corpus. Unsupported synthesized rows use RustySynth and their measured FluidSynth difference is reported separately from the fixed-template replay. The reference registry does not forward sample_rate to the follower constructor; the port preserves the fixed 16 kHz follower rate while using the configured hop. The reference chooses `skf` templates by trying FluidSynth with Partitura's bundled MuseScore SoundFont and falling back to Gaussian peaks on any error; this port makes the mode explicit and uses the pinned SF2 for the benchmark reference. Audio ground-truth CLI evaluation accepts direct `perf_sec`/`score_beat` TSV; note-ID `.match` evaluation still requires an explicit MIDI performance. `cqt` does not reproduce `soxr_hq` bit for bit (libsoxr is LGPL C with SIMD paths, so there is no pure-Rust port to reuse): a Kaiser-windowed sinc fitted to its measured response halves the rate, everything is computed in `f64`, and the features meet the reference to the amended per-frame limit of the contract instead of the `float32` per-element one; early downsampling by more than 2 (above about 70 kHz) is rejected instead of ported (research R13). `CQTSpectralFluxProcessor` keeps `fmin`, `n_bins` and `bins_per_octave` at the 88 keys and reports a shape error where NumPy would broadcast a one-frame history. The reference score-audio pipeline remains blocked for `arzt`, `dixon`, `arzt_tempo`, `arzt_multi_fold` and `skf`: the score's matrix stays as flux history after no-op reset; flux is accepted only by standalone extraction and the note-array-based `outerhmm` (research C07). The CI workflows download the pinned FreePats archive from a fixed HTTPS URL, verify its archive and extracted SF2 hashes, and provide the SoundFont to audio tests as a temporary artifact.

Parangonar `OTM`, `OPTM`, `SLT_OLTW`, and `SL_OLTW` are MIDI-only with processor
`parangonar` and no polling. Offline `DualDTWNoteMatcher` is not a
registered pymatchmaker score follower. OTM mirrors the released `__call__`, including its
hardcoded initial 0.5-second beat period and greedy branch's grace-note tempo update.
NumPy 2 and SciPy's weighted-endpoint interpolation are reproduced in float32; public event
times remain float64. Source/compact hashes and tensor shapes are recorded in
`assets/otm/model.json`; `tools/golden/export_otm_weights.py` deterministically converts the
checkpoint without shipping optimizer state or unused positional rows. `NOTICE` and
`assets/otm/LICENSE` retain Apache-2.0 attribution, following the project owner's explicit
license determination recorded in `specs/006-online-transformer-matcher/research.md`.
The inspected upstream package does not document the model's training corpus.

## pymatchmaker 0.5.2 change map

The upstream comparison is `v0.5.1...v0.5.2` (four changed files). The Python
implementation remains the source of truth; each release change is mapped here to the
Rust equivalent, a version/provenance update, or an explicit no-code disposition.

| Upstream change | Rust counterpart / disposition | Evidence |
|---|---|---|
| `CHANGES.md`: pymatchmaker 0.5.2 requires python-hiddenmarkov 0.3.0 for the optimized forward step. | `src/prob/hidden_markov.rs` ports `forward_algorithm_step_optimized`; `src/prob/pthmm.rs` and `src/prob/hmm.rs` share it. | Focused optimized-vs-full tests, HMM goldens, and benchmark procedure below. |
| `matchmaker/__init__.py`: fallback package version changes to 0.5.2. | Version is reported from the installed reference in `tools/golden/generate.py` and `tools/golden/generate_audio.py`; Rust binding and OTM export metadata identify their validated tag in `crates/matchmaker-py/src/lib.rs` and `src/export.rs`. | Generated metadata and `crates/matchmaker-py/tests/test_drop_in.py`. |
| `matchmaker/prob/hmm.py`: observation-model formatting only; `PitchIOIHMM.step` delegates to the inherited optimized forward step. | Formatting-only change needs no Rust port. The behavior is implemented once in `src/prob/hidden_markov.rs` and used by both HMM methods. | `src/prob/hidden_markov.rs` unit tests and the `pthmm`/`hmm` golden tests. |
| `pyproject.toml`: package version 0.5.2 and minimum python-hiddenmarkov 0.3.0. | Pin and resolved dependency are recorded in `tools/golden/pyproject.toml` and `tools/golden/uv.lock`. | Generated reference metadata records the exact package versions and lock digest. |

## Procedure: a new upstream release

1. Bump the pins in `tools/golden/pyproject.toml`, then run `uv lock` and `uv sync --locked` from `tools/golden/` (setup and regeneration: `specs/001-midi-score-following/quickstart.md`).
2. Compare the complete upstream releases (`git diff --name-status v<old>...v<new>` and inspect every changed file, including additions and deletions). The module map is a starting point, not a filter: classify each upstream change as ported, behaviorally equivalent, or intentionally not ported, and add mappings for new upstream code.
3. For each changed Python function, update the corresponding Rust implementation and its source/version trace. Keep the Python structure easy to compare with the Rust port.
4. Regenerate MIDI and audio reference data with `tools/golden/generate.py` and `tools/golden/generate_audio.py` (layout: `specs/001-midi-score-following/contracts/golden-data.md`), then review all metadata and golden diffs. Every changed number must trace back to an upstream or locked-dependency change.
5. `cargo test --workspace --exclude matchmaker-py` (golden tests compare stage by stage), then `cargo clippy --all-targets --all-features -- -D warnings` and `cargo fmt`.
6. Run the existing benchmark harness on identical inputs and record median results for affected methods; investigate regressions greater than 10%.
7. Update the version table above and add a release change map showing each changed upstream file, its Rust counterpart or disposition, and the validation evidence. Update the defaults in `src/matchmaker.rs` and `default_kwargs` in `crates/matchmaker-py/src/lib.rs` if `methods.yaml` changed.
7. If the Python API changed, mirror it in `crates/matchmaker-py/src/lib.rs` and `crates/matchmaker-py/tests/`.

New upstream code that is not in the table: add a Rust module with the `Port of` / `Upstream:` header and a row here.
