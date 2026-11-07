# Contract: Remaining Audio Methods and Features

This extends the existing 003 audio contract. Existing MIDI/audio names, constructors, defaults, generator behavior, path units, error classes, and exports remain valid.

## Facade defaults

Defaults below are taken from released methods.yaml and Matchmaker's configuration derivation, not merely raw class constructors. An explicit hop overrides frame_rate; otherwise hop is floor(sample_rate/frame_rate).

| Audio method | Processor | Rate / hop | Other default behavior |
|---|---|---|---|
| arzt_tempo | chroma | 44100 / 1470 (30 fps) | window_size=10 s, start_window_size=0.1 s, step_size=3, ArztTempoModel |
| arzt_multi_fold | chroma | 44100 / 1470 (30 fps) | same frame windows/step, selection_window=50, switch_margin=0.05 |
| skf | raw_spectrum | 8000 / 128 (62.5 fps) | n_fft=512; `template_mode` synthesized (default, needs the SoundFont) or gaussian, recorded in every export, never a silent fallback (research R12) |
| outerhmm | cqt_spectral_flux | 16000 / 640 (25 fps) | s_j=0.0; weighted chord and 88-pitch template model |
| pfkorz | korzeniowski | 44100 / 1470 (30 fps) | n_fft=4096, win_length=2048, num_particles=500; seed=1984 as reproducibility extension |

The direct KorzeniowskiAudioProcessor class defaults hop=512; the facade passes its derived hop instead. Preserve both. Audio-OuterHMM's reference registry forwards hop_length but not sample_rate to its constructor: capture the actual override semantics in US8 tests before accepting overrides; do not silently correct the released behavior.

## Processor contracts

| Key | Observation / defaults | Reference-critical operations |
|---|---|---|
| raw_spectrum | n_fft/2+1 bins; n_fft=512 | center=False STFT, magnitude, transposed float32 rows |
| cqt | 88 bins A0-C8; norm=infinity, fixed (the facade forwards only sample_rate and hop, so `norm` and `n_fft` are rejected); 44100 Hz, hop floor(rate/frame_rate) | complex CQT magnitude, float32, drop first/last rows; a performance call sees two hops (cache of one hop) and yields one row, the score is one whole-signal call; stateless, reset is a no-op |
| cqt_spectral_flux | 89 values by default (88 bins plus the flux), 88 if flux disabled; the facade forwards only sample_rate and hop, so 44100 Hz / hop 1470 and the 88 keys from A0 | same edge trim; flux = sum over the bins of the positive part of (current - previous call's matrix), frame by frame before the trim; first-call flux zero; inherited reset is a no-op (the history survives it); a call with another frame count than the previous one is a typed shape error |
| mel | 128 mel bands; n_fft=2*hop; norm=infinity | center=False power spectrum, exact librosa mel bank/scaling |
| mfcc | 13 coefficients; n_fft=2*hop; norm="backward" | exact locked librosa log-power/DCT normalization; error if reference default is unusable, never silently substitute |
| korzeniowski | structured spectrum/onset/loudness; optional note fields absent | periodic Hann, cropped win_length, n_fft padding, L2 spectrum, unnormalized causal flux, RMS-to-dB floor |

## Compatibility and exposure

- Vector processors are selectable on the already delivered Arzt/Dixon frame followers; tempo/multi-fold variants inherit only combinations whose reference construction/replay succeeds. Score and performance processing settings must match.
- `cqt` runs on `arzt` and `dixon` only; `arzt_tempo` and `arzt_multi_fold` reject it until their replay evidence exists. A rate/hop pair for which the reference would downsample the input by more than 2 before the transform (above about 70 kHz with a hop that is a multiple of 512) is an `AudioConfig` error: the 2:1 decimator is the only factor ported (research R13).
- SKF accepts raw_spectrum observations; Audio-OuterHMM accepts its CQT(+flux) layout; PF accepts structured Korzeniowski observations. Reject dimensionally or semantically incompatible pairs at construction.
- SKF follows the score's chord sequence (not a rendered score), so its audio reference is empty and `window_size`, `distance_func`, `norm`, `step_size`, `max_run_count` are rejected. Its keys are the shared ones (`processor`=raw_spectrum, `sample_rate`, `hop_length`/`frame_rate`, `n_fft`, `soundfont_path`) plus `template_mode`; the beam width and noise scales are constructor arguments of the library type only, as in the reference registry.
- CQT flux tests must capture inherited no-op reset and the score-to-performance matrix shape interaction. Fresh instances isolate separate runs. If the locked reference's default pipeline fails on those shapes, report a reference blocker rather than silently clearing history or fabricating successful goldens.
- **Outcome (C07): reference blocker.** The released pipeline fails on those shapes: the registry runs the rendered score through the same processor as the performance and calls its no-op `reset`, so the score's matrix stays as flux history and the first performance call raises NumPy's broadcast error (`(3,88)` against `(N,88)`) in the stream thread, after which `Matchmaker.run()` waits forever. Audio `arzt`, `dixon`, `arzt_tempo` and `arzt_multi_fold` therefore cannot follow with `cqt_spectral_flux`, and neither can `skf`: `processor=cqt_spectral_flux` is rejected for all of them with an `AudioConfig` error that names this (`no frame follower`). The processor is available through standalone extraction (library `export::write_cqt_spectral_flux_features` and `CqtSpectralFluxProcessor`, CLI `features --processor cqt_spectral_flux [--no-spectral-flux]`, Python `extract_audio_features(path, "cqt_spectral_flux", {"include_spectral_flux": bool})`); Audio-OuterHMM (C08) is its follower, and its reference builds no score audio, so it is not affected. `fmin`, `n_bins` and `bins_per_octave` of the reference constructor are not ported, and a one-frame history (a call shorter than a hop), which NumPy would broadcast, is a shape error.
- US4 introduces minimal standalone extraction through library processor types, a Python extract_audio_features(performance_path, processor, kwargs=None) helper, and CLI `features PERFORMANCE --processor KEY --export DIRECTORY`. Reuse AudioStream and stage-export machinery; no alternate decoder/pipeline. Each later processor is enabled in its own commit. US11 extends extraction to structured output, without inventing a vector-compatible follower.
- Each new method extends AudioMethod parsing/configuration and the existing AudioMatchmaker entry points; Python uses existing Matchmaker(..., input_type="audio", method=KEY, kwargs=...). Preserve run iterator semantics and score_follower.alignment_path shape.
- CLI follow accepts the new audio method names independently of the MIDI Method enum; otherwise audio-only names would fail before dispatch. Preserve MIDI validation and help. Add only current-story config keys; reject unknown or incompatible keys.
- Python errors remain ValueError for configuration and existing typed/path-specific conversions for I/O. Document public processors, constructors, extraction helper and emitted structured fields with runnable examples.

## Score audio conventions

Cross-cutting for every audio method and processor; they follow the pinned reference, not the quirks of the Rust synthesizer.

| Convention | Contract | Replace in |
|---|---|---|
| Level | The rendered score is peak-normalized to 1.0 before padding/cropping (Partitura `synthesize_fluidsynth`). Unnormalized processors (`raw_spectrum`, `cqt`, later `mel`, ...) depend on it. A silent piece stays unchanged. | `normalize_level` in `src/score/audio.rs` |
| Framing | Score features come from one whole-signal `center=False` pass: frame `j` covers `j*hop .. j*hop + n_fft`, no padding, trailing partial window dropped; chroma tuning is estimated once over the score; a score shorter than one window is an `AudioConfig` error. The performance stream is unaffected (cache of `n_fft - hop`, first block primed). `cqt` is the exception: one centered whole-signal transform with the first and last frame dropped, so row `j` is centered on sample `(j + 1) * hop`; a score shorter than two hops is an `AudioConfig` error. | `extract_score_features` in `src/score/audio.rs` |

Known deviation: Partitura divides by the peak of the last (track, channel) signal; Rust renders one mix, so multi-track scores are unverified. Replacing either function is allowed when the audio benchmark's parity against the reference stays within the limits below; research R11 records the alternatives that were weighed.

## Evaluation

Enable `follow ... --input-type audio --ground-truth ground_truth.tsv` using GroundTruth and the existing exact evaluate_alignment. Direct TSV is `perf_sec<TAB>score_beat`; retain existing header/loading rules and metric JSON schema. The existing eval command remains usable for audio-derived paths with direct TSV. Note-ID match input requires an explicit MIDI performance via the existing evaluation flow; an audio file must never be passed to load_performance_midi.

Validate ground truth and required inputs before writing results. Error paths do not truncate existing outputs. Include ground-truth identity and metrics in benchmark/export evidence without distributing restricted source datasets.

## Exports

Retain existing export file layouts; add story-specific arrays/metadata without renaming old keys. Record processor/method/rates/hop/seed/template mode and exact asset hashes. SKF writes its chord model to `score/skf/` (`templates`, `chords_flat`, `chord_offsets`, `lengths`, `onset_beats`), the template mode and SoundFont path to `score/meta.json`, and per frame the hypotheses as CSR arrays (`state_frames`, `state_offsets`, `state_chord`, `state_age`, `state_probability`, `state_tempo_mean`, `state_tempo_variance`) plus `candidates` (hypotheses before the beam) to `skf/`; it omits `score/audio/`. The same names are used by the goldens under `tests/golden/audio/skf/`. Vector arrays, reference features, timestamps, positions, paths, cost/probability windows, tempo/hypothesis/particle state are NumPy-readable. Structured output exports spectrum matrix plus onset/loudness arrays and explicit absent note-field metadata. Bound large model-state exports using existing recording choices; do not force particle dumps into every normal run.

## Validation and comparison limits

These are initial acceptance limits, declared before implementation; they are not observed results. First generate exact reference fixtures and confirm their modes/shapes. If an acceptance limit cannot be met, stop and report the first differing stage; changing a limit requires an explicit reviewed design change, not an automatic tolerance increase.

| Comparison | Required result |
|---|---|
| Names, shapes, counts, integer state indices, repeat selection with identical observations | EXACT |
| Frame timestamps and deterministic alignment beats from identical references/observations | Absolute error <=1e-9 |
| New float32 vector features | Per-element abs(a-b) <= 1e-5 + 1e-5*abs(reference) |
| cqt float32 features (C06, reviewed design change, research R13) | Per-element abs(a-b) <= 1e-5 + 1e-5*abs(reference) + 1e-5*max(abs(reference frame)); sparse kernel coefficients: the same with the maximum of their row; the 2:1 decimation stage keeps the strict row above. cqt_spectral_flux rows (C07) use the same limit for all 89 values with the maximum taken over the 88 bins, so the flux is judged as strictly as the bins |
| Deterministic float64 model diagnostics and structured fields with identical inputs/templates/draws | Per-element abs(a-b) <= 1e-9 + 1e-7*abs(reference) |
| Ground-truth mapping and evaluation primitives on identical paths | Exact discrete mapping, floating metrics absolute error <=1e-9 |
| Full pipeline with differing synthesizers | Absolute accuracy-rate difference <=0.01; absolute median beat-error difference <=0.01 beat; absolute p95 beat-error difference <=0.05 beat |
| PF full-pipeline comparison across seeds 1984-1988 | Same metric limits on pooled errors/accuracy; publish per-seed results as well |
| PF repeat in same Rust runtime/config/seed | EXACT positions, path, and recorded particle state |

Do not compare NumPy and Rust random streams for exact equality. Test normalization/resampling with fixed external draw fixtures to isolate deterministic algorithm errors. Full-pipeline metric exceptions do not excuse feature or fixed-input recurrence failures. Threshold definitions and interpolation must use the reference evaluator; metric differences above the limits block delivery.

Every method/processor has tiny fixed feature/replay goldens, interface tests, empty/silent/short/reset/error tests, and at least one full compatible run. New methods cover tempo/repeat/degenerate likelihood cases as appropriate. Fixed-template replay isolates model behavior from synthesis. Compare and report EXACT/WITHIN_TOL/EXCEEDS with first-differing-frame diagnostics.

## Commit gate

One story per authorized commit, C01-C12 in order. Required evidence: narrow tests, all Rust/CLI/doc/Python regression gates, fmt/clippy, MSRV, WASM, cargo-deny, Linux/Windows/macOS CI, five-run benchmark and delta report, golden/provenance review, and updated docs. No current-source work or commits are authorized by this planning request. Unrun or failed gates remain uncompleted tasks.
