# Research: Remaining Audio Methods and Features

**Date**: 2026-11-03
**Reference**: Released pymatchmaker 0.4.0 in tools/golden/.venv/Lib/site-packages/matchmaker; environment pinned by tools/golden/uv.lock.
**Evidence**: The initial sections record read-only research. The dated implementation entries below contain the later fixture, numerical, benchmark, and gate evidence.

## R1 - Incremental delivery

- **Decision**: One story per coherent commit, ordered C01-C12; goldens, facades, exports, docs, benchmarks, and gates belong to that commit.
- **Rationale**: Existing audio infrastructure is already implemented; an infrastructure megacommit or simultaneous new algorithms would violate the user's delivery constraint.
- **Alternatives considered**: Parallel method ports, separate core/facade commits, and a generic registry are rejected. Test-first edits within a story are allowed; unfinished algorithm commits are not.

## R2 - Tempo-aware Arzt and multiple unfolding hypotheses

- **Decision**: Reuse ArztFrame; port ArztTempoModel into utils/tempo_models.rs, expose arzt_tempo in its own method module. Port the multi-fold selection owner into dp/arzt_multi_fold.rs and use existing score repeat/unfold structures.
- **Rationale**: methods.yaml maps arzt_tempo to OnlineTimeWarpingArztFrame plus ArztTempoModel. dp/oltw_arzt.py contains before_step, tempo-dependent advance, reset, and state-history behavior; utils/tempo_models.py owns the tempo model. dp/oltw_arzt_multi_fold.py owns selection_window=50 and switch_margin=0.05.
- **Alternatives considered**: Duplicating ArztFrame or treating multi-fold as merely a different pre-unfolded score loses the controlling reference behavior.
- **Gate**: US1 captures tempo insertion/deletion/history state; US2 captures candidate unfolding, score-position domain, switching, and ties before changing the shared follower.

## R3 - Existing evaluation, no second evaluator

- **Decision**: Reuse src/utils/eval.rs and GroundTruth. Extend the audio follow branch that currently rejects --ground-truth; support direct perf_sec/score_beat TSV without asking the MIDI loader to read audio.
- **Rationale**: CLI already has follow/eval and the exact reference mapping primitives. The reference definition is utils/eval.py:evaluate_alignment, not the paper's alternative interpretation.
- **Alternatives considered**: A new audio accuracy formula is rejected. MIDI-note match files require a separately supplied MIDI performance and are not inferred from audio.

## R4 - FFT-based processor reuse

- **Decision**: Reuse realfft and nearby chroma/LSE conventions for raw spectrum, mel, MFCC, and Korzeniowski features; share small helpers only when actual duplication appears.
- **Rationale**: features/audio.py specifies center=False for raw spectrum, mel, and MFCC; mel/MFCC use n_fft=2*hop. Defaults are N_MELS=128, N_MFCC=13, NORM=np.inf; MFCC norm="backward" is passed to librosa and must be checked in the locked environment rather than silently replaced by "ortho".
- **Alternatives considered**: A new DSP dependency for a short FFT-bank/DCT computation is not justified. A maintained DCT dependency is evaluated only if its implementation meaningfully reduces complexity and matches scipy's scaling.
- **Gate**: If the released MFCC configuration fails in the locked environment, report a reference compatibility blocker; do not invent successful default goldens.

## R5 - Constant-Q parity and state

- **Decision**: CQT must reproduce the locked librosa transform, including filter norm, resampling, padding, complex magnitude, and trimmed edge frames. Evaluate maintained pure-Rust alternatives in US6 before writing kernels; use installed realfft/rubato only if the candidate evaluation justifies a targeted port.
- **Rationale**: CQTProcessor returns abs(cqt).T.astype(float32)[1:-1] over 88 bins from A0, norm=np.inf. CQTSpectralFluxProcessor also trims [1:-1]; its default 89th bin is half-wave flux. It compares the full current CQT matrix to the previous call's matrix, not simply adjacent rows within the call; the initial call's flux is zero. The class has no reset override; Processor.reset is a no-op. Do not assume the score-to-performance reset clears its matrix.
- **Alternatives considered**: No parity-compatible maintained CQT crate was verified during this planning pass. An approximate log filter bank, a C-backed backend, and Python/FFI in the default runtime are rejected. Absence of a verified crate is not proof that no crate exists.
- **Gate**: Compare a small pure-Rust prototype against fixed librosa inputs at default and overridden hops/rates. Record candidate version/license/MSRV/WASM status and max deviations before US6 registration. A failed probe pauses C06 and the sequence. US7 also probes full score-to-performance matrix shapes and inherited no-op reset; a failing released pipeline is a reference blocker, not permission to silently change flux/reset semantics. Use fresh instances for independent runs.

## R6 - SKF spectral templates

- **Decision**: Port the reference's synthesized templates and Gaussian-peak fallback as distinct, recorded modes; reuse RustySynth for runtime synthesis, FluidSynth only for reference generation.
- **Rationale**: prob/skf.py:build_spectral_templates first attempts synthesized chord spectra and otherwise uses harmonic Gaussian templates. SKF consumes raw magnitudes (257 default bins) with sample_rate=8000, hop=128, n_fft=512. Synthesis differences cannot be hidden as state-recurrence errors.
- **Alternatives considered**: Using Gaussian mode permanently or catching every error as permission to switch modes would change the reference's successful path. User asset/config failures remain typed errors, not fallback triggers.
- **Gate**: Pin the reference SoundFont and template mode; compare replay using identical templates, then full-pipeline metrics. Only reproduce a fallback condition observed and documented in the reference.

## R7 - Audio outer-product HMM templates

- **Decision**: Keep audio outerhmm distinct from MIDI OuterProductHmm. Port weighted sustained-note chord templates, the two-layer sound/pause state, and the pitch-template overlay rule.
- **Rationale**: prob/outer_product_hmm_audio.py:build_pitch_cqt_templates synthesizes all 88 MIDI pitches 21-108 (velocities 50/80/110), then overlays rows from templates/pitch_templates_{sample_rate}_{hop}.npz where the learned row sum exceeds 0.5. Default facade hop is 640 at 16000 Hz / 25 fps; a corresponding bundled archive exists. This is a synthesis-plus-overlay model, not an interchangeable synthesized-only fallback.
- **Alternatives considered**: Reusing MIDI emissions, silently omitting learned rows, or replacing the template SoundFont without metric evidence are rejected.
- **Gate**: US8 audits exact asset origin, license, hash, dimensions, and support mask before copying/redistributing anything. Archive license was not established in planning. Compare fixed-template replay separately from rendering. methods.yaml does not forward sample_rate into the follower constructor; US8 must capture override behavior rather than silently repair that upstream quirk.

## R8 - Structured observations and score model

- **Decision**: Keep existing vector Observation unchanged. US11 adds KorzeniowskiObservation and a dedicated processor with its own typed output; US12 adds the score model and a direct consumer path. Standalone extraction exports the structured fields separately.
- **Rationale**: features/processor.py defines spectrum, active_notes, onset, onset_notes, loudness; audio sets the two note fields to None. audio.py uses periodic Hann, L2-normalized rFFT magnitude, positive flux of unnormalized spectra divided by current sum+1e-8, and librosa RMS/amplitude_to_db. The onset is not clamped to [0,1]. The score processor builds beat_grid, onset_positions, nearest_onsets, rest_mask, templates, and active_notes analytically; it does not require FluidSynth or learned assets.
- **Alternatives considered**: Flattening onset/loudness into an undocumented Vec<f32> would weaken the observation contract; changing every existing vector caller for one structured processor adds unnecessary blast radius.
- **Gate**: Preserve reference frame cropping/padding and silence dB floor. Default direct processor hop is 512, but the Matchmaker pfkorz facade derives hop=1470 from 44100/30 unless overridden; test both entry points.

## R9 - Seeded particle filtering

- **Decision**: Expose an explicit seed with default 1984, use a maintained deterministic seeded Rust RNG, and validate Rust repeatability plus reference accuracy across seeds 1984-1988. Do not require NumPy-identical random streams.
- **Rationale**: prob/particle_filter_korzeniowski.py has a per-instance RandomState(1984); the registry facade defaults num_particles=500 whereas a raw constructor/provider can have a different default. Constitution I permits metric comparison for stochastic methods; IV requires selectable seeds and repeatability.
- **Alternatives considered**: Global mutable RNG, unseeded platform entropy, or claiming exact posterior parity between distinct generators are rejected.
- **Gate**: US12 evaluates RNG license/MSRV/WASM portability and replay checks for deterministic suboperations (weights, normalization, resampling with fixed draws). Reference seed-set generation can override its RNG in the fixture harness without modifying the pinned package or release.

## R10 - Verification policy

- **Decision**: Use the explicit comparison limits in contracts/audio-api.md. Establish reference fixtures and first-difference reporting before porting each slice; limits are not adjusted merely to make a port pass.
- **Rationale**: Full-pipeline synthesis and random-stream differences need the exact evaluator; deterministic feature/state comparisons still require their own goldens. Existing benchmark infrastructure already supports feature/inference timing and five-run reports.
- **Alternatives considered**: A single end-to-end tolerance hides the root stage. Automatically relaxing limits, skipping inconvenient platforms, or marking unrun gates complete is rejected.

## Implementation Evidence Log

C01 local and remote delivery gates are complete. C01 publication and multi-platform CI are recorded below; C02-C12 have not started.

### T001 - Implementation scope authorized (2026-11-03)

- The user invoked speckit-implement after requesting one separate commit per algorithm. This authorizes implementation and the planned C01-C12 commit sequence, one coherent story at a time; it does not authorize pushing, creating another branch, changing the reference release, or bypassing gates.
- Existing branch: 004-audio-methods. Initial worktree status: only specs/004-audio-methods/ is untracked from the preceding planning step; no source changes are present. Preserve those planning artifacts and all subsequent user edits.
- Prerequisite script resolved this feature and all design documents. The requirements checklist has 16 checked and zero unchecked criteria; implementation leaves its markers unchanged. No .specify/extensions.yml exists, so there are no before/after implementation hooks to dispatch.
- Existing .gitignore covers the actual Rust target/backups/logs/environment files and Python virtualenv/cache/binding distribution outputs. No Docker, Node, Terraform, or Helm setup is present at the project root; no unrelated ignore file is added.
- Next blocking task: T002 verifies the locked reference, SoundFont and unchanged baseline. No algorithm port begins before that task passes.

### T002 - Local baseline passed (2026-11-03)

- Baseline HEAD: bf864a0 (docs(003): record T035 benchmark verification). No algorithm source has been changed during this baseline run.
- Existing audio-only golden targets: six tests passed with `cargo test -p matchmaker-rs --all-features --test golden_audio_arzt --test golden_audio_dixon --test golden_audio_processors`.
- Full Rust/CLI/doc baseline passed with `cargo test --workspace --exclude matchmaker-py --all-features --no-fail-fast`; all 28 doc tests passed, including MIDI probability and audio end-to-end goldens.
- `cargo fmt --all --check`, core/CLI clippy with all targets/features and warnings denied, binding clippy with warnings denied, MSRV `cargo +1.88 check -p matchmaker-rs -p matchmaker-cli --all-targets`, and `cargo build -p matchmaker-rs --target wasm32-unknown-unknown` all passed.
- `cargo deny check` passed advisories, bans, licenses, and sources. Existing unmatched license allowances and duplicate syn versions were warnings, not failures.
- `maturin develop` in crates/matchmaker-py succeeded; its existing `.venv` on Python 3.13.5 passed all 39 pytest tests. Other Python/platform CI jobs have not been claimed run locally.
- Actual SF2 SHA-256 matched CF2A98EB38A32C4954B4B6E2CAAE4112D62DD8E892ECEEFDD7942B0E7D01AC2F. Explicit tools/golden/.venv/Scripts/python.exe imported pymatchmaker 0.4.0, partitura 1.9.0 and librosa 1.0.0. The Pylance snippet tool failed/timed out without details; direct execution of that exact interpreter passed.
- GitHub checks/status lookup for bf864a0 returned HTTP 422/404; multi-platform baseline CI cannot be established from those endpoints. The user selected "Lokal implementieren; CI vor Lieferung": local baseline completes T002 and permits local story implementation, while multi-platform CI remains an unexecuted required delivery gate for every story. No push was authorized and no completed C01-C12 delivery is claimed.

### T003 - Seeded tempo reference captured (2026-11-03)

- The reference ArztTempoModel is stochastic: MT19937 seed 1984 controls score-frame insertions/deletions. It requires backpointers, rectified onset-anchored backward paths, score/cost/backpointer row edits, and the tempo-adjusted position estimator, not just a different advance clamp.
- `generate_audio.py --tempo-replay-only` writes renderer-free fixtures under tests/golden/audio/tempo_replay/: steady (480 frames, zero insertions/deletions), accelerating (490 frames, 16 insertions/7 deletions), slowing (490 frames, 10 insertions/13 deletions). The generator asserts exact repeated positions/state/random decisions after reference reset; meta.json records every step's uniform draws and eight state columns.
- `generate_audio.py --arzt-tempo-only` writes separate Chroma/LSE audio cases under tests/golden/audio/tempo_pipeline/, 226 consumed frames each. Initial and altered score features/mappings and tempo state are recorded separately; prior Arzt/Dixon goldens were not regenerated.
- Reference generation used the already present FluidSynth 2.6.2 archive (SHA-256 808295345D43C9C428F3FA87D29FBF97A9926F356DF95559FF2689F3BD4072D7) with its bin directory on PATH/current DLL search directory and the existing pinned SF2. No installed reference source or runtime native linkage was modified. Explicit reference kwargs must include ArztTempoModel because the released facade replaces, rather than merges, method defaults.
- Dependency evaluation: rand_mt 6.1.0 is a maintained MIT OR Apache-2.0, pure-Rust MT19937 implementation with MSRV 1.88.0 and no build script. It is used without optional default rand-traits; NumPy's two-u32 53-bit uniform conversion is implemented directly. Final license/MSRV/WASM gates for this dependency are still required before delivery.

### C01 - Initial implementation evidence (before renderer calibration)

- Implemented the optional tempo state/backpointer path in ArztFrame, dedicated ArztTempoFrame, seeded frame alteration, AudioMethod/config/constructor, CLI audio-name dispatch, Python construction/defaults/method list, and additive state/cost-window exports. No next algorithm is registered.
- Green focused checks: three synthetic replay scenarios and both actual-audio feature replays (positions <=1e-9 absolute; float64 state <=1e-9 + 1e-7 relative; final synthetic score frames exact), CLI default/seed/error test, Python tempo default/seed/path test, export state/cost-window test, and existing Arzt/Dixon/processor/score goldens after compatibility restoration.
- Two processor/reference-path issues were verified independently: NumPy tuning argmax uses the first histogram winner (four real-frame oracle checks); Score Chroma uses a common batch tuning (-0.07 on the existing three-tone oracle). Unprimed center=False LSE score windows were also validated against the same RustySynth waveform at 30/50 fps.
- Existing Arzt/Dixon score feature construction remains unchanged under an explicit legacy path, because migrating it changed delivered end-to-end goldens beyond their existing tolerances. Batch score tuning and unprimed score windows are scoped to the new Tempo-Arzt method; no old golden or tolerance was relaxed.
- The declared full-pipeline accuracy limits remain unchanged. The focused gate still fails on Chroma accuracy_0.05b: Rust 0.8982 versus reference 0.9735 in the last focused run (difference 0.0753, allowed 0.01). Fixed-reference/observation replays pass, but a cause attributable solely to synthesizer differences has not been proven. This is a real open delivery blocker, not an accepted deviation.
- Benchmark measured five serial Release runs per method/side on the bundled SF2 fixture, without RUST_LOG/subscriber and without parallel measurement load. Latest local Windows report: target/bench/audio-methods/report/report.json and index.md; records and run-0 paths are under its sibling rust/ and python/ directories. Tempo-Arzt: feature median 1.723 ms/frame Rust vs 4.189 Python; inference 0.01635 vs 1.341; median total speed-up 3.4x, Rust RTF 0.05412. Report accuracy_0.1b is Rust 0.9558 vs Python 0.9912; strict metric parity is marked false, while the older broad audio position check passes at max difference 0.5333 beats.
- Arzt/Dixon were measured alongside it for regression visibility. The existing local report-fixed baseline reports no >10% inference slowdown for those methods; Tempo-Arzt has no prior same-method baseline. These Windows measurements are not the required Ubuntu master-baseline result or multi-platform CI evidence.
- Core/CLI all-target/all-feature Clippy with warnings denied passed before the final parser/test-only repairs. The full workspace run completed (target/tmp/004-audio-tests.log): three failing targets, namely CLI unknown-name exit code, dynamic Tempo-Arzt overrun bound, and the declared Chroma accuracy limit; all other targets and 29 doc tests passed. The first two were repaired and their exact focused checks then passed; the accuracy gate remains reproducibly red. A green full-suite claim is not made.
- Exact Rust reset/seed replay passed for steady, accelerating and slowing fixtures after core integration. T003-T005 are complete (reference capture, test creation and reference-replay core port). Final Python rebuild/regressions, rand_mt license/MSRV/WASM, documentation/stage-comparison completion, and multi-platform CI remain open. T006-T009 stay unchecked; no C01 commit, push, or C02 work is performed.

### C01 - Renderer cause verified and local completion (2026-11-03)

- Four-way feature replay isolated the entire accuracy loss to the score reference: Python score + either performance feature set gives accuracy_0.05b=0.9735; Rust score + either performance feature set gives 0.8982. Python and Rust followers on the same inputs have exactly identical positions. Performance features differ by at most 1.13249e-6; Chroma on identical Rust waveform samples differs by at most 5.36442e-7, both within existing feature tolerances.
- The dominant cause is RustySynth's channel reset default Reverb Send CC91=40, whereas the pinned FluidSynth reference channel has CC91=0. Explicitly initializing CC91=0 produces exactly the same Rust waveform as disabling effects for this fixture, raising accuracy_0.05b to 0.9690. Peak normalization, left-vs-average stereo, interpolation order and the maximum one-sample event-clock difference did not explain the loss. No new synthesizer dependency or relaxed accuracy limit is needed.
- The production renderer now sends CC91=0 immediately after creation. Both processor cases pass the unchanged full-pipeline median, all six accuracy thresholds and p95 comparison gates. Chroma accuracy is 0.9690 vs reference 0.9735 (0.0045 difference); accuracy_0.1b is identical at 0.9912; p95 difference is approximately 3.3e-16 beats. Existing Arzt/Dixon golden, framing, I/O and export tests remain green.
- T006-T008 are complete. Stage comparison recognizes Tempo-Arzt, explicitly reports synthesis feature differences and enforces the declared renderer metrics; its test and report-boundary tests pass. Benchmark reports preserve strict metric identity as a separate flag and expose renderer_accuracy_parity. Public method/seed/export docs, module map, asset provenance and existing benchmark CI method lists were updated.
- Full `cargo test --workspace --exclude matchmaker-py --all-features --no-fail-fast` passed, including 29 doc tests. `maturin develop` and all 40 Python 3.13.5 tests passed. Core/CLI all-target/all-feature Clippy and binding Clippy with warnings denied, fmt check, MSRV 1.88, WASM core build and cargo-deny all passed. Report/stage-comparison tests: 5/5 passed. The YAML editor's schema fetch is network-blocked; this is not a YAML content validation failure or a successful remote CI claim.
- Fresh serial Release benchmark: five runs per method per side, identical pinned fixture/config, no tracing subscriber and no concurrent compiler load. Final report under target/bench/audio-methods/report/: Tempo-Arzt feature median 1.719 ms/frame Rust vs 3.155 Python, inference 0.0163 vs 1.153, median total speed-up 2.4x, Rust RTF 0.05286. Renderer accuracy and path checks pass (maximum beat-path difference 0.06667); strict metric identity remains false because residual synthesizer metrics differ. Existing local baseline marks no >10% inference slowdown for Arzt/Dixon; Tempo-Arzt has no prior same-method baseline.
- Actual origin is Meisterschueler/matchmaker-rs. C01 commit ed3bfe18d1f9e597219d7ac37b140ffd4eb99734 was pushed on `004-audio-methods` and is the only branch-ahead commit. The user created draft PR #5 with an intentionally empty body; it remains unmerged.

### C01 - Commit/publication authorization

- The user explicitly selected "C01 committen und Featurebranch pushen" after all local gates passed. Create one coherent C01 commit and push the existing 004-audio-methods branch; do not merge, skip CI or begin C02 before CI is verified.
- A local commit and branch publication precede remote CI by necessity. T009 remains open until remote checks pass and the C01 hash is recorded; publication is not a claim of completed delivery.

### T009 - Remote C01 gates passed (2026-11-04)

- PR #5 (`004 audio methods`) remains a draft with no body. No merge was performed.
- CI run 41732450196 completed successfully in 3m37s: pinned SoundFont, WASM, MSRV 1.88, cargo-deny, Rust tests on Ubuntu/Windows/macOS, and Python bindings on 3.10/3.13 all passed.
- Benchmark run 41732450143 completed successfully in 2h06m: all four dataset jobs and the report job passed; deployment was skipped for the pull request. The workflow's configured five-run comparisons therefore completed.
- With the C01 hash recorded above and every remote check green, T009 is complete and C02 may begin. The PR remains unmerged; merge is not authorized by this completion.

### C02 - Repeat candidate semantics captured (2026-11-04)

- Released reference: `matchmaker.dp.oltw_arzt_multi_fold.OnlineTimeWarpingArztMultiFold`; candidates are ordered maximal unfold, input score as-is, then the first four `partitura.score.iter_unfolded_parts` results, deduplicated by `(note_count, round(max_onset_beat, 2))`.
- On the hand-authored `fixture_repeats_voltas.musicxml`, Partitura emits two iterator variants: 9 notes ending at beat 18, then 6 notes ending at beat 10. Including the original folded score (7 notes, beat 14) yields three candidate hypotheses in order. The generator records note arrays and the corresponding input condition.
- Released facade quirk verified by source and execution: `unfold_score=True` gives the multi-fold class an already maximally unfolded score, collapsing this fixture to one unique hypothesis; `unfold_score=False` gives the three intended repeat hypotheses. Keep the public unfold flag behavior-compatible and make repeat-focused tests/benchmarks explicitly use the folded input.
- All candidates advance on every observation. Candidate cost is the distance at its current reference frame; selection uses the mean of at most the trailing 50 costs, preserves the first candidate on ties, and switches only when `best < current * (1 - 0.05)`. Output maps each candidate's unfolded frame beat through repeated note IDs back to the input score's notated beats.
- `generate_audio.py --multi-fold-only` wrote taken/skipped FluidSynth performances and reference features, candidate frame mappings, costs, selected indices and paths. Both cases use the verified CC0 SF2 (`CF2A98...AC2F`) and score fixture hash in their metadata; takes/skips consumed 362/202 observations. WAV hashes are recorded in `tests/data/audio/README.md`.
- Metric ground truth consists of unique score note-onset events in performance seconds with their corresponding folded-score beats. It does not duplicate each intermediate audio frame; the frame-dense diagnostic overweights repeated near-threshold positions.
- Rust's bounded repeat-path enumeration matches the Partitura candidates and order in a focused unit test. Synthetic replay, margin/tie, reset, short-score, CLI/Python interface, and diagnostic export tests pass. T010-T014 are complete; T015-T016 remain open.

### C02 - Initial parity and benchmark probe (2026-11-04)

- Rust and the pinned Python reference both ran the expanded audio manifest on one run per method/side. The default maximally unfolded score produced one candidate; the folded taken/skipped fixtures produced three each and consumed 362/202 observations.
- With the final note-onset TSVs, both repeat cases have matching median beat error and all six accuracy metrics. The benchmark report marks renderer accuracy parity `ok`; alignment position parity also passes (max path difference 0.05 beat on each repeat case). Strict metric identity remains false due renderer-dependent secondary statistics.
- Five serial Release runs per side are recorded under `target/bench/audio-methods-c02/`. The report has five records per piece/side, renderer accuracy parity `ok` for both repeat pieces, and no alignment-parity violation. The new method is not present in the previous master baseline; this Windows run is not an Ubuntu performance-regression claim.
- Final five serial Release runs per side are recorded under `target/bench/audio-methods-c02/`; the report has five records per piece/side. Rust feature/inference medians are 1.737/0.0041 ms per observation on the bundled score, 1.731/0.0126 ms for repeats-skipped, and 1.731/0.0131 ms for repeats-taken. Rust RTF is 0.05457, 0.05303, and 0.05313 respectively. Both repeat rows pass the 0.01 renderer-accuracy and 0.6-beat path gates; strict metric identity remains false for renderer-dependent secondary statistics. No prior same-method master baseline exists, so no Ubuntu regression claim is made.
- A one-run full audio workflow matrix produced six rows: Arzt/Dixon/Tempo-Arzt remain on the bundled score, Multi-Fold covers all three pieces, and all alignment checks pass. Full local Rust/CLI tests (including 30 doctests), all 41 Python tests, core/binding clippy, fmt, MSRV 1.88, WASM, cargo-deny, report tests, and C02 stage comparisons pass.
- C02 is commit `56c67d7db082f2757515023fab354bb6b895defc`, pushed as one coherent algorithm commit. PR #5 remains a draft and unmerged. T016 is complete after remote CI run `41997608783` and benchmark run `41997608781` both succeeded; the benchmark's four dataset jobs and report passed, with deployment skipped for the PR.

### C03 - Audio ground-truth reference captured (2026-11-04)

- `GroundTruth` TSV columns are `perf_sec` and `score_beat`; `evaluate_alignment` and its six beat thresholds remain the sole metric implementation. The fixture samples eight unique note onsets from the bundled rendered score using the reference `frame_to_beat` mapping.
- `tools/golden/generate_audio.py --audio-ground-truth-only` writes `tests/data/audio/ground_truth/score_onsets.tsv` and `tests/golden/audio/ground_truth/{ground_truth.tsv,ground_truth.npy,alignment_path.npy,evaluation.json,meta.json}`. Metadata pins pymatchmaker 0.4.0, Partitura 1.9.0, the locked environment, the score/performance hashes, and the verified CC0 SF2 hash.
- Reference metric golden: median 0.025 beats, mean 0.0354; accuracy is 0.75 at 0.05 beats, 0.875 at 0.1 beats, and 1.0 at each larger threshold. T017 is complete; T018-T023 remain open.

### C03 - Direct audio ground-truth integration (2026-11-04)

- The CLI accepts direct `perf_sec`/`score_beat` TSV with `--ground-truth` and `--export`, reusing `GroundTruth::load` and `evaluate_alignment`. Note-ID `.match` is rejected with a separate-MIDI-performance message; the audio path is never passed to the MIDI loader.
- Ground truth is parsed before any path/export writes. The malformed-TSV CLI test preserves a pre-existing output file. Valid follow writes `arzt/evaluation.json` and `arzt/evaluation_meta.json` with the exact source path and TSV format. Legacy audio exports remain covered.
- `tools/golden/compare.py` verifies identity and median/all six accuracy metrics against the locked golden. End-to-end bundled run matches all seven tested metric values exactly; the CLI test permits the existing 0.01 renderer threshold. Direct parser tests retain exact mapping/metric checks and malformed typed errors.
- Rust and Python audio benchmark records retain both the evaluation result and direct TSV identity; the C02 repeat benchmark exercises both repeat TSVs. `test_report.py` passes all five report checks. T017-T022 are complete; T023 remains open for commit, push, and its remote-CI gate.
- Final local Rust/CLI suite passes, including all 30 doctests; all 41 Python tests pass. Format, core/CLI and binding Clippy, MSRV 1.88, and WASM pass. `cargo deny check bans licenses sources` passes with the existing allowance/duplicate-syn warnings. Full advisory checking is currently blocked by proxy HTTP 407 fetching RustSec; C03 adds no dependency. C02 PR #5's remote CI and benchmark workflows are both green. T023 remains open until C03 is committed as one coherent increment; that push will start its remote CI.
- C03 full Rust/CLI rerun includes 4 audio-follow tests, 4 evaluator tests, 4 audio export/config tests, all prior golden/edge/benchmark targets, and 30/30 doctests. `maturin develop` and all 41 Python 3.13 tests pass; fmt, core/CLI and binding Clippy, MSRV 1.88, WASM, report tests (5/5), and deny bans/licenses/sources pass. Full advisory refresh is blocked by proxy HTTP 407, with no dependency change.
- Five serial Rust/reference runs per side for the bundled and direct-ground-truth audio rows are under `target/bench/audio-ground-truth-c03/`. On the GT row, feature/inference medians are 1.743/0.0040 ms Rust and 2.389/0.0714 ms Python (1.4x total speed-up); path parity is within audio tolerance, median/accuracy metrics match exactly, and the report marks GT accuracy `ok`.
- T023 remains open until the one C03 commit is created, pushed, and its remote CI completes; C04 does not begin before that boundary.
