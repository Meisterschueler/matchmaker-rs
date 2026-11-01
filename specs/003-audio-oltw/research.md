# Research: Audio Online Time Warping (arzt, dixon)

Phase 0 of `/speckit-plan`. The reference source was inspected in the installed pymatchmaker v0.4.0 environment (`tools/golden/uv.lock` pins the release). Dependency facts were checked against current crate documentation on 2026-11-01.

## R1. Reference method and processor defaults

| | Audio `arzt` | Audio `dixon` |
|---|---|---|
| Follower | `OnlineTimeWarpingArztFrame` (`matchmaker/dp/oltw_arzt.py`) | `OnlineTimeWarpingDixonFrame` (`matchmaker/dp/oltw_dixon.py`) |
| Default processor | `chroma` | `lse` |
| Sample rate | 44,100 Hz | 44,100 Hz |
| Default frame rate / hop | 30 fps / 1,470 samples | 50 fps / 882 samples |
| Window | 10 seconds, rounded to frames | 10 seconds, converted to frames |
| Warmup window | 0.1 seconds, rounded to frames | not applicable |
| Other defaults | `step_size = 3`, Manhattan distance | `max_run_count = 3`, Euclidean distance |

Both followers consume frame-level feature vectors and a score reference feature sequence. The score reference also has a beat value per reference frame. Positions are reported by mapping the selected reference frame through that beat sequence. The frame-level `arzt` search is a rolling two-column DTW recurrence with a step-size clamp. Frame-level `dixon` is a bounded, normalized-cost search over a sparse cost map and advances the score, input, or both.

**Decision**: Model audio method defaults separately from the existing MIDI `MethodConfig`, so `arzt` and `dixon` remain selectable under both input types without changing existing MIDI defaults or validation.

**Rationale**: The same user-facing method names have different frame/event algorithms, feature defaults, distance functions, and parameter units by input type.

**Alternatives considered**: Reuse the current MIDI method enum/config unchanged; that would conflate seconds and event counts and risk changing the tested MIDI API.

## R2. Audio loading, framing, and timestamps

`AudioStream` in `matchmaker/io/audio.py` uses `librosa.load(file_path, sr=None)`, which loads mono audio, then resamples to `target_sr` (44,100 Hz by default). It pads the final partial hop with zeros. The stream retains `n_fft - hop_length` samples of context when the processor provides `n_fft`; otherwise it retains one hop. The first call primes the cache and does not enqueue a result. Emitted `perf_time` is `emit_count * hop_length / sample_rate`.

The upstream extension check accepts `.wav`, `.mp3`, `.flac`, `.aac`, `.ogg`, and `.m4a`, case-insensitively; successful decoding depends on the installed librosa backend. The checked Rust decoder path must use a documented pure-Rust format set and return a typed error for unsupported containers. Implementation-time verification with `cargo info symphonia@0.6.1` corrected the initial research: Symphonia does offer `isomp4` and `aac` features. They remain disabled because AAC/MP4 containers are outside this feature's agreed initial decoder set, not because the crate lacks them.

**Decision**: Decode prerecorded files through Symphonia to mono `f32`, resample non-44.1-kHz inputs with Rubato, then process fixed hops with the same cache, timestamp, and final-padding rules. Initially document and test WAV, FLAC, MP3, and Ogg/Vorbis; reject unsupported AAC/MP4 containers with an actionable error unless a pure-Rust decoder path is proven during implementation.

**Rationale**: This retains the reference's target sample rate and offline behavior without introducing C/C++ dependencies. WAV/MP3/FLAC/Ogg cover the checked reference assets and common open container paths.

**Alternatives considered**: Link to system FFmpeg/libsndfile for every upstream format (violates the default-build constraint); ship a decoder that silently falls back to system codecs (not reproducible).

## R3. Feature processor definitions

`ChromagramProcessor` in `matchmaker/features/audio.py` calls `librosa.feature.chroma_stft` with `center=False`, `n_fft = 2 * hop_length` unless overridden, 12 bins and float32 output. Per input hop it yields a `(frames, 12)` matrix and the input timestamp. At the default 30 fps this is a 2,940-sample FFT.

`LogSpectralEnergyProcessor` uses an approximately 46-ms Hamming-windowed STFT (`n_fft = 2 ** round(log2(2048 / 44100 * sample_rate))`), groups squared magnitudes into a linear/log frequency scale capped at MIDI 127, takes a half-wave-rectified first difference, and normalizes each frame with L2 norm by default. At 44.1 kHz it yields 84 float32 feature values per frame. It is stateful and must be reset between reference-score and performance processing.

**Decision**: Implement processors as separate modules using existing `realfft`; preserve upstream windows, centering, bin mapping, normalization, state reset, timestamps, and float32 output. Golden tests compare raw feature arrays before testing follower parity.

**Rationale**: Audio processor errors propagate into every frame-level alignment decision; isolated feature goldens make those failures diagnosable.

**Alternatives considered**: Use different generic chroma/LSE libraries; their filter banks, FFT conventions, and normalization would add unmeasured differences.

## R4. Score-side reference features and renderer

The registry's `score_audio` reference builder calls `generate_score_audio(score_part, tempo, sample_rate)`, casts to float32, runs the same processor as the performance, and resets the processor. `generate_score_audio` uses Partitura's FluidSynth renderer with its default MuseScore General SF3 soundfont, pads before the first score onset, and crops 0.1 seconds beyond the final onset. `_build_ref_frame_to_beat` converts each reference frame to beat using the score's beat map, tempo, PPQ, and frame rate.

`rustysynth` is a maintained, pure-Rust synthesizer that reads SF2 SoundFonts; it does not establish support for Partitura's default SF3 asset. The project has no SoundFont asset. The reference package license is Apache-2.0, but that does not establish rights for its bundled MP3s or Partitura's separately distributed soundfont.

**Decision**: Use RustySynth with an explicitly configured, pinned SF2 SoundFont; do not bundle or download an unreviewed asset automatically. Record the chosen asset's URL, checksum, and license before implementation. Apply the reference's score timing/padding/cropping rules. Compare audio features and end-to-end alignment to reference tolerances and evaluation metrics; do not claim waveform parity across FluidSynth and RustySynth.

**Rationale**: The Constitution allows score-synthesis differences only when the measured alignment metrics and deviation are documented. A user-configured soundfont avoids silent redistribution or network-download assumptions.

**Alternatives considered**: Link FluidSynth (C dependency, violates the default pure-Rust build); use a sine oscillator (would make the synthesized reference materially unlike a piano); bundle the upstream SF3 or MP3 assets without checking their separate asset licenses.

## R5. Decoder and resampler dependencies

- **Symphonia 0.6.1**: pure Rust, feature-selectable decoders/demuxers for WAV, FLAC, MP3, and Ogg/Vorbis. Its MPL-2.0 license is compatible for distribution in a larger Apache-2.0 work under MPL terms (Mozilla MPL 2.0 FAQ, Q13), but the repository's `cargo-deny` allow list must explicitly include it. Its WASM build must be verified; pure Rust alone does not guarantee every target configuration.
- **Rubato 5.0.1**: pure-Rust resampling, MIT OR Apache-2.0, MSRV 1.87 (below this workspace's Rust 1.88). Use fixed-ratio offline resampling. Its output is not assumed bit-identical to the `soxr_hq` backend used by the installed librosa; golden fixtures should use 44.1-kHz input to isolate feature parity, and resampled inputs need a separately documented tolerance.
- **RustySynth 1.3.8**: pure Rust, MIT, reads SF2, and has no transitive dependencies. It is suitable for the existing default-build constraint once an appropriately licensed SF2 asset is pinned.
- **realfft 3.5.1**: already present and used by the project; reuse it for the STFT work rather than adding another FFT implementation.

**Decision**: Use Symphonia, Rubato, RustySynth, and the existing realfft, with only required Symphonia features enabled and an explicit MPL-2.0 allow-list entry. Check `cargo deny`, native tests, and `wasm32-unknown-unknown` before accepting the stack.

**Rationale**: This is the smallest single decoder/resampler/synth stack that covers the initial prerecorded workflow with no C/C++ build dependency and compatible licenses.

**Alternatives considered**: Hound alone (WAV only); many format-specific decoders (more dependencies and inconsistent behavior); Symphonia plus a system FFmpeg backend (not a reproducible pure-Rust default).

## R6. Golden fixtures and validation scope

The reference distribution includes MP3 assets, but they are not checked into this repository, and no separate asset license was established during this planning research. `tools/golden/generate.py` currently builds MIDI-only goldens. The installed pymatchmaker reference is pinned and has an Apache-2.0 package license; score audio relies on Partitura's optional FluidSynth and separately licensed soundfont.

**Decision**: Add a deterministic, redistributable mono WAV fixture at 44.1 kHz generated from a short score/performance pair using a SoundFont whose license is verified and recorded. Generate processor goldens for `chroma` and `lse`; generate full reference runs for audio `arzt` and `dixon` using their default processor pairs. Test each follower by replaying reference feature sequences and test the full pipeline separately. Add benchmark pieces only after matching audio/score annotation provenance is established.

**Rationale**: This keeps CI offline and avoids copying an unverified upstream MP3. Separate feature-replay and end-to-end tests identify whether a deviation comes from synthesis, decoding, features, or alignment.

**Alternatives considered**: Use package MP3s without provenance (licensing risk); download a benchmark recording in CI (network, license, and reproducibility risk); use only hand-constructed feature matrices (does not exercise the audio pipeline).

## Decisions that constrain the design

1. Audio methods have their own method/config model and preserve MIDI API/defaults.
2. Initial audio files are prerecorded; no live capture/device I/O is included.
3. Initial decoder set is WAV, FLAC, MP3, and Ogg/Vorbis; other reference extensions return typed unsupported-format errors.
4. Audio input is normalized to mono and 44.1 kHz; source-rate conversion is explicitly tested separately from 44.1-kHz golden fixtures.
5. Runtime score synthesis requires a configured, pinned SF2 asset; asset license and checksum are release gates.
6. Audio fidelity from RustySynth is not treated as waveform parity with FluidSynth; alignment parity/evaluation is the acceptance measure for that stage.

## Implementation verification (2026-11-01)

### T001: SoundFont provenance blocked

The user supplied a small SF2 archive named
`UprightPianoKW-small-SF2-20190703.7z`. Its included readme identifies version
2019-07-03 and its included `cc0.txt` accompanies a statement that the sound bank
is published under CC0. The archive passed a 7-Zip integrity test. Archive and
extracted SF2 hashes, sizes, and provenance are recorded in
[the audio fixture provenance document](../../tests/data/audio/README.md).
The current publisher page identifies the bank as 2022-02-21, whereas this
archive explicitly identifies itself as 2019-07-03. Pin and provision the exact
2019 archive by its filename and SHA-256; do not silently substitute another
revision. GeneralUser GS was rejected because its license explicitly leaves
some sample origins uncertain.

T001 is complete for this exact asset pin. The SoundFont itself remains outside
the repository; derived audio fixtures and reference goldens are tracked under
`tests/data/audio/` and `tests/golden/audio/` after T011.

### T003: Locked audio generator

`tools/golden/generate_audio.py` is implemented for all four `arzt`/`dixon` and
`chroma`/`lse` combinations. It verifies the pinned SF2 SHA-256, records the
reference and renderer hashes, and emits feature arrays, frame times, score
beat mapping, positions, and alignment paths. Python syntax, CLI help, and the
asset-hash check pass in the locked environment. Full score rendering requires
the native FluidSynth library; processor-only synthetic references do not.
T011 later generated the rendered WAV and all four end-to-end reference outputs.
The optional FluidSynth prerequisite is not a Rust runtime dependency.

### T004: Audio core types

Added separate `AudioMethod`, `AudioProcessorKind`, `AudioDistance`, and
`AudioMethodConfig` types with the reference's arzt/chroma/30-fps/Manhattan and
dixon/lse/50-fps/Euclidean defaults. Added typed audio decode, render, and
configuration errors; existing MIDI method and config defaults are unchanged.
`cargo test --test method_config` passes (13 tests), as does
`cargo fmt --all -- --check`. Audio stream and processor behavior are recorded
under T006-T008.

### T006: Audio file stream

Added Symphonia decoding for WAV, FLAC, MP3, and Ogg/Vorbis, channel averaging,
Rubato FFT resampling, first-hop cache priming, zero-padded final hops, and
reference-compatible emitted timestamps. The local tests cover generated WAV,
FLAC, MP3, and Ogg/Vorbis fixtures, stereo downmixing, rate conversion, cache
priming, padding, empty/corrupt input, and timestamps. `cargo test --test
audio_io --test method_config` passes (19 tests), as do library Clippy and fmt
checks. The compressed-format fixtures are synthetic and generated by the locked
`soundfile` dependency; no third-party recording is used. T005 is complete.

### T005: Decoder and framing tests

Added four generated 44.1-kHz mono tone files and exercised each through
Symphonia. The exact byte sizes and hashes are printed by
`generate_audio.py --codec-fixtures-only`; fixture source and format coverage are
documented in the audio provenance README.

### T007: Synthetic processor references

Extended `generate_audio.py --processors-only` to produce a deterministic
three-tone signal and locked pymatchmaker outputs for `chroma`/`lse` at both
method frame rates. `tests/golden_audio_processors.rs` checks feature arrays,
frame counts and times for all four combinations, plus LSE reset and silence.
The checked-in comparisons pass at the documented `2e-4` relative and `1e-6`
absolute feature tolerances.

### T008: Audio feature processors

Added the Librosa-compatible chroma STFT/filterbank with per-frame tuning and
stateful LSE FFT/frequency mapping, normalization, reset, and DEBUG spans.
`AudioStream::prime` runs the first block through the processor without yielding
an alignment observation, preserving the reference's LSE history. Processor
goldens pass for all four method/processor configurations.

### T009/T010: Synthesized score reference

Added `ScoreAudioReference` and an explicit SF2-backed RustySynth renderer. The
builder processes the rendered samples through the selected audio processor,
keeps per-frame beat mapping and score onset positions, and adds a DEBUG render
span. Tests verify the local pinned SF2 render, onset padding, final 0.1-second
crop, feature/map cardinality, and deterministic processor reset. The local SF2
tests pass; they skip when that ignored local asset is absent. WASM, Rust 1.88,
format, clippy, and the 22 focused audio/MIDI regression tests pass. T011 has
generated the FluidSynth reference outputs; waveform and end-to-end parity
between FluidSynth and RustySynth remains unmeasured.

### T011: FluidSynth reference fixtures and goldens

The locked Python reference rendered
`tests/data/audio/simple_mozart_k265_var1_sf2.wav` with the pinned SF2 and
FluidSynth 2.6.2. The official Windows x64 FluidSynth archive was SHA-256
verified before use; its hash, along with the WAV hash, is recorded in
`tests/data/audio/README.md`. `tests/golden/audio/` contains all four
`arzt`/`dixon` and `chroma`/`lse` runs, including performance features/times,
score features, frame-to-beat mapping, positions, and alignment paths. Metadata
records the native FluidSynth version and the Python reference/SoundFont pins.
Focused validation passed: audio I/O 6/6, processor goldens 2/2, score-audio
tests 2/2, and MIDI score goldens 3/3. RustySynth-to-FluidSynth parity remains a
later follower acceptance check.

### T002: Dependency and target checks complete

- Symphonia 0.6.1: MPL-2.0, declared MSRV 1.85; defaults disabled, with only
	`flac`, `mp3`, `ogg`, `pcm`, `vorbis`, and `wav` enabled.
- Rubato 5.0.1: MIT OR Apache-2.0, declared MSRV 1.87; defaults disabled and
	`fft_resampler` enabled for fixed-ratio offline resampling using existing
	realfft. RustySynth 1.3.8: MIT, no transitive dependencies.
- `cargo check -p matchmaker-rs --lib`: PASS.
- `cargo build -p matchmaker-rs --target wasm32-unknown-unknown`: PASS.
- `cargo +1.88 check -p matchmaker-rs -p matchmaker-cli --all-targets`: PASS.
- `cargo deny check licenses bans sources`: PASS, with warnings for unused
	license allowances and duplicate `syn` major versions.
- `cargo tree -p matchmaker-rs --edges normal,build`: inspected the default
	dependency graph; the new audio stack adds no C/C++ binding or compiler crate.
- Existing `.github/workflows/ci.yml` already covers native builds, MSRV 1.88,
	WASM, and cargo-deny; no redundant workflow changes were needed.

The full `cargo deny check` did not reach advisory validation: fetching the
RustSec database failed with proxy HTTP 407. The full advisory gate remains
unverified and must be rerun when network access is available. Processor goldens
and local SF2 score-render smoke tests now pass; FluidSynth waveform and
end-to-end alignment parity have not been compared, and the benchmark has not
run.
