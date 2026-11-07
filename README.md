# matchmaker-rs

[![CI](https://github.com/Meisterschueler/matchmaker-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Meisterschueler/matchmaker-rs/actions/workflows/ci.yml)
[![Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2FMeisterschueler%2Fmatchmaker-rs%2Fmaster%2FCargo.toml&query=%24.package.version&label=version)](Cargo.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Benchmark](https://img.shields.io/endpoint?url=https%3A%2F%2Fmeisterschueler.github.io%2Fmatchmaker-rs%2Fbadge.json)](https://meisterschueler.github.io/matchmaker-rs/)

Pure-Rust port of [pymatchmaker](https://github.com/pymatchmaker/matchmaker), a library for real-time music alignment (score following).

Status: work in progress, developed spec-driven with [Spec Kit](https://github.com/github/spec-kit) (see `specs/`).

## Quickstart

```text
git clone https://github.com/Meisterschueler/matchmaker-rs
cd matchmaker-rs
cargo run --release -p matchmaker-cli -- follow tests/data/simple_mozart_k265_var1.musicxml tests/data/simple_mozart_k265_var1.mid --output path.tsv
```

Prints `perf_time<TAB>beat` per position and writes the alignment path to `path.tsv`.
`cargo test --workspace --exclude matchmaker-py` checks parity with the reference;
[quickstart.md](specs/001-midi-score-following/quickstart.md) has all scenarios.

## Methods and processors

| Method | Input | Processor | Note |
|---|---|---|---|
| `pthmm` (default) | MIDI | `pitch`, `chord_onset` | pitch-vector HMM |
| `hmm` | MIDI | `pitch`, `chord_onset` | pitch lists, tempo-aware (`beat_period`) |
| `outerhmm` | MIDI | `pitch`, `chord_onset` | outer-product HMM with pitch-error model |
| `arzt` | MIDI | `chord_onset`, `pitch` | event-level online time warping over the onset pianoroll |
| `dixon` | MIDI | `chord_onset`, `pitch` | event-level online time warping with sparse path-normalized costs |
| `arzt` | Audio | `chroma`, `lse`, `raw_spectrum`, `cqt` | frame-level online time warping; defaults to `chroma` at 30 fps |
| `dixon` | Audio | `chroma`, `lse`, `raw_spectrum`, `cqt` | frame-level online time warping; defaults to `lse` at 50 fps |
| `arzt_tempo` | Audio | `chroma`, `lse` | seeded tempo-aware Arzt; defaults to `chroma` at 30 fps |
| `arzt_multi_fold` | Audio | `chroma`, `lse` | parallel repeat hypotheses; defaults to `chroma` at 30 fps, 50-cost window and 5% switch margin |
| `skf` | Audio | `raw_spectrum` | switching Kalman filter over chords with hidden tempo; 8000 Hz, hop 128, FFT 512 |

Select with `--method` and `--processor`; override parameters with `--set KEY=VALUE`.
For `arzt_multi_fold`, pass `--no-unfold` when the score has repeats so the follower receives
the folded score and can compare repeat hypotheses. With the default maximal unfolding, this
reference-compatible method may have only one unique candidate.
Tempo-aware Arzt uses seed 1984 by default; `--set random_seed=VALUE` selects another
reproducible MT19937 stream. Its export includes tempo state and accumulated cost windows.
`skf` follows chords, not a rendered score, and compares every frame with one spectral template
per chord. The template mode is explicit and recorded in the export (`score/meta.json`):
`synthesized` (default; chords rendered with the SoundFont, so `--soundfont-path` is required) or
`gaussian` (harmonic peaks, no SoundFont; far less accurate): `--set template_mode=gaussian`.
The reference falls back to Gaussian peaks silently on any synthesis error; this port reports the
error instead. See research R12 in
[specs/004-audio-methods/research.md](specs/004-audio-methods/research.md).
`cqt` is the magnitude of `librosa.cqt` over the 88 piano keys (A0 to C8). It keeps 44100 Hz and
takes its hop from `frame_rate` (1470 for `arzt`, 882 for `dixon`); `norm` and `n_fft` are not
options, and a configuration for which the reference would downsample by more than 2 before the
transform (above about 70 kHz with a hop that is a multiple of 512) is rejected.
`cqt_spectral_flux` appends the half-wave rectified flux against the previous call to every
`cqt` row (89 values, 88 without the flux). The reference keeps the rendered score's CQT matrix as
flux history (its `reset` is a no-op) and then fails on the first performance frame, so audio
`arzt` and `dixon` cannot follow with it and `--processor cqt_spectral_flux` is rejected there:
it is for feature extraction and for the audio `outerhmm` that is not ported yet. A processor
keeps its history across `reset`, so use one per run.
Audio files support WAV, FLAC, MP3 and Ogg/Vorbis. For score synthesis, provision the pinned
CC0 SF2 SoundFont described in [tests/data/audio/README.md](tests/data/audio/README.md), then run:

```text
cargo run --release -p matchmaker-cli -- follow score.musicxml performance.wav --input-type audio --method arzt --soundfont-path target/audio-assets/UprightPianoKW-small-SF2-20190703/UprightPianoKW-small-20190703.sf2 --output path.tsv --export export/
```

Standalone feature extraction does not require a score or SoundFont. `raw_spectrum` defaults to
8000 Hz, hop 128, and FFT size 512, `cqt` and `cqt_spectral_flux` to 44100 Hz and hop 1470; use
`--sample-rate`, `--hop-length`, (`raw_spectrum` only) `--n-fft` and (`cqt_spectral_flux` only)
`--no-spectral-flux` for overrides:

```text
cargo run --release -p matchmaker-cli -- features tests/data/audio/simple_mozart_k265_var1_sf2.wav --processor raw_spectrum --export target/raw-spectrum/
cargo run --release -p matchmaker-cli -- features tests/data/audio/simple_mozart_k265_var1_sf2.wav --processor cqt --export target/cqt/
cargo run --release -p matchmaker-cli -- features tests/data/audio/simple_mozart_k265_var1_sf2.wav --processor cqt_spectral_flux --export target/cqt-flux/
```

Audio `--ground-truth` accepts a direct TSV with `perf_sec` and `score_beat` columns and writes
the reference metrics to `<method>/evaluation.json` with `--export`. Note-ID `.match` files need
a separate MIDI performance and are rejected by audio follow. The runtime is pure Rust;
FluidSynth is only needed to generate Python reference fixtures and benchmark records. The CI workflows
download the pinned FreePats archive from a fixed HTTPS URL, verify its archive and extracted SF2
hashes, and provide the SoundFont to audio tests as a temporary artifact.

```text
cargo run --release -p matchmaker-cli -- follow tests/data/simple_mozart_k265_var1.musicxml tests/data/audio/simple_mozart_k265_var1_sf2.wav --input-type audio --method arzt --soundfont-path target/audio-assets/UprightPianoKW-small-SF2-20190703/UprightPianoKW-small-20190703.sf2 --ground-truth tests/data/audio/ground_truth/score_onsets.tsv --export target/audio-follow/
```

## Python

`Matchmaker` supports MIDI and prerecorded audio input (needs Python 3.10–3.13 and `maturin`):

```text
cd crates/matchmaker-py
python -m uv run --with maturin maturin develop --release
```

```python
from matchmaker_rs import Matchmaker
mm = Matchmaker("score.musicxml", "performance.mid", input_type="midi", method="pthmm")
positions = list(mm.run(verbose=False))
```

Audio `arzt`, `dixon`, `arzt_tempo`, `arzt_multi_fold` and `skf` use the same generator API. Supply the pinned SF2 path in `kwargs` (`skf` accepts `template_mode="gaussian"` instead). For repeat hypotheses, set `unfold_score=False`; the default maximal unfolding may leave one unique candidate:

```python
mm = Matchmaker(
  "score.musicxml",
  "performance.wav",
  input_type="audio",
  method="dixon",
  kwargs={"soundfont_path": "target/audio-assets/UprightPianoKW-small-SF2-20190703/UprightPianoKW-small-20190703.sf2"},
)
positions = list(mm.run(verbose=False))
alignment_path = mm.score_follower.alignment_path
mm.export("export/")
```

Standalone feature arrays and their timestamps are available without constructing a follower:

```python
from matchmaker_rs import extract_audio_features

features, times = extract_audio_features("performance.wav", "raw_spectrum")  # or "cqt", "cqt_spectral_flux"
# cqt_spectral_flux without the flux column (88 values): kwargs={"include_spectral_flux": False}
```

## Parity and accuracy

Golden tests compare against the reference with documented tolerances; each comparison reports
its class (`EXACT`, `WITHIN_TOL`, `EXCEEDS`) and the maximum deviation, see
[golden-data.md](specs/001-midi-score-following/contracts/golden-data.md#deviation-classes).

The score renderer initializes Reverb Send (CC91) to zero, matching the pinned FluidSynth
reference rather than RustySynth's default 40. It also peak-normalizes the rendered score to
±1.0 and extracts score features in one whole-signal pass (frame `j` starts at `j*hop`, chroma
tuning is estimated once for the whole score), exactly like the reference; the performance stream
still frames per hop with a cache. Both conventions matter for level-sensitive processors such as
`raw_spectrum`. Each is a single function in `src/score/audio.rs` (`normalize_level`,
`extract_score_features`) and can be replaced locally, see research R11 in
[specs/004-audio-methods/research.md](specs/004-audio-methods/research.md).
Renderer metrics are checked separately from exact feature-replay parity, see
[the audio contract](specs/004-audio-methods/contracts/audio-api.md).
`cqt` features are compared with `|a - b| <= 1e-5 + 1e-5 * |ref| + 1e-5 * max(frame)`: the
reference computes in `float32` and halves the sample rate with `soxr`, which leaves noise that
scales with the largest bin of a frame and that an implementation that is not bit-identical cannot
match per element. On the bundled performance the deviation is at most 3 % of that limit for the
bins and 7 % for the flux column of `cqt_spectral_flux` (research R13, C07).

## Development

Enable the pre-commit hook once per clone; it runs the CI lint gates (`cargo fmt --check`,
`cargo clippy -D warnings`) before every commit:

```text
git config core.hooksPath .githooks
```

## Reference implementation

- pymatchmaker release: tag [`v0.4.0`](https://github.com/pymatchmaker/matchmaker/tree/v0.4.0).
  Only released tags are used; see the constitution (`.specify/memory/constitution.md`, I).
- Python versions for the bindings follow its `requires-python` (currently 3.10–3.13).

## Golden data

`tests/golden/` holds reference outputs generated by the release above, from the inputs in
`tests/data/` (layout: `specs/001-midi-score-following/contracts/golden-data.md`). The Python
environment is locked in `tools/golden/uv.lock`. To regenerate (Windows needs a C compiler for
pymatchmaker's Cython extensions):

```text
cd tools/golden
python -m uv sync --locked
python -m uv run generate.py --data ../../tests/data --out ../../tests/golden
```

Moving to a newer reference release: bump the tag in `tools/golden/pyproject.toml`, run
`python -m uv lock`, regenerate, and review the diff.

## Roadmap

1. MIDI score following: methods `pthmm`, `hmm`, `outerhmm`, `arzt`, `dixon`; processors `pitch`, `chord_onset`.
2. Audio score following: frame-level `arzt`, `dixon`, `arzt_tempo`, `arzt_multi_fold`, `skf`; processors `lse`, `chroma`, `raw_spectrum`, `cqt` (implemented; `cqt_spectral_flux` for extraction only).
3. Later: remaining methods (audio `outerhmm`, `pfkorz`) and parangonar methods.

## Benchmark datasets

The [Benchmark workflow](.github/workflows/benchmark.yml) compares Rust and the Python reference
(latency, accuracy, alignment parity) and publishes the
[report](https://meisterschueler.github.io/matchmaker-rs/) on GitHub Pages; the benchmark badge
above shows the median speed-up over all pieces of the latest `master` run (red on a parity
violation). Datasets are
downloaded in CI at pinned commits ([tools/datasets/datasets.toml](tools/datasets/datasets.toml),
[fetch.py](tools/datasets/fetch.py)); they are not part of this repository and keep their own
licenses — only derived metrics are published:

The bundled audio fixture can be benchmarked with `workflow_dispatch` by supplying a direct
HTTPS URL to the pinned CC0 SF2 SoundFont; CI verifies its SHA-256 and does not retain the asset.

- [(n)ASAP](https://github.com/CPJKU/asap-dataset) (CC BY-NC-SA 4.0): S. D. Peter et al.,
  TISMIR 2023, doi:10.5334/tismir.149
- [Batik-plays-Mozart](https://github.com/huispaty/batik_plays_mozart) (CC BY-NC-SA 4.0):
  P. Hu and G. Widmer, ISMIR 2023
- [Vienna 4x22](https://github.com/CPJKU/vienna4x22) (CC BY 4.0): W. Goebl, 1999,
  doi:10.21939/4X22
- The bundled Mozart K265 examples for fast tests.

## Paper

Park, Cancino-Chacón, Chiruthapudi, Nam: *Matchmaker: An Open-Source Library for Real-Time Piano Score Following and Systematic Evaluation*, ISMIR 2025. [arXiv:2510.10087](https://arxiv.org/abs/2510.10087)
