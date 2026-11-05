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
| `arzt` | Audio | `chroma`, `lse` | frame-level online time warping; defaults to `chroma` at 30 fps |
| `dixon` | Audio | `chroma`, `lse` | frame-level online time warping; defaults to `lse` at 50 fps |
| `arzt_tempo` | Audio | `chroma`, `lse` | seeded tempo-aware Arzt; defaults to `chroma` at 30 fps |
| `arzt_multi_fold` | Audio | `chroma`, `lse` | parallel repeat hypotheses; defaults to `chroma` at 30 fps, 50-cost window and 5% switch margin |

Select with `--method` and `--processor`; override parameters with `--set KEY=VALUE`.
For `arzt_multi_fold`, pass `--no-unfold` when the score has repeats so the follower receives
the folded score and can compare repeat hypotheses. With the default maximal unfolding, this
reference-compatible method may have only one unique candidate.
Tempo-aware Arzt uses seed 1984 by default; `--set random_seed=VALUE` selects another
reproducible MT19937 stream. Its export includes tempo state and accumulated cost windows.
Audio files support WAV, FLAC, MP3 and Ogg/Vorbis. For score synthesis, provision the pinned
CC0 SF2 SoundFont described in [tests/data/audio/README.md](tests/data/audio/README.md), then run:

```text
cargo run --release -p matchmaker-cli -- follow score.musicxml performance.wav --input-type audio --method arzt --soundfont-path target/audio-assets/UprightPianoKW-small-SF2-20190703/UprightPianoKW-small-20190703.sf2 --output path.tsv --export export/
```

The runtime is pure Rust; FluidSynth is only needed to generate Python reference fixtures and
benchmark records. Audio `--ground-truth` evaluation is not implemented yet. The CI workflows
download the pinned FreePats archive from a fixed HTTPS URL, verify its archive and extracted SF2
hashes, and provide the SoundFont to audio tests as a temporary artifact.

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

Audio `arzt`, `dixon`, `arzt_tempo` and `arzt_multi_fold` use the same generator API. Supply the pinned SF2 path in `kwargs`. For repeat hypotheses, set `unfold_score=False`; the default maximal unfolding may leave one unique candidate:

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

## Parity and accuracy

Golden tests compare against the reference with documented tolerances; each comparison reports
its class (`EXACT`, `WITHIN_TOL`, `EXCEEDS`) and the maximum deviation, see
[golden-data.md](specs/001-midi-score-following/contracts/golden-data.md#deviation-classes).

The score renderer initializes Reverb Send (CC91) to zero, matching the pinned FluidSynth
reference rather than RustySynth's default 40. Tempo-Arzt uses batch score tuning and
unprimed score FFT windows; existing Arzt/Dixon score framing is retained for compatibility.
Renderer metrics are checked separately from exact feature-replay parity, see
[the audio contract](specs/004-audio-methods/contracts/audio-api.md).

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
2. Audio score following: frame-level `arzt`, `dixon`, `arzt_tempo`, `arzt_multi_fold`; processors `lse`, `chroma` (implemented).
3. Later: remaining methods (audio `outerhmm`, `skf`, `pfkorz`) and parangonar methods.

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
