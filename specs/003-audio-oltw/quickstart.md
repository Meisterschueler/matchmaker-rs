# Audio OLTW Quickstart

This guide validates prerecorded audio following on the bundled fixture after the feature is implemented.

## Prerequisites

- Rust stable matching the workspace MSRV (1.88 or newer).
- The locked Python 3.12 environment in `tools/golden/` for reference golden generation.
- A pinned, license-verified SF2 SoundFont available at a local path. Pass it as `--soundfont-path` or `kwargs["soundfont_path"]`; the project does not download it automatically.
- A fixture performance in a supported WAV, FLAC, MP3, or Ogg/Vorbis container and its MusicXML/MIDI score.

## Generate Reference Goldens

From the repository root:

```text
cd tools/golden
python -m uv run --locked generate_audio.py --data ../../tests/data/audio --out ../../tests/golden/audio
```

Expected result: processor arrays, frame timestamps, score reference arrays, and follower outputs for the two methods are written under `tests/golden/audio/`. The generator records the pymatchmaker tag, lockfile hash, method config, and comparison tolerances.

## Run Rust Parity Tests

```text
cargo test --test golden_audio_processors
cargo test --test golden_audio_arzt
cargo test --test golden_audio_dixon
```

Expected result: feature frames and timestamps match their goldens, replayed observations match frame-follower outputs within documented tolerances, and end-to-end tests report renderer deviations with the first differing frame and evaluation metrics.

## Run the CLI

```text
cargo run --release -p matchmaker-cli -- follow tests/data/simple_mozart_k265_var1.musicxml tests/data/audio/simple_mozart_k265_var1.wav --input-type audio --method arzt --soundfont-path path/to/soundfont.sf2 --output path.tsv --export target/audio-export
```

Expected result: `path.tsv` contains performance seconds and score beats; the export directory contains audio frame, feature, score-reference, beat-map, and alignment data readable by NumPy.

Repeat with `--method dixon` to validate the second follower. Run the Python facade tests after `maturin develop` and use the equivalent `Matchmaker(..., input_type="audio", method="dixon", kwargs={"soundfont_path": ...})` call.

## Platform and Quality Gates

```text
cargo fmt --check
cargo clippy --workspace --exclude matchmaker-py --all-targets --all-features -- -D warnings
cargo test --workspace --exclude matchmaker-py --no-fail-fast
cargo deny check
cargo build -p matchmaker-rs --target wasm32-unknown-unknown
```

Expected result: existing MIDI gates remain green, audio decoding/rendering uses no system C/C++ libraries, and the default core still builds for WASM. The benchmark report compares the same fixture inputs over at least five runs.
