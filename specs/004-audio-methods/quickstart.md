# Quickstart Validation: Remaining Audio Methods and Features

These scenarios become runnable as their story is implemented. No new method or extraction command exists merely because it is listed here.

## Prerequisites

- Rust stable and MSRV 1.88; wasm32-unknown-unknown target, clippy, rustfmt, cargo-deny.
- Locked reference environment from tools/golden, Python binding environment from crates/matchmaker-py, verified SoundFont provisioned as in tests/data/audio/README.md.
- Current story's checked-in score/audio/golden fixtures and permitted template assets. Check research.md for asset and CQT gates.
- Explicit authorization to implement and create the twelve separate commits; this planning request does neither.

On this Windows GNU setup, before Cargo:

```powershell
$env:Path = "$env:USERPROFILE\tools\mingw64\bin;$env:USERPROFILE\.cargo\bin;" + $env:Path
```

## Per-story focused checks

Run from the repository root after the named test target is created. Processor goldens stay in the existing golden_audio_processors target; method replay tests get one new target per method.

| Commit | Focused Rust command | Additional scenario |
|---|---|---|
| C01 | cargo test --test golden_audio_arzt_tempo | Accelerating/slowing audio; tempo reset; existing Arzt regression |
| C02 | cargo test --test golden_audio_arzt_multi_fold | Repeat taken/skipped, switch-margin ties, selected hypothesis export |
| C03 | cargo test -p matchmaker-cli --test follow_audio ground_truth | Direct TSV reference metrics; corrupt input; existing output preserved |
| C04 | cargo test --test golden_audio_processors raw_spectrum | 257 default bins; standalone extraction through all interfaces |
| C05 | cargo test --test golden_audio_skf | Identical-template replay; synthesized/fallback mode; degenerate likelihood |
| C06 | cargo test --test golden_audio_processors cqt | CQT candidate probe, 88 bins, edge trim, rates/hops |
| C07 | cargo test --test golden_audio_processors cqt_spectral_flux | 89/88 bins, initial zero flux, matrix history/reset |
| C08 | cargo test --test golden_audio_outerhmm | Permitted template overlay, sound/pause posteriors, MIDI regression |
| C09 | cargo test --test golden_audio_processors mel | 128 bands; power/norm; compatible Arzt/Dixon runs |
| C10 | cargo test --test golden_audio_processors mfcc | 13 coefficients; reference default norm compatibility |
| C11 | cargo test --test golden_audio_processors korzeniowski | Structured export; direct/facade hop distinction; silence dB floor |
| C12 | cargo test --test golden_audio_pfkorz | Exact fixed-seed Rust repeat; fixed-draw substeps; five-seed metrics |

Expect EXACT or WITHIN_TOL under contracts/audio-api.md; EXCEEDS blocks the current story. Every focused filter must execute at least one test, not pass by selecting zero tests. Run CLI/Python current-story tests and old method regression tests next.

After C01, a CLI smoke run uses the existing verified SoundFont path:

```text
cargo run -p matchmaker-cli -- follow tests/data/simple_mozart_k265_var1.musicxml tests/data/audio/simple_mozart_k265_var1_sf2.wav --input-type audio --method arzt_tempo --soundfont-path target/audio-assets/UprightPianoKW-small-SF2-20190703/UprightPianoKW-small-20190703.sf2 --output target/tmp/arzt_tempo.tsv --export target/tmp/arzt_tempo/
```

Use the fixture filename recorded by the generator if it differs; do not fabricate a missing asset. The method and export must report positions, performance seconds/score beats, and tempo state. C03 adds --ground-truth with its checked-in direct TSV. C04 adds standalone `features PERFORMANCE --processor raw_spectrum --export DIRECTORY`; subsequent processors extend that same path.

## Full gate before each commit

```text
cargo fmt --all --check
cargo clippy --workspace --exclude matchmaker-py --all-targets --all-features -- -D warnings
cargo test --workspace --exclude matchmaker-py --all-features --no-fail-fast
cargo build -p matchmaker-rs --target wasm32-unknown-unknown
cargo +1.88 check -p matchmaker-rs -p matchmaker-cli --all-targets
cargo deny check
```

Build the bindings with the existing maturin workflow and run pytest tests from crates/matchmaker-py; CI covers Python 3.10/3.13 and linting of the binding crate. Run the repository quickstart/CI workflows for Linux/Windows/macOS; local Windows results alone do not establish all-platform success.

Extend and invoke the existing tools/bench/audio.toml manifest, reference runner, and report workflow for the current story. At least five identical-input repetitions; no tracing subscriber during timing. Publish feature latency, inference latency, real-time factor, accuracy/error metrics, and >10% slowdown flags. For PF use seeds 1984-1988, publishing individual and aggregate outcomes. Do not invent new benchmark CLI options before adding/testing them.

## Close one increment

Append exact commands/results, fixture/template hashes, comparison maxima, report location and delta, and platform-gate status to research.md. Stage only the current story's implementation/tests/fixtures/docs/evidence and create its separate authorized commit. Record its hash and mark its tasks complete. Only then begin the next story; pause if a tool, asset right, reference behavior, or parity result blocks delivery.
