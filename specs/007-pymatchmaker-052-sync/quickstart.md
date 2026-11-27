# Quickstart: Validate the pymatchmaker 0.5.2 Sync

Commands run from the repository root unless stated.

## Prerequisites

- Rust stable, including MSRV 1.88; `cargo-deny` for dependency checks.
- Python 3.12 and `uv` for the golden/reference environment.
- The pinned SoundFont described in `tests/data/audio/README.md` for audio golden
  regeneration, plus native FluidSynth support for score-rendered fixtures.

## 1. Lock and verify the reference

Edit the pymatchmaker source tag in `tools/golden/pyproject.toml`, then:

```text
cd tools/golden
uv lock
uv sync --locked
uv run python -c "from importlib.metadata import version; print('pymatchmaker', version('pymatchmaker')); print('python-hiddenmarkov', version('python-hiddenmarkov'))"
```

Expected: pymatchmaker reports `0.5.2`; python-hiddenmarkov reports a version at least
`0.3.0`; the lockfile records the exact resolved versions.

## 2. Regenerate reference outputs

```text
cd tools/golden
uv run generate.py --data ../../tests/data --out ../../tests/golden
uv run generate_audio.py --out ../../tests/golden/audio
uv run generate_audio.py --processors-only --out ../../tests/golden/audio
uv run generate_audio.py --raw-spectrum-only --out ../../tests/golden/audio
uv run generate_audio.py --mel-only --out ../../tests/golden/audio
uv run generate_audio.py --mfcc-only --out ../../tests/golden/audio
uv run generate_audio.py --korzeniowski-only --out ../../tests/golden/audio
uv run generate_audio.py --pfkorz-only --out ../../tests/golden/audio
uv run generate_audio.py --cqt-only --out ../../tests/golden/audio
uv run generate_audio.py --cqt-spectral-flux-only --out ../../tests/golden/audio
uv run generate_audio.py --skf-only --out ../../tests/golden/audio
uv run generate_audio.py --outerhmm-only --out ../../tests/golden/audio
uv run generate_audio.py --tempo-replay-only --out ../../tests/golden/audio
uv run generate_audio.py --arzt-tempo-only --out ../../tests/golden/audio
uv run generate_audio.py --multi-fold-only --out ../../tests/golden/audio
uv run generate_audio.py --audio-ground-truth-only --out ../../tests/golden/audio
```

Expected: generated metadata records pymatchmaker `v0.5.2`, the resolved
python-hiddenmarkov version, and the current lockfile digest. Review every changed value
against the release diff and lockfile changes; unexplained changes block completion.
The multi-fold and ground-truth modes also regenerate their tracked audio/TSV fixtures.
On Windows, add the pinned FluidSynth `bin` directory to `PATH` before starting `uv`.

## 3. Verify HMM parity and edge cases

```text
cargo test --workspace --exclude matchmaker-py
```

Expected: optimized-vs-full forward-step tests cover the initial observation, a narrow
reachable range, non-contiguous non-zero states, and an all-zero previous vector. Existing
`pthmm` and `hmm` golden results, positions, and paths pass within their documented
tolerances.

## 4. Audit release references

Search for current and historical references:

```text
rg -n "0\.5\.1|v0\.5\.1|0\.5\.2|v0\.5\.2|python-hiddenmarkov" README.md UPSTREAM.md tools src crates tests specs NOTICE
```

Expected: active pins, generators, and documentation identify 0.5.2 and its required
dependency; older release mentions are retained only where they accurately describe
historical work. Review each match rather than mechanically replacing all version strings.

## 5. Benchmark the HMM optimization

Use the existing Rust and reference benchmark procedure from
[specs/001-midi-score-following/quickstart.md](../../001-midi-score-following/quickstart.md),
on identical inputs, with at least five runs.

```text
cargo run --release -p matchmaker-cli -- bench tools/bench/bundled.toml --out target/bench/0.5.2/rust/records.json --runs 5 --methods pthmm,hmm,outerhmm,arzt,dixon,dannenberg
cd tools/golden
uv run python ../bench/run_reference.py ../bench/bundled.toml --out ../../target/bench/0.5.2/python/records.json --runs 5 --methods pthmm,hmm,outerhmm,arzt,dixon,dannenberg
uv run python ../bench/report.py --rust ../../target/bench/0.5.2/rust/records.json --python ../../target/bench/0.5.2/python/records.json --out ../../target/bench/0.5.2/report
```

Expected: report median `pthmm` and `hmm` inference timings, the overall benchmark delta,
and any parity flags. The six-method selector omits `pfkorz` because the current reference
runner cannot identify its `ParticleFilterMidiProcessor` processor class. Existing measured
methods must not regress by more than 10%; any regression or unexpected HMM output difference
is investigated before acceptance.

## 6. Quality gates

```text
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --target wasm32-unknown-unknown
cargo deny check
```

Expected: all commands pass; CI also verifies the MSRV and supported platforms.
