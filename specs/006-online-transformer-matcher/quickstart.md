# Quickstart: Validate Online Transformer Matcher (OTM)

Validation guide for the OTM port. Run each code block from the repository root; `cd` applies
to the remaining commands in that block only. See [contracts/otm-api.md](contracts/otm-api.md) for the public contract and
[data-model.md](data-model.md) for the entities and state invariants.

## Prerequisites

- Stable Rust and MSRV 1.88 toolchains; `wasm32-unknown-unknown` target for the WASM gate.
- Python 3.10–3.13 and `uv` for locked reference generation and benchmarking.
- The model artifact converted by the locked reference tool. Its source, checksum and
  redistribution permission must be verified before committing it or including it in a
  release. Do not fetch a model at runtime.

## 1. Prepare and verify the reference

```text
cd tools/golden
python -m uv sync --locked
python -m uv run export_otm_weights.py
python -m uv run generate.py --data ../../tests/data --out ../../tests/golden --methods OTM
```

Expected:

- The exporter writes the documented compact inference artifact and manifest, retaining the
  upstream source and checkpoint SHA-256 from [research.md](research.md#r3-model-architecture-and-weights).
- Golden metadata identifies pymatchmaker `v0.5.1`, the locked dependency versions and the
  model hash.
- Reference inference runs on CPU even when CUDA is available, so fixture and benchmark
  outputs use the same execution device.
- New OTM fixtures cover score notes including grace status, event observations, token
  windows, model logits, tempo state, positions and alignment paths. Review the generated
  diff; unexpected changes to existing method goldens block the port.

## 2. Run focused parity tests

```text
cargo test --test golden_otm
cargo test --test otm_api --test otm_errors --test otm_score --test otm_processor --test otm_model --test otm_export
cargo test --lib dp::otm
```

Expected: every fixture passes the documented score/feature/model tolerances, beat positions
stay within `1e-6`, selected encoder-layer outputs match the small reference fixtures, and
five repeated Rust runs produce identical positions and paths. A failure names the fixture,
stage and first differing event.

## 3. Exercise the CLI and export

```text
cargo run --release -p matchmaker-cli -- follow tests/data/simple_mozart_k265_var1.musicxml tests/data/simple_mozart_k265_var1.mid --method OTM --output target/otm.tsv --export target/otm-export/
```

Expected: the CLI prints one position per consumed note event and writes a public
`perf_sec<TAB>score_beat` alignment path. The export can be loaded by the existing Python
golden comparison tools and contains the score reference, processed observations, tokens,
token/model-score masks and logits, tempo diagnostics, positions and alignment path.

Invalid cases must fail explicitly:

```text
cargo run -p matchmaker-cli -- follow tests/data/simple_mozart_k265_var1.musicxml tests/data/simple_mozart_k265_var1.mid --method OTM --input-type audio
cargo run -p matchmaker-cli -- follow tests/data/simple_mozart_k265_var1.musicxml tests/data/simple_mozart_k265_var1.mid --method OTM --set polling_period=0.01
cargo run -p matchmaker-cli -- follow tests/data/simple_mozart_k265_var1.musicxml tests/data/simple_mozart_k265_var1.mid --method OTM --set unknown_option=1
```

Expected: audio, non-event polling and unknown options are rejected with actionable errors;
no fallback method or alignment is returned.

## 4. Exercise the Python binding

```text
cd crates/matchmaker-py
python -m uv run --with maturin maturin develop --release
python -m uv run python -c "from matchmaker_rs import Matchmaker; mm = Matchmaker('../../tests/data/simple_mozart_k265_var1.musicxml', '../../tests/data/simple_mozart_k265_var1.mid', input_type='midi', method='OTM'); print(list(mm.run(verbose=False)))"
python -m uv run pytest tests
```

Expected: Python exposes `OTM` as a MIDI method, positions match the library/CLI within
`1e-6` beats, the public alignment path has shape `(2, T)`, and binding tests pass.

## 5. Benchmark Rust against the pinned reference

```text
cargo run --release -p matchmaker-cli -- bench tools/bench/bundled.toml --out target/bench/otm-rust/records.json --runs 5 --methods OTM
cd tools/golden
python -m uv run ../bench/run_reference.py ../bench/bundled.toml --out ../../target/bench/otm-python/records.json --runs 5 --methods OTM
python -m uv run ../bench/report.py --rust ../../target/bench/otm-rust/records.json --python ../../target/bench/otm-python/records.json --out ../../target/bench/otm-report
```

Expected: the report compares identical inputs, includes per-event inference latency and
real-time factor for both implementations, and Rust's median RTF is below `1.0` over at
least five runs.

## 6. Run repository quality gates

```text
cargo fmt --all --check
cargo clippy --workspace --exclude matchmaker-py --all-targets --all-features
cargo test --workspace --exclude matchmaker-py --all-features
cargo build -p matchmaker-rs --target wasm32-unknown-unknown
cargo deny check
```

Also run the CI MSRV check (`cargo check -p matchmaker-rs -p matchmaker-cli --all-targets`
with Rust 1.88) and the Python binding checks on Python 3.10 and 3.13. Confirm existing
non-OTM golden outputs remain unchanged.

## Release gate

Before any release packages the converted weights, verify and record the artifact's
redistribution terms and required attribution. If that cannot be established, stop before
distribution and obtain maintainer/legal guidance; do not silently bundle, replace, or
download another model.

## Recorded implementation validation

Validated on Windows x86-64 CPU, 2026-11-24:

| Check | Result |
|---|---|
| `cargo test --workspace --exclude matchmaker-py --all-features -j 1` | Passed, including unchanged MIDI/audio goldens and doctests. Serial linking avoided a transient Windows LNK1104 failure. |
| `cargo test --test golden_otm --test otm_model --test otm_processor` | Passed after the final sequence-softmax tie fix; six stage/E2E fixtures. |
| `cargo test --lib dp::otm` | Passed: tempo, Python set ordering, sequence-softmax ties, and all six encoder layers. |
| `cargo test --test otm_api` | Passed, including five identical paths across file/push/repeated runs. |
| `cargo test --test edge_cases empty_performance_gives_an_empty_path` | Passed with OTM included. |
| `python -m pytest crates/matchmaker-py/tests -q` | 74 passed on each of Python 3.10, 3.12 and 3.13 using the final ABI3 wheel. Task-local 3.10/3.13 environments were provisioned because those interpreters were absent. |
| `cargo fmt --all --check` | Passed. |
| `cargo clippy --workspace --exclude matchmaker-py --all-targets --all-features -- -D warnings` | Passed. |
| `cargo build -p matchmaker-rs --target wasm32-unknown-unknown` | Passed. |
| `cargo +1.88 check -p matchmaker-rs -p matchmaker-cli --all-targets` | Passed. |
| `cargo deny check` | Passed; existing unused-license and duplicate-`syn` warnings remain. |
| `cargo package --list --allow-dirty -p matchmaker-rs` | Includes converted weights, manifest and Apache-2.0 license. |
| `compare.py tests/golden/simple_mozart_k265_var1 target/otm-export` | All 18 shared files/stages equal. |
| Existing tracked goldens | Unchanged; new OTM fixtures added separately. |

Paired benchmark: five runs per implementation and piece (15 Rust and 15 Python runs),
exact position/path parity and matching metrics. Median Rust RTFs:

| Piece | Median Rust RTF |
|---|---:|
| simple_mozart_k265_var1 | 0.01283 |
| mozart_k265_var1 | 0.01193 |
| Bach-fugue_bwv_858 | 0.006499 |

All are below `1.0`; these are local measurements, not hardware-independent guarantees.
The report is saved under `target/bench/otm-report/`. The six existing MIDI methods were
also measured over five runs per piece against the prior local baseline: all 18 comparison
rows retain exact paths and pass the 10% inference-regression threshold
(`target/bench/otm-existing-report/`).

License gate: cleared by the owner's explicit Apache-2.0 determination for source and
weights, recorded in [research.md](research.md#r5-model-asset-licensing-and-provenance).
No extension-hook file is present, so there were no pre/post hooks to dispatch.
