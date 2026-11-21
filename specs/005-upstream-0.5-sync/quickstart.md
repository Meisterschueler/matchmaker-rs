# Quickstart: Validate the Upstream 0.5.1 Sync

Validation guide per increment ([plan.md](plan.md) C01/C02). Commands run from the
repository root unless stated. Contracts: [contracts/sync-api.md](contracts/sync-api.md);
entities: [data-model.md](data-model.md).

## Prerequisites

- Rust toolchain (stable, MSRV 1.88 for the MSRV check) and the usual gates installed.
- Python with `uv` for `tools/golden` and `tools/bench`.
- For audio golden regeneration: the pinned FreePats SF2 provisioned as documented in
  `tests/data/audio/README.md`.

## C01 — Pin bump, SPARC, regenerated goldens

### 1. Re-pin and regenerate the reference environment

```text
cd tools/golden
# pyproject.toml: pymatchmaker @ git+...@v0.5.1
python -m uv lock
python -m uv sync --locked
```

Expected: `uv.lock` records `pymatchmaker` at `rev=v0.5.1`; review the lockfile diff for
transitive moves (research R1 risk).

### 2. Regenerate goldens

```text
cd tools/golden
python -m uv run generate.py --data ../../tests/data --out ../../tests/golden
python -m uv run generate_audio.py --data ../../tests/data/audio --out ../../tests/golden/audio
```

Expected: every `meta.json` says tag `v0.5.1`. Review `git diff tests/golden/`:
`evaluation.json` `smoothness` values change everywhere (SPARC 3000 Hz); `hmm` arrays may
move at ulp level (upstream forward-step restructuring); all other numbers unchanged
unless the lockfile moved a dependency. Any untraceable change blocks the sync.

### 3. Rust-side SPARC and equivalence notes

```text
cargo test --test golden_eval
cargo test -p matchmaker-rs --lib utils::eval
cargo test --workspace --exclude matchmaker-py
```

Expected: regenerated expectations pass; the new unit test showing the sliding and
per-window SPARC routes agree passes; full suite green (goldens compare stage by stage).

### 4. Gates

```text
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo build --target wasm32-unknown-unknown
cargo deny check
```

Plus MSRV check and Python binding tests in CI. Update `UPSTREAM.md`/`README.md` version
surfaces before committing (contract §5).

## C02 — `dannenberg` follower

### 1. Golden replay

```text
cargo test --test golden_dannenberg
```

Expected: positions, `current_index`, rating column, and alignment path match
`tests/golden/<piece>/dannenberg/` within the documented tolerances for all pieces,
including the wrong-note variant (no lock-on from a wrong first note).

### 2. CLI

```text
cargo run --release -p matchmaker-cli -- follow tests/data/simple_mozart_k265_var1.musicxml tests/data/simple_mozart_k265_var1.mid --method dannenberg --output dann.tsv
cargo run -p matchmaker-cli -- evaluate tests/data/simple_mozart_k265_var1.musicxml dann.tsv tests/data/simple_mozart_k265_var1.match --performance tests/data/simple_mozart_k265_var1.mid --json
```

Expected: a position path; evaluation keys equal
`tests/golden/simple_mozart_k265_var1/dannenberg/evaluation.json` within `1e-4`.
`--method dannenberg --set piano_range=false` (or `return_pitch_list`) is rejected with a
typed parameter error (contract §1).

### 3. Python bindings

```text
cd crates/matchmaker-py
python -m uv run --with maturin maturin develop --release
python -m uv run python -c "from matchmaker_rs import Matchmaker; mm = Matchmaker('../../tests/data/simple_mozart_k265_var1.musicxml', '../../tests/data/simple_mozart_k265_var1.mid', input_type='midi', method='dannenberg'); print(list(mm.run(verbose=False))[:5])"
python -m uv run pytest tests
```

Expected: positions print; `default_kwargs` for `dannenberg` equals `methods.yaml`
(contract §1); binding tests pass.

### 4. Benchmark

```text
cargo run --release -p matchmaker-cli -- bench tools/bench/bundled.toml --out target/bench/rust/records.json
cd tools/golden
python -m uv run ../bench/run_reference.py ../bench/bundled.toml --out ../../target/bench/python/records.json
python -m uv run ../bench/report.py --rust ../../target/bench/rust/records.json --python ../../target/bench/python/records.json --out ../../target/bench/report
```

Expected: the report contains a `dannenberg` row beside the reference; no >10% latency
regression for existing methods (five-run median rule).

### 5. Gates

Same gate set as C01, plus `tests/method_config.rs` coverage of the new keys (accepted:
`c_m`, `c_e`, `epsilon`, `epsilon_fraction`, `window`, `smoothing`, `processor`,
`polling_period`; rejected: `piano_range`, `return_pitch_list`, unknown keys).

## Done when

- Both increments' expected outcomes above hold locally and in CI (Linux, Windows, macOS).
- `UPSTREAM.md` traces every 0.4.0→0.5.1 upstream change to a port change or a not-ported
  entry (SC-005), and the golden diff review found no untraceable number (SC-001).
