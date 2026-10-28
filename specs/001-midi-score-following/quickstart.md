# Quickstart & Validation: MIDI Score Following

Runnable scenarios that prove the feature end to end. Details: [contracts/](contracts/),
[data-model.md](data-model.md).

## Prerequisites

- Rust stable ≥ MSRV (`rust-version` in `Cargo.toml`).
- Only for golden-data regeneration, Python bindings and the benchmark: Python 3.10–3.13 and
  `uv` (`python -m uv`), plus `maturin` for the bindings.

## 1. First alignment from a fresh checkout (SC-005, US1)

```text
git clone https://github.com/Meisterschueler/matchmaker-rs
cd matchmaker-rs
cargo run --release -p matchmaker-cli -- follow tests/data/simple_mozart_k265_var1.musicxml tests/data/simple_mozart_k265_var1.mid --output path.tsv
```

Expected: positions (`perf_time<TAB>beat`) printed while running; `path.tsv` with header
`perf_sec	score_beat`; every beat is one of the score states.

## 2. Parity with the reference (US2-1, SC-001…SC-003, SC-007)

```text
cargo test --workspace --exclude matchmaker-py
```

Expected: golden tests for score loading, stream, `pitch`, `chord_onset`, `pthmm`, `hmm`,
`outerhmm` and evaluation pass. Introduce a deliberate change (e.g. scale `1.0` → `0.9` in
`stable_transition_matrix`) and re-run: the failure names method, piece, file and first
differing index.

## 3. Each method and the MIDI score (US1-2, US3, US4)

```text
cargo run -p matchmaker-cli -- follow tests/data/simple_mozart_k265_var1.mid tests/data/simple_mozart_k265_var1.mid --method pthmm --quiet --output a.tsv
cargo run -p matchmaker-cli -- follow tests/data/mozart_k265_var1.musicxml tests/data/mozart_k265_var1.mid --method hmm --export out/hmm
cargo run -p matchmaker-cli -- follow tests/data/Bach-fugue_bwv_858.musicxml tests/data/Bach-fugue_bwv_858.mid --method outerhmm --export out/outerhmm
```

Expected: `out/hmm/…/hmm/beat_period.npy` exists (US3-2); `outerhmm` export contains
`is_continuation.npy`.

## 4. Evaluation (US2-2)

```text
cargo run -p matchmaker-cli -- evaluate tests/data/simple_mozart_k265_var1.musicxml path.tsv tests/data/simple_mozart_k265_var1.match --performance tests/data/simple_mozart_k265_var1.mid --json
```

Expected: keys `mean_b`, `median_b`, `std_b`, `accuracy_0.05b` … `accuracy_2b`, `skewness`,
`kurtosis`, `smoothness`; values equal `tests/golden/simple_mozart_k265_var1/pthmm/evaluation.json`
within `1e-4`.

## 5. Debug a disagreement (US6)

```text
cd tools/golden
python -m uv run compare.py ../../tests/golden/mozart_k265_var1 ../../out/hmm/mozart_k265_var1
```

Expected: "all stages equal" or the first differing stage (`score` → `stream` → `processor` →
method), file and index. With `RUST_LOG=matchmaker_rs=debug` the CLI logs per-stage timing and the
current position per observation (US6-2).

## 6. Python drop-in (US5)

```text
cd crates/matchmaker-py
python -m uv run --with maturin maturin develop --release
python -m uv run python -c "from matchmaker_rs import Matchmaker; mm = Matchmaker('../../tests/data/simple_mozart_k265_var1.musicxml', '../../tests/data/simple_mozart_k265_var1.mid', input_type='midi', method='pthmm'); print(list(mm.run(verbose=False))[:5]); print(mm.score_follower.alignment_path.shape)"
```

Expected: same positions as scenario 1; shape `(2, T)`. `method="arzt"` raises `ValueError`
listing `['pthmm', 'hmm', 'outerhmm']` (US5-2).

## 7. Benchmark (US2-3, US2-4)

```text
cargo run --release -p matchmaker-cli -- bench tools/bench/bundled.toml --out rust.json
cd tools/bench
python -m uv run run_reference.py bundled.toml --out python.json
python -m uv run report.py ../../rust.json python.json --out report/
```

Expected: `report/index.md` with Rust vs Python latency per stage, speed-up, real-time factor and
metrics per method and piece; Rust inference latency ≥ 5× lower (SC-004 target). In CI the same
runs on the full datasets (`tools/datasets/fetch.py`) and the report is published; the README
links to it (SC-006).

## 8. Platform gates

```text
cargo build --target wasm32-unknown-unknown
cargo deny check
```

Expected: both succeed (V).
