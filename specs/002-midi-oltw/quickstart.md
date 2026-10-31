# Quickstart: MIDI Online Time Warping

Validation scenarios; setup as in the [001 quickstart](../001-midi-score-following/quickstart.md).

## 1. Follow with `arzt` / `dixon` (library parity)

```text
cargo test --workspace --exclude matchmaker-py --test golden_arzt --test golden_dixon
```

Expected: both methods match `tests/golden/*/{arzt,dixon}/` (positions `< 1e-6` beats,
`current_index` exact); a mismatch names method, piece and first differing observation.

## 2. CLI

```text
cargo run --release -p matchmaker-cli -- follow tests/data/simple_mozart_k265_var1.musicxml tests/data/simple_mozart_k265_var1.mid --method dixon --set window_size=20 --output path.tsv
```

Expected: `perf_time<TAB>beat` per position. `--method arzt --set window_size=5` fails with
`InvalidParameter` ("not used by the reference's MIDI arzt").

## 3. Python

```python
from matchmaker_rs import Matchmaker
positions = list(Matchmaker("score.musicxml", "performance.mid", input_type="midi", method="arzt").run(verbose=False))
```

Expected: same positions as scenario 1; `method="foo"` raises `ValueError` listing the five MIDI
methods.

## 4. Regenerate golden data

```text
cd tools/golden
python -m uv run generate.py --data ../../tests/data --out ../../tests/golden
```

Expected: the diff of `tests/golden/` only adds `arzt/`, `dixon/` and the two onset-pianoroll files.

## 5. Benchmark

```text
cargo run --release -p matchmaker-cli -- bench --help
```

Expected: `arzt` and `dixon` appear in the default method list and in the report next to the
other methods.
