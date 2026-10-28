# Contract: CLI (`matchmaker`)

Binary of `crates/matchmaker-cli`. Errors are printed with context (via `anyhow`) and exit code
1; usage errors exit with code 2 (clap). Logging via `RUST_LOG` (e.g.
`RUST_LOG=matchmaker_rs=debug` prints per-observation positions and stage timings).

## `matchmaker follow`

```text
matchmaker follow <SCORE> <PERFORMANCE>
    [--method pthmm|hmm|outerhmm]        default: pthmm
    [--tempo <BPM>] [--no-unfold]
    [--output <FILE.tsv>]                alignment path (perf_sec<TAB>score_beat)
    [--export <DIR>]                     intermediate data, layout of contracts/golden-data.md (US6)
    [--quiet]                            do not print positions
```

stdout (unless `--quiet`): one line per alignment-path entry, `<perf_time_s>\t<score_beat>`,
printed as computed (US1-4). Unknown method → exit 2 with the supported methods.
`--processor` / `--set KEY=VALUE` (FR-011) are added together with the methods whose parameters
they override.

## `matchmaker evaluate`

```text
matchmaker evaluate <SCORE> <ALIGNMENT> <GROUND_TRUTH> [--performance <MIDI>] [--json]
```

`ALIGNMENT`: `.tsv`/`.npy` written by `follow`. `GROUND_TRUTH`: `.match` or TSV with columns
`perf_sec`, `score_beat`. Output: the `EvaluationResult` keys of the reference, as a table or
JSON (`--json`).

## `matchmaker bench`

```text
matchmaker bench <MANIFEST> --out <FILE.json> [--runs <N=5>] [--methods pthmm,hmm,outerhmm]
```

`MANIFEST`: TOML listing pieces (`dataset`, `piece`, `score`, `performance`, `ground_truth`).
Runs each method `N` times per piece, timing processor and follower calls per observation with a
monotonic clock and **no tracing subscriber**; writes records per
[benchmark-report.md](benchmark-report.md) with `implementation = "rust"`.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | runtime error (invalid file, unsupported format, write failure) |
| 2 | invalid command line |
