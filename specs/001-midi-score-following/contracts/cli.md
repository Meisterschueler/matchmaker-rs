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

`ALIGNMENT`: `.tsv` written by `follow --output`. `GROUND_TRUTH`: `.match` (versions 1.x.y and
legacy 3.0-5.0; score notes by shared id, else by pitch and onset beat like the reference) or
TSV with columns `perf_sec`, `score_beat`; other match files
are converted with `tools/golden/match_to_tsv.py`. Output: the `EvaluationResult` keys of the
reference, as a table or JSON (`--json`); NaN is printed as `null`.

## `matchmaker bench`

```text
matchmaker bench <MANIFEST> --out <FILE.json> [--runs <N=5>] [--methods pthmm,hmm,outerhmm]
```

`MANIFEST`: TOML with `[[pieces]]` entries (`dataset`, `piece`, `score`, `performance`,
optional `ground_truth`; paths relative to the manifest), e.g. `tools/bench/bundled.toml`.
Runs each method `N` times per piece (default: all available methods), timing processor and
follower calls per observation with a monotonic clock and **no tracing subscriber**; writes
records per [benchmark-report.md](benchmark-report.md) with `implementation = "rust"` and the
alignment path of run 0 as `<out dir>/paths/<dataset>/<piece>/<method>.tsv`.
`tools/bench/run_reference.py` does the same for the reference; `tools/bench/report.py`
merges both.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | runtime error (invalid file, unsupported format, write failure) |
| 2 | invalid command line |
