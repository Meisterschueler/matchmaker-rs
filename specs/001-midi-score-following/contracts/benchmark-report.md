# Contract: Benchmark Records and Report

Both `matchmaker bench` (Rust) and `tools/bench/run_reference.py` (Python reference) write a
JSON array of records with this schema. `tools/bench/report.py` merges them into
`report.json` + `index.html`/`index.md`, published to GitHub Pages and linked from `README.md`
(FR-016, FR-020, SC-006). It also writes `badge.json`, a
[shields.io endpoint](https://shields.io/badges/endpoint-badge) (`label` "vs pymatchmaker",
`message` = median speed-up over all rows, `color` red on a position-parity violation) shown
as a badge in `README.md`.

## Record

```json
{
  "implementation": "rust",                 // "rust" | "python"
  "implementation_version": "0.1.0",        // crate version or reference tag
  "method": "pthmm",
  "processor": "pitch",
  "dataset": "bundled",                     // "bundled" | "nasap" | "batik" | "vienna4x22"
  "piece": "simple_mozart_k265_var1",
  "run": 0,                                 // 0..runs-1
  "n_observations": 41,
  "performance_duration_s": 7.93,
  "latency_ms": {
    "feature":   {"mean": 0.002, "median": 0.002, "p95": 0.003, "max": 0.01},
    "inference": {"mean": 0.010, "median": 0.009, "p95": 0.015, "max": 0.05}
  },
  "total_processing_s": 0.0005,
  "real_time_factor": 0.00006,              // total_processing_s / performance_duration_s
  "evaluation": { "mean_b": 0.0, "median_b": 0.0, "std_b": 0.0,
                  "accuracy_0.05b": 1.0, "accuracy_0.1b": 1.0, "accuracy_0.3b": 1.0,
                  "accuracy_0.5b": 1.0, "accuracy_1b": 1.0, "accuracy_2b": 1.0,
                  "skewness": 0.0, "kurtosis": 0.0, "smoothness": -1.5 },
  "environment": {"os": "ubuntu-latest", "cpu": "…", "commit": "…", "timestamp": "…"}
}
```

Timing rules (III): feature latency = time inside the processor call per frame that produced an
observation; inference latency = time inside the follower call per observation; no tracing
subscriber active; median over `runs ≥ 5` is the reported value.

## Report

Per method and piece, side by side: Rust vs Python feature/inference latency (median of runs),
speed-up factor, real-time factor, all evaluation metrics, parity flag (metrics within the
golden tolerances). Position parity compares the run-0 alignment paths (`paths/<dataset>/<piece>/<method>.tsv`); a step where Rust's best state leads the runner-up by less than 1e-9 (relative, `<method>.gap.tsv`) is a numerical tie and counts as equal, because the reference's BLAS rounding picks one of exactly tied states. Per method and dataset: aggregate (mean of per-piece medians, mean metrics).
Regression flag: Rust latency > 1.10 × the latest `master` report for the same method/piece.
Cached Python reference results are marked with the cache key and date.
