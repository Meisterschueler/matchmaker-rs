"""Merge Rust and Python benchmark records into report.json and index.md.

Per method and piece: median latencies over runs, speed-up, real-time factor, metrics and two
parity flags (contracts/benchmark-report.md):

- metric parity: every evaluation metric of Rust within 1e-4 of the reference (golden tolerance)
- position parity: alignment paths (``paths/<dataset>/<piece>/<method>.tsv`` next to each
  records file) agree within 1e-6 at every step for pthmm/outerhmm (SC-001) and at >= 99.9 %
  of the steps for hmm (SC-002); the first differing step is reported (SC-007)

Exits with 1 on a position-parity violation. ``--baseline`` (the previous master report.json)
flags Rust inference latency above 1.10 x the baseline. Standard library + NumPy only; the two
records files must live in different directories (each has its own ``paths/``):

    python report.py rust/records.json python/records.json --out report
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

METRIC_TOL = 1e-4
POSITION_TOL = 1e-6
HMM_MIN_SHARE = 0.999
REGRESSION = 1.10


def load(path: Path) -> dict:
    """Records grouped by (method, dataset, piece)."""
    groups: dict = {}
    for r in json.loads(path.read_text(encoding="utf-8")):
        groups.setdefault((r["method"], r["dataset"], r["piece"]), []).append(r)
    return groups


def median(records: list, *keys: str):
    values = []
    for r in records:
        v = r
        for k in keys:
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, (int, float)) and not math.isnan(v):
            values.append(v)
    return statistics.median(values) if values else None


def same(a, b) -> bool:
    nan = lambda x: x is None or (isinstance(x, float) and math.isnan(x))  # noqa: E731
    return (nan(a) and nan(b)) or (not nan(a) and not nan(b) and abs(a - b) <= METRIC_TOL)


def position_parity(method: str, rust: Path, python: Path) -> dict | None:
    if not rust.exists() or not python.exists():
        return None
    r = np.loadtxt(rust, delimiter="\t", skiprows=1, ndmin=2)
    p = np.loadtxt(python, delimiter="\t", skiprows=1, ndmin=2)
    n = min(len(r), len(p))
    close = np.all(np.abs(r[:n] - p[:n]) <= POSITION_TOL, axis=1)
    differing = np.flatnonzero(~close)
    first = int(differing[0]) if len(differing) else (n if len(r) != len(p) else None)
    share = float(close.sum()) / max(len(r), len(p), 1)
    ok = share >= HMM_MIN_SHARE if method == "hmm" else first is None
    return {"ok": bool(ok), "share": share, "first_differing_step": first, "steps": [len(r), len(p)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rust", type=Path)
    parser.add_argument("python", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, help="report.json of the latest master run")
    args = parser.parse_args()

    rust, python = load(args.rust), load(args.python)
    baseline = {}
    if args.baseline and args.baseline.exists():
        for row in json.loads(args.baseline.read_text(encoding="utf-8"))["rows"]:
            baseline[(row["method"], row["dataset"], row["piece"])] = row["rust"]["inference_ms"]

    rows = []
    for key in sorted(set(rust) | set(python)):
        method, dataset, piece = key
        side = {}
        for name, groups in (("rust", rust), ("python", python)):
            records = groups.get(key, [])
            side[name] = {
                "runs": len(records),
                "feature_ms": median(records, "latency_ms", "feature", "median"),
                "inference_ms": median(records, "latency_ms", "inference", "median"),
                "total_s": median(records, "total_processing_s"),
                "real_time_factor": median(records, "real_time_factor"),
                "evaluation": records[0]["evaluation"] if records else None,
            }
        r, p = side["rust"], side["python"]
        metric_parity = None
        if r["evaluation"] and p["evaluation"]:
            metric_parity = all(same(r["evaluation"].get(k), v) for k, v in p["evaluation"].items())
        rel = Path("paths") / dataset / piece / f"{method}.tsv"
        previous = baseline.get(key)
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "piece": piece,
                "rust": r,
                "python": p,
                "speed_up": p["total_s"] / r["total_s"] if r["total_s"] and p["total_s"] else None,
                "metric_parity": metric_parity,
                "position_parity": position_parity(method, args.rust.parent / rel, args.python.parent / rel),
                "regression": bool(previous and r["inference_ms"] and r["inference_ms"] > REGRESSION * previous),
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps({"rows": rows}, indent=2) + "\n", encoding="utf-8")

    fmt = lambda v, f=".4g": "-" if v is None else format(v, f)  # noqa: E731
    flag = lambda v: "-" if v is None else ("ok" if v else "FAIL")  # noqa: E731
    lines = [
        "# matchmaker-rs benchmark",
        "",
        "Medians over runs. Latency in ms per observation; speed-up = Python / Rust total processing time.",
        "",
        "| method | dataset | piece | feature ms (rs / py) | inference ms (rs / py) | speed-up | RTF rs | mean_b | acc 0.1b | smoothness | metrics | positions | regression |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        r, p, ev = row["rust"], row["python"], row["rust"]["evaluation"] or row["python"]["evaluation"] or {}
        pos = row["position_parity"]
        pos_text = "-" if pos is None else flag(pos["ok"]) + (
            "" if pos["first_differing_step"] is None else f" (step {pos['first_differing_step']})"
        )
        lines.append(
            f"| {row['method']} | {row['dataset']} | {row['piece']} "
            f"| {fmt(r['feature_ms'])} / {fmt(p['feature_ms'])} | {fmt(r['inference_ms'])} / {fmt(p['inference_ms'])} "
            f"| {fmt(row['speed_up'], '.1f')} | {fmt(r['real_time_factor'])} | {fmt(ev.get('mean_b'))} "
            f"| {fmt(ev.get('accuracy_0.1b'))} | {fmt(ev.get('smoothness'))} | {flag(row['metric_parity'])} "
            f"| {pos_text} | {'REGRESSION' if row['regression'] else ''} |"
        )
    (args.out / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    violations = [row for row in rows if row["position_parity"] and not row["position_parity"]["ok"]]
    for row in violations:
        print(f"position parity violated: {row['method']} {row['dataset']}/{row['piece']}: {row['position_parity']}")
    raise SystemExit(1 if violations else 0)


if __name__ == "__main__":
    main()
