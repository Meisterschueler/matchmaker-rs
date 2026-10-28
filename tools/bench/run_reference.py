"""Python side of the benchmark: run the reference on a bench manifest and write records in the
schema of specs/001-midi-score-following/contracts/benchmark-report.md (implementation "python").

Processor and follower ``__call__`` are timed with ``time.perf_counter()`` through subclasses
created at run time, so the reference itself is unchanged. The alignment path of run 0 goes to
``<out dir>/paths/<dataset>/<piece>/<method>.tsv`` like ``matchmaker bench``. Run in the
tools/golden environment (from tools/golden):

    python -m uv run ../bench/run_reference.py ../bench/bundled.toml --out ../../target/bench/python/records.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
import tomllib
from pathlib import Path

import numpy as np

import matchmaker
from matchmaker import Matchmaker
from matchmaker.utils.eval import evaluate_alignment, resolve_gt
from matchmaker.utils.symbolic import midi_messages_from_performance

METHODS = ["pthmm", "hmm", "outerhmm"]
PROCESSORS = {"PitchProcessor": "pitch", "ChordOnsetProcessor": "chord_onset"}


def timed(obj, times: list, only_results: bool = False) -> None:
    """Record the duration of every ``obj(...)`` call (only non-None results if asked)."""
    base = type(obj)

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        result = base.__call__(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        if not only_results or result is not None:
            times.append(elapsed)
        return result

    obj.__class__ = type(f"Timed{base.__name__}", (base,), {"__call__": __call__})


def latency(times: list) -> dict:
    if not times:
        return {"mean": None, "median": None, "p95": None, "max": None}
    ms = np.asarray(times) * 1e3
    return {
        "mean": float(ms.mean()),
        "median": float(np.median(ms)),
        "p95": float(np.percentile(ms, 95)),
        "max": float(ms.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()

    manifest = tomllib.loads(args.manifest.read_text(encoding="utf-8"))
    base = args.manifest.resolve().parent
    out_dir = args.out.resolve().parent
    environment = {
        "os": platform.system().lower(),
        "arch": platform.machine(),
        "commit": os.environ.get("GITHUB_SHA"),
        "timestamp": int(time.time()),
    }

    records = []
    for p in manifest["pieces"]:
        score, perf = base / p["score"], base / p["performance"]
        _, msg_times = midi_messages_from_performance(str(perf))
        duration = float(msg_times[-1]) if len(msg_times) else 0.0
        for method in args.methods.split(","):
            for run in range(args.runs):
                mm = Matchmaker(score_file=str(score), performance_file=str(perf), input_type="midi", method=method)
                feature, inference = [], []
                processor = PROCESSORS[type(mm.processor).__name__]
                timed(mm.processor, feature, only_results=True)
                timed(mm.score_follower, inference)
                for _ in mm.run(verbose=False):
                    pass
                path = np.asarray(mm.score_follower.alignment_path, dtype=float).reshape(2, -1)
                if run == 0:
                    tsv = out_dir / "paths" / p["dataset"] / p["piece"] / f"{method}.tsv"
                    tsv.parent.mkdir(parents=True, exist_ok=True)
                    np.savetxt(tsv, path.T, delimiter="\t", fmt="%.17g", header="perf_sec\tscore_beat", comments="")
                evaluation = None
                if p.get("ground_truth"):
                    gt_perf, gt_score = resolve_gt(
                        str(base / p["ground_truth"]), mm.score_part.note_array(), performance=str(perf)
                    )
                    evaluation = evaluate_alignment(path[1], path[0], gt_score, gt_perf)
                total = sum(feature) + sum(inference)
                records.append(
                    {
                        "implementation": "python",
                        "implementation_version": matchmaker.__version__,
                        "method": method,
                        "processor": processor,
                        "dataset": p["dataset"],
                        "piece": p["piece"],
                        "run": run,
                        "n_observations": len(inference),
                        "performance_duration_s": duration,
                        "latency_ms": {"feature": latency(feature), "inference": latency(inference)},
                        "total_processing_s": total,
                        "real_time_factor": total / duration if duration else None,
                        "evaluation": evaluation,
                        "environment": environment,
                    }
                )
                print(f"{p['piece']} {method} run {run}: {len(inference)} observations, {total:.4f} s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
