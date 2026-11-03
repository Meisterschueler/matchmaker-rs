"""Compare an export of matchmaker-rs (``matchmaker follow --export DIR``) with a golden
directory, stage by stage (score, stream, processor, method), and report the first difference
(US6, SC-007).

    python -m uv run compare.py ../../tests/golden/<piece> <export_dir>

Only files present on both sides are compared. Tolerances follow contracts/golden-data.md:
positions and alignment path 1e-6, probabilities rtol per method (pthmm 1.5e-13, hmm 1e-13,
outerhmm 1e-14; arzt and dixon have none), beat_period exact, evaluation
metrics 1e-4 (hmm: 0.01), the rest exact; messages are compared as a multiset (the reference sorts them with an unstable argsort)
and note arrays on the fields both sides have.
"""

import json
import sys
from pathlib import Path

import numpy as np

METHODS = ("pthmm", "hmm", "outerhmm", "arzt", "dixon")
STAGES = ("score", "stream", "processor", *METHODS)
# file name -> (rtol, atol); everything else must be equal (`beat_period.npy` included)
TOLERANCES = {
    "positions.npy": (0.0, 1e-6),
    "alignment_path.npy": (0.0, 1e-6),
}
# rtol of probabilities.npy per method (tests/common/mod.rs::probabilities_rtol)
PROBABILITIES_RTOL = {"pthmm": 1.5e-13, "hmm": 1e-13, "outerhmm": 1e-14}


def first_difference(name, expected, actual, stage=None, tolerance=None):
    if expected.shape != actual.shape:
        return f"shape differs: expected {expected.shape}, actual {actual.shape}"
    rtol, atol = tolerance or TOLERANCES.get(name, (0.0, 0.0))
    if name == "probabilities.npy":
        rtol, atol = PROBABILITIES_RTOL[stage], 1e-300
    bad = ~np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    if not bad.any():
        return None
    index = tuple(np.argwhere(bad)[0])
    return f"first difference at index {index}: expected {expected[index]!r}, actual {actual[index]!r}"


def metric_difference(expected, actual, tolerance):
    """NaN (Python) and null (Rust) both mean "no value"."""
    missing = lambda v: v is None or v != v  # noqa: E731
    for key, e in expected.items():
        a = actual.get(key)
        if missing(e) or missing(a):
            same = missing(e) and missing(a)
        else:
            same = abs(e - a) <= tolerance
        if not same:
            return f"{key}: expected {e!r}, actual {a!r}"
    return None


def compare_file(stage, rel, expected_path, actual_path):
    if rel.suffix == ".json":
        expected, actual = (json.loads(p.read_text()) for p in (expected_path, actual_path))
        if rel.name == "evaluation.json":
            return metric_difference(expected, actual, 0.01 if stage == "hmm" else 1e-4)
        if rel.name != "note_ids.json":
            return None
        if expected == actual:
            return None
        index = next((i for i, (e, a) in enumerate(zip(expected, actual)) if e != a), min(len(expected), len(actual)))
        return f"note ids differ at index {index} ({len(expected)} expected, {len(actual)} actual)"
    expected, actual = np.load(expected_path), np.load(actual_path)
    if expected.dtype.names is None:
        return first_difference(rel.name, expected, actual, stage)
    fields = [f for f in expected.dtype.names if f in actual.dtype.names]
    if rel.name == "messages.npy":
        order = ["time", *(f for f in fields if f != "time")]
        expected, actual = np.sort(expected, order=order), np.sort(actual, order=order)
    for field in fields:
        difference = first_difference(rel.name, expected[field], actual[field], stage)
        if difference:
            return f"field {field}: {difference}"
    return None


def compare_audio_case(golden_dir, export_dir, metadata):
    method = metadata["method"]
    processor = metadata["processor"]
    mapping = {
        "performance_features.npy": Path("processor") / processor / "features.npy",
        "performance_times.npy": Path("processor") / processor / "times.npy",
        "score_features.npy": Path("score/audio/features.npy"),
        "score_frame_to_beat.npy": Path("score/audio/frame_to_beat.npy"),
        "score_positions.npy": Path("score/audio/score_positions.npy"),
        "positions.npy": Path(method) / "positions.npy",
        "alignment_path.npy": Path(method) / "alignment_path.npy",
    }
    for golden_name, export_rel in mapping.items():
        expected = np.load(golden_dir / golden_name)
        exported_path = export_dir / export_rel
        if not exported_path.is_file():
            print(f"first differing stage: {export_rel.parent}, missing {export_rel}")
            return 1
        actual = np.load(exported_path)

        if golden_name == "alignment_path.npy":
            if expected.ndim != 2 or actual.ndim != 2 or expected.shape[0] != 2 or actual.shape[0] != 2:
                print(f"first differing stage: {method}, file {golden_name}: expected a (2, T) path")
                return 1
            frame_count = min(expected.shape[1], actual.shape[1])
            frame_delta = abs(expected.shape[1] - actual.shape[1])
            if frame_delta > 1:
                print(f"first differing stage: {method}, file {golden_name}: frame count differs by {frame_delta}")
                return 1
            differences = (
                first_difference("perf_time.npy", expected[0, :frame_count], actual[0, :frame_count], tolerance=(0.0, 1e-6)),
                first_difference("score_beat.npy", expected[1, :frame_count], actual[1, :frame_count], tolerance=(0.0, 0.6)),
            )
        else:
            frame_series = golden_name != "score_positions.npy"
            if frame_series and expected.ndim and actual.ndim and expected.shape[1:] == actual.shape[1:]:
                frame_count = min(expected.shape[0], actual.shape[0])
                frame_delta = abs(expected.shape[0] - actual.shape[0])
                if frame_delta > 1:
                    print(f"first differing stage: {export_rel.parent}, file {golden_name}: frame count differs by {frame_delta}")
                    return 1
                expected, actual = expected[:frame_count], actual[:frame_count]
            tolerance = {
                "performance_features.npy": (2e-4, 1e-6),
                "performance_times.npy": (0.0, 0.0),
                "score_frame_to_beat.npy": (0.0, 1e-6),
                "positions.npy": (0.0, 0.5),
            }.get(golden_name, (0.0, 0.0))
            differences = (first_difference(golden_name, expected, actual, tolerance=tolerance),)

        difference = next((value for value in differences if value), None)
        if difference:
            print(f"first differing stage: {export_rel.parent}, file {golden_name}: {difference}")
            return 1
    print("all audio stages within configured tolerances (7 files compared)")
    return 0


def main(golden_dir, export_dir):
    meta = golden_dir / "meta.json"
    if meta.is_file():
        metadata = json.loads(meta.read_text())
        if metadata.get("method") in ("arzt", "dixon") and metadata.get("processor") in ("chroma", "lse"):
            return compare_audio_case(golden_dir, export_dir, metadata)

    compared = 0
    for stage in STAGES:
        golden, export = golden_dir / stage, export_dir / stage
        if not (golden.is_dir() and export.is_dir()):
            continue
        for expected in sorted(p for p in golden.rglob("*") if p.is_file()):
            rel = expected.relative_to(golden)
            actual = export / rel
            if not actual.is_file():
                continue
            compared += 1
            difference = compare_file(stage, rel, expected, actual)
            if difference:
                print(f"first differing stage: {stage}, file {stage}/{rel.as_posix()}: {difference}")
                return 1
    if compared == 0:
        print("nothing to compare: no file exists on both sides")
        return 1
    print(f"all stages equal ({compared} files compared)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
