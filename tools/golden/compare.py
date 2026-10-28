"""Compare an export of matchmaker-rs (``matchmaker follow --export DIR``) with a golden
directory, stage by stage (score, stream, processor, method), and report the first difference
(US6, SC-007).

    python -m uv run compare.py ../../tests/golden/<piece> <export_dir>

Only files present on both sides are compared. Tolerances follow contracts/golden-data.md:
positions and alignment path 1e-6, probabilities rtol 1e-12, beat_period rtol 1e-9, the rest
exact; messages are compared as a multiset (the reference sorts them with an unstable argsort)
and note arrays on the fields both sides have.
"""

import json
import sys
from pathlib import Path

import numpy as np

METHODS = ("pthmm", "hmm", "outerhmm")
STAGES = ("score", "stream", "processor", *METHODS)
# file name -> (rtol, atol); everything else must be equal
TOLERANCES = {
    "probabilities.npy": (1e-12, 1e-300),
    "beat_period.npy": (0.0, 1e-9),
    "positions.npy": (0.0, 1e-6),
    "alignment_path.npy": (0.0, 1e-6),
}


def first_difference(name, expected, actual):
    if expected.shape != actual.shape:
        return f"shape differs: expected {expected.shape}, actual {actual.shape}"
    rtol, atol = TOLERANCES.get(name, (0.0, 0.0))
    bad = ~np.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    if not bad.any():
        return None
    index = tuple(np.argwhere(bad)[0])
    return f"first difference at index {index}: expected {expected[index]!r}, actual {actual[index]!r}"


def compare_file(rel, expected_path, actual_path):
    if rel.suffix == ".json":
        if rel.name != "note_ids.json":
            return None
        expected, actual = (json.loads(p.read_text()) for p in (expected_path, actual_path))
        if expected == actual:
            return None
        index = next((i for i, (e, a) in enumerate(zip(expected, actual)) if e != a), min(len(expected), len(actual)))
        return f"note ids differ at index {index} ({len(expected)} expected, {len(actual)} actual)"
    expected, actual = np.load(expected_path), np.load(actual_path)
    if expected.dtype.names is None:
        return first_difference(rel.name, expected, actual)
    fields = [f for f in expected.dtype.names if f in actual.dtype.names]
    if rel.name == "messages.npy":
        order = ["time", *(f for f in fields if f != "time")]
        expected, actual = np.sort(expected, order=order), np.sort(actual, order=order)
    for field in fields:
        difference = first_difference(rel.name, expected[field], actual[field])
        if difference:
            return f"field {field}: {difference}"
    return None


def main(golden_dir, export_dir):
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
            difference = compare_file(rel, expected, actual)
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
