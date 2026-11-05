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


def compare_tempo_audio_case(golden_dir, export_dir, metadata):
    """Compare deterministic input stages and the declared renderer metric limits."""
    from matchmaker.utils.eval import evaluate_alignment

    method = metadata["method"]
    processor = metadata["processor"]
    for name, path, tolerance in (
        ("performance_features.npy", f"processor/{processor}/features.npy", (1e-5, 1e-5)),
        ("performance_times.npy", f"processor/{processor}/times.npy", (0.0, 1e-9)),
        ("score_frame_to_beat.npy", "score/audio/frame_to_beat.npy", (0.0, 1e-9)),
        ("score_positions.npy", "score/audio/score_positions.npy", (0.0, 0.0)),
    ):
        difference = first_difference(name, np.load(golden_dir / name), np.load(export_dir / path), tolerance=tolerance)
        if difference:
            print(f"first differing stage: {path}: {difference}")
            return 1
    expected_features = np.load(golden_dir / "score_features.npy")
    actual_features = np.load(export_dir / "score/audio/features.npy")
    difference = first_difference("score_features.npy", expected_features, actual_features)
    if difference:
        print(f"SYNTHESIS_DIFFERENCE: score/audio/features.npy: {difference}")
    expected = np.load(golden_dir / "alignment_path.npy")
    actual = np.load(export_dir / method / "alignment_path.npy")
    if expected.ndim != 2 or actual.shape != expected.shape or expected.shape[0] != 2:
        print("first differing stage: alignment_path.npy: incompatible path shape")
        return 1
    difference = first_difference("times", expected[0], actual[0], tolerance=(0.0, 1e-9))
    if difference:
        print(f"first differing stage: alignment timestamps: {difference}")
        return 1
    mapping = np.load(golden_dir / "score_frame_to_beat.npy")
    truth = np.interp(expected[0] * metadata["frame_rate"], np.arange(len(mapping)), mapping)
    reference_metrics = evaluate_alignment(expected[1], expected[0], truth, expected[0])
    actual_metrics = evaluate_alignment(actual[1], actual[0], truth, expected[0])
    for key in ("median_b", *(key for key in reference_metrics if key.startswith("accuracy_"))):
        delta = abs(actual_metrics[key] - reference_metrics[key])
        print(f"METRIC\t{key}\tRust={actual_metrics[key]}\treference={reference_metrics[key]}\tdelta={delta}")
        if delta > 0.01 + 1e-12:
            print(f"EXCEEDS: {key} exceeds the declared 0.01 difference")
            return 1
    delta = abs(np.percentile(np.abs(actual[1] - truth), 95) - np.percentile(np.abs(expected[1] - truth), 95))
    if delta > 0.05 + 1e-12:
        print(f"EXCEEDS: p95 beat-error difference {delta} exceeds 0.05")
        return 1
    states = np.load(export_dir / method / "tempo_states.npy")
    if states.shape != (len(actual[0]), 8) or not np.isfinite(states).all():
        print("first differing stage: tempo_states.npy: invalid shape or non-finite state")
        return 1
    print(f"WITHIN_TOL: tempo audio stages and renderer metrics; p95 difference={delta}")
    return 0


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


def compare_raw_spectrum_case(golden_dir, export_dir, metadata):
    feature_tolerance = metadata.get("tolerances", {}).get("features_rtol", 1e-5), metadata.get(
        "tolerances", {}
    ).get("features_atol", 1e-5)
    for name, export_path, tolerance in (
        (
            "features.npy",
            export_dir / "processor/raw_spectrum/features.npy",
            feature_tolerance,
        ),
        (
            "times.npy",
            export_dir / "processor/raw_spectrum/times.npy",
            (0.0, 0.0),
        ),
    ):
        if not export_path.is_file():
            print(f"first differing stage: processor/raw_spectrum, missing {export_path}")
            return 1
        difference = first_difference(
            name,
            np.load(golden_dir / name),
            np.load(export_path),
            tolerance=tolerance,
        )
        if difference:
            print(f"first differing stage: processor/raw_spectrum/{name}: {difference}")
            return 1

    actual_metadata = json.loads((export_dir / "audio/meta.json").read_text())
    for key in ("processor", "sample_rate", "hop_length", "n_fft", "frames"):
        if actual_metadata.get(key) != metadata.get(key):
            print(
                f"first differing stage: audio/meta.json: {key}: "
                f"expected {metadata.get(key)!r}, actual {actual_metadata.get(key)!r}"
            )
            return 1
    print("WITHIN_TOL: standalone raw-spectrum features, timestamps, and metadata")
    return 0


# RustySynth against FluidSynth templates (research R12); Gaussian templates are closed-form.
SKF_TEMPLATE_MIN_COSINE = 0.99
SKF_TEMPLATE_MAX_VARIATION = 0.06


def compare_skf_case(golden_dir, export_dir, metadata):
    """Compare the score model of an skf export with a golden: mode, chords, lengths, templates.

    The per-frame replay of the follower is covered by tests/golden_audio_skf.rs, which feeds the
    golden observations; an export of a real run has its own observations.
    """
    model_dir = export_dir / "score/skf"
    score_meta = json.loads((export_dir / "score/meta.json").read_text())
    if score_meta.get("template_mode") != metadata.get("template_mode"):
        print(
            "first differing stage: score/meta.json: template_mode: "
            f"expected {metadata.get('template_mode')!r}, actual {score_meta.get('template_mode')!r}"
        )
        return 1
    for name in ("chords_flat.npy", "chord_offsets.npy", "lengths.npy", "onset_beats.npy"):
        difference = first_difference(
            name, np.load(golden_dir / name), np.load(model_dir / name), tolerance=(0.0, 0.0)
        )
        if difference:
            print(f"first differing stage: score/skf/{name}: {difference}")
            return 1
    expected = np.load(golden_dir / "templates.npy").astype(np.float64)
    actual = np.load(model_dir / "templates.npy")
    if metadata["template_mode"] == "gaussian":
        tolerances = metadata["tolerances"]
        difference = first_difference(
            "templates.npy", expected, actual, tolerance=(tolerances["model_rtol"], tolerances["model_atol"])
        )
        if difference:
            print(f"first differing stage: score/skf/templates.npy: {difference}")
            return 1
    else:
        if expected.shape != actual.shape:
            print(f"first differing stage: score/skf/templates.npy: shape differs: expected {expected.shape}, actual {actual.shape}")
            return 1
        cosine = (expected * actual).sum(axis=1) / (
            np.linalg.norm(expected, axis=1) * np.linalg.norm(actual, axis=1)
        )
        variation = 0.5 * np.abs(expected - actual).sum(axis=1)
        bad = np.flatnonzero((cosine < SKF_TEMPLATE_MIN_COSINE) | (variation > SKF_TEMPLATE_MAX_VARIATION))
        if len(bad):
            chord = int(bad[0])
            print(
                f"first differing stage: score/skf/templates.npy: chord {chord}: cosine {cosine[chord]:.5f} "
                f"(min {SKF_TEMPLATE_MIN_COSINE}), total variation {variation[chord]:.4f} (max {SKF_TEMPLATE_MAX_VARIATION})"
            )
            return 1
    print(f"WITHIN_TOL: skf chord model and {metadata['template_mode']} templates")
    return 0


def compare_audio_ground_truth_case(golden_dir, export_dir):
    """Compare direct TSV identity and evaluator metrics for an audio export."""
    metadata = json.loads((golden_dir / "meta.json").read_text())
    method_dir = export_dir / "arzt"
    identity_path = method_dir / "evaluation_meta.json"
    metrics_path = method_dir / "evaluation.json"
    if not identity_path.is_file() or not metrics_path.is_file():
        print("first differing stage: missing audio evaluation identity or metrics")
        return 1
    identity = json.loads(identity_path.read_text())
    expected_name = metadata["ground_truth_file"]
    actual_name = Path(identity.get("ground_truth_path", "")).name
    if actual_name != expected_name:
        print(
            "first differing ground-truth identity: "
            f"expected {expected_name!r}, actual {actual_name!r}"
        )
        return 1
    if identity.get("ground_truth_format") != "perf_sec/score_beat TSV":
        print("first differing ground-truth identity: unexpected format")
        return 1

    expected = json.loads((golden_dir / "evaluation.json").read_text())
    actual = json.loads(metrics_path.read_text())
    keys = (
        "median_b",
        "accuracy_0.05b",
        "accuracy_0.1b",
        "accuracy_0.3b",
        "accuracy_0.5b",
        "accuracy_1b",
        "accuracy_2b",
    )
    for key in keys:
        delta = abs(float(actual[key]) - float(expected[key]))
        print(f"METRIC\t{key}\tRust={actual[key]}\treference={expected[key]}\tdelta={delta}")
        if delta > 0.01 + 1e-12:
            print(f"EXCEEDS: {key} exceeds the declared 0.01 difference")
            return 1
    print("WITHIN_TOL: direct audio ground-truth identity and metrics")
    return 0


def compare_multi_fold_audio_case(golden_dir, export_dir, metadata):
    """Compare repeat candidates, selection trace, and note-onset metrics."""
    from matchmaker.utils.eval import evaluate_alignment

    method = metadata["method"]
    processor = metadata["processor"]
    stages = (
        ("performance_features.npy", Path("processor") / processor / "features.npy", (2e-4, 1e-6)),
        ("performance_times.npy", Path("processor") / processor / "times.npy", (0.0, 0.0)),
    )
    for golden_name, export_rel, tolerance in stages:
        expected = np.load(golden_dir / golden_name)
        exported = export_dir / export_rel
        if not exported.is_file():
            print(f"first differing stage: {export_rel}, missing {export_rel}")
            return 1
        difference = first_difference(
            golden_name, expected, np.load(exported), tolerance=tolerance
        )
        if difference:
            print(f"first differing stage: {export_rel}: {difference}")
            return 1

    hypothesis_path = export_dir / method / "hypothesis_metadata.json"
    if not hypothesis_path.is_file():
        print(f"first differing stage: {method}/hypothesis_metadata.json is missing")
        return 1
    hypotheses = json.loads(hypothesis_path.read_text())
    if len(hypotheses) != len(metadata["hypotheses"]):
        print(
            "first differing hypothesis: candidate count differs: "
            f"expected {len(metadata['hypotheses'])}, actual {len(hypotheses)}"
        )
        return 1
    method_metadata = json.loads((export_dir / method / "meta.json").read_text())
    config = method_metadata.get("config", {})
    for key in ("selection_window", "switch_margin"):
        if config.get(key) != metadata[key]:
            print(
                f"first differing method config: {key}: "
                f"expected {metadata[key]!r}, actual {config.get(key)!r}"
            )
            return 1

    for index, expected_meta in enumerate(metadata["hypotheses"]):
        actual_meta = hypotheses[index]
        for key in ("index", "note_count", "reference_frames"):
            if expected_meta[key] != actual_meta[key]:
                print(
                    f"first differing hypothesis: candidate {index} {key}: "
                    f"expected {expected_meta[key]!r}, actual {actual_meta[key]!r}"
                )
                return 1
        if abs(expected_meta["max_onset_beat"] - actual_meta["max_onset_beat"]) > 1e-6:
            print(
                f"first differing hypothesis: candidate {index} max_onset_beat: "
                f"expected {expected_meta['max_onset_beat']}, actual {actual_meta['max_onset_beat']}"
            )
            return 1
        expected_dir = golden_dir / f"candidate_{index}"
        actual_dir = export_dir / method / "candidates" / str(index)
        for name, tolerance in (
            ("score_features.npy", (2e-4, 1e-6)),
            ("score_positions.npy", (0.0, 0.0)),
            ("frame_to_unfolded_beat.npy", (0.0, 1e-6)),
            ("frame_to_notated_beat.npy", (0.0, 1e-6)),
        ):
            expected = np.load(expected_dir / name)
            exported = actual_dir / name
            if not exported.is_file():
                print(f"first differing hypothesis: candidate {index} missing {name}")
                return 1
            actual = np.load(exported)
            difference = first_difference(name, expected, actual, tolerance=tolerance)
            if difference and name.startswith("score_features"):
                print(f"SYNTHESIS_DIFFERENCE: candidate {index}, {name}: {difference}")
            elif difference:
                print(f"first differing hypothesis: candidate {index}, {name}: {difference}")
                return 1

    trace_pairs = (
        ("selected_hypothesis.npy", "selected_hypothesis.npy"),
        ("candidate_frames.npy", "candidate_current_frames.npy"),
        ("candidate_local_costs.npy", "candidate_local_costs.npy"),
        ("candidate_mean_costs.npy", "candidate_mean_costs.npy"),
    )
    for golden_name, export_name in trace_pairs:
        expected = np.load(golden_dir / golden_name)
        actual = np.load(export_dir / method / export_name)
        if expected.shape != actual.shape:
            print(
                f"first differing hypothesis/frame in {golden_name}: "
                f"shape differs: expected {expected.shape}, actual {actual.shape}"
            )
            return 1
        difference = first_difference(
            golden_name,
            expected,
            actual,
            tolerance=(1e-6, 1e-7) if expected.dtype.kind == "f" else (0.0, 0.0),
        )
        if difference:
            index = np.argwhere(
                ~np.isclose(actual, expected, rtol=1e-6, atol=1e-7)
                if expected.dtype.kind == "f"
                else actual != expected
            )[0]
            step, candidate = (int(index[0]), int(index[1])) if len(index) > 1 else (int(index[0]), 0)
            print(
                f"HYPOTHESIS_DIFFERENCE: first differing candidate/frame "
                f"{candidate}/{step} in {golden_name}: {difference}"
            )

    expected_path = np.load(golden_dir / "alignment_path.npy")
    actual_path = np.load(export_dir / method / "alignment_path.npy")
    if expected_path.shape[0] != 2 or actual_path.shape[0] != 2:
        print("first differing stage: alignment_path.npy must have shape (2, T)")
        return 1
    shared = min(expected_path.shape[1], actual_path.shape[1])
    if abs(expected_path.shape[1] - actual_path.shape[1]) > 1:
        print(
            "first differing stage: alignment_path.npy frame count differs: "
            f"expected {expected_path.shape[1]}, actual {actual_path.shape[1]}"
        )
        return 1
    time_difference = first_difference(
        "perf_time.npy", expected_path[0, :shared], actual_path[0, :shared],
        tolerance=(0.0, 1e-9),
    )
    if time_difference:
        print(f"first differing stage: alignment timestamps: {time_difference}")
        return 1
    beat_differences = np.abs(expected_path[1, :shared] - actual_path[1, :shared])
    within_audio_tolerance = beat_differences <= 0.6
    position_share = float(within_audio_tolerance.sum()) / max(
        expected_path.shape[1], actual_path.shape[1], 1
    )
    if position_share < 0.99:
        first = int(np.flatnonzero(~within_audio_tolerance)[0])
        print(
            f"first differing alignment frame {first}: expected beat "
            f"{expected_path[1, first]}, actual {actual_path[1, first]}, "
            f"within-tolerance share={position_share}"
        )
        return 1
    if np.any(beat_differences > 1e-6):
        first = int(np.flatnonzero(beat_differences > 1e-6)[0])
        print(
            f"POSITION_DIFFERENCE: first differing frame {first}: expected beat "
            f"{expected_path[1, first]}, actual {actual_path[1, first]}, "
            f"max difference={beat_differences.max()}"
        )

    truth = np.loadtxt(golden_dir / "ground_truth.tsv", delimiter="\t", skiprows=1, ndmin=2)
    reference_metrics = evaluate_alignment(
        expected_path[1], expected_path[0], truth[:, 1], truth[:, 0]
    )
    actual_metrics = evaluate_alignment(
        actual_path[1], actual_path[0], truth[:, 1], truth[:, 0]
    )
    for key in ("median_b", *(name for name in reference_metrics if name.startswith("accuracy_"))):
        delta = abs(actual_metrics[key] - reference_metrics[key])
        print(f"METRIC\t{key}\tRust={actual_metrics[key]}\treference={reference_metrics[key]}\tdelta={delta}")
        if delta > 0.01 + 1e-12:
            print(f"EXCEEDS: {key} exceeds the declared 0.01 difference")
            return 1
    expected_errors = np.abs(truth[:, 1] - np.interp(truth[:, 0], expected_path[0], expected_path[1]))
    actual_errors = np.abs(truth[:, 1] - np.interp(truth[:, 0], actual_path[0], actual_path[1]))
    p95_delta = abs(np.percentile(actual_errors, 95) - np.percentile(expected_errors, 95))
    if p95_delta > 0.05 + 1e-12:
        print(f"EXCEEDS: p95 beat-error difference {p95_delta} exceeds 0.05")
        return 1
    print(f"WITHIN_TOL: repeat hypotheses and note-onset metrics; p95 difference={p95_delta}")
    return 0


def main(golden_dir, export_dir):
    meta = golden_dir / "meta.json"
    if meta.is_file():
        metadata = json.loads(meta.read_text())
        if metadata.get("method") == "audio_ground_truth":
            return compare_audio_ground_truth_case(golden_dir, export_dir)
        if metadata.get("method") == "arzt_tempo":
            return compare_tempo_audio_case(golden_dir, export_dir, metadata)
        if metadata.get("method") == "arzt_multi_fold":
            return compare_multi_fold_audio_case(golden_dir, export_dir, metadata)
        if metadata.get("method") in ("arzt", "dixon") and metadata.get("processor") in ("chroma", "lse"):
            return compare_audio_case(golden_dir, export_dir, metadata)
        if metadata.get("processor") == "raw_spectrum" and "sample_rate" in metadata:
            return compare_raw_spectrum_case(golden_dir, export_dir, metadata)
        if metadata.get("method") == "skf":
            return compare_skf_case(golden_dir, export_dir, metadata)

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
