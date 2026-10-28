"""Generate golden data for matchmaker-rs from the locked reference release.

Layout and dtypes: specs/001-midi-score-following/contracts/golden-data.md

Usage (from tools/golden):
    python -m uv run generate.py --data ../../tests/data --out ../../tests/golden
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

import mido
import numpy as np
import partitura
from partitura.score import merge_parts

import matchmaker
from matchmaker import Matchmaker
from matchmaker.features.midi import ChordOnsetProcessor, PitchProcessor
from matchmaker.matchmaker import DEFAULT_TEMPO
from matchmaker.utils.eval import evaluate_alignment, resolve_gt
from matchmaker.utils.misc import get_tempo_from_score
from matchmaker.utils.symbolic import (
    midi_messages_from_performance,
    midi_messages_to_framed_midi,
)

REFERENCE_TAG = "v0.4.0"
PIECES = ["simple_mozart_k265_var1", "mozart_k265_var1", "Bach-fugue_bwv_858"]
METHODS = ["pthmm", "hmm", "outerhmm"]
MAX_PROBABILITY_VALUES = 5_000_000

KIND_CODES = {"note_on": 0, "note_off": 1, "control_change": 2, "program_change": 3}
MESSAGE_DTYPE = np.dtype(
    [
        ("kind", "<i4"),
        ("channel", "<i4"),
        ("note", "<i4"),
        ("velocity", "<i4"),
        ("control", "<i4"),
        ("value", "<i4"),
        ("program", "<i4"),
        ("time", "<f8"),
    ]
)
NOTE_FIELDS = [
    ("onset_beat", "<f4"),
    ("duration_beat", "<f4"),
    ("onset_quarter", "<f4"),
    ("duration_quarter", "<f4"),
    ("onset_div", "<i4"),
    ("duration_div", "<i4"),
    ("pitch", "<i4"),
    ("voice", "<i4"),
]
TOLERANCES = {
    "positions": 1e-6,
    "probabilities_rtol": 1e-12,
    "beat_period_rtol": 1e-9,
    "f32_rtol": 1e-6,
    "metrics_abs": 1e-4,
}


# --------------------------------------------------------------------------- helpers
def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save(path: Path, array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(array))


def lockfile_sha256() -> str:
    lock = Path(__file__).with_name("uv.lock")
    return hashlib.sha256(lock.read_bytes()).hexdigest()


def reference_meta() -> dict:
    return {
        "package": "pymatchmaker",
        "tag": REFERENCE_TAG,
        "version": matchmaker.__version__,
        "partitura": version("partitura"),
        "numpy": np.__version__,
        "lockfile_sha256": lockfile_sha256(),
    }


def pad_rows(rows: list[np.ndarray], fill: float = -1.0) -> np.ndarray:
    """Stack 1-D float32 arrays of different lengths, padding with ``fill``."""
    width = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), width), fill, dtype=np.float32)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


# --------------------------------------------------------------------------- score
def load_score_like_matchmaker(path: Path, unfold: bool = True):
    """Same steps as ``Matchmaker.__init__``; also report whether unfolding succeeded."""
    ext = path.suffix.lower()
    if ext in (".musicxml", ".xml", ".mxl"):
        score = partitura.load_musicxml(
            str(path), ignore_invisible_objects=True, force_note_ids="keep"
        )
    else:
        score = partitura.load_score(str(path))
    if unfold:
        sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
        try:
            unfolded = partitura.score.unfold_part_maximal(score, ignore_leaps=False)
            return score, merge_parts(unfolded.parts), True
        except Exception:
            pass
    return score, merge_parts(score.parts), False


def coverage_report(name: str, score) -> dict:
    """Which score features (FR-002, US1-3) this input exercises."""
    sc = partitura.score
    parts = list(score.parts)
    features = {
        "multiple_parts": len(parts) > 1,
        "repeats": any(True for p in parts for _ in p.iter_all(sc.Repeat)),
        "endings": any(True for p in parts for _ in p.iter_all(sc.Ending)),
        "segno_coda_dacapo": any(
            True
            for p in parts
            for cls in (sc.Segno, sc.Coda, sc.ToCoda, sc.DaCapo, sc.DalSegno, sc.Fine)
            for _ in p.iter_all(cls)
        ),
        "grace_notes": any(True for p in parts for _ in p.iter_all(sc.GraceNote)),
        "ties": any(n.tie_next is not None for p in parts for n in p.notes),
        "multiple_voices": any(len({n.voice for n in p.notes}) > 1 for p in parts),
        "time_signature_changes": any(
            len(list(p.iter_all(sc.TimeSignature))) > 1 for p in parts
        ),
        "pickup_measure": False,
    }
    for p in parts:
        measures = list(p.iter_all(sc.Measure))
        ts = next(p.iter_all(sc.TimeSignature), None)
        if measures and ts is not None:
            m = measures[0]
            q = p.quarter_duration_map(m.start.t)
            dur_quarters = (m.end.t - m.start.t) / q
            if dur_quarters < ts.beats * 4 / ts.beat_type:
                features["pickup_measure"] = True
    print(f"  coverage {name}: " + ", ".join(f"{k}={'Y' if v else '-'}" for k, v in features.items()))
    return features


def dump_score(out: Path, score_path: Path) -> dict:
    score, part, unfolded = load_score_like_matchmaker(score_path)
    na = part.note_array()
    arr = np.zeros(len(na), dtype=NOTE_FIELDS)
    for field, _ in NOTE_FIELDS:
        arr[field] = na[field]
    save(out / "score" / "note_array.npy", arr)
    write_json(out / "score" / "note_ids.json", [str(i) for i in na["id"]])
    features = coverage_report(score_path.name, score)
    score_tempo = get_tempo_from_score(part, str(score_path))
    write_json(
        out / "score" / "meta.json",
        {
            "score_tempo": score_tempo,
            "tempo": float(score_tempo) if score_tempo is not None else float(DEFAULT_TEMPO),
            "unfolded": unfolded,
            "features": features,
        },
    )
    return {"part": part, "unfolded": unfolded, "features": features, "note_array": na}


# --------------------------------------------------------------------------- stream + processors
def message_record(msg, t: float):
    kind = KIND_CODES[msg.type]
    return (
        kind,
        getattr(msg, "channel", 0),
        getattr(msg, "note", -1),
        getattr(msg, "velocity", -1),
        getattr(msg, "control", -1),
        getattr(msg, "value", -1),
        getattr(msg, "program", -1),
        float(t),
    )


def dump_stream(out: Path, perf_path: Path) -> dict:
    msgs, times = midi_messages_from_performance(str(perf_path))
    event = np.array([message_record(m, t) for m, t in zip(msgs, times)], dtype=MESSAGE_DTYPE)
    save(out / "stream" / "event" / "messages.npy", event)
    framed = {}
    for pp, label in ((0.01, "windowed_0.01"), (0.001, "windowed_0.001")):
        frames, frame_times = midi_messages_to_framed_midi(msgs, times, pp)
        records, offsets = [], [0]
        for buf in frames:
            records.extend(message_record(m, t) for m, t in buf.frame)
            offsets.append(len(records))
        save(out / "stream" / label / "messages.npy", np.array(records, dtype=MESSAGE_DTYPE))
        save(out / "stream" / label / "frame_offsets.npy", np.array(offsets, dtype="<i8"))
        save(out / "stream" / label / "frame_times.npy", np.asarray(frame_times, dtype="<f8"))
        framed[pp] = frames
    return {"messages": msgs, "times": times, "frames": framed}


def run_processor_windowed(proc, frames):
    outputs = []
    for buf in frames:
        o = proc((buf.frame[:], buf.time))
        if o is not None:
            outputs.append(o)
    if hasattr(proc, "flush_remaining"):
        last = proc.flush_remaining()
        if last is not None:
            outputs.append(last)
    return outputs


def run_processor_event(proc, msgs, times):
    outputs = []
    for m, t in zip(msgs, times):
        o = proc(([(m, t)], t))
        if o is not None:
            outputs.append(o)
    return outputs


def save_observations(directory: Path, outputs, padded: bool) -> None:
    feats = [np.asarray(f, dtype=np.float32) for f, _ in outputs]
    if padded:
        features = pad_rows(feats)
    else:
        features = np.stack(feats) if feats else np.zeros((0, 0), dtype=np.float32)
    save(directory / "features.npy", features)
    save(directory / "times.npy", np.array([t for _, t in outputs], dtype="<f8"))


def dump_processors(out: Path, stream: dict) -> None:
    f10, f1 = stream["frames"][0.01], stream["frames"][0.001]
    configs = {
        "pitch_piano_vector": (lambda: PitchProcessor(True, False), f10, False),
        "pitch_piano_list": (lambda: PitchProcessor(True, True), f10, True),
        "pitch_full_vector": (lambda: PitchProcessor(False, False), f10, False),
        "chord_onset_piano": (lambda: ChordOnsetProcessor(piano_range=True), f1, False),
    }
    for name, (make, frames, padded) in configs.items():
        save_observations(out / "processor" / name, run_processor_windowed(make(), frames), padded)
    outputs = run_processor_event(PitchProcessor(True, False), stream["messages"], stream["times"])
    save_observations(out / "processor" / "pitch_piano_vector_event", outputs, False)


# --------------------------------------------------------------------------- methods
class Recorder:
    """Records what the follower receives and its state after every call."""

    def __init__(self):
        self.features, self.times = [], []
        self.positions, self.indices = [], []
        self.probabilities, self.beat_period, self.is_continuation = [], [], []

    def install(self, follower) -> None:
        recorder = self
        base = type(follower)

        def __call__(self, observation, perf_time, *args, **kwargs):
            position = base.__call__(self, observation, perf_time, *args, **kwargs)
            recorder.features.append(np.asarray(observation, dtype=np.float32).copy())
            recorder.times.append(float(perf_time))
            recorder.positions.append(float(position))
            recorder.indices.append(int(self.current_index))
            probs = getattr(self, "forward_variable", None)
            if probs is None:
                probs = getattr(self, "state_probabilities", None)
            recorder.probabilities.append(None if probs is None else np.array(probs, dtype="<f8"))
            tempo = getattr(self, "tempo_model", None)
            recorder.beat_period.append(float(tempo.beat_period) if tempo is not None else np.nan)
            recorder.is_continuation.append(bool(getattr(self, "_is_continuation", False)))
            return position

        follower.__class__ = type(f"Recording{base.__name__}", (base,), {"__call__": __call__})


def evaluation(mm: Matchmaker, path: np.ndarray, match_path: Path, perf_path: Path) -> dict:
    gt_perf, gt_score = resolve_gt(str(match_path), mm.score_part.note_array(), performance=str(perf_path))
    return evaluate_alignment(path[1], path[0], gt_score, gt_perf)


def dump_method(out: Path, method: str, score_path: Path, perf_path: Path, match_path: Path | None) -> None:
    mm = Matchmaker(
        score_file=str(score_path),
        performance_file=str(perf_path),
        input_type="midi",
        method=method,
    )
    rec = Recorder()
    rec.install(mm.score_follower)
    yielded = list(mm.run(verbose=False))
    assert yielded == rec.positions, "recorded positions differ from yielded positions"

    d = out / method
    follower = mm.score_follower
    save(d / "score_positions.npy", np.asarray(follower.score_positions))
    is_list = method == "hmm"
    save(d / "observations_features.npy", pad_rows(rec.features) if is_list else (np.stack(rec.features) if rec.features else np.zeros((0, 88), np.float32)))
    save(d / "observations_times.npy", np.array(rec.times, dtype="<f8"))
    save(d / "positions.npy", np.array(rec.positions, dtype="<f8"))
    save(d / "current_index.npy", np.array(rec.indices, dtype="<i8"))
    n_states = len(follower.score_positions)
    if rec.probabilities and rec.probabilities[0] is not None and len(rec.probabilities) * n_states <= MAX_PROBABILITY_VALUES:
        save(d / "probabilities.npy", np.stack(rec.probabilities))
    if method == "hmm":
        save(d / "beat_period.npy", np.array(rec.beat_period, dtype="<f8"))
    if method == "outerhmm":
        save(d / "is_continuation.npy", np.array(rec.is_continuation, dtype=bool))
    path = np.asarray(follower.alignment_path, dtype="<f8").reshape(2, -1)
    save(d / "alignment_path.npy", path)
    if match_path is not None and match_path.exists():
        write_json(d / "evaluation.json", evaluation(mm, path, match_path, perf_path))
    write_json(
        d / "meta.json",
        {
            "reference": reference_meta(),
            "generator": "tools/golden/generate.py",
            "inputs": {
                "score": score_path.name,
                "performance": perf_path.name,
                "ground_truth": match_path.name if match_path is not None else None,
            },
            "config": {
                "method": method,
                "processor": type(mm.processor).__name__,
                "piano_range": bool(getattr(mm.processor, "piano_range", True)),
                "return_pitch_list": bool(getattr(mm.processor, "return_pitch_list", False)),
                "polling_period": mm.polling_period,
                "tempo": mm.tempo,
            },
            "tolerances": TOLERANCES,
        },
    )
    print(f"  {method}: {len(rec.positions)} steps, path {path.shape[1]}, states {n_states}")


# --------------------------------------------------------------------------- variants
def make_wrongnote_midi(src: Path, dst: Path, index: int = 4, shift: int = 1) -> None:
    """Copy ``src`` with the ``index``-th note (by note-on order) shifted by ``shift`` semitones."""
    mid = mido.MidiFile(str(src))
    count, target = 0, None
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                if count == index:
                    target = (msg.channel, msg.note)
                    msg.note += shift
                    break
                count += 1
        if target is not None:
            break
    assert target is not None, "not enough notes"
    for track in mid.tracks:
        seen_on = False
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0 and (msg.channel, msg.note - shift) == target and not seen_on:
                seen_on = True
                continue
            if seen_on and msg.type in ("note_off", "note_on") and getattr(msg, "velocity", 0) == 0 and (msg.channel, msg.note) == target:
                msg.note += shift
                break
    mid.save(str(dst))


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data, out = args.data.resolve(), args.out.resolve()

    for piece in PIECES:
        print(piece)
        score_path, perf_path, match_path = (data / f"{piece}{e}" for e in (".musicxml", ".mid", ".match"))
        po = out / piece
        info = dump_score(po, score_path)
        stream = dump_stream(po, perf_path)
        dump_processors(po, stream)
        for method in METHODS:
            dump_method(po, method, score_path, perf_path, match_path)

    print("simple_mozart_k265_var1_midiscore")
    piece = "simple_mozart_k265_var1"
    po = out / f"{piece}_midiscore"
    dump_score(po, data / f"{piece}.mid")
    dump_method(po, "pthmm", data / f"{piece}.mid", data / f"{piece}.mid", None)

    print("simple_mozart_k265_var1_wrongnote")
    wrong = data / f"{piece}_wrongnote.mid"
    if not wrong.exists():
        make_wrongnote_midi(data / f"{piece}.mid", wrong)
    po = out / f"{piece}_wrongnote"
    dump_stream(po, wrong)
    dump_method(po, "outerhmm", data / f"{piece}.musicxml", wrong, None)

    for fixture in sorted(data.glob("fixture_*.musicxml")):
        print(fixture.stem)
        po = out / fixture.stem
        dump_score(po, fixture)


if __name__ == "__main__":
    main()
