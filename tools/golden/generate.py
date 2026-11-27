"""Generate golden data for matchmaker-rs from the locked reference release.

Layout and dtypes: specs/001-midi-score-following/contracts/golden-data.md

Usage (from tools/golden):
    python -m uv run generate.py --data ../../tests/data --out ../../tests/golden
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import types
from importlib.metadata import version
from pathlib import Path

def otm_requested() -> bool:
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("--methods="):
            return "OTM" in arg.split("=", 1)[1].split(",")
        if arg == "--methods" and i + 2 < len(sys.argv):
            return "OTM" in sys.argv[i + 2].split(",")
    return False


if otm_requested():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import mido
import numpy as np
import partitura
from partitura.score import merge_parts

import matchmaker
from matchmaker import Matchmaker
from matchmaker.features.midi import ChordOnsetProcessor, PitchProcessor, onset_pianoroll
from matchmaker.matchmaker import DEFAULT_TEMPO
from matchmaker.utils.eval import evaluate_alignment, resolve_gt
from matchmaker.utils.misc import get_tempo_from_score
from matchmaker.utils.symbolic import (
    midi_messages_from_performance,
    midi_messages_to_framed_midi,
)

REFERENCE_TAG = "v0.5.2"
PIECES = ["simple_mozart_k265_var1", "mozart_k265_var1", "Bach-fugue_bwv_858"]
METHODS = ["pthmm", "hmm", "outerhmm", "arzt", "dixon", "dannenberg"]
ALL_METHODS = [*METHODS, "OTM", "OPTM", "SLT_OLTW", "SL_OLTW"]
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
        "python_hiddenmarkov": version("python-hiddenmarkov"),
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


def dump_annotations(out: Path, part, tempo: float) -> None:
    """``Matchmaker.build_score_annotations`` for every level, unit and musical-beat setting."""
    for musical in (False, True):
        for level in ("beat", "note"):
            for return_type in ("beats", "seconds"):
                if musical and level == "note":
                    continue
                # use_musical_beat() mutates the part, so every call gets its own copy
                stub = types.SimpleNamespace(score_part=copy.deepcopy(part), tempo=tempo)
                name = f"{level}_{return_type}{'_musical' if musical else ''}.npy"
                try:
                    result = Matchmaker.build_score_annotations(stub, level, musical, return_type)
                except Exception as error:  # the reference cannot annotate every score
                    print(f"  annotations {name}: skipped ({error})")
                    continue
                save(out / "score" / "annotations" / name, np.asarray(result))


def dump_score(out: Path, score_path: Path) -> dict:
    score, part, unfolded = load_score_like_matchmaker(score_path)
    na = part.note_array()
    arr = np.zeros(len(na), dtype=NOTE_FIELDS)
    for field, _ in NOTE_FIELDS:
        arr[field] = na[field]
    save(out / "score" / "note_array.npy", arr)
    write_json(out / "score" / "note_ids.json", [str(i) for i in na["id"]])
    # reference features of arzt/dixon: registry._onset_pianoroll
    features88, onsets = onset_pianoroll(na, onset_key="onset_beat", piano_range=True)
    save(out / "score" / "onset_pianoroll.npy", features88)
    save(out / "score" / "onset_pianoroll_full.npy", onset_pianoroll(na, onset_key="onset_beat", piano_range=False)[0])
    save(out / "score" / "onset_positions.npy", onsets)
    features = coverage_report(score_path.name, score)
    score_tempo = get_tempo_from_score(part, str(score_path))
    tempo = float(score_tempo) if score_tempo is not None else float(DEFAULT_TEMPO)
    dump_annotations(out, part, tempo)
    write_json(
        out / "score" / "meta.json",
        {
            "score_tempo": score_tempo,
            "tempo": tempo,
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
        "pitch_piano_vector_0.001": (lambda: PitchProcessor(True, False), f1, False),
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
            try:
                recorder.features.append(np.asarray(observation, dtype=np.float32).copy())
            except (TypeError, ValueError):
                recorder.features.append(
                    np.asarray([observation["pitch"]], dtype="<i4").copy()
                )
            recorder.times.append(float(perf_time))
            recorder.positions.append(float(position))
            recorder.indices.append(int(self.current_index))
            probs = getattr(self, "forward_variable", None)
            if probs is None:
                probs = getattr(self, "state_probabilities", None)
            if probs is None:
                # dannenberg: the DP rating column plays the state-distribution role
                probs = getattr(self, "_values", None)
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
    if method == "OTM":
        dump_otm(out, score_path, perf_path, match_path)
        return
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
    is_list = method in ("hmm", "dannenberg")
    if method in ("OPTM", "SLT_OLTW", "SL_OLTW"):
        save(d / "observations_features.npy", np.asarray(rec.features, dtype="<i4").reshape(-1, 1))
    else:
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


def dump_otm(out: Path, score_path: Path, perf_path: Path, match_path: Path | None) -> None:
    from parangonar.match.online_matchers import ALIGNMENT_TRANSFORMER_CHECKPOINT

    if matchmaker.__version__ != "0.5.2" or version("parangonar") != "3.4.1":
        raise RuntimeError("OTM goldens require pymatchmaker 0.5.2 and parangonar 3.4.1")
    checkpoint_hash = hashlib.sha256(Path(ALIGNMENT_TRANSFORMER_CHECKPOINT).read_bytes()).hexdigest()
    if checkpoint_hash != "9ec8a1b8e6731c01a9bb359635dab3fa0b9e60045bbbf18f27227237b249af97":
        raise RuntimeError(f"unexpected OTM checkpoint SHA-256: {checkpoint_hash}")
    mm = Matchmaker(
        score_file=str(score_path), performance_file=str(perf_path),
        input_type="midi", method="OTM",
    )
    follower = mm.score_follower
    matcher = follower.matcher
    windows, logits, pitches, times, positions, indices, tempo = [], [], [], [], [], [], []
    current = []

    def capture(_model, inputs, output):
        current.append((
            inputs[0].detach().cpu().numpy()[0].copy(),
            output.detach().cpu().numpy().reshape(26, 2).copy(),
        ))

    handle = matcher.model.register_forward_hook(capture)
    base = type(follower)

    def record(self, observation, perf_time):
        current.clear()
        position = base.__call__(self, observation, perf_time)
        assert len(current) <= 1
        windows.append(current[0][0] if current else None)
        logits.append(current[0][1] if current else None)
        pitches.append(int(observation["pitch"]))
        times.append(float(perf_time))
        positions.append(float(position))
        indices.append(int(self.current_index))
        tempo.append((float(matcher._prev_score_onset), float(matcher.tempo_model.beat_period)))
        return position

    follower.__class__ = type("RecordingOTM", (base,), {"__call__": record})
    try:
        yielded = list(mm.run(verbose=False))
    finally:
        handle.remove()
    assert yielded == positions
    d = out / "OTM"
    # Keep upstream's full structured sort: extra fields break ties in np.sort.
    notes = matcher.score_note_array_full
    compact = np.empty(len(notes), dtype=[
        ("pitch", "<i4"), ("onset_beat", "<f4"), ("duration_beat", "<f4"),
        ("voice", "<i4"), ("id", "<U256"), ("is_grace", "?"),
    ])
    for field in compact.dtype.names:
        compact[field] = notes[field]
    present = np.array([window is not None for window in windows], dtype=bool)
    token_rows = np.zeros((len(windows), 26, 7), dtype="<i8")
    logit_rows = np.zeros((len(logits), 26, 2), dtype="<f4")
    for i, window in enumerate(windows):
        if window is not None:
            token_rows[i] = window
            logit_rows[i] = logits[i]
    save(d / "score_positions.npy", np.asarray(follower.score_positions, dtype="<f4"))
    save(d / "score_notes.npy", compact)
    save(d / "observations_features.npy", np.asarray(pitches, dtype="<i4"))
    save(d / "observations_times.npy", np.asarray(times, dtype="<f8"))
    save(d / "token_windows.npy", token_rows)
    save(d / "token_windows_present.npy", present)
    save(d / "model_logits.npy", logit_rows)
    save(d / "model_logits_present.npy", present)
    save(d / "tempo_state.npy", np.asarray(tempo, dtype="<f8").reshape(-1, 2))
    save(d / "current_index.npy", np.asarray(indices, dtype="<i8"))
    save(d / "positions.npy", np.asarray(positions, dtype="<f8"))
    save(d / "internal_path.npy", np.asarray(matcher._warping_path, dtype="<i8").reshape(-1, 2).T)
    path = np.asarray(follower.alignment_path, dtype="<f8").reshape(2, -1)
    save(d / "alignment_path.npy", path)
    if match_path is not None and match_path.exists():
        write_json(d / "evaluation.json", evaluation(mm, path, match_path, perf_path))
    write_json(d / "meta.json", {
        "reference": {**reference_meta(), "parangonar": version("parangonar")},
        "generator": "tools/golden/generate.py",
        "inputs": {"score": score_path.name, "performance": perf_path.name,
                   "ground_truth": match_path.name if match_path else None},
        "config": {"method": "OTM", "processor": "ParangonarProcessor",
                   "polling_period": None, "device": "cpu"},
        "model": {"format": "pytorch checkpoint", "format_version": "parangonar-3.4.1",
                  "source": "https://github.com/sildater/parangonar/blob/3.4.1/parangonar/assets/alignment_transformer_checkpoint.pt",
                  "sha256": checkpoint_hash},
        "tolerances": {"model_logits_atol": 1e-4, "model_logits_rtol": 1e-4,
                       "positions_atol": 1e-6, "alignment_path_atol": 1e-6,
                       "tempo_state_atol": 1e-6, "tokens": "exact"},
    })
    print(f"  OTM: {len(positions)} steps, {present.sum()} model calls, path {path.shape[1]}")


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
    parser.add_argument("--methods", default=",".join(METHODS),
                        help="comma-separated methods; OTM is opt-in")
    args = parser.parse_args()
    data, out = args.data.resolve(), args.out.resolve()
    methods = args.methods.split(",")
    if not methods or any(method not in ALL_METHODS for method in methods):
        parser.error(f"--methods must contain only {', '.join(ALL_METHODS)}")
    if "OTM" in methods:
        import torch
        torch.set_num_threads(1)
    otm_only = methods == ["OTM"]

    for piece in PIECES:
        print(piece)
        score_path, perf_path, match_path = (data / f"{piece}{e}" for e in (".musicxml", ".mid", ".match"))
        po = out / piece
        if not otm_only:
            dump_score(po, score_path)
            stream = dump_stream(po, perf_path)
            dump_processors(po, stream)
        for method in methods:
            dump_method(po, method, score_path, perf_path, match_path)

    print("simple_mozart_k265_var1_midiscore")
    piece = "simple_mozart_k265_var1"
    po = out / f"{piece}_midiscore"
    if not otm_only:
        dump_score(po, data / f"{piece}.mid")
    for method in methods:
        dump_method(po, method, data / f"{piece}.mid", data / f"{piece}.mid", None)

    print("simple_mozart_k265_var1_wrongnote")
    wrong = data / f"{piece}_wrongnote.mid"
    if not wrong.exists():
        if otm_only:
            raise FileNotFoundError(f"OTM generation requires the existing fixture {wrong}")
        make_wrongnote_midi(data / f"{piece}.mid", wrong)
    po = out / f"{piece}_wrongnote"
    if not otm_only:
        dump_stream(po, wrong)
    for method in methods:
        dump_method(po, method, data / f"{piece}.musicxml", wrong, None)

    if not otm_only:
        for fixture in sorted(data.glob("fixture_*.musicxml")):
            print(fixture.stem)
            po = out / fixture.stem
            dump_score(po, fixture)
    if "OTM" in methods:
        fixture = data / "otm" / "grace.musicxml"
        dump_otm(out / "otm_grace", fixture, fixture.with_suffix(".mid"), None)


if __name__ == "__main__":
    main()
