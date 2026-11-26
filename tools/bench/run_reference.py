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
import hashlib
import json
import os
import platform
import sys
import time
import tomllib
from ctypes import c_int
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

def transformer_requested() -> bool:
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith("--methods="):
            return bool({"OTM", "OPTM"} & set(arg.split("=", 1)[1].split(",")))
        if arg == "--methods" and i + 2 < len(sys.argv):
            return bool({"OTM", "OPTM"} & set(sys.argv[i + 2].split(",")))
    return False


if transformer_requested():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import partitura
import soundfile as sf

import matchmaker
from matchmaker import Matchmaker
from matchmaker.features.audio import KorzeniowskiAudioProcessor
from matchmaker.utils.eval import evaluate_alignment, resolve_gt
from matchmaker.utils.symbolic import midi_messages_from_performance

DEFAULT_METHODS = ["pthmm", "hmm", "outerhmm", "arzt", "dixon", "pfkorz", "dannenberg"]
PARANGONAR_METHODS = ("OTM", "OPTM", "SLT_OLTW", "SL_OLTW")
SOUNDFONT_SHA256 = "CF2A98EB38A32C4954B4B6E2CAAE4112D62DD8E892ECEEFDD7942B0E7D01AC2F"
PROCESSORS = {
    "PitchProcessor": "pitch",
    "ChordOnsetProcessor": "chord_onset",
    "ParangonarProcessor": "parangonar",
    "ChromagramProcessor": "chroma",
    "LogSpectralEnergyProcessor": "lse",
    "RawSpectrumProcessor": "raw_spectrum",
    "CQTProcessor": "cqt",
    "CQTSpectralFluxProcessor": "cqt_spectral_flux",
    "MelSpectrogramProcessor": "mel",
    "MFCCProcessor": "mfcc",
    "KorzeniowskiAudioProcessor": "korzeniowski",
}


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


@contextmanager
def use_soundfont(soundfont: Path):
    renderer = partitura.save_wav_fluidsynth

    def pinned_renderer(*args, **kwargs):
        kwargs["soundfont"] = str(soundfont)
        return renderer(*args, **kwargs)

    with patch.object(partitura, "save_wav_fluidsynth", pinned_renderer):
        yield


def fluidsynth_version() -> str:
    import fluidsynth

    major, minor, micro = c_int(), c_int(), c_int()
    fluidsynth.fluid_version(major, minor, micro)
    return f"{major.value}.{minor.value}.{micro.value}"


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


def run_audio_piece(piece, base, out_dir, methods, runs, environment):
    if piece.get("processor") == "korzeniowski":
        selected = [
            method
            for method in methods
            if not piece.get("methods") or method in piece["methods"]
        ]
        if any(method != "korzeniowski" for method in selected):
            raise ValueError("standalone Korzeniowski extraction cannot use a vector follower")
        return run_korzeniowski_piece(piece, base, selected, runs, environment)

    score = base / piece["score"]
    performance = base / piece["performance"]
    soundfont = base / piece["soundfont"] if piece.get("soundfont") else None
    soundfont_hash = None
    if soundfont:
        soundfont_hash = hashlib.sha256(soundfont.read_bytes()).hexdigest().upper()
        if soundfont_hash != SOUNDFONT_SHA256:
            raise ValueError(f"unexpected SoundFont SHA-256 {soundfont_hash}")
    duration = float(sf.info(performance).duration)
    records = []

    for method in methods:
        if piece.get("methods") and method not in piece["methods"]:
            continue
        kwargs = dict(matchmaker.DEFAULT_KWARGS["audio"][method])
        kwargs.update(piece.get("kwargs", {}))
        if soundfont:
            kwargs["soundfont_path"] = str(soundfont)
        if piece.get("processor"):
            kwargs["processor"] = piece["processor"]
        seeds = piece.get("seeds", [1984]) if method == "pfkorz" else [None]
        if not seeds:
            raise ValueError("pfkorz benchmark seeds must not be empty")
        record_start = len(records)
        seed_paths = []
        from matchmaker.prob import particle_filter_korzeniowski as reference_pfkorz

        for seed in seeds:
            for run in range(runs):
                feature, inference = [], []
                seed_context = (
                    patch.object(reference_pfkorz, "SEED", seed)
                    if seed is not None
                    else nullcontext()
                )
                renderer_context = use_soundfont(soundfont) if soundfont else nullcontext()
                with seed_context, renderer_context:
                    mm = Matchmaker(
                        score_file=str(score),
                        performance_file=str(performance),
                        input_type="audio",
                        method=method,
                        kwargs=kwargs,
                        unfold_score=piece.get("unfold_score", True),
                    )
                    if method == "pfkorz":
                        # AudioStream uses n_fft, but this processor consumes win_length.
                        mm.stream.cache_size = mm.processor.win_length - mm.hop_length
                    processor = PROCESSORS[type(mm.processor).__name__]
                    frame_to_beat = None if piece.get("ground_truth") else mm.ref_frame_to_beat().copy()
                    tempo_model = getattr(mm.score_follower, "tempo_model", None)
                    if method == "arzt_tempo" and tempo_model is None:
                        raise RuntimeError("arzt_tempo benchmark requires an active tempo model")
                    timed(mm.processor, feature, only_results=True)
                    timed(mm.score_follower, inference)
                    for _ in mm.run(verbose=False):
                        pass
                path = np.asarray(mm.score_follower.alignment_path, dtype=float).reshape(2, -1)
                if piece.get("ground_truth"):
                    gt_perf, gt_score = resolve_gt(
                        str(base / piece["ground_truth"]),
                        mm.score_part.note_array(),
                    )
                    evaluation = evaluate_alignment(path[1], path[0], gt_score, gt_perf)
                    p95_b = float(
                        np.percentile(
                            np.abs(gt_score - np.interp(gt_perf, path[0], path[1])), 95
                        )
                    )
                    if method == "pfkorz" and run == 0:
                        seed_paths.append((seed, path, gt_perf, gt_score, evaluation, p95_b))
                else:
                    frame_indices = np.clip(path[0] * mm.frame_rate, 0, len(frame_to_beat) - 1)
                    expected_beats = np.interp(
                        frame_indices,
                        np.arange(len(frame_to_beat), dtype=float),
                        frame_to_beat,
                    )
                    evaluation = evaluate_alignment(
                        path[1], path[0], expected_beats, path[0]
                    )
                    p95_b = float(
                        np.percentile(
                            np.abs(expected_beats - np.interp(path[0], path[0], path[1])),
                            95,
                        )
                    )
                if run == 0:
                    filename = f"{method}-seed-{seed}.tsv" if seed is not None else f"{method}.tsv"
                    tsv = out_dir / "paths" / piece["dataset"] / piece["piece"] / filename
                    tsv.parent.mkdir(parents=True, exist_ok=True)
                    np.savetxt(
                        tsv,
                        path.T,
                        delimiter="\t",
                        fmt="%.17g",
                        header="perf_sec\tscore_beat",
                        comments="",
                    )
                total = sum(feature) + sum(inference)
                records.append({
                    "implementation": "python",
                    "implementation_version": matchmaker.__version__,
                    "input_type": "audio",
                    "method": method,
                    "random_seed": seed if seed is not None else getattr(tempo_model, "random_seed", None),
                    "seed": seed,
                    "unfold_score": piece.get("unfold_score", True),
                    "candidate_count": len(mm.score_follower.hypotheses) if method == "arzt_multi_fold" else 0,
                    "selection_window": getattr(mm.score_follower, "selection_window", None),
                    "switch_margin": getattr(mm.score_follower, "switch_margin", None),
                    "processor": processor,
                    "dataset": piece["dataset"],
                    "piece": piece["piece"],
                    "run": run,
                    "n_observations": len(inference),
                    "performance_duration_s": duration,
                    "latency_ms": {"feature": latency(feature), "inference": latency(inference)},
                    "total_processing_s": total,
                    "real_time_factor": total / duration if duration else None,
                    "evaluation": evaluation,
                    "p95_b": p95_b,
                    "ground_truth": bool(piece.get("ground_truth")),
                    "ground_truth_path": piece.get("ground_truth"),
                    "renderer": "none" if method == "pfkorz" else "FluidSynth",
                    "renderer_version": None if method == "pfkorz" else fluidsynth_version(),
                    "soundfont_sha256": soundfont_hash,
                    "environment": environment,
                })
                print(f"{piece['piece']} {method} seed {seed} run {run}: {len(inference)} observations, {total:.4f} s")
        if method == "pfkorz":
            pooled_path = []
            pooled_gt_perf = []
            pooled_gt_score = []
            per_seed = {}
            for index, (seed, path, gt_perf, gt_score, evaluation, p95_b) in enumerate(seed_paths):
                per_seed[str(seed)] = {"evaluation": evaluation, "p95_b": p95_b}
                offset = index * (duration + 10.0)
                shifted_path = path.copy()
                shifted_path[0] += offset
                pooled_path.append(shifted_path)
                pooled_gt_perf.append(gt_perf + offset)
                pooled_gt_score.append(gt_score)
            combined_path = np.concatenate(pooled_path, axis=1)
            combined_gt_perf = np.concatenate(pooled_gt_perf)
            combined_gt_score = np.concatenate(pooled_gt_score)
            pooled_evaluation = evaluate_alignment(
                combined_path[1], combined_path[0], combined_gt_score, combined_gt_perf
            )
            pooled_p95 = float(
                np.percentile(
                    np.abs(
                        combined_gt_score
                        - np.interp(combined_gt_perf, combined_path[0], combined_path[1])
                    ),
                    95,
                )
            )
            for record in records[record_start:]:
                record["per_seed_metrics"] = per_seed
                record["pooled_evaluation"] = pooled_evaluation
                record["pooled_p95_b"] = pooled_p95
    return records


def run_korzeniowski_piece(piece, base, methods, runs, environment):
    performance = base / piece["performance"]
    kwargs = dict(piece.get("kwargs", {}))
    sample_rate = kwargs.get("sample_rate", 44_100)
    hop_length = kwargs.get("hop_length", 512)
    win_length = kwargs.get("win_length", 2_048)
    n_fft = kwargs.get("n_fft", 4_096)
    samples, source_rate = sf.read(performance, dtype=np.float32, always_2d=True)
    if source_rate != sample_rate:
        raise ValueError(
            f"standalone Korzeniowski benchmark expects {sample_rate} Hz audio, got {source_rate}"
        )
    samples = samples.mean(axis=1, dtype=np.float32)
    padded = np.pad(samples, (0, (-len(samples)) % hop_length), mode="constant")
    duration = float(sf.info(performance).duration)
    records = []

    for method in methods:
        for run in range(runs):
            processor = KorzeniowskiAudioProcessor(
                sample_rate=sample_rate,
                hop_length=hop_length,
                win_length=win_length,
                n_fft=n_fft,
            )
            cache_size = max(0, win_length - hop_length)
            cache = np.zeros(cache_size, dtype=np.float32)
            feature = []
            for chunk_index, chunk in enumerate(padded.reshape(-1, hop_length)):
                block = np.concatenate((cache, chunk))
                frame_time = max(0, chunk_index - 1) * hop_length / sample_rate
                start = time.perf_counter()
                processor((block, frame_time))
                elapsed = time.perf_counter() - start
                if chunk_index > 0:
                    feature.append(elapsed)
                if cache_size:
                    cache = block[-cache_size:].copy()
            total = sum(feature)
            records.append({
                "implementation": "python",
                "implementation_version": matchmaker.__version__,
                "input_type": "audio",
                "method": method,
                "processor": "korzeniowski",
                "dataset": piece["dataset"],
                "piece": piece["piece"],
                "run": run,
                "n_observations": len(feature),
                "performance_duration_s": duration,
                "latency_ms": {"feature": latency(feature), "inference": latency([])},
                "total_processing_s": total,
                "real_time_factor": total / duration if duration else None,
                "evaluation": None,
                "ground_truth": False,
                "environment": environment,
            })
            print(f"{piece['piece']} {method} run {run}: {len(feature)} structured observations, {total:.4f} s")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    args = parser.parse_args()
    methods = args.methods.split(",")
    if {"OTM", "OPTM"} & set(methods):
        import torch
        torch.set_num_threads(1)

    manifest = tomllib.loads(args.manifest.read_text(encoding="utf-8"))
    base = args.manifest.resolve().parent
    out_dir = args.out.resolve().parent
    environment = {
        "os": platform.system().lower(),
        "arch": platform.machine(),
        "commit": os.environ.get("GITHUB_SHA"),
        "timestamp": int(time.time()),
        "numpy_disabled_cpu_features": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
    }

    records = []
    for p in manifest["pieces"]:
        if p.get("input_type") == "audio":
            records.extend(
                run_audio_piece(
                    p,
                    base,
                    out_dir,
                    [method for method in methods if method not in PARANGONAR_METHODS],
                    args.runs,
                    environment,
                )
            )
            continue
        score, perf = base / p["score"], base / p["performance"]
        _, msg_times = midi_messages_from_performance(str(perf))
        duration = float(msg_times[-1]) if len(msg_times) else 0.0
        for method in methods:
            first_path = None
            for run in range(args.runs):
                mm = Matchmaker(score_file=str(score), performance_file=str(perf), input_type="midi", method=method)
                feature, inference = [], []
                processor = PROCESSORS[type(mm.processor).__name__]
                timed(mm.processor, feature, only_results=True)
                timed(mm.score_follower, inference)
                for _ in mm.run(verbose=False):
                    pass
                path = np.asarray(mm.score_follower.alignment_path, dtype=float).reshape(2, -1)
                if method in PARANGONAR_METHODS:
                    if first_path is None:
                        first_path = path.copy()
                    elif not np.array_equal(path, first_path):
                        raise RuntimeError(f"{method} path changed across runs for {p['piece']}")
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
